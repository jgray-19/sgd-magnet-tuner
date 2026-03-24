"""Abstract base class for all worker process types.

This module defines the core worker interface that all specific worker
implementations must follow. It handles the process lifecycle, MAD-NG
interface setup, and communication protocol with the main process.
"""

from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import TYPE_CHECKING, Generic, TypeVar

from aba_optimiser.mad import GradientDescentMadInterface

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy as np
    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig
    from aba_optimiser.workers.protocol import WorkerErrorPayload

LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")


class AbstractWorker(Process, ABC, Generic[WorkerDataType]):
    """Abstract base class for all worker process implementations.

    This class provides the core infrastructure for running optimisation workers
    in separate processes. It handles:
    - Process lifecycle management
    - MAD-NG interface initialization
    - Communication with the main process via pipes
    - Common configuration and logging

    Subclasses must implement:
    - setup_mad_sequence(): Configure MAD-NG sequence parameters
    - send_initial_conditions(): Initialize particle states in MAD-NG
    - compute_gradients_and_loss(): Core computation logic
    - prepare_data(): Process and prepare input data

    Type Parameters:
        WorkerDataType: The type of data structure this worker uses
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: WorkerDataType,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        """Initialize the worker process.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: Worker-specific data (tracking or optics)
            config: Configuration parameters
            simulation_config: Simulation configuration settings
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn
        self.config = config
        self.simulation_config = simulation_config
        self.bpm_range = f"{config.start_bpm}/{config.end_bpm}"

        self.tracking_range = self.bpm_range
        if config.sdir < 0:
            self.tracking_range = f"{config.end_bpm}/{config.start_bpm}"

        LOGGER.debug(
            f"Initializing worker {worker_id} for BPM range {config.start_bpm} -> {config.end_bpm}"
        )

        # Let subclasses process their specific data
        self.prepare_data(data)

    @abstractmethod
    def prepare_data(self, data: WorkerDataType) -> None:
        """Process and prepare worker-specific data.

        This method should extract relevant data from the input structure,
        compute weights, split into batches, etc.

        Args:
            data: Worker-specific data structure
        """
        pass

    @abstractmethod
    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for this worker type.

        This method should set worker-specific MAD-NG variables like
        number of turns, tracking range, etc.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def send_initial_conditions(self, mad: MAD) -> None:
        """Send initial particle/optics conditions to MAD-NG.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute gradients and loss for given knob values.

        This is the core computation method that runs tracking/optics
        calculations and computes the gradient of the loss function.

        Args:
            mad: MAD-NG interface object
            knob_updates: Dictionary of knob names to values
            batch: Batch index for multi-batch processing

        Returns:
            Tuple of (gradient_array, loss_value)
        """
        pass

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """Create a base differential algebra (DA) map in MAD-NG.

        The DA map is used for automatic differentiation of tracking
        with respect to optimisation knobs.

        Args:
            mad: MAD-NG interface object
            knob_order: Order of the DA expansion (1 for linear, 2 for quadratic)
        """
        mad.send("coord_names = {'x', 'px', 'y', 'py', 't', 'pt'}")
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, "
            f"mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def build_error_payload(self, exc: BaseException, *, phase: str) -> WorkerErrorPayload:
        """Build a structured error payload for parent-side handling."""
        return {
            "worker_id": self.worker_id,
            "status": "error",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }

    def send_error_payload(self, exc: BaseException, *, phase: str) -> None:
        """Best-effort send of a structured worker failure message."""
        payload = self.build_error_payload(exc, phase=phase)
        LOGGER.error(
            "Worker %s failed during %s: %s",
            self.worker_id,
            phase,
            payload["error"],
        )
        try:
            self.conn.send(payload)
        except (BrokenPipeError, EOFError, OSError):
            LOGGER.exception(
                "Worker %s could not send error payload to parent during %s",
                self.worker_id,
                phase,
            )

    def _resolve_per_worker_logfile(self, logfile_path):
        """Return a per-worker logfile path derived from a base logfile path."""
        if logfile_path is None:
            return None

        if logfile_path.suffix:
            return logfile_path.with_name(
                f"{logfile_path.stem}_worker_{self.worker_id}{logfile_path.suffix}"
            )
        return logfile_path.with_name(f"{logfile_path.name}_worker_{self.worker_id}")

    def configure_python_worker_logging(self) -> None:
        """Attach a file handler so worker Python logs land in the worker logfile."""
        worker_logfile = self._resolve_per_worker_logfile(
            self.config.python_logfile or self.config.mad_logfile
        )
        if worker_logfile is None:
            return

        worker_logfile.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        level = self.simulation_config.worker_logging_level
        root_logger.setLevel(level)

        if any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == str(worker_logfile)
            for handler in root_logger.handlers
        ):
            return

        file_handler = logging.FileHandler(worker_logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "PYTHON: %(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

        LOGGER.warning(
            "Worker %s Python logging attached to %s",
            self.worker_id,
            worker_logfile,
        )

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """Initialize and configure the MAD-NG interface.

        This method uses the accelerator's factory method to create a properly
        configured MAD interface, eliminating the need to manually pass many
        individual parameters.

        Args:
            init_knobs: Initial values for all optimisation knobs

        Returns:
            Tuple of (MAD interface object, number of BPMs)

        Raises:
            ValueError: If knob names from MAD don't match initial knobs
        """
        LOGGER.debug(f"Worker {self.worker_id}: Setting up MAD interface")
        LOGGER.debug(f"Worker {self.worker_id}: Using BPM range {self.bpm_range}")

        worker_logfile = self._resolve_per_worker_logfile(self.config.mad_logfile)

        # Use accelerator factory to create MAD interface
        init_bpm = self.config.start_bpm if self.config.sdir > 0 else self.config.end_bpm
        mad_iface = GradientDescentMadInterface(
            accelerator=self.config.accelerator,
            magnet_range=self.config.magnet_range,
            bpm_range=self.bpm_range,
            corrector_strengths=self.config.corrector_strengths,
            tune_knobs_file=self.config.tune_knobs_file,
            bad_bpms=self.config.bad_bpms,
            debug=self.config.debug,
            mad_logfile=worker_logfile,
            py_name="python",
            start_bpm=init_bpm,  # Workers use hardcoded "python" in their MAD scripts
        )

        knob_names = mad_iface.knob_names
        if knob_names != list(init_knobs.keys()):
            raise ValueError(
                f"Worker {self.worker_id}: Knob names from MAD {knob_names} "
                f"do not match initial knobs {list(init_knobs.keys())}"
            )

        mad = mad_iface.mad
        mad["knob_names"] = knob_names
        mad["nbpms"] = mad_iface.nbpms
        mad["sdir"] = self.config.sdir

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix", "vector")
        mad.load("MAD.utility", "tblcat")

        # Call worker-specific sequence setup
        self.setup_mad_sequence(mad)

        # Setup differential algebra maps
        self._setup_da_maps(mad)

        return mad, mad_iface.nbpms

    @abstractmethod
    def _setup_da_maps(self, mad: MAD) -> None:
        """Setup differential algebra maps specific to worker type.

        Args:
            mad: MAD-NG interface object
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Main worker run loop.

        This method handles the communication protocol:
        1. Wait for initial handshake
        2. Receive initial knob values
        3. Setup MAD interface
        4. Loop: receive knobs -> compute -> send results
        5. Cleanup on termination signal (None received)
        """
        pass

    @abstractmethod
    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for computation.

        This method should load any initialisation scripts needed
        before the main computation loop.

        Args:
            mad: MAD-NG interface object
        """
        pass
