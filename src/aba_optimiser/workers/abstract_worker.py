"""Abstract base class for all worker process types.

This module defines the core worker interface that all specific worker
implementations must follow. It handles the process lifecycle, MAD-NG
interface setup, and communication protocol with the main process.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import TYPE_CHECKING, Generic, TypeVar

from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy as np
    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig
LOGGER = logging.getLogger(__name__)

# Type variable for worker data type
WorkerDataType = TypeVar("WorkerDataType")


class AbstractWorker(Process, ABC, Generic[WorkerDataType]):
    """Abstract base class for all worker process implementations.

    This class provides the core infrastructure for running optimization workers
    in separate processes. It handles:
    - Process lifecycle management
    - MAD-NG interface initialization
    - Communication with the main process via pipes
    - Common configuration and logging

    Subclasses must implement:
    - get_bpm_range(): Define BPM range for tracking
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
    def get_bpm_range(self, sdir: int) -> str:
        """Get the BPM range string for MAD-NG tracking.

        Args:
            sdir: Direction of propagation (+1 forward, -1 backward)

        Returns:
            BPM range string in MAD-NG format (e.g., "BPM1/BPM2")
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
        with respect to optimization knobs.

        Args:
            mad: MAD-NG interface object
            knob_order: Order of the DA expansion (1 for linear, 2 for quadratic)
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, "
            f"mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """Initialize and configure the MAD-NG interface.

        This method creates the OptimisationMadInterface, loads required
        modules, and sets up the differential algebra maps for gradient
        computation.

        Args:
            init_knobs: Initial values for all optimization knobs

        Returns:
            Tuple of (MAD interface object, number of BPMs)

        Raises:
            ValueError: If knob names from MAD don't match initial knobs
        """
        LOGGER.debug(f"Worker {self.worker_id}: Setting up MAD interface")

        bpm_range = self.get_bpm_range(sdir=1)
        LOGGER.debug(f"Worker {self.worker_id}: Using BPM range {bpm_range}")

        mad_iface = OptimisationMadInterface(
            self.config.sequence_file_path,
            py_name="python",
            seq_name=self.config.seq_name,
            magnet_range=self.config.magnet_range,
            bpm_range=bpm_range,
            simulation_config=self.simulation_config,
            bad_bpms=self.config.bad_bpms,
            corrector_strengths=self.config.corrector_strengths,
            tune_knobs_file=self.config.tune_knobs_file,
            beam_energy=self.config.beam_energy,
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

    def run(self) -> None:
        """Main worker run loop.

        This method handles the communication protocol:
        1. Wait for initial handshake
        2. Receive initial knob values
        3. Setup MAD interface
        4. Loop: receive knobs -> compute -> send results
        5. Cleanup on termination signal (None received)
        """
        # Initial handshake
        self.conn.recv()
        knob_values, batch = self.conn.recv()

        # Setup MAD interface
        mad, nbpms = self.setup_mad_interface(knob_values)

        # Send initial conditions
        self.send_initial_conditions(mad)

        # Initialize MAD environment for computation
        self._initialize_mad_computation(mad)

        LOGGER.debug(f"Worker {self.worker_id}: Ready for computation with {nbpms} BPMs")

        # Main computation loop
        while knob_values is not None:
            # Compute gradients and loss
            grad, loss = self.compute_gradients_and_loss(mad, knob_values, batch)

            # Normalize and send results
            self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))

            # Wait for next knob values
            knob_values, batch = self.conn.recv()

        # Cleanup
        LOGGER.debug(f"Worker {self.worker_id}: Terminating")
        del mad

    @abstractmethod
    def _initialize_mad_computation(self, mad: MAD) -> None:
        """Initialize MAD-NG environment for computation.

        This method should load any initialization scripts needed
        before the main computation loop.

        Args:
            mad: MAD-NG interface object
        """
        pass
