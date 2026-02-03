"""Base controller class with shared functionality for all controllers."""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tensorboardX import SummaryWriter

from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.utils import filter_bad_bpms, normalize_true_strengths

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import OptimiserConfig, SimulationConfig

LOGGER = logging.getLogger(__name__)


class BaseController(ABC):
    """Base class for all optimisation controllers.

    Provides shared functionality for:
    - Configuration management
    - Optimisation loop setup
    - Result management
    - Logging setup
    - Common initialization patterns
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        magnet_range: str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        first_bpm: str | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
        optimise_knobs: list[str] | None = None,
    ):
        """Initialize base controller.

        Args:
            accelerator: Accelerator instance defining machine configuration
            optimiser_config: Gradient descent optimiser configuration
            simulation_config: Simulation and worker configuration
            magnet_range: Magnet range specification
            bpm_start_points: Start BPMs for each range
            bpm_end_points: End BPMs for each range
            show_plots: Whether to show plots
            initial_knob_strengths: Initial knob strengths
            true_strengths: True strengths (Path, dict, or None)
            bad_bpms: List of bad BPMs to exclude
            first_bpm: First BPM in the sequence
            optimise_knobs: List of global knob names to optimise, or None
        """
        self.optimiser_config = optimiser_config
        self.simulation_config = simulation_config
        self.accelerator = accelerator
        self.show_plots = show_plots
        self.debug = debug
        self.mad_logfile = mad_logfile

        # Filter bad BPMs
        bpm_start_points, bpm_end_points = filter_bad_bpms(
            bpm_start_points, bpm_end_points, bad_bpms
        )

        # Initialize configuration manager
        self.config_manager = ConfigurationManager(
            accelerator,
            simulation_config,
            magnet_range,
            bpm_start_points,
            bpm_end_points,
            optimise_knobs,
        )
        self.config_manager.setup_mad_interface(
            first_bpm,
            bad_bpms,
            debug,
            mad_logfile,
        )

        # Normalize and initialize knob strengths
        true_strengths_dict = normalize_true_strengths(true_strengths)
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(
                true_strengths_dict, initial_knob_strengths
            )
        )

        # Use initial knobs as true strengths if none provided
        if not true_strengths_dict:
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Initialize managers
        self._init_managers()

    def _init_managers(self) -> None:
        """Initialize optimisation loop and result manager."""
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.optimiser_config,
            self.simulation_config,
        )

        self.result_manager = ResultManager(
            self.config_manager.knob_names,
            self.config_manager.elem_spos,
            show_plots=self.show_plots,
            accelerator=self.accelerator,
        )

    def setup_logging(self, log_suffix: str = "opt") -> SummaryWriter:
        """Set up TensorBoard logging.

        Args:
            log_suffix: Suffix for the log directory name

        Returns:
            TensorBoard SummaryWriter instance
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{timestamp}_{log_suffix}")

    @abstractmethod
    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process.

        Returns:
            Tuple of (final_knobs, uncertainties)
        """
        pass


# class LHCControllerMixin:
#     """Mixin providing LHC-specific configuration helpers."""

#     @staticmethod
#     def get_lhc_config(beam: int, sequence_path: Path | None = None) -> dict[str, str]:
#         """Get LHC-specific configuration for a beam.

#         Args:
#             beam: Beam number (1 or 2)
#             sequence_path: Optional custom sequence path

#         Returns:
#             Dictionary with 'sequence_file_path', 'first_bpm', and 'seq_name'
#         """
#         from aba_optimiser.io.utils import get_lhc_file_path

#         sequence_file = str(get_lhc_file_path(beam) if sequence_path is None else sequence_path)
#         first_bpm = "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"
#         seq_name = f"lhcb{beam}"

#         return {
#             "sequence_file_path": sequence_file,
#             "first_bpm": first_bpm,
#             "seq_name": seq_name,
#         }
