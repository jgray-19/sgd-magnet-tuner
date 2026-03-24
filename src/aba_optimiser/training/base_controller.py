"""Base controller class with shared functionality for all controllers."""

from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tensorboardX import SummaryWriter

from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.controller_config import OutputConfig
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.utils import filter_bad_bpms, normalise_true_strengths

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

    Design: Delta-space only internally. All user inputs are expected in absolute-space
    and are converted to delta-space during initialization. All internal algorithms work
    exclusively in delta-space. Results in subclasses are converted back to absolute-space
    at the user-facing boundary.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        magnet_range: str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        first_bpm: str | None = None,
        debug: bool = False,
        optimise_knobs: list[str] | None = None,
        output_config: OutputConfig | None = None,
    ):
        """Initialize base controller.

        User inputs (initial_knob_strengths, true_strengths) are expected in absolute-space
        and are automatically converted to delta-space for internal use.

        Args:
            accelerator: Accelerator instance defining machine configuration
            optimiser_config: Gradient descent optimiser configuration
            simulation_config: Simulation and worker configuration
            magnet_range: Magnet range specification
            bpm_start_points: Start BPMs for each range
            bpm_end_points: End BPMs for each range
            initial_knob_strengths: Initial knob strengths (absolute-space). Missing keys
                default to 1e-7 in delta-space after conversion.
            true_strengths: True strengths (Path, dict, or None) in absolute-space
            bad_bpms: List of bad BPMs to exclude
            first_bpm: First BPM in the sequence
            debug: Enable debug mode
            optimise_knobs: List of global knob names to optimise, or None
            output_config: Output and logging configuration. Defaults to OutputConfig().
        """
        self.optimiser_config = optimiser_config
        self.simulation_config = simulation_config
        self.accelerator = accelerator
        self.debug = debug
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.show_plots = self.output_config.show_plots
        self.mad_logfile: Path | None = self.output_config.mad_logfile
        self.python_logfile: Path | None = self.output_config.python_logfile

        if not accelerator.has_any_optimisation():
            raise ValueError("No optimisation types enabled in the accelerator configuration.")

        # Filter bad BPMs
        bpm_start_points, bpm_end_points = filter_bad_bpms(
            bpm_start_points, bpm_end_points, bad_bpms
        )
        LOGGER.warning(f"After filtering bad BPMs, using BPM start points: {bpm_start_points}, end points: {bpm_end_points}")

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
            self.mad_logfile,
        )

        # Convert user-space (absolute) inputs to internal delta-space
        true_strengths_dict = normalise_true_strengths(true_strengths)
        true_strengths_delta = self._convert_true_strengths_to_delta(true_strengths_dict)
        initial_knobs_delta = self._convert_initial_knobs_to_delta(initial_knob_strengths)

        # Initialize knob strengths in delta-space
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(true_strengths_delta, initial_knobs_delta)
        )
        self._validate_knob_initialisation()

        # Use initial knobs as true strengths if none provided
        if not true_strengths_delta:
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Initialize managers
        self._init_managers()

    def _convert_true_strengths_to_delta(
        self, true_strengths: dict[str, float]
    ) -> dict[str, float]:
        """Convert user-provided true strengths (absolute) to delta-space for internal use.

        Subclasses can override to add special handling (e.g., energy parameter conversions).
        """
        if not true_strengths:
            return {}
        # Convert from absolute to delta-space
        return self.config_manager.mad_iface.absolute_to_optimisation_knobs(true_strengths)

    def _convert_initial_knobs_to_delta(
        self, initial_knob_strengths: dict[str, float] | None
    ) -> dict[str, float] | None:
        """Convert user-provided initial knob strengths (absolute) to delta-space for internal use.

        Subclasses can override to add special handling (e.g., energy parameter conversions).
        """
        if initial_knob_strengths is None:
            return None
        # Convert from absolute to delta-space
        return self.config_manager.mad_iface.absolute_to_optimisation_knobs(initial_knob_strengths)

    def _init_managers(self) -> None:
        """Initialize optimisation loop and result manager."""
        abs_offsets, dabs_dopt = self.config_manager.mad_iface.optimisation_to_absolute_affine(
            self.config_manager.knob_names
        )
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.optimiser_config,
            self.simulation_config,
            abs_offsets=abs_offsets,
            dabs_dopt=dabs_dopt,
        )

        output_knob_names = self.config_manager.mad_iface.format_knob_names_for_output(
            self.config_manager.knob_names
        )
        output_knob_names = self.accelerator.format_result_knob_names(output_knob_names)

        self.result_manager = ResultManager(
            output_knob_names,
            self.config_manager.elem_spos,
            show_plots=self.show_plots,
            accelerator=self.accelerator,
        )

    def _validate_knob_initialisation(self) -> None:
        """Validate that controller setup produced a usable knob set."""
        knob_names = self.config_manager.knob_names
        if not knob_names:
            raise ValueError(
                "No optimisation knobs were created for this controller configuration. "
                f"Optimisation is enabled, but the MAD model returned zero knobs for "
                f"magnet range '{self.config_manager.magnet_range}'. Check that the "
                "selected optimisation flags match elements present in the loaded "
                "sequence and range."
            )

        if len(self.initial_knobs) != len(knob_names):
            raise ValueError(
                "Knob initialisation produced an inconsistent result: "
                f"{len(knob_names)} knob names but {len(self.initial_knobs)} initial values."
            )

    def setup_logging(self, log_suffix: str = "opt") -> SummaryWriter | None:
        """Set up TensorBoard logging.

        Args:
            log_suffix: Suffix for the log directory name

        Returns:
            TensorBoard SummaryWriter instance or None when disabled
        """
        if not self.output_config.write_tensorboard_logs:
            LOGGER.info("TensorBoard logging disabled")
            return None

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
