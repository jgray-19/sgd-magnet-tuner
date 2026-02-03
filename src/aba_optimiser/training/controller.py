"""Refactored controller for orchestrating multi-process knob optimisation."""

from __future__ import annotations

import dataclasses
import gc
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.utils import normalize_true_strengths
from aba_optimiser.training.worker_manager import WorkerManager

if TYPE_CHECKING:
    from tensorboardX import SummaryWriter

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.controller_config import (
        MeasurementConfig,
        SequenceConfig,
    )

logger = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


class Controller(BaseController):
    """
    Orchestrates multi-process knob optimisation using MAD-NG.

    Extends BaseController with tracking-specific functionality including
    data management and worker coordination for multi-turn tracking.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        sequence_config: SequenceConfig,
        measurement_config: MeasurementConfig,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        plots_dir: Path | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
    ):
        """
        Initialise the controller with all required managers.

        Args:
            accelerator (Accelerator): Accelerator instance defining machine configuration.
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration.
            simulation_config (SimulationConfig): Simulation and worker configuration.
            sequence_config (SequenceConfig): Sequence and beam configuration.
            measurement_config (MeasurementConfig): Measurement data file configuration.
            bpm_start_points (list[str]): Starting BPM names for each range.
            bpm_end_points (list[str]): Ending BPM names for each range.
            show_plots (bool, optional): Whether to show plots. Defaults to True.
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths.
            true_strengths (Path | dict[str, float], optional): True strengths file or dict.
            plots_dir (Path | None, optional): Directory to save plots. Defaults to plots/.
            debug (bool, optional): Enable debug mode. Defaults to False.
            mad_logfile (Path | None, optional): Path to MAD log file. Defaults to None.
        """

        # Log optimisation targets
        accelerator.log_optimisation_targets()
        if simulation_config.optimise_momenta:
            logger.info("Including momenta (px, py) in loss function")
        else:
            logger.info("Using position-only optimisation (x, y only)")

        # Normalize and validate multi-config inputs
        measurement_files, corrector_files, tune_knobs_files, machine_deltaps = (
            self._validate_config_inputs(
                measurement_config.measurement_files,
                measurement_config.corrector_files,
                measurement_config.tune_knobs_files,
                measurement_config.machine_deltaps,
            )
        )
        self.measurement_files = measurement_files
        self.corrector_files = corrector_files
        self.tune_knobs_files = tune_knobs_files
        self.machine_deltaps = machine_deltaps
        self.num_configs = len(measurement_files)
        self.plots_dir = plots_dir

        # Initialize base controller (handles config manager, optimisation loop, result manager)
        super().__init__(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config.magnet_range,
            bpm_start_points,
            bpm_end_points,
            show_plots,
            initial_knob_strengths=None,  # Will be set later after true_strengths processing
            true_strengths=None,  # Will be processed separately for tracking-specific logic
            bad_bpms=sequence_config.bad_bpms,
            first_bpm=sequence_config.first_bpm,
            debug=debug,
            mad_logfile=mad_logfile,
        )

        # Initialize tracking-specific managers
        self._init_data_manager(
            measurement_config.bunches_per_file, measurement_config.flattop_turns
        )

        self._init_worker_manager(
            sequence_config.magnet_range,
            sequence_config.bad_bpms,
            measurement_config.flattop_turns,
            measurement_config.bunches_per_file,
        )
        # Process true strengths with tracking-specific transformations
        true_strengths_dict = self._process_true_strengths(true_strengths)
        initial_knob_strengths = self._process_initial_knobs(initial_knob_strengths)

        # Initialize knobs using processed strengths
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(
                true_strengths_dict, initial_knob_strengths
            )
        )

        if not true_strengths_dict:
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Re-initialize managers with tracking-specific configurations
        self._init_managers_with_tracking_config()

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process."""
        run_start = time.time()
        writer = self.setup_logging("tracking_opt")
        total_turns = self.data_manager.get_total_turns()

        try:
            self.worker_manager.start_workers(
                self.data_manager.track_data,
                self.data_manager.turn_batches,
                self.data_manager.file_map,
                self.config_manager.start_bpms,
                self.config_manager.end_bpms,
                self.simulation_config,
                self.machine_deltaps,
            )

            # Clean up memory after workers are started
            self._cleanup_memory()

            self.final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs,
                self.worker_manager.parent_conns,
                writer,
                run_start,
                total_turns,
            )

            total_hessian = self.worker_manager.termination_and_hessian(len(self.final_knobs))
        except RuntimeError as e:
            logger.error(f"optimisation failed: {e}")
            self.worker_manager.terminate_workers()
            raise RuntimeError(f"Worker error during optimisation: {e}") from e
        except KeyboardInterrupt:
            logger.warning("\nKeyboardInterrupt detected. Terminating early and writing results.")
            self.worker_manager.terminate_workers()
            self.final_knobs = self.initial_knobs
            total_hessian = np.zeros((len(self.final_knobs), len(self.final_knobs)))
        uncertainties = self._save_results(total_hessian, writer)
        uncertainties = dict(zip(self.final_knobs.keys(), uncertainties))

        return self.final_knobs, uncertainties

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager
        gc.collect()

    def _convert_pt2dp(self, uncertainties: np.ndarray) -> None:
        self.final_knobs["deltap"] = self.config_manager.mad_iface.pt2dp(self.final_knobs.pop("pt"))
        if "pt" in self.config_manager.initial_strengths:
            self.filtered_true_strengths["deltap"] = self.config_manager.mad_iface.pt2dp(
                self.filtered_true_strengths.pop("pt")
            )

        uncertainties[-1] = self.config_manager.mad_iface.pt2dp(uncertainties[-1])

    def _save_results(
        self,
        total_hessian: np.ndarray,
        writer: SummaryWriter,
    ) -> np.ndarray:
        """Clean up resources and save final results."""
        # Calculate uncertainties
        cov = np.linalg.inv(total_hessian + 1e-8 * np.eye(total_hessian.shape[0]))
        uncertainties = np.sqrt(np.diag(cov))

        # Close logging and save results
        writer.close()

        # Convert the knobs back from pt to dp if we had optimised energy
        if "pt" in self.final_knobs:
            self._convert_pt2dp(uncertainties)

        # Save and plot using the final knobs (not the initial ones)
        self.result_manager.save_results(
            self.final_knobs, uncertainties, self.filtered_true_strengths
        )
        self.result_manager.generate_plots(
            self.final_knobs,
            self.config_manager.initial_strengths,
            self.filtered_true_strengths,
            uncertainties,
        )

        logger.info("Optimisation complete.")
        return uncertainties

    @staticmethod
    def _validate_config_inputs(
        measurement_files, corrector_files, tune_knobs_files, machine_deltaps
    ) -> tuple:
        """Validate and normalise multi-configuration inputs."""
        # Validate and expand lists
        num_configs = len(measurement_files)
        for name, lst in [
            ("corrector_files", corrector_files),
            ("tune_knobs_files", tune_knobs_files),
            ("machine_deltaps", machine_deltaps),
        ]:
            if len(lst) == 1:
                lst *= num_configs
            elif len(lst) != num_configs:
                raise ValueError(
                    f"Number of {name} ({len(lst)}) must match number of measurement files ({num_configs}) or be 1"
                )

        return measurement_files, corrector_files, tune_knobs_files, machine_deltaps

    def _init_data_manager(self, num_tracks: int, flattop_turns: int) -> None:
        """Initialize data manager and load track data."""
        self.data_manager = DataManager(
            self.config_manager.bpms_in_range,
            self.config_manager.all_bpms,
            self.simulation_config,
            self.measurement_files,
            num_bunches=num_tracks,
            flattop_turns=flattop_turns,
        )

        # Load track data and prepare batches
        self.data_manager.load_track_data()
        self.data_manager.prepare_turn_batches(self.config_manager)

        # Adjust num_batches to not exceed tracks_per_worker
        self.simulation_config = dataclasses.replace(
            self.simulation_config,
            num_batches=min(
                self.simulation_config.num_batches, self.data_manager.tracks_per_worker
            ),
        )

    def _init_worker_manager(
        self,
        magnet_range: str,
        bad_bpms: list[str] | None,
        flattop_turns: int,
        num_tracks: int,
    ) -> None:
        """Initialize worker manager for tracking workers."""
        # Set worker logging level
        import logging

        logging.getLogger("aba_optimiser.workers").setLevel(
            self.simulation_config.worker_logging_level
        )

        self.worker_manager = WorkerManager(
            self.config_manager.calculate_n_data_points(),
            ybpm=magnet_range.split("/")[0],  # Assume start bpm has largest vertical kick
            magnet_range=magnet_range,
            fixed_start=self.config_manager.fixed_start,
            fixed_end=self.config_manager.fixed_end,
            accelerator=self.accelerator,
            corrector_strengths_files=self.corrector_files,
            tune_knobs_files=self.tune_knobs_files,
            all_bpms=self.config_manager.all_bpms,
            bad_bpms=bad_bpms,
            flattop_turns=flattop_turns,
            num_tracks=num_tracks,
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            debug=self.debug,
            mad_logfile=self.mad_logfile,
        )

    def _process_true_strengths(self, true_strengths) -> dict[str, float]:
        """Process true strengths with tracking-specific transformations."""
        true_strengths_dict = normalize_true_strengths(true_strengths)

        # Handle deltap to pt conversion
        if isinstance(true_strengths, Path):
            true_strengths_dict["pt"] = 0.0

        if "deltap" in true_strengths_dict:
            if len(self.measurement_files) > 1:
                logger.warning(
                    "Ignoring provided 'deltap' in true strengths, can differ per measurement file, setting to 0.0"
                )
                true_strengths_dict["pt"] = 0.0
            else:
                true_strengths_dict["pt"] = self.config_manager.mad_iface.dp2pt(
                    true_strengths_dict.pop("deltap")
                )
        return self.accelerator.normalise_true_strengths(
            true_strengths_dict, self.config_manager.bend_lengths
        )

    def _process_initial_knobs(self, initial_knob_strengths) -> dict[str, float] | None:
        """Process initial knob strengths, converting deltap to pt if present."""
        if initial_knob_strengths is not None and "deltap" in initial_knob_strengths:
            initial_knob_strengths = initial_knob_strengths.copy()
            initial_knob_strengths["pt"] = self.config_manager.mad_iface.dp2pt(
                initial_knob_strengths.pop("deltap")
            )
        return initial_knob_strengths

    def _init_managers_with_tracking_config(self) -> None:
        """Re-initialize optimisation and result managers with tracking-specific config."""
        from aba_optimiser.training.optimisation_loop import OptimisationLoop
        from aba_optimiser.training.result_manager import ResultManager

        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.optimiser_config,
            self.simulation_config,
        )

        # Replace "pt" with "deltap" in result manager if optimizing energy
        deltap_knob_names = self.accelerator.format_result_knob_names(
            self.config_manager.knob_names
        )

        self.result_manager = ResultManager(
            deltap_knob_names,
            self.config_manager.elem_spos,
            show_plots=self.show_plots,
            accelerator=self.accelerator,
            plots_dir=self.plots_dir,
        )
