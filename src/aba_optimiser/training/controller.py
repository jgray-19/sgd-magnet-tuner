"""Refactored controller for orchestrating multi-process knob optimisation."""

from __future__ import annotations

import dataclasses
import gc
import logging
import random
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.controller_config import OutputConfig
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.worker_manager import WorkerManager

if TYPE_CHECKING:
    from pathlib import Path

    from tensorboardX import SummaryWriter

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.controller_config import (
        CheckpointConfig,
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

    Design: Delta-space only internally. All user inputs (initial_knob_strengths, true_strengths)
    are converted from absolute-space to delta-space at entry. All internal algorithms operate
    exclusively in delta-space. Results are converted back to absolute-space only at exit
    (in _save_results). This ensures the optimization logic never branches on absolute vs delta
    assumptions and remains simple and unified.

    Default behavior: Any knobs not explicitly provided by the user default to 1e-7 in delta-space
    to avoid flat optimization starts.
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
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        debug: bool = False,
        optimise_knobs: list[str] | None = None,
        output_config: OutputConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
    ):
        """
        Initialise the controller with all required managers.

        User inputs are in absolute-space; they are automatically converted to delta-space
        for internal optimization and converted back to absolute-space when returning results.

        Args:
            accelerator (Accelerator): Accelerator instance defining machine configuration.
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration.
            simulation_config (SimulationConfig): Simulation and worker configuration.
            sequence_config (SequenceConfig): Sequence and beam configuration.
            measurement_config (MeasurementConfig): Measurement data file configuration.
            output_config (OutputConfig): Output/logging behaviour.
            bpm_start_points (list[str]): Starting BPM names for each range.
            bpm_end_points (list[str]): Ending BPM names for each range.
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths (absolute-space).
                Missing keys default to 1e-7 in delta-space after conversion.
            true_strengths (Path | dict[str, float], optional): True strengths file or dict (absolute-space).
            debug (bool, optional): Enable debug mode. Defaults to False.
            optimise_knobs (list[str] | None, optional): List of global knob names to optimise.
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
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.checkpoint_config = checkpoint_config

        # Initialise base controller with actual values.
        # BaseController will call _convert_true_strengths_to_delta and _convert_initial_knobs_to_delta
        # (which this class overrides) to handle tracking-specific energy parameter conversions.
        super().__init__(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config.magnet_range,
            bpm_start_points,
            bpm_end_points,
            initial_knob_strengths=initial_knob_strengths,  # Pass actual values
            true_strengths=true_strengths,  # Pass actual values
            bad_bpms=sequence_config.bad_bpms,
            first_bpm=sequence_config.first_bpm,
            debug=debug,
            optimise_knobs=optimise_knobs,
            output_config=self.output_config,
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
        # Re-initialize managers with tracking-specific configurations
        self._init_managers_with_tracking_config()

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process.

        Returns:
            Tuple of (final_knobs, uncertainties) in absolute-space (user-facing format).
        """
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
                self.initial_knobs,
            )

            # Pre-loop diagnostics: mask BPM and worker outliers before optimisation
            if self.simulation_config.enable_preloop_outlier_screening:
                self.worker_manager.screen_initial_outliers(
                    self.initial_knobs,
                    bpm_sigma_threshold=self.simulation_config.bpm_loss_outlier_sigma,
                    worker_sigma_threshold=self.simulation_config.worker_loss_outlier_sigma,
                )

            # Clean up memory after workers are started
            self._cleanup_memory()
            channels = self.worker_manager.channels
            if channels is None:
                raise RuntimeError("Worker channels are not initialised")

            self.final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs,
                channels,
                writer,
                run_start,
                total_turns,
                checkpoint_config=self.checkpoint_config,
                validation_loss_fn=self.worker_manager.compute_validation_loss,
            )

            total_hessian = self.worker_manager.termination_and_hessian(
                len(self.final_knobs),
                estimate_hessian=self.output_config.include_uncertainty,
            )
        except RuntimeError as e:
            logger.error(f"optimisation failed: {e}")
            self.worker_manager.terminate_workers()
            raise RuntimeError(f"Worker error during optimisation: {e}") from e
        except KeyboardInterrupt:
            logger.warning("\nKeyboardInterrupt detected. Terminating early and writing results.")
            self.worker_manager.terminate_workers()
            self.final_knobs = self.optimisation_loop.best_knobs
            total_hessian = None
        uncertainties = self._save_results(total_hessian, writer)
        uncertainties = dict(zip(self.final_knobs.keys(), uncertainties))

        return self.final_knobs, uncertainties

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager
        gc.collect()

    def _save_results(
        self,
        total_hessian: np.ndarray | None,
        writer: SummaryWriter | None,
    ) -> np.ndarray:
        """Save final results and convert from delta-space to user-facing absolute-space."""
        # Calculate uncertainties only when explicitly requested.
        if self.output_config.include_uncertainty and total_hessian is not None:
            cov = np.linalg.inv(total_hessian + 1e-8 * np.eye(total_hessian.shape[0]))
            uncertainties = np.sqrt(np.diag(cov))
        else:
            uncertainties = np.zeros(len(self.final_knobs), dtype=np.float64)

        # Close logging
        if writer is not None:
            writer.close()

        # All results are currently in delta-space; convert to absolute for output.
        # Create initial knobs dict in delta-space for conversion.
        initial_knobs_delta = dict(
            zip(self.config_manager.knob_names, self.config_manager.initial_strengths, strict=False)
        )

        # Convert all results from delta-space to absolute-space
        initial_knobs_abs = self.config_manager.mad_iface.optimisation_to_absolute_knobs(
            initial_knobs_delta
        )
        final_knobs_abs = self.config_manager.mad_iface.optimisation_to_absolute_knobs(
            self.final_knobs
        )
        true_strengths_abs = self.config_manager.mad_iface.optimisation_to_absolute_knobs(
            self.filtered_true_strengths
        )
        uncertainties_abs = self.config_manager.mad_iface.convert_uncertainties_to_absolute(
            self.config_manager.knob_names,
            uncertainties,
        )

        # Replace "pt" with "deltap" if needed for output
        output_knob_names = self.result_manager.knob_names
        if "deltap" in output_knob_names and "pt" in initial_knobs_abs:
            initial_knobs_abs["deltap"] = self.config_manager.mad_iface.pt2dp(
                initial_knobs_abs.pop("pt")
            )
        initial_strengths_abs = np.array(
            [initial_knobs_abs[name] for name in output_knob_names], dtype=np.float64
        )

        # Save and plot using the final knobs
        self.result_manager.save_results(
            final_knobs_abs,
            uncertainties_abs,
            true_strengths_abs,
        )
        self.result_manager.generate_plots(
            final_knobs_abs,
            initial_strengths_abs,
            true_strengths_abs,
            uncertainties_abs,
        )

        # Keep controller outputs user-facing (absolute-space) after save.
        self.final_knobs = final_knobs_abs
        self.filtered_true_strengths = true_strengths_abs

        logger.info("Optimisation complete.")
        return uncertainties_abs

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

        # Adjust num_batches to not exceed the smallest worker allocation.
        min_tracks_per_worker = min(len(batch) for batch in self.data_manager.turn_batches)
        self.simulation_config = dataclasses.replace(
            self.simulation_config,
            num_batches=min(self.simulation_config.num_batches, min_tracks_per_worker),
        )
        self.data_manager.simulation_config = self.simulation_config

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
            file_kick_planes=self.data_manager.file_kick_planes,
            bad_bpms=bad_bpms,
            flattop_turns=flattop_turns,
            num_tracks=num_tracks,
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            debug=self.debug,
            mad_logfile=self.mad_logfile,
            python_logfile=self.python_logfile,
        )

    def _convert_true_strengths_to_delta(
        self, true_strengths: dict[str, float]
    ) -> dict[str, float]:
        """Override parent to handle tracking-specific energy parameter conversion."""
        if not true_strengths:
            return {}

        true_strengths = true_strengths.copy()
        if "deltap" in true_strengths:
            if len(self.measurement_files) > 1:
                logger.warning(
                    "Ignoring provided 'deltap' in true strengths (differs per measurement file), setting to 0.0"
                )
                true_strengths["pt"] = 0.0
            else:
                true_strengths["pt"] = self.config_manager.mad_iface.dp2pt(
                    true_strengths.pop("deltap")
                )

        true_strengths = self.accelerator.normalise_true_strengths(true_strengths)
        return self.config_manager.mad_iface.absolute_to_optimisation_knobs(true_strengths)

    def _convert_initial_knobs_to_delta(
        self, initial_knob_strengths: dict[str, float] | None
    ) -> dict[str, float] | None:
        """Override parent to handle tracking-specific energy parameter conversion."""
        if initial_knob_strengths is None:
            return None

        initial_knob_strengths = initial_knob_strengths.copy()
        if "deltap" in initial_knob_strengths:
            initial_knob_strengths["pt"] = self.config_manager.mad_iface.dp2pt(
                initial_knob_strengths.pop("deltap")
            )

        return self.config_manager.mad_iface.absolute_to_optimisation_knobs(initial_knob_strengths)

    def _init_managers_with_tracking_config(self) -> None:
        """Re-initialize optimisation and result managers with tracking-specific config."""
        from aba_optimiser.training.optimisation_loop import OptimisationLoop
        from aba_optimiser.training.result_manager import ResultManager

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

        # Replace "pt" with "deltap" in result manager if optimizing energy
        output_knob_names = self.config_manager.mad_iface.format_knob_names_for_output(
            self.config_manager.knob_names
        )
        deltap_knob_names = self.accelerator.format_result_knob_names(output_knob_names)

        self.result_manager = ResultManager(
            deltap_knob_names,
            self.config_manager.elem_spos,
            show_plots=self.show_plots,
            accelerator=self.accelerator,
            include_uncertainty=self.output_config.include_uncertainty,
            plot_real_values=self.output_config.plot_real_values,
            save_prefix=self.output_config.save_prefix,
            plots_dir=self.output_config.plots_dir,
        )
