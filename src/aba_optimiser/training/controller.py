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
from aba_optimiser.training.config.models import KickerConfig, OutputConfig
from aba_optimiser.training.config.tracking import build_tracking_plan
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.workers.manager import WorkerManager

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tensorboardX import SummaryWriter

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.config.models import (
        CheckpointConfig,
        MeasurementConfig,
        SequenceConfig,
    )

logger = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


def _estimate_uncertainties_from_hessian(
    total_hessian: np.ndarray,
    *,
    min_eigenvalue: float = 1e-8,
) -> np.ndarray:
    """Convert an approximate Hessian into 1-sigma parameter uncertainties.

    The tracking Hessian is assembled as a sum of weighted Jacobian outer products,
    so it should be symmetric positive semidefinite. In practice, accumulated
    numerical noise can introduce asymmetry or slightly negative modes, which can
    yield negative entries on the covariance diagonal after a direct inversion.

    To keep the uncertainty estimate physically meaningful, we symmetrise the
    Hessian, floor non-positive eigenvalues to a small positive curvature, and
    build the covariance from the resulting eigendecomposition.
    """
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym_hessian)
    clipped_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)

    n_clipped = int(np.count_nonzero(eigenvalues < min_eigenvalue))
    if n_clipped:
        logger.warning(
            "Hessian had %d eigenvalue(s) below %.3e; using the floor to keep "
            "uncertainties finite and non-negative.",
            n_clipped,
            min_eigenvalue,
        )

    covariance = (eigenvectors / clipped_eigenvalues) @ eigenvectors.T
    variances = np.clip(np.diag(covariance), 0.0, None)
    return np.sqrt(variances)


class Controller(BaseController):
    """
    Orchestrates multi-process knob optimisation using MAD-NG.

    Extends BaseController with tracking-specific functionality including
    data management and worker coordination for multi-turn tracking.

    Design: optimisation-space only internally and externally. All user inputs,
    internal algorithms, and reported results use the same knob coordinates.
    """

    _defer_managers = True

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
        initial_conditions_callback: Callable[[dict[str, float]], np.ndarray] | None = None,
        kicker_config: KickerConfig | None = None,
    ):
        """
        Initialise the controller with all required managers.

        User inputs are in optimisation space and remain there throughout the run.

        Args:
            accelerator (Accelerator): Accelerator instance defining machine configuration.
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration.
            simulation_config (SimulationConfig): Simulation and worker configuration.
            sequence_config (SequenceConfig): Sequence and beam configuration.
            measurement_config (MeasurementConfig): Measurement data file configuration.
            output_config (OutputConfig): Output/logging behaviour.
            bpm_start_points (list[str]): Starting BPM names for each range.
            bpm_end_points (list[str]): Ending BPM names for each range.
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths in optimisation space.
            true_strengths (Path | dict[str, float], optional): True strengths file or dict in optimisation space.
            debug (bool, optional): Enable debug mode. Defaults to False.
            optimise_knobs (list[str] | None, optional): List of global knob names to optimise.
        """

        # Log optimisation targets
        accelerator.log_optimisation_targets()
        if simulation_config.optimise_momenta:
            logger.info("Including momenta (px, py) in loss function")
        else:
            logger.info("Using position-only optimisation (x, y only)")

        if kicker_config is not None:
            kicker_config.log_state()
            sequence_config = dataclasses.replace(
                sequence_config,
                first_bpm=sequence_config.first_bpm or kicker_config.kicker_name,
            )
            simulation_config = dataclasses.replace(
                simulation_config,
                tracks_per_worker=1,
                num_workers=1,
                num_batches=1,
                run_arc_by_arc=False,
                n_run_turns=kicker_config.turns_after_kicker,
                different_turns_per_range=False,
            )
            bpm_start_points = [kicker_config.kicker_name]
            bpm_end_points = []
            logger.info(
                "Kicker mode enabled: start=%s, turns=%d",
                kicker_config.kicker_name,
                kicker_config.turns_after_kicker,
            )

        # Normalize and validate multi-config inputs
        measurement_config = measurement_config.expanded_for_measurements()
        self.measurement_config = measurement_config
        self.measurement_files = measurement_config.measurement_files
        self.corrector_files = measurement_config.corrector_files
        self.tune_knobs_files = measurement_config.tune_knobs_files
        self.machine_deltaps = measurement_config.machine_deltaps
        self.num_configs = len(self.measurement_files)
        self.output_config = output_config if output_config is not None else OutputConfig()
        self.checkpoint_config = checkpoint_config
        self.initial_conditions_callback = initial_conditions_callback
        self.kicker_config = kicker_config
        self.tracking_plan = build_tracking_plan(
            kicker_config=kicker_config,
            simulation_config=simulation_config,
        )

        # BaseController will normalise the optimisation-space inputs and handle
        # tracking-specific energy parameter conversions in this subclass.
        super().__init__(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            bpm_start_points,
            bpm_end_points,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
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
        # Initialize OptimisationLoop and ResultManager now that _init_data_manager
        # has finalised simulation_config.num_batches.
        BaseController._init_managers(self)

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process.

        Returns:
            Tuple of (final_knobs, uncertainties) in optimisation space.
        """
        run_start = time.time()
        writer = self.setup_logging("tracking_opt")
        total_turns = self.data_manager.get_total_turns()
        self.final_knobs = None  # Will be set after optimisation loop

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
                enable_validation=self.tracking_plan.enable_validation,
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

            epoch_end_hook = self._make_epoch_end_hook()
            self.final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs,
                channels,
                writer,
                run_start,
                total_turns,
                checkpoint_config=self.checkpoint_config,
                validation_loss_fn=self.worker_manager.compute_validation_loss,
                epoch_end_hook=epoch_end_hook,
            )

            total_hessian = self.worker_manager.termination_and_hessian(
                len(self.final_knobs),
                estimate_hessian=self.output_config.include_uncertainty,
                parallelism=self.output_config.parallel_hessian,
            )
        except RuntimeError as e:
            logger.error(f"optimisation failed: {e}")
            self.worker_manager.terminate_workers()
            raise RuntimeError(f"Worker error during optimisation: {e}") from e
        except KeyboardInterrupt:
            logger.warning(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
            self.worker_manager.terminate_workers()
            self.final_knobs = self.optimisation_loop.best_knobs
            total_hessian = None
        finally:
            if self.final_knobs is None:
                self.final_knobs = self.optimisation_loop.best_knobs

        self.final_knobs = self._format_result_knobs(self.final_knobs)
        self.filtered_true_strengths = self._format_result_knobs(self.filtered_true_strengths)

        uncertainties = self._finalise_results(total_hessian, writer)
        uncertainties = dict(zip(self.final_knobs.keys(), uncertainties))

        return self.final_knobs, uncertainties

    def _make_epoch_end_hook(self):
        """Return an epoch-end callable that updates worker initial conditions, or None."""
        if self.initial_conditions_callback is None:
            return None

        worker_manager = self.worker_manager

        def hook(current_knobs: dict[str, float]) -> None:
            new_px_py = self.initial_conditions_callback(current_knobs)
            if new_px_py is None:
                return
            worker_manager.send_init_condition_updates(new_px_py)

        return hook

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager
        gc.collect()

    def _format_result_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Map internal optimisation-space knob names to user-facing result names."""
        formatted = knobs.copy()
        if not self.accelerator.optimise_energy or "pt" not in formatted:
            return formatted

        return self.convert_pt_to_deltap(formatted)

    def _format_result_uncertainties(self, uncertainties: np.ndarray) -> np.ndarray:
        """Align uncertainty values with the formatted output knob ordering."""
        uncertainty_by_knob = dict(zip(self.config_manager.knob_names, uncertainties, strict=True))
        if self.accelerator.optimise_energy and "pt" in uncertainty_by_knob:
            uncertainty_by_knob["deltap"] = abs(
                self.config_manager.mad_iface.pt2dp(uncertainty_by_knob.pop("pt"))
            )
        return np.array(
            [uncertainty_by_knob[name] for name in self.output_knob_names],
            dtype=np.float64,
        )

    def _finalise_results(
        self,
        total_hessian: np.ndarray | None,
        writer: SummaryWriter | None,
    ) -> np.ndarray:
        """Save final results in optimisation space."""
        # Calculate uncertainties only when explicitly requested.
        if self.output_config.include_uncertainty and total_hessian is not None:
            uncertainties = _estimate_uncertainties_from_hessian(total_hessian)
        else:
            uncertainties = np.zeros(len(self.final_knobs), dtype=np.float64)

        # Close logging
        if writer is not None:
            writer.close()

        uncertainties_abs = self._format_result_uncertainties(uncertainties)

        logger.info("Optimisation complete.")
        return uncertainties_abs

    def _init_data_manager(self, num_tracks: int, flattop_turns: int) -> None:
        """Initialize data manager and load track data."""
        observed_bpms = self.tracking_plan.observed_bpms(
            self.config_manager.bpms_in_range,
            self.config_manager.all_bpms,
        )
        self.data_manager = DataManager(
            observed_bpms,
            self.config_manager.all_bpms,
            self.simulation_config,
            self.measurement_files,
            num_bunches=num_tracks,
            flattop_turns=flattop_turns,
            tracking_plan=self.tracking_plan,
            extra_markers=self.tracking_plan.extra_markers(),
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
            tracking_plan=self.tracking_plan,
        )

    def _get_controller_mad_setup_kwargs(self) -> dict:
        """Mirror the worker MAD setup when building the expected knob list."""
        return {
            "corrector_strengths": next((p for p in self.corrector_files if p is not None), None),
            "tune_knobs_file": next((p for p in self.tune_knobs_files if p is not None), None),
        }

    def _get_configuration_manager_kwargs(self) -> dict:
        """Pass kicker-mode planning information into configuration setup."""
        return {"tracking_plan": self.tracking_plan}
