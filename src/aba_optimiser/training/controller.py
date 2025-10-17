"""Refactored controller for orchestrating multi-process knob optimisation."""

from __future__ import annotations

import datetime
import gc
import logging

# import multiprocessing as mp
import random
import time
from typing import TYPE_CHECKING

import numpy as np
from tensorboardX import SummaryWriter

from aba_optimiser.config import (
    BPM_END_POINTS,
    BPM_START_POINTS,
    MACHINE_DELTAP,
    MAGNET_RANGE,
    RUN_ARC_BY_ARC,
    TRUE_STRENGTHS_FILE,
)
from aba_optimiser.io.utils import read_knobs
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.worker_manager import WorkerManager

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings

logger = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


class Controller:
    """
    Orchestrates multi-process knob optimisation using MAD-NG.

    This refactored version uses a composition-based approach with specialised
    managers for different aspects of the optimisation process.
    """

    def __init__(
        self,
        opt_settings: OptSettings,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        true_strengths_file: str | None = TRUE_STRENGTHS_FILE,
        machine_deltap: float = MACHINE_DELTAP,
        magnet_range: str = MAGNET_RANGE,
        bpm_start_points: list[str] = BPM_START_POINTS,
        bpm_end_points: list[str] = BPM_END_POINTS,
        measurement_file: str | None = None,
        bad_bpms: list[str] | None = None,
    ):
        """Initialise the controller with all required managers."""

        logger.info("Optimising energy")
        if opt_settings.optimise_quadrupoles:
            logger.info("Optimising quadrupoles")
        if opt_settings.optimise_sextupoles:
            logger.info("Optimising sextupoles")
        self.opt_settings = opt_settings

        # Remove all the bad bpms from the start and end points
        if bad_bpms is not None:
            for bpm in bad_bpms:
                if bpm in bpm_start_points:
                    bpm_start_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from start points")
                if bpm in bpm_end_points:
                    bpm_end_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from end points")

        # Initialise managers
        self.config_manager = ConfigurationManager(
            opt_settings, magnet_range, bpm_start_points, bpm_end_points
        )
        self.config_manager.setup_mad_interface(bad_bpms)
        self.config_manager.determine_worker_and_bpms()

        self.data_manager = DataManager(
            self.config_manager.all_bpms,
            opt_settings,
            measurement_file,
        )
        self.data_manager.prepare_turn_batches(self.config_manager)

        # Determine the exact set of turns needed and load filtered data
        # needed_turns = None
        # if RUN_ARC_BY_ARC:
        #     needed_turns = {
        #         t for batch in self.data_manager.turn_batches for t in batch
        #     } | {t + 1 for batch in self.data_manager.turn_batches for t in batch}
        self.data_manager.load_track_data(
            # needed_turns=needed_turns,
            use_off_energy_data=opt_settings.use_off_energy_data,
        )

        self.worker_manager = WorkerManager(
            self.config_manager.Worker,
            self.config_manager.calculate_n_data_points(),
            # Assume the start bpm has is largest vertical kick
            ybpm=magnet_range.split("/")[0],
            magnet_range=magnet_range,
            bad_bpms=bad_bpms,
        )
        if true_strengths_file is None:
            true_strengths = {}
            true_strengths["pt"] = self.config_manager.mad_iface.dp2pt(machine_deltap)
        else:
            true_strengths = read_knobs(TRUE_STRENGTHS_FILE)
            true_strengths["pt"] = self.config_manager.mad_iface.dp2pt(machine_deltap)
        # For all the main bending magnets, remove the a/b/c/d suffixes
        # and set the only knob to the c suffix value (the only one that has a reasonable gradient).
        # bending_magnets: dict[str, float] = {}
        # mag_names_to_remove = []
        # for mag_name in true_strengths:
        #     if mag_name.startswith("MB."):
        #         if (mag_name[3] == "C" and "R" in mag_name) or (
        #             mag_name[3] == "A" and "L" in mag_name
        #         ):
        #             new_mag_name = (
        #                 mag_name[:3] + mag_name[4:]
        #             )  # Remove the a/b/c/d suffix
        #             bending_magnets[new_mag_name] = true_strengths[mag_name]
        #         mag_names_to_remove.append(mag_name)

        # for mag_name in mag_names_to_remove:
        #     del true_strengths[mag_name]
        # for mag_name in bending_magnets:
        #     true_strengths[mag_name] = bending_magnets[mag_name]

        # Initialise knobs
        if initial_knob_strengths is not None and "deltap" in initial_knob_strengths:
            initial_knob_strengths["pt"] = self.config_manager.mad_iface.dp2pt(
                initial_knob_strengths.pop("deltap")
            )
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(
                true_strengths, initial_knob_strengths
            )
        )

        # Setup optimisation and result managers
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            opt_settings,
            optimiser_type=opt_settings.optimiser_type,
        )

        deltap_knob_names = self.config_manager.knob_names[:-1] + ["deltap"]
        self.result_manager = ResultManager(
            deltap_knob_names,
            self.config_manager.elem_spos,
            show_plots=show_plots,
            opt_settings=opt_settings,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process."""
        run_start = time.time()
        writer = self._setup_logging()
        total_turns = self.data_manager.get_total_turns()

        try:
            self.worker_manager.start_workers(
                self.data_manager.track_data,
                self.data_manager.turn_batches,
                self.data_manager.energy_map,
                self.config_manager.bpm_ranges,
                self.data_manager.var_x,
                self.data_manager.var_y,
                self.opt_settings,
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
        except KeyboardInterrupt:
            logger.warning(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
            self.final_knobs = self.initial_knobs
        finally:
            total_hessian = self.worker_manager.terminate_workers()
        uncertainties = self._save_results(total_hessian, writer)
        uncertainties = dict(zip(self.final_knobs.keys(), uncertainties))

        return self.final_knobs, uncertainties

    def _setup_logging(self) -> SummaryWriter:
        """Sets up TensorBoard logging."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{ts}_opt")

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager
        gc.collect()

    def _convert_pt2dp(self, uncertainties: np.ndarray):
        self.final_knobs["deltap"] = self.config_manager.mad_iface.pt2dp(
            self.final_knobs.pop("pt")
        )
        if "pt" in self.config_manager.initial_strengths:
            self.filtered_true_strengths["deltap"] = (
                self.config_manager.mad_iface.pt2dp(
                    self.filtered_true_strengths.pop("pt")
                )
            )

        uncertainties[-1] = self.config_manager.mad_iface.pt2dp(uncertainties[-1])

    def _save_results(
        self,
        total_hessian: np.ndarray,
        writer: SummaryWriter,
    ) -> None:
        """Clean up resources and save final results."""
        # Calculate uncertainties
        cov = np.linalg.inv(total_hessian + 1e-8 * np.eye(total_hessian.shape[0]))
        uncertainties = np.sqrt(np.diag(cov))

        # Close logging and save results
        writer.close()

        # Convert the knobs back from pt to dp.
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
