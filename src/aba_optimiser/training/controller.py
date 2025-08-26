"""Refactored controller for orchestrating multi-process knob optimisation."""

from __future__ import annotations

import datetime
import gc
import logging
import multiprocessing as mp
import random
import time

import numpy as np
from tensorboardX import SummaryWriter

from aba_optimiser.config import OPTIMISER_TYPE, TRUE_STRENGTHS
from aba_optimiser.io.utils import read_knobs
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.worker_manager import WorkerManager

LOGGER = logging.getLogger(__name__)
random.seed(42)  # For reproducibility


class Controller:
    """
    Orchestrates multi-process knob optimisation using MAD-NG.

    This refactored version uses a composition-based approach with specialised
    managers for different aspects of the optimisation process.
    """

    def __init__(self):
        """Initialise the controller with all required managers."""
        # Load true strengths first
        self.true_strengths = read_knobs(TRUE_STRENGTHS)

        # Initialise managers
        self.config_manager = ConfigurationManager()
        self.config_manager.setup_mad_interface()
        self.config_manager.determine_worker_and_bpms()

        self.data_manager = DataManager(
            self.config_manager.all_bpms, self.config_manager.bpm_start_points
        )
        self.data_manager.prepare_turn_batches()

        # Determine the exact set of turns needed and load filtered data
        needed_turns = {t for batch in self.data_manager.turn_batches for t in batch}
        self.data_manager.load_track_data(needed_turns=needed_turns)

        self.worker_manager = WorkerManager(
            self.config_manager.Worker, self.config_manager.calculate_n_data_points()
        )

        # Initialise knobs
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(self.true_strengths)
        )

        # Setup optimisation and result managers
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            OPTIMISER_TYPE,
        )

        self.result_manager = ResultManager(
            self.config_manager.knob_names, self.config_manager.elem_spos
        )

    def run(self) -> None:
        """Execute the optimisation process."""
        run_start = time.time()
        writer = self._setup_logging()
        total_turns = self.data_manager.get_total_turns()

        try:
            parent_conns = self.worker_manager.start_workers(
                self.data_manager.get_indexed_comparison_data(),
                self.data_manager.turn_batches,
                self.config_manager.bpm_start_points,
                self.data_manager.var_x,
                self.data_manager.var_y,
            )

            # Clean up memory after workers are started
            self._cleanup_memory()

            self.final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs, parent_conns, writer, run_start, total_turns
            )
        except KeyboardInterrupt:
            LOGGER.warning(
                "\nKeyboardInterrupt detected. Terminating early and writing results."
            )
            self.final_knobs = self.initial_knobs
        finally:
            total_hessian = self.worker_manager.terminate_workers()
        self._save_results(total_hessian, writer)

    def _setup_logging(self) -> SummaryWriter:
        """Sets up TensorBoard logging."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{ts}_opt")

    def _cleanup_memory(self) -> None:
        """Clean up memory after worker initialisation."""
        del self.data_manager.comparison_data
        del self.data_manager.turn_batches
        gc.collect()

    def _save_results(
        self,
        total_hessian: np.ndarray,
        writer: SummaryWriter,
    ) -> None:
        """Clean up resources and save final results."""
        # Calculate uncertainties
        cov = np.linalg.inv(total_hessian)
        uncertainties = np.sqrt(np.diag(cov))

        # Close logging and save results
        writer.close()
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

        LOGGER.info("Optimisation complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("spawn")
    ctrl = Controller()
    ctrl.run()
