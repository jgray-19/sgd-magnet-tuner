"""Refactored controller for orchestrating multi-process knob optimisation."""

from __future__ import annotations

import dataclasses
import datetime
import gc
import logging

# import multiprocessing as mp
import random
import re
import time
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

from aba_optimiser.config import OptSettings
from aba_optimiser.io.utils import get_lhc_file_path, read_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.worker_manager import WorkerManager

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
        sequence_file_path: str | Path,
        measurement_files: list[str] | str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        magnet_range: str,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_files: list[Path] | Path | None = None,
        tune_knobs_files: list[Path] | Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        machine_deltaps: list[float] | float = 0.0,
        bad_bpms: list[str] | None = None,
        first_bpm: str | None = None,
        seq_name: str | None = None,
        beam_energy: float = 6800.0,
        num_tracks: int = 3,
        flattop_turns: int = 6600,
    ):
        """
        Initialise the controller with all required managers.

        Args:
            opt_settings (OptSettings): Optimisation settings.
            sequence_file_path (str | Path): Path to the sequence file.
            show_plots (bool, optional): Whether to show plots. Defaults to True.
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths.
            corrector_files (list[Path] | Path, optional): List of corrector strength files or single file.
            tune_knobs_files (list[Path] | Path, optional): List of tune knob files or single file.
            true_strengths (Path | dict[str, float], optional): True strengths file or dict.
            machine_deltaps (list[float] | float, optional): List of machine deltaps or single value.
            beam_energy (float, optional): Beam energy in GeV.
            magnet_range (str, optional): Magnet range.
            bpm_start_points (list[str], optional): BPM start points.
            bpm_end_points (list[str], optional): BPM end points.
            measurement_files (list[str] | str | None, optional): List of measurement files or single file.
            bad_bpms (list[str] | None, optional): List of bad BPMs.
            first_bpm (str | None, optional): First BPM.
            seq_name (str | None, optional): Sequence name.
            num_tracks (int, optional): Number of tracks per measurement file. Defaults to NUM_TRACKS.
            flattop_turns (int, optional): Number of turns on the flat top. Defaults to FLATTOP_TURNS.
        """

        logger.info("Optimising energy")
        if opt_settings.optimise_quadrupoles:
            logger.info("Optimising quadrupoles")
        if opt_settings.optimise_bends:
            logger.info("Optimising bends")
        self.opt_settings = dataclasses.replace(opt_settings)

        # Normalise inputs to lists
        if isinstance(measurement_files, Path | str) or measurement_files is None:
            measurement_files = [measurement_files]
        if isinstance(corrector_files, Path | str) or corrector_files is None:
            corrector_files = [corrector_files]
        if isinstance(tune_knobs_files, Path | str) or tune_knobs_files is None:
            tune_knobs_files = [tune_knobs_files]
        if isinstance(machine_deltaps, float):
            machine_deltaps = [machine_deltaps]

        # Validate and expand lists
        num_configs = len(measurement_files)
        for name, lst in [("corrector_files", corrector_files), ("tune_knobs_files", tune_knobs_files), ("machine_deltaps", machine_deltaps)]:
            if len(lst) == 1:
                lst *= num_configs
            elif len(lst) != num_configs:
                raise ValueError(f"Number of {name} ({len(lst)}) must match number of measurement files ({num_configs}) or be 1")

        self.measurement_files = measurement_files
        self.corrector_files = corrector_files
        self.tune_knobs_files = tune_knobs_files
        self.machine_deltaps = machine_deltaps
        self.num_configs = num_configs

        # Remove all the bad bpms from the start and end points
        if bad_bpms is not None:
            for bpm in bad_bpms:
                if bpm in bpm_start_points:
                    bpm_start_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from start points")
                if bpm in bpm_end_points:
                    bpm_end_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from end points")

        bpm_order = OptimisationMadInterface(
            sequence_file_path,
            seq_name=seq_name,
            bad_bpms=bad_bpms,
            start_bpm=first_bpm,
            tune_knobs_file=None,
            corrector_strengths=None,
            beam_energy=beam_energy,
        ).all_bpms
        # Initialise managers
        self.config_manager = ConfigurationManager(
            opt_settings, magnet_range, bpm_start_points, bpm_end_points
        )
        self.config_manager.setup_mad_interface(
            sequence_file_path,
            bad_bpms,
            seq_name,
            beam_energy,
        )
        self.config_manager.determine_worker_and_bpms()

        self.data_manager = DataManager(
            self.config_manager.all_bpms,
            opt_settings,
            measurement_files,
            bpm_order=bpm_order,
            num_tracks=num_tracks,
            flattop_turns=flattop_turns,
        )
        self.data_manager.prepare_turn_batches(self.config_manager)

        # Adjust num_batches to not exceed tracks_per_worker for consistency
        self.opt_settings = dataclasses.replace(
            self.opt_settings,
            num_batches=min(self.opt_settings.num_batches, self.data_manager.tracks_per_worker)
        )

        # Load track data from all measurement files
        self.data_manager.load_track_data()

        self.worker_manager = WorkerManager(
            self.config_manager.calculate_n_data_points(),
            # Assume the start bpm has is largest vertical kick
            ybpm=magnet_range.split("/")[0],
            magnet_range=magnet_range,
            sequence_file_path=sequence_file_path,
            corrector_strengths_files=corrector_files,
            tune_knobs_files=tune_knobs_files,
            bad_bpms=bad_bpms,
            seq_name=seq_name,
            beam_energy=beam_energy,
            flattop_turns=flattop_turns,
            num_tracks=num_tracks,
        )
        if true_strengths is None:
            true_strengths = {}
        elif isinstance(true_strengths, dict):
            true_strengths = true_strengths.copy()
        elif isinstance(true_strengths, Path):
            true_strengths = read_knobs(true_strengths)
            # Use the first machine deltap for true strengths (typically all configs should match)
            true_strengths["pt"] = 0.0

        if "deltap" in true_strengths:
            logger.warning("Ignoring provided 'deltap' in true strengths, can different per measurement file, setting to 0.0")
            true_strengths["pt"] = 0.0
        # Update bend keys to remove [ABCD] and average over them
        pattern = r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.k0"
        new_true_strengths = {}
        for key, value in true_strengths.items():
            match = re.match(pattern, key)
            if match:
                new_key = match.group(1) + match.group(3) + ".k0"
                if new_key not in new_true_strengths:
                    new_true_strengths[new_key] = []
                new_true_strengths[new_key].append(value)
            else:
                new_true_strengths[key] = value
        true_strengths = {k: np.mean(v) if isinstance(v, list) else v for k, v in new_true_strengths.items()}

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
        if not true_strengths:
            # Set the true strengths to the initial strengths if not provided
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Setup optimisation and result managers
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.opt_settings,
            optimiser_type=self.opt_settings.optimiser_type,
        )
        # Replace "pt" with "deltap" in the result manager if optimising energy
        deltap_knob_names = self.config_manager.knob_names
        if self.opt_settings.optimise_energy:
            deltap_knob_names = deltap_knob_names + ["deltap"]
            deltap_knob_names.remove("pt")
    
        self.result_manager = ResultManager(
            deltap_knob_names,
            self.config_manager.elem_spos,
            show_plots=show_plots,
            opt_settings=self.opt_settings,
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
                self.data_manager.file_map,
                self.config_manager.bpm_ranges,
                self.opt_settings,
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


class LHCController(Controller):
    """
    LHC-specific controller that automatically sets sequence file path and first BPM based on beam number.
    """

    def __init__(self, beam: int, **kwargs):
        """
        Initialise the LHC controller with beam number and other parameters.

        Args:
            beam (int): The beam number (1 or 2).
            **kwargs: Additional keyword arguments passed to the parent Controller class.
                See Controller.__init__ for the full list of available parameters.
        """
        sequence_file_path = get_lhc_file_path(beam)
        first_bpm = "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"
        seq_name = f"lhcb{beam}"

        super().__init__(
            sequence_file_path=sequence_file_path,
            first_bpm=first_bpm,
            seq_name=seq_name,
            **kwargs,
        )
