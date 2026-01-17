"""Controller for optics optimisation (beta functions using quadrupole strengths)."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omc3.optics_measurements.constants import (
    AMP_BETA_NAME,
    BETA_NAME,
    DISPERSION_NAME,
    ORBIT_NAME,
    PHASE_NAME,
)

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_controller import BaseController, LHCControllerMixin
from aba_optimiser.training.utils import (
    extract_bpm_range_names,
    find_common_bpms,
    load_tfs_files,
)
from aba_optimiser.training.worker_lifecycle import WorkerLifecycleManager
from aba_optimiser.workers import OpticsData, OpticsWorker, WorkerConfig

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.training.controller_config import BPMConfig, SequenceConfig

X = "x"
Y = "y"

logger = logging.getLogger(__name__)


class OpticsController(BaseController):
    """
    Orchestrates optics optimisation using MAD-NG.

    This controller is specialised for beta function optimisation using quadrupole
    strengths. It reads beta function measurements from TFS files and uses a single
    worker to optimise quadrupole strengths.
    """

    def __init__(
        self,
        sequence_config: SequenceConfig,
        optics_folder: str | Path,
        bpm_config: BPMConfig,
        optimiser_config: OptimiserConfig,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
    ):
        """
        Initialise the optics controller.

        Args:
            sequence_config (SequenceConfig): Sequence and beam configuration
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_config (BPMConfig): BPM range configuration
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            use_errors (bool, optional): Whether to use measurement errors in optimisation. Defaults to True
        """
        logger.info("Optimising quadrupoles for beta functions")

        # Create optics-specific simulation config
        opt_quads = sequence_config.optimise_knobs is None
        simulation_config = SimulationConfig(
            tracks_per_worker=1,
            num_workers=1,
            num_batches=1,
            optimise_energy=False,
            optimise_quadrupoles=opt_quads,
            optimise_bends=False,
            use_fixed_bpm=True,
        )

        # Initialize base controller
        super().__init__(
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_file_path=sequence_config.sequence_file_path,
            magnet_range=sequence_config.magnet_range,
            bpm_start_points=bpm_config.start_points,
            bpm_end_points=bpm_config.end_points,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            bad_bpms=sequence_config.bad_bpms,
            first_bpm=sequence_config.first_bpm,
            seq_name=sequence_config.seq_name,
            beam_energy=sequence_config.beam_energy,
            optimise_knobs=sequence_config.optimise_knobs,
        )

        # Store optics-specific attributes
        self.optics_folder = Path(optics_folder)
        self.corrector_file = corrector_file
        self.tune_knobs_file = tune_knobs_file
        self.use_errors = use_errors

        # Create optics-specific worker payloads
        optics_path = Path(optics_folder)
        template_config = WorkerConfig(
            start_bpm="TEMP",
            end_bpm="TEMP",
            magnet_range=sequence_config.magnet_range,
            sequence_file_path=str(sequence_config.sequence_file_path),
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            beam_energy=sequence_config.beam_energy,
            sdir=np.nan,
            bad_bpms=sequence_config.bad_bpms,
            seq_name=sequence_config.seq_name,
            optimise_knobs=sequence_config.optimise_knobs,
        )

        # Use explicit BPM (start, end) pairs from config manager
        self.worker_payloads = create_worker_payloads(
            optics_path,
            self.config_manager.bpm_pairs,
            sequence_config.bad_bpms,
            template_config,
            self.use_errors,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process using optics workers."""
        writer = self.setup_logging("optics_opt")

        # Create and start workers
        worker_manager = WorkerLifecycleManager(OpticsWorker)
        worker_manager.create_and_start_workers(
            [(data, config, self.simulation_config) for config, data in self.worker_payloads],
            send_handshake=True,
        )

        # Run optimisation
        self.final_knobs = self.optimisation_loop.run_optimisation(
            self.initial_knobs,
            worker_manager.parent_conns,
            writer,
            run_start=time.time(),
            total_turns=1,
        )

        # Terminate workers
        worker_manager.terminate_workers()

        # Generate results
        uncertainties = np.zeros(len(self.initial_knobs))
        self.result_manager.generate_plots(
            self.final_knobs,
            self.config_manager.initial_strengths,
            self.filtered_true_strengths,
            uncertainties,
        )

        writer.close()
        return self.final_knobs, dict(zip(self.final_knobs.keys(), uncertainties))


def load_optics_data_and_find_common_bpms(
    optics_dir: Path,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load optics TFS files and find common BPMs across all files.

    Args:
        optics_dir: Path to directory containing TFS optics measurement files

    Returns:
        Tuple of (tfs_data dict, common_bpms list)
    """
    # Load all TFS files - both phase-based and amplitude-based beta measurements
    file_specs = {
        "beta_phase_x": (BETA_NAME, X),  # Beta from phase
        "beta_phase_y": (BETA_NAME, Y),
        "beta_amplitude_x": (AMP_BETA_NAME, X),  # Beta from amplitude
        "beta_amplitude_y": (AMP_BETA_NAME, Y),
        "phase_x": (PHASE_NAME, X),
        "phase_y": (PHASE_NAME, Y),
        "disp_x": (DISPERSION_NAME, X),
        "disp_y": (DISPERSION_NAME, Y),
        "orbit_x": (ORBIT_NAME, X),
        "orbit_y": (ORBIT_NAME, Y),
    }
    tfs_data = load_tfs_files(optics_dir, file_specs)
    beta_phase_x = tfs_data["beta_phase_x"]
    beta_phase_y = tfs_data["beta_phase_y"]
    beta_amplitude_x = tfs_data["beta_amplitude_x"]
    beta_amplitude_y = tfs_data["beta_amplitude_y"]
    phase_x = tfs_data["phase_x"]
    phase_y = tfs_data["phase_y"]
    disp_x = tfs_data["disp_x"]
    disp_y = tfs_data["disp_y"]
    orbit_x = tfs_data["orbit_x"]
    orbit_y = tfs_data["orbit_y"]

    # Find common BPMs across all files
    # For phase files, we need BPMs that appear in NAME or NAME2 columns
    phase_x_bpms = set(phase_x["NAME"].tolist()) | set(phase_x["NAME2"].tolist())
    phase_y_bpms = set(phase_y["NAME"].tolist()) | set(phase_y["NAME2"].tolist())
    common_bpms = find_common_bpms(
        beta_phase_x,
        beta_phase_y,
        beta_amplitude_x,
        beta_amplitude_y,
        disp_x,
        disp_y,
        orbit_x,
        orbit_y,
    )
    # Intersect with phase measurement BPMs
    common_bpms_before_phase = len(common_bpms)
    common_bpms = [bpm for bpm in common_bpms if bpm in phase_x_bpms and bpm in phase_y_bpms]
    logger.info(
        f"Found {common_bpms_before_phase} BPMs in beta/disp/orbit files, {len(common_bpms)} after requiring phase measurements"
    )
    if len(common_bpms) == 0:
        logger.warning(
            "No common BPMs found! Check that beta_amplitude, phase, dispersion, and orbit files all have overlapping BPMs"
        )
        logger.warning(
            f"Beta phase x BPMs: {len(beta_phase_x)}, Beta amplitude x BPMs: {len(beta_amplitude_x)}"
        )
        logger.warning(
            f"Phase x BPMs in measurements: {len(phase_x_bpms)}, Phase y BPMs: {len(phase_y_bpms)}"
        )

    return tfs_data, common_bpms


def filter_bpm_list_for_phase_coverage(
    bpm_list: list[str],
    phase_df: pd.DataFrame,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Filter BPM list to only include BPMs that have phase measurements.

    This function removes BPMs from the list that don't have phase measurements
    to their neighbors, keeping only BPMs that form a connected chain.

    Args:
        bpm_list: Ordered list of BPM names
        phase_df: Phase measurement dataframe with NAME, NAME2 columns

    Returns:
        Tuple of (filtered_bpm_list, missing_pairs_indices) where missing_pairs_indices
        is always empty (for backwards compatibility)
    """
    if len(bpm_list) <= 1:
        return bpm_list, []

    # Build a set of available phase measurement pairs for quick lookup
    phase_pairs = set(zip(phase_df["NAME"].tolist(), phase_df["NAME2"].tolist()))

    # Get all BPMs that appear in phase measurements
    bpms_with_measurements = set(phase_df["NAME"].tolist() + phase_df["NAME2"].tolist())

    # Filter to only include BPMs that:
    # 1. Have phase measurements (appear in the phase dataframe)
    # 2. Form a connected chain (have measurement to next BPM in filtered list)
    filtered_bpm_list = []

    for bpm in bpm_list:
        # Only include BPMs that appear in phase measurements
        if bpm in bpms_with_measurements:
            filtered_bpm_list.append(bpm)

    # Now remove BPMs that don't have measurements to their neighbors in the filtered list
    # Keep iterating until we have a connected chain
    max_iterations = len(filtered_bpm_list)  # Prevent infinite loop
    for _ in range(max_iterations):
        connected_list = []
        removed_any = False

        for i, bpm in enumerate(filtered_bpm_list):
            # Check if this BPM has measurement to previous or next BPM
            has_connection = False

            # Check connection to previous BPM
            if i > 0 and (filtered_bpm_list[i - 1], bpm) in phase_pairs:
                has_connection = True

            # Check connection to next BPM
            if i < len(filtered_bpm_list) - 1 and (bpm, filtered_bpm_list[i + 1]) in phase_pairs:
                has_connection = True

            # First and last BPMs only need one connection, others need at least one
            if has_connection or len(filtered_bpm_list) == 1:
                connected_list.append(bpm)
            else:
                removed_any = True
                logger.debug(f"Removing BPM {bpm} - no phase measurements to neighbors")

        filtered_bpm_list = connected_list

        # If we didn't remove anything, we have a stable list
        if not removed_any:
            break

    # Log if we removed BPMs
    removed_count = len(bpm_list) - len(filtered_bpm_list)
    if removed_count > 0:
        logger.info(
            f"Filtered BPM list from {len(bpm_list)} to {len(filtered_bpm_list)} BPMs "
            f"(removed {removed_count} BPMs without phase measurements)"
        )

    # Return empty missing_pairs list since we filtered out problematic BPMs
    return filtered_bpm_list, []


def find_phase_advance_between_bpms(
    phase_df: pd.DataFrame,
    bpm_start: str,
    bpm_end: str,
    bpm_list: list[str],
    phase_col: str = "PHASEX",
    err_col: str = "ERRPHASEX",
) -> tuple[float, float]:
    """Find phase advance between two BPMs, computing from intermediate measurements if needed.

    Args:
        phase_df: Phase measurement dataframe with NAME, NAME2, PHASEX/PHASEY columns
        bpm_start: Starting BPM name
        bpm_end: Ending BPM name
        bpm_list: Ordered list of BPMs in the tracking range
        phase_col: Column name for phase values (PHASEX or PHASEY)
        err_col: Column name for phase errors (ERRPHASEX or ERRPHASEY)

    Returns:
        Tuple of (phase_advance, error) computed by summing intermediate measurements

    Raises:
        ValueError: If no path can be found between the BPMs
    """
    # Check for direct measurement
    direct_match = phase_df[(phase_df["NAME"] == bpm_start) & (phase_df["NAME2"] == bpm_end)]
    if not direct_match.empty:
        return direct_match[phase_col].values[0], direct_match[err_col].values[0]

    # Find indices in BPM list
    try:
        idx_start = bpm_list.index(bpm_start)
        idx_end = bpm_list.index(bpm_end)
    except ValueError as e:
        raise ValueError(f"BPM not found in list: {e}")

    if idx_start >= idx_end:
        raise ValueError(f"Invalid BPM order: {bpm_start} must come before {bpm_end}")

    # Sum phase advances through intermediate BPMs
    total_phase = 0.0
    total_error_sq = 0.0

    for i in range(idx_start, idx_end):
        bpm1 = bpm_list[i]
        bpm2 = bpm_list[i + 1]

        # Find measurement between consecutive BPMs (try both directions)
        match = phase_df[(phase_df["NAME"] == bpm1) & (phase_df["NAME2"] == bpm2)]

        if match.empty:
            # Try reverse direction
            match = phase_df[(phase_df["NAME"] == bpm2) & (phase_df["NAME2"] == bpm1)]
            if not match.empty:
                # Reverse direction - negate the phase
                phase = -match[phase_col].values[0]
                error = match[err_col].values[0]
            else:
                raise ValueError(
                    f"No phase measurement found between consecutive BPMs {bpm1} and {bpm2}"
                )
        else:
            phase = match[phase_col].values[0]
            error = match[err_col].values[0]

        total_phase += phase
        total_error_sq += error**2

    # Take modulo 1 (phase is in units of 2π)
    total_phase = total_phase % 1.0

    # Errors add in quadrature
    total_error = np.sqrt(total_error_sq)

    return total_phase, total_error


def _get_initial_conditions(
    bpm: str,
    beta_x: pd.DataFrame,
    beta_y: pd.DataFrame,
    disp_x: pd.DataFrame,
    disp_y: pd.DataFrame,
    orbit_x: pd.DataFrame,
    orbit_y: pd.DataFrame,
) -> dict[str, float]:
    """Extract initial Twiss parameters and orbit for a BPM."""
    return {
        "beta11": beta_x.loc[bpm, "BETX"],
        "beta22": beta_y.loc[bpm, "BETY"],
        "alfa11": beta_x.loc[bpm, "ALFX"],
        "alfa22": beta_y.loc[bpm, "ALFY"],
        "dx": disp_x.loc[bpm, "DX"],
        "dpx": disp_x.loc[bpm, "DPX"],
        "dy": disp_y.loc[bpm, "DY"],
        "dpy": disp_y.loc[bpm, "DPY"],
        "x": orbit_x.loc[bpm, "X"],
        "y": orbit_y.loc[bpm, "Y"],
    }


def _extract_phase_advances(
    bpm_list: list[str],
    phase_x: pd.DataFrame,
    phase_y: pd.DataFrame,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Extract phase advances between consecutive BPMs.

    Returns:
        Tuple of (phase_x_list, phase_y_list, err_x_list, err_y_list, missing_count)
    """
    phase_adv_x_list = []
    phase_adv_y_list = []
    err_phase_adv_x_list = []
    err_phase_adv_y_list = []
    missing_count = 0

    for i in range(len(bpm_list) - 1):
        bpm1, bpm2 = bpm_list[i], bpm_list[i + 1]

        try:
            phase_adv_x, err_phase_adv_x = find_phase_advance_between_bpms(
                phase_x, bpm1, bpm2, bpm_list, "PHASEX", "ERRPHASEX"
            )
            phase_adv_y, err_phase_adv_y = find_phase_advance_between_bpms(
                phase_y, bpm1, bpm2, bpm_list, "PHASEY", "ERRPHASEY"
            )
            phase_adv_x_list.append(phase_adv_x)
            phase_adv_y_list.append(phase_adv_y)
            err_phase_adv_x_list.append(err_phase_adv_x)
            err_phase_adv_y_list.append(err_phase_adv_y)
        except ValueError:
            # No phase measurement found - set to 0 with infinite error to ignore
            missing_count += 1
            phase_adv_x_list.extend([0.0])
            phase_adv_y_list.extend([0.0])
            err_phase_adv_x_list.extend([float("inf")])
            err_phase_adv_y_list.extend([float("inf")])

    return (
        phase_adv_x_list,
        phase_adv_y_list,
        err_phase_adv_x_list,
        err_phase_adv_y_list,
        missing_count,
    )


def create_worker_payloads(
    optics_dir: Path,
    bpm_pairs: list[tuple[str, str]],
    bad_bpms: list[str] | None,
    template_config: WorkerConfig,
    use_errors: bool = True,
) -> list[tuple[WorkerConfig, OpticsData]]:
    """Create worker payloads for optics optimisation.

    Args:
        optics_dir: Path to directory containing TFS optics measurement files
        bpm_pairs: List of (start_bpm, end_bpm) tuples defining tracking ranges
        bad_bpms: Optional list of BPM names to exclude from analysis
        template_config: Template configuration to use for all workers
        use_errors: Whether to use measurement errors in optimisation

    Returns:
        List of (WorkerConfig, OpticsData) tuples for each worker
    """

    logger.info(f"Loading beta measurements from {optics_dir}")

    tfs_data, common_bpms = load_optics_data_and_find_common_bpms(optics_dir)
    # Extract and filter data frames to common BPMs
    data_frames = {
        key: tfs_data[key].loc[common_bpms] if key not in ("phase_x", "phase_y") else tfs_data[key]
        for key in tfs_data
    }

    # Validate no bad BPMs in measurements
    if bad_bpms and any(
        bpm in data_frames[key].index
        for key in [
            "beta_phase_x",
            "beta_phase_y",
            "beta_amplitude_x",
            "beta_amplitude_y",
            "disp_x",
            "disp_y",
            "orbit_x",
            "orbit_y",
        ]
        for bpm in bad_bpms
    ):
        raise ValueError("Bad BPMs found in optics measurement data.")

    if not bpm_pairs:
        return []

    # Get BPMs that have phase measurements in both planes
    phase_x, phase_y = data_frames["phase_x"], data_frames["phase_y"]
    phase_x_bpms = set(phase_x["NAME"]) | set(phase_x["NAME2"])
    phase_y_bpms = set(phase_y["NAME"]) | set(phase_y["NAME2"])
    bpms_with_phase = [bpm for bpm in common_bpms if bpm in phase_x_bpms and bpm in phase_y_bpms]

    worker_payloads = []

    for start_bpm, end_bpm in bpm_pairs:
        for sdir in (1, -1):
            init_bpm = start_bpm if sdir == 1 else end_bpm

            # Get BPMs in range with phase measurements
            bpm_list = extract_bpm_range_names(bpms_with_phase, start_bpm, end_bpm, sdir)

            if len(bpm_list) < 2:
                logger.warning(
                    f"Skipping BPM range {start_bpm} to {end_bpm} (sdir={sdir}): insufficient BPMs"
                )
                continue

            logger.info(
                f"Using {len(bpm_list)} BPMs in range {start_bpm} to {end_bpm} (sdir={sdir})"
            )

            # Extract phase advances
            phase_x_list, phase_y_list, err_x_list, err_y_list, missing = _extract_phase_advances(
                bpm_list, phase_x, phase_y
            )

            if missing > 0:
                logger.info(
                    f"BPM range {start_bpm} to {end_bpm} (sdir={sdir}): "
                    f"{missing}/{len(bpm_list) - 1} phase measurements missing"
                )

            # Extract beta function measurements at each BPM
            beta_amplitude_x = data_frames["beta_amplitude_x"]
            beta_amplitude_y = data_frames["beta_amplitude_y"]

            beta_x_list = [beta_amplitude_x.loc[bpm, "BETX"] for bpm in bpm_list]
            beta_y_list = [beta_amplitude_y.loc[bpm, "BETY"] for bpm in bpm_list]
            err_beta_x_list = [beta_amplitude_x.loc[bpm, "ERRBETX"] for bpm in bpm_list]
            err_beta_y_list = [beta_amplitude_y.loc[bpm, "ERRBETY"] for bpm in bpm_list]

            # Prepare arrays for phase advances
            comp = np.hstack(
                [np.array(phase_x_list).reshape(-1, 1), np.array(phase_y_list).reshape(-1, 1)]
            )
            err_comp = np.hstack(
                [np.array(err_x_list).reshape(-1, 1), np.array(err_y_list).reshape(-1, 1)]
            )

            # Prepare arrays for beta functions
            beta_comp = np.hstack(
                [np.array(beta_x_list).reshape(-1, 1), np.array(beta_y_list).reshape(-1, 1)]
            )
            err_beta_comp = np.hstack(
                [np.array(err_beta_x_list).reshape(-1, 1), np.array(err_beta_y_list).reshape(-1, 1)]
            )

            if not use_errors or np.all(err_comp == 0):
                logger.warning(
                    f"No valid phase errors for {start_bpm} to {end_bpm}. Using uniform errors."
                )
                err_comp = np.ones_like(err_comp)

            if not use_errors or np.all(err_beta_comp == 0):
                logger.warning(
                    f"No valid beta errors for {start_bpm} to {end_bpm}. Using 10% of beta values as default errors."
                )
                err_beta_comp = 0.1 * beta_comp  # Use 10% of beta values as default errors

            # Create payload
            config = replace(template_config, start_bpm=start_bpm, end_bpm=end_bpm, sdir=sdir)
            data = OpticsData(
                comparisons=comp,
                variances=err_comp**2,
                beta_comparisons=beta_comp,
                beta_variances=err_beta_comp**2,
                init_coords=_get_initial_conditions(
                    init_bpm,
                    data_frames["beta_phase_x"],
                    data_frames["beta_phase_y"],
                    data_frames["disp_x"],
                    data_frames["disp_y"],
                    data_frames["orbit_x"],
                    data_frames["orbit_y"],
                ),
            )
            worker_payloads.append((config, data))
    return worker_payloads


class LHCOpticsController(LHCControllerMixin, OpticsController):
    """
    LHC-specific optics controller that automatically sets sequence file path
    and first BPM based on beam number.
    """

    def __init__(
        self,
        beam: int,
        optics_folder: str | Path,
        bpm_config: BPMConfig,
        magnet_range: str,
        optimiser_config: OptimiserConfig,
        sequence_path: Path | None = None,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        beam_energy: float = 6800.0,
        use_errors: bool = True,
        optimise_knobs: list[str] | None = None,
    ):
        """
        Initialise the LHC optics controller.

        Args:
            beam (int): The beam number (1 or 2)
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_config (BPMConfig): BPM range configuration
            magnet_range (str): Magnet range specification
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration
            sequence_path (Path | None, optional): Path to sequence file. If None, uses default
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            bad_bpms (list[str] | None, optional): List of bad BPMs
            beam_energy (float, optional): Beam energy in GeV. Defaults to 6800.0
            use_errors (bool, optional): Whether to use measurement errors in optimisation. Defaults to True
            optimise_knobs (list[str] | None, optional): List of knob names to optimise. If None, optimises all knobs in magnet_range
        """
        # Create SequenceConfig using mixin helper
        sequence_config = self.create_sequence_config(
            beam=beam,
            magnet_range=magnet_range,
            sequence_path=sequence_path,
            bad_bpms=bad_bpms,
            beam_energy=beam_energy,
            optimise_knobs=optimise_knobs,
        )

        # Load TFS files to find common BPMs and extend bad_bpms
        optics_path = Path(optics_folder)
        tfs_data, common_bpms = load_optics_data_and_find_common_bpms(optics_path)

        # Create temporary MAD to get all BPMs in range
        from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

        temp_mad = OptimisationMadInterface(
            str(sequence_config.sequence_file_path),
            seq_name=sequence_config.seq_name,
            magnet_range=sequence_config.magnet_range,
            bpm_range=sequence_config.magnet_range,  # Use magnet_range as bpm_range
            bad_bpms=sequence_config.bad_bpms,
            beam_energy=sequence_config.beam_energy,
        )
        all_bpms_in_range = temp_mad.all_bpms
        del temp_mad  # Clean up temporary MAD instance

        # Find BPMs in range that don't have measurements
        extra_bad_bpms = [bpm for bpm in all_bpms_in_range if bpm not in common_bpms]
        if extra_bad_bpms:
            sequence_config.bad_bpms = (sequence_config.bad_bpms or []) + extra_bad_bpms
            logger.info(
                f"Extended bad BPMs list with {len(extra_bad_bpms)} BPMs without complete measurements."
            )

        super().__init__(
            sequence_config=sequence_config,
            optics_folder=optics_folder,
            bpm_config=bpm_config,
            optimiser_config=optimiser_config,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            corrector_file=corrector_file,
            tune_knobs_file=tune_knobs_file,
            true_strengths=true_strengths,
            use_errors=use_errors,
        )

        # Filter to optimise only specified knobs if provided
        if optimise_knobs is not None:
            logger.info(f"Filtering optimisation to {len(optimise_knobs)} specified knobs")
            # Filter knob_names
            original_knob_names = self.config_manager.knob_names
            self.config_manager.knob_names = [k for k in original_knob_names if k in optimise_knobs]
            # Filter initial_knobs
            self.initial_knobs = {
                k: v for k, v in self.initial_knobs.items() if k in optimise_knobs
            }
            # Filter true_strengths
            self.filtered_true_strengths = {
                k: v for k, v in self.filtered_true_strengths.items() if k in optimise_knobs
            }
            logger.info(
                f"Optimising {len(self.config_manager.knob_names)} knobs: {self.config_manager.knob_names}"
            )
