"""Controller for optics optimisation (beta functions using quadrupole strengths)."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omc3.optics_measurements.constants import PHASE_ADV
from tmom_recon import build_twiss_from_measurements

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad import get_mad_interface
from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.utils import extract_bpm_range_names
from aba_optimiser.training.worker_lifecycle import WorkerLifecycleManager
from aba_optimiser.workers import OpticsData, OpticsWorker, WorkerConfig

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.controller_config import SequenceConfig

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
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        optimiser_config: OptimiserConfig,
        optics_folder: str | Path,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        use_amplitude_beta: bool = True,
    ):
        """
        Initialise the optics controller.

        Args:
            accelerator (Accelerator): Accelerator instance defining machine configuration.
            sequence_config (SequenceConfig): Sequence configuration of BPMs and magnets.
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration.
            optics_folder (str | Path): Path to directory containing TFS optics measurement files.
            bpm_start_points (list[str]): Start BPMs for each range.
            bpm_end_points (list[str]): End BPMs for each range.
            show_plots (bool): Whether to show plots.
            initial_knob_strengths (dict[str, float] | None): Initial knob strengths.
            corrector_file (Path | None): Path to corrector strengths file.
            tune_knobs_file (Path | None): Path to tune knobs file.
            true_strengths (Path | dict[str, float] | None): True strengths (Path, dict, or None).
            use_errors (bool): Whether to use measurement errors in optimisation.
            use_amplitude_beta (bool): Use beta from amplitude (True) or phase (False
        """
        logger.info("Optimising quadrupoles for beta functions")

        # Create optics-specific simulation config
        simulation_config = SimulationConfig(
            tracks_per_worker=1,
            num_workers=1,
            num_batches=1,
            use_fixed_bpm=True,
        )

        # Initialize base controller
        super().__init__(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            magnet_range=sequence_config.magnet_range,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            bad_bpms=sequence_config.bad_bpms,
            first_bpm=sequence_config.first_bpm,
        )

        # Store optics-specific attributes
        self.optics_folder = Path(optics_folder)
        self.corrector_file = corrector_file
        self.tune_knobs_file = tune_knobs_file
        self.use_errors = use_errors
        self.use_amplitude_beta = use_amplitude_beta

        # Create optics-specific worker payloads
        optics_path = Path(optics_folder)
        template_config = WorkerConfig(
            accelerator=accelerator,
            start_bpm="TEMP",
            end_bpm="TEMP",
            magnet_range=sequence_config.magnet_range,
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            sdir=0,
            bad_bpms=sequence_config.bad_bpms,
        )

        # Use explicit BPM (start, end) pairs from config manager
        self.worker_payloads = create_worker_payloads(
            accelerator,
            optics_path,
            self.config_manager.bpm_pairs,
            sequence_config.bad_bpms,
            template_config,
            self.use_errors,
            self.use_amplitude_beta,
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


def load_optics_data(
    optics_dir: Path,
    use_amplitude_beta: bool = True,
) -> pd.DataFrame:
    """Load optics measurements using build_twiss_from_measurements.

    Args:
        optics_dir: Path to directory containing TFS optics measurement files
        use_amplitude_beta: Use beta from amplitude (True) or phase (False)

    Returns:
        Twiss dataframe with measurement data
    """
    logger.info(f"Loading optics measurements from {optics_dir}")
    logger.info(f"Using beta from {'amplitude' if use_amplitude_beta else 'phase'} measurements")
    twiss_df, has_dispersion = build_twiss_from_measurements(
        optics_dir, include_errors=True, use_amplitude_beta=use_amplitude_beta
    )

    if not has_dispersion:
        logger.warning("Dispersion data not found in measurements")

    return twiss_df


def find_phase_advance_between_bpms(
    twiss_df: pd.DataFrame,
    bpm_start: str,
    bpm_end: str,
    plane: str = "X",
) -> tuple[float, float]:
    """Find phase advance between two BPMs using cumulative phase from twiss.

    Args:
        twiss_df: Twiss dataframe with PHASEADVX/PHASEADVY and mu1_var/mu2_var columns
        bpm_start: Starting BPM name
        bpm_end: Ending BPM name
        plane: Plane ('X' or 'Y')

    Returns:
        Tuple of (phase_advance, error)

    Raises:
        ValueError: If BPMs not found in dataframe
    """
    if bpm_start not in twiss_df.index:
        raise ValueError(f"BPM {bpm_start} not found in twiss dataframe")
    if bpm_end not in twiss_df.index:
        raise ValueError(f"BPM {bpm_end} not found in twiss dataframe")

    phase_col = f"{PHASE_ADV}{plane}"
    var_col = "mu1_var" if plane == "X" else "mu2_var"
    total_var_key = "MU1_TOTAL_VAR" if plane == "X" else "MU2_TOTAL_VAR"

    # Calculate phase advance as difference in cumulative phase
    phase_start = twiss_df.loc[bpm_start, phase_col]
    phase_end = twiss_df.loc[bpm_end, phase_col]
    phase_advance = (phase_end - phase_start) % 1.0

    # Get cumulative variances and total variance
    var_start = twiss_df.loc[bpm_start, var_col]
    var_end = twiss_df.loc[bpm_end, var_col]
    total_var = twiss_df.headers.get(total_var_key, 0.0)

    # Check if we wrap around the ring (start phase > end phase)
    variance = total_var - var_start + var_end if phase_start > phase_end else var_end - var_start

    # Handle potential negative values from numerical precision
    variance = max(0.0, variance)
    error = np.sqrt(variance)

    return phase_advance, error


def _get_initial_conditions(
    bpm: str,
    twiss_df: pd.DataFrame,
) -> dict[str, float]:
    """Extract initial Twiss parameters and orbit for a BPM."""
    row = twiss_df.loc[bpm]
    return {
        "beta11": float(row["BETX"]),
        "beta22": float(row["BETY"]),
        "alfa11": float(row["ALFX"]),
        "alfa22": float(row["ALFY"]),
        "dx": float(row.get("DX", 0.0)),
        "dpx": float(row.get("DPX", 0.0)),
        "dy": float(row.get("DY", 0.0)),
        "dpy": float(row.get("DPY", 0.0)),
        "x": float(row["X"]),
        "y": float(row["Y"]),
    }


def _extract_phase_advances(
    bpm_list: list[str],
    twiss_df: pd.DataFrame,
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
                twiss_df, bpm1, bpm2, "X"
            )
            phase_adv_y, err_phase_adv_y = find_phase_advance_between_bpms(
                twiss_df, bpm1, bpm2, "Y"
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
    accelerator: Accelerator,
    optics_dir: Path,
    bpm_pairs: list[tuple[str, str]],
    bad_bpms: list[str] | None,
    template_config: WorkerConfig,
    use_errors: bool = True,
    use_amplitude_beta: bool = True,
) -> list[tuple[WorkerConfig, OpticsData]]:
    """Create worker payloads for optics optimisation.

    Args:
        optics_dir: Path to directory containing TFS optics measurement files
        bpm_pairs: List of (start_bpm, end_bpm) tuples defining tracking ranges
        bad_bpms: Optional list of BPM names to exclude from analysis
        template_config: Template configuration to use for all workers
        use_errors: Whether to use measurement errors in optimisation
        use_amplitude_beta: Use beta from amplitude (True) or phase (False)

    Returns:
        List of (WorkerConfig, OpticsData) tuples for each worker
    """

    logger.info(f"Loading beta measurements from {optics_dir}")

    twiss_df = load_optics_data(optics_dir, use_amplitude_beta)

    if not bpm_pairs:
        raise ValueError("No BPM pairs provided for worker payload creation")

    worker_payloads = []

    for start_bpm, end_bpm in bpm_pairs:
        for sdir in (1, -1):
            # Choose init_bpm as the first good BPM in the list
            init_bpm = start_bpm if sdir == 1 else end_bpm

            # Get all BPMs in the sequence for range extraction
            temp_mad = get_mad_interface(accelerator)(
                accelerator,
                magnet_range=template_config.magnet_range,
                bpm_range=f"{start_bpm}/{end_bpm}",
                bad_bpms=bad_bpms,  # Filter out bad BPMs
            )
            all_bpms = temp_mad.all_bpms
            del temp_mad  # Clean up

            additional_bad_bpms = list(set(all_bpms) - set(twiss_df.index))
            all_bpms = [bpm for bpm in all_bpms if bpm in twiss_df.index]

            try:
                # Get all BPMs in range from sequence (includes bad BPMs)
                bpm_list = extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir)
            except ValueError:
                logger.warning(
                    f"Skipping BPM range {start_bpm} to {end_bpm} (sdir={sdir}): BPM(s) not found in model"
                )
                continue

            if len(bpm_list) < 2:
                logger.warning(
                    f"Skipping BPM range {start_bpm} to {end_bpm} (sdir={sdir}): insufficient BPMs"
                )
                continue

            logger.info(
                f"Using {len(bpm_list)} BPMs in range {all_bpms[0]} to {all_bpms[-1]} (sdir={sdir})"
            )

            # Extract phase advances
            phase_x_list, phase_y_list, err_x_list, err_y_list, missing = _extract_phase_advances(
                bpm_list, twiss_df
            )

            if missing > 0:
                logger.info(
                    f"BPM range {start_bpm} to {end_bpm} (sdir={sdir}): "
                    f"{missing}/{len(bpm_list) - 1} phase measurements missing"
                )

            # Extract beta function measurements at each BPM
            beta_x_list = [twiss_df.loc[bpm, "BETX"] for bpm in bpm_list]
            beta_y_list = [twiss_df.loc[bpm, "BETY"] for bpm in bpm_list]
            err_beta_x_list = [twiss_df.loc[bpm, "ERRBETX"] for bpm in bpm_list]
            err_beta_y_list = [twiss_df.loc[bpm, "ERRBETY"] for bpm in bpm_list]

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
                    f"No valid phase errors for {start_bpm} to {end_bpm}. Using 10% of phase values."
                )
                err_comp = 0.001 * comp

            if not use_errors or np.all(err_beta_comp == 0):
                logger.warning(
                    f"No valid beta errors for {start_bpm} to {end_bpm}. Using 10% of beta values."
                )
                err_beta_comp = 0.1 * beta_comp

            # Create payload
            config = replace(template_config, start_bpm=start_bpm, end_bpm=end_bpm, sdir=sdir)
            if additional_bad_bpms:
                if config.bad_bpms is None:
                    config = replace(config, bad_bpms=additional_bad_bpms)
                else:
                    config.bad_bpms = additional_bad_bpms + config.bad_bpms

            data = OpticsData(
                comparisons=comp,
                variances=err_comp**2,
                beta_comparisons=beta_comp,
                beta_variances=err_beta_comp**2,
                init_coords=_get_initial_conditions(init_bpm, twiss_df),
            )
            worker_payloads.append((config, data))
    return worker_payloads
