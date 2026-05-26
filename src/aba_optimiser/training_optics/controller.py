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
from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.utils import extract_bpm_range_names
from aba_optimiser.training.worker_lifecycle import WorkerLifecycleManager
from aba_optimiser.workers import OpticsData, OpticsWorker, WorkerConfig

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.controller_config import OutputConfig, SequenceConfig

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
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        use_amplitude_beta: bool = True,
        include_phase_advances: bool = False,
        output_config: OutputConfig | None = None,
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
            initial_knob_strengths (dict[str, float] | None): Initial knob strengths.
            corrector_file (Path | None): Path to corrector strengths file.
            tune_knobs_file (Path | None): Path to tune knobs file.
            true_strengths (Path | dict[str, float] | None): True strengths (Path, dict, or None).
            use_errors (bool): Whether to use measurement errors in optimisation.
            use_amplitude_beta (bool): Use beta from amplitude (True) or phase (False
            include_phase_advances (bool): Include phase-advance targets alongside beta targets.
            output_config (OutputConfig | None): Output and logging configuration.
        """
        logger.info("Optimising quadrupoles to match measurement-built optics")

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
            sequence_config=sequence_config,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            output_config=output_config,
        )

        # Store optics-specific attributes
        self.optics_folder = Path(optics_folder)
        self.corrector_file = corrector_file
        self.tune_knobs_file = tune_knobs_file
        self.use_errors = use_errors
        self.use_amplitude_beta = use_amplitude_beta
        self.include_phase_advances = include_phase_advances
        self.target_twiss = load_optics_data(self.optics_folder, use_amplitude_beta)

        # Create optics-specific worker payloads
        template_config = WorkerConfig(
            accelerator=accelerator,
            tracking_start_bpm="TEMP",
            tracking_end_bpm="TEMP",
            magnet_range=sequence_config.magnet_range,
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            sdir=0,
            bad_bpms=sequence_config.bad_bpms,
            mad_logfile=self.mad_logfile,
            python_logfile=self.python_logfile,
        )

        # Use explicit BPM (start, end) pairs from config manager
        self.worker_payloads = create_worker_payloads(
            self.target_twiss,
            self.config_manager.all_bpms,
            self.config_manager.bpm_pairs,
            sequence_config.bad_bpms,
            template_config,
            self.use_errors,
            self.include_phase_advances,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process using optics workers."""
        writer = self.setup_logging("optics_opt")
        worker_manager = WorkerLifecycleManager(OpticsWorker)
        self.final_knobs = None

        try:
            worker_manager.create_and_start_workers(
                [(data, config, self.simulation_config) for config, data in self.worker_payloads],
                send_handshake=False,
            )
            channels = worker_manager.channels
            if channels is None:
                raise RuntimeError("Worker channels are not initialised")

            self.final_knobs = self.optimisation_loop.run_optimisation(
                self.initial_knobs,
                channels,
                writer,
                run_start=time.time(),
                total_turns=1,
            )
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected. Terminating optics optimisation early.")
            self.final_knobs = self.optimisation_loop.best_knobs
        finally:
            worker_manager.terminate_workers()
            initial_knobs_abs = self._deltas_to_abs()

        uncertainties = self._finalise_results(initial_knobs_abs, writer)
        return self.final_knobs, dict(zip(self.final_knobs.keys(), uncertainties, strict=False))

    def _deltas_to_abs(self) -> dict[str, float]:
        """Keep optics optimisation results in optimisation space."""
        initial_knobs_delta = dict(
            zip(
                self.config_manager.knob_names,
                self.config_manager.initial_strengths,
                strict=False,
            )
        )
        if self.final_knobs is None:
            self.final_knobs = self.optimisation_loop.best_knobs

        self.final_knobs = self.final_knobs.copy()
        self.filtered_true_strengths = self.filtered_true_strengths.copy()
        return initial_knobs_delta

    def _finalise_results(
        self,
        initial_knobs_abs: dict[str, float],
        writer,
    ) -> np.ndarray:
        """Save optics optimisation results in optimisation space."""
        if writer is not None:
            writer.close()

        uncertainties_abs = np.zeros(len(self.config_manager.knob_names), dtype=np.float64)
        logger.info("Optics optimisation complete.")
        return uncertainties_abs


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
    twiss_df: pd.DataFrame,
    all_bpms: list[str],
    bpm_pairs: list[tuple[str, str]],
    bad_bpms: list[str] | None,
    template_config: WorkerConfig,
    use_errors: bool = True,
    include_phase_advances: bool = False,
) -> list[tuple[WorkerConfig, OpticsData]]:
    """Create worker payloads for optics optimisation.

    Args:
        twiss_df: Measurement-built optics Twiss used as the optimisation target.
        all_bpms: Full model BPM ordering.
        bpm_pairs: List of (start_bpm, end_bpm) tuples defining tracking ranges
        bad_bpms: Optional list of BPM names to exclude from analysis
        template_config: Template configuration to use for all workers
        use_errors: Whether to use measurement errors in optimisation
        include_phase_advances: Include phase advances in the loss alongside beta targets

    Returns:
        List of (WorkerConfig, OpticsData) tuples for each worker
    """
    logger.info(
        "Preparing optics worker payloads from measurement-built Twiss (%sphase targets)",
        "with " if include_phase_advances else "without ",
    )

    if not bpm_pairs:
        raise ValueError("No BPM pairs provided for worker payload creation")

    filtered_model_bpms = [bpm for bpm in all_bpms if bad_bpms is None or bpm not in bad_bpms]
    worker_payloads = []

    unique_bpm_pairs = list(dict.fromkeys(bpm_pairs))

    for start_bpm, end_bpm in unique_bpm_pairs:
        for sdir in (1, -1):
            try:
                bpm_list = extract_bpm_range_names(filtered_model_bpms, start_bpm, end_bpm, sdir)
            except ValueError:
                logger.warning(
                    f"Skipping BPM range {start_bpm} to {end_bpm} (sdir={sdir}): BPM(s) not found in model"
                )
                continue

            additional_bad_bpms = [bpm for bpm in bpm_list if bpm not in twiss_df.index]
            bpm_list = [bpm for bpm in bpm_list if bpm in twiss_df.index]

            if len(bpm_list) < 2:
                logger.warning(
                    f"Skipping BPM range {start_bpm} to {end_bpm} (sdir={sdir}): insufficient BPMs"
                )
                continue

            init_bpm = bpm_list[0]
            logger.info(
                f"Using {len(bpm_list)} measured BPMs in range {bpm_list[0]} to {bpm_list[-1]} (sdir={sdir})"
            )

            if include_phase_advances:
                phase_x_list, phase_y_list, err_x_list, err_y_list, missing = _extract_phase_advances(
                    bpm_list, twiss_df
                )
                if missing > 0:
                    logger.info(
                        f"BPM range {start_bpm} to {end_bpm} (sdir={sdir}): "
                        f"{missing}/{len(bpm_list) - 1} phase measurements missing"
                    )
            else:
                phase_x_list = [0.0] * (len(bpm_list) - 1)
                phase_y_list = [0.0] * (len(bpm_list) - 1)
                err_x_list = [float("inf")] * (len(bpm_list) - 1)
                err_y_list = [float("inf")] * (len(bpm_list) - 1)

            beta_x_list = [twiss_df.loc[bpm, "BETX"] for bpm in bpm_list]
            beta_y_list = [twiss_df.loc[bpm, "BETY"] for bpm in bpm_list]
            err_beta_x_list = [twiss_df.loc[bpm, "ERRBETX"] for bpm in bpm_list]
            err_beta_y_list = [twiss_df.loc[bpm, "ERRBETY"] for bpm in bpm_list]

            comp = np.hstack(
                [np.array(phase_x_list).reshape(-1, 1), np.array(phase_y_list).reshape(-1, 1)]
            )
            err_comp = np.hstack(
                [np.array(err_x_list).reshape(-1, 1), np.array(err_y_list).reshape(-1, 1)]
            )

            beta_comp = np.hstack(
                [np.array(beta_x_list).reshape(-1, 1), np.array(beta_y_list).reshape(-1, 1)]
            )
            err_beta_comp = np.hstack(
                [np.array(err_beta_x_list).reshape(-1, 1), np.array(err_beta_y_list).reshape(-1, 1)]
            )

            if include_phase_advances and (
                not use_errors or not np.any(np.isfinite(err_comp) & (err_comp > 0))
            ):
                logger.warning(
                    f"No valid phase errors for {start_bpm} to {end_bpm}. Using 10% of phase values."
                )
                err_comp = 0.001 * np.maximum(np.abs(comp), 1e-12)

            if not use_errors or not np.any(np.isfinite(err_beta_comp) & (err_beta_comp > 0)):
                logger.warning(
                    f"No valid beta errors for {start_bpm} to {end_bpm}. Using 10% of beta values."
                )
                err_beta_comp = 0.1 * np.maximum(np.abs(beta_comp), 1e-12)

            config = replace(template_config, start_bpm=start_bpm, end_bpm=end_bpm, sdir=sdir)
            if additional_bad_bpms:
                config = replace(
                    config,
                    bad_bpms=additional_bad_bpms + ([] if config.bad_bpms is None else config.bad_bpms),
                )

            data = OpticsData(
                comparisons=comp,
                variances=err_comp**2,
                beta_comparisons=beta_comp,
                beta_variances=err_beta_comp**2,
                init_coords=_get_initial_conditions(init_bpm, twiss_df),
            )
            worker_payloads.append((config, data))
    return worker_payloads
