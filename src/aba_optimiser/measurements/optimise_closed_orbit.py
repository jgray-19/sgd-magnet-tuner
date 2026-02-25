from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nxcals.spark_session_builder import get_or_create
from omc3.machine_data_extraction.nxcals_knobs import get_energy
from pymadng_utils.io.utils import save_knobs

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.measurements.create_datafile import (
    process_measurements,
    save_online_knobs,
)
from aba_optimiser.measurements.squeeze_helpers import get_or_make_sequence
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import SequenceConfig
from aba_optimiser.training.controller_helpers import create_arc_measurement_config

logger = logging.getLogger(__name__)


@dataclass
class RangeConfig:
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


@dataclass
class MeasurementSetupConfig:
    beam: int
    model_dir: str
    arc_config: RangeConfig
    folder: str
    name_prefix: str
    times: list[str]
    title: str


def weighted_mean(values: list[float], uncertainties: list[float]) -> float:
    """Compute weighted mean where weights are 1/sigma^2."""
    finite_pairs = [(v, u) for v, u in zip(values, uncertainties) if u > 0]
    if not finite_pairs:
        raise ValueError("Cannot compute weighted mean without positive uncertainties")
    weights = [1 / (u**2) for _, u in finite_pairs]
    numerator = sum(v * w for (v, _), w in zip(finite_pairs, weights))
    return numerator / sum(weights)


def compute_weighted_mean_and_variance(
    sub: pd.DataFrame, value_col: str, var_col: str
) -> tuple[float, float]:
    """Compute inverse-variance weighted mean and its variance.

    Args:
        sub: DataFrame subset for a single BPM
        value_col: Column name for values to average
        var_col: Column name for variances

    Returns:
        Tuple of (weighted_mean, variance_of_mean)
    """
    vals = sub[value_col].to_numpy()
    vars_ = sub[var_col].to_numpy()
    mask = np.isfinite(vals) & np.isfinite(vars_) & (vars_ > 0)
    vals = vals[mask]
    vars_ = vars_[mask]

    # Fallback when no valid variances are available
    if vals.size == 0:
        mu = float(sub[value_col].mean())
        n = sub[value_col].count()
        if n >= 2:
            v_unw = float(np.var(sub[value_col].to_numpy(), ddof=1))
            var_mean = v_unw / n
        else:
            var_mean = np.nan
        return mu, var_mean

    w = 1.0 / vars_
    sum_w = float(np.sum(w))
    mu = float(np.sum(w * vals) / sum_w)

    # Variance of weighted mean from reported measurement variances
    var_mean = 1.0 / sum_w

    return mu, var_mean


def optimise_ranges(
    range_config: RangeConfig,
    range_type: str,
    beam: int,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    sequence_path: Path,
    corrector_knobs_file: Path,
    tune_knobs_file: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    energy: float,
    write_tensorboard_logs: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    """Optimise for a given range configuration.

    Returns:
        Tuple of (deltap_wrt_6800_list, deltap_uncertainties, fitted_deltap_list).
    """
    results = []
    uncertainties = []
    fitted_deltaps = []
    num_ranges = len(range_config.magnet_ranges)
    for i in range(num_ranges):
        logger.info(f"Starting optimisation for {range_type} {i + 1}/{num_ranges} for {title}")

        # Create LHC accelerator instance
        accelerator = LHC(
            beam=beam,
            beam_energy=energy,
            sequence_file=sequence_path,
            optimise_energy=True,  # Since we're optimizing deltap/pt
        )

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
            first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        )

        measurement_config = create_arc_measurement_config(
            measurement_file,
            num_tracks=1,
            flattop_turns=3,
            corrector_files=corrector_knobs_file,
            tune_knobs_files=tune_knobs_file,
        )

        controller = Controller(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            measurement_config,
            range_config.bpm_starts[i],
            range_config.bpm_end_points[i],
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
            write_tensorboard_logs=write_tensorboard_logs,
        )
        final_knobs, uncs = controller.run()
        fitted_deltap = final_knobs["deltap"]
        fitted_deltaps.append(fitted_deltap)
        # Convert to reference energy 6800 GeV (assume beta is 1 and in GeV)
        e_ref = 6800
        e_meas = energy * (1 + fitted_deltap)
        deltap_wrt_6800 = (e_meas - e_ref) / e_ref
        results.append(deltap_wrt_6800)
        # results.append(fitted_deltap)
        uncertainties.append(uncs["deltap"])  # Assuming uncs is a dict with 'deltap'
        logger.info(f"{range_type.capitalize()} {i + 1}: deltap = {results[-1]}")
        logger.info(f"Finished optimisation for {range_type} {i + 1}/{num_ranges} for {title}")
    return results, uncertainties, fitted_deltaps


def optimise_corrector_ranges(
    range_config: RangeConfig,
    range_type: str,
    beam: int,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    sequence_path: Path,
    corrector_knobs_file: Path,
    tune_knobs_file: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    energy: float,
    machine_deltap: float,
) -> list[dict[str, float]]:
    """Optimise correctors for a given range configuration."""
    results = []
    num_ranges = len(range_config.magnet_ranges)
    for i in range(num_ranges):
        logger.info(
            f"Starting corrector optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )

        # Create LHC accelerator instance
        accelerator = LHC(
            beam=beam,
            beam_energy=energy,
            sequence_file=sequence_path,
            optimise_correctors=True,
        )

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
            first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        )

        meas_config = create_arc_measurement_config(
            measurement_file,
            machine_deltap=machine_deltap,
            num_tracks=1,
            flattop_turns=3,
            corrector_files=corrector_knobs_file,
            tune_knobs_files=tune_knobs_file,
        )

        controller = Controller(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            meas_config,
            range_config.bpm_starts[i],
            range_config.bpm_end_points[i],
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
        )
        final_knobs, _ = controller.run()
        results.append(final_knobs)
        logger.info(
            f"Finished corrector optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )
    return results


def create_beam1_configs(
    folder: str, name_prefix: str, fixed_bpm: bool
) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 1."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    skip_step = 3 if fixed_bpm else 5
    arc_magnet_ranges_b1 = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    arc_bpm_starts_b1 = [[f"BPM.{i}R{s}.B1" for i in range(9, 35, skip_step)] for s in range(1, 9)]
    arc_bpm_end_points_b1 = [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in range(9, 34, skip_step)] for s in range(1, 9)
    ]

    arc_config_b1 = RangeConfig(
        magnet_ranges=arc_magnet_ranges_b1,
        bpm_starts=arc_bpm_starts_b1,
        bpm_end_points=arc_bpm_end_points_b1,
    )

    return [
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_53_05_820", "07_54_13_858"],
            title="0",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_08_02_826", "08_09_11_940"],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_11_13_745", "08_12_25_817"],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_18_09_980", "08_19_16_847"],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_23_20_980", "08_24_32_020"],
            title="m0p2",
        ),
    ]


def create_beam2_configs(
    folder: str, name_prefix: str, use_fixed_bpm: bool
) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 2."""
    model_dir_b2 = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    )
    # Arc settings
    skip_step = 2 if use_fixed_bpm else 4
    arc_magnet_ranges_b2 = [f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
    arc_bpm_starts_b2 = [
        [f"BPM.{i}L{s}.B2" for i in range(9, 34, skip_step)] for s in range(8, 0, -1)
    ]
    arc_bpm_end_points_b2 = [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 35, skip_step)] for s in range(8, 0, -1)
    ]
    arc_config_b2 = RangeConfig(
        magnet_ranges=arc_magnet_ranges_b2,
        bpm_starts=arc_bpm_starts_b2,
        bpm_end_points=arc_bpm_end_points_b2,
    )

    return [
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_35_27_940", "07_36_39_380", "07_38_44_035"],
            # times=["07_36_39_380", "07_38_44_035"],
            title="0",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_57_30_885", "08_00_44_900"],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_04_55_798", "08_06_06_900", "08_07_13_900"],
            # times=["08_06_06_900", "08_07_13_900"],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_15_06_860", "08_16_13_980"],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_19_35_860", "08_22_57_752", "08_18_27_900"],
            title="m0p2",
        ),
    ]


def process_single_config(
    config: MeasurementSetupConfig,
    temp_analysis_dir: Path,
    date: str,
    skip_reload: bool,
    optimise_correctors: bool,
    use_fixed_bpm: bool = True,
) -> None:
    """Process a single measurement configuration.

    Args:
        config: Measurement configuration for this run
        temp_analysis_dir: Temporary directory for analysis outputs
        date: Date string in YYYY-MM-DD format
        skip_reload: If True, skip reloading strengths from LSA and reuse existing analysis
        optimise_correctors: If True, optimise correctors after energy optimisation
        use_fixed_bpm: If True (default), use fixed reference BPM approach.
                       If False, create all combinations of start/end BPMs (Cartesian product).
    """
    results_dir = PROJECT_ROOT / f"b{config.beam}co_results"
    tune_knobs_file = results_dir / f"tune_knobs_{config.title}.txt"
    corrector_knobs_file = results_dir / f"corrector_knobs_{config.title}.txt"
    results_dir.mkdir(exist_ok=True)

    # Delete temp_analysis_dir if it exists
    temp_analysis_dir.mkdir(exist_ok=True)

    bad_bpms_file = results_dir / f"bad_bpms_{config.title}.txt"
    measurement_filename = "pz_data.parquet"
    measurement_file = temp_analysis_dir / measurement_filename

    # Compute meas_time always
    if not config.times:
        logger.warning(f"No times specified for config {config.title}, skipping.")
        return
    earliest_time = min(config.times)
    # e.g., "07_53_05_820" -> "07:53:05"
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{date} {time_str}"

    tz = ZoneInfo("UTC")
    meas_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)

    # Get beam energy from NXCALS always
    spark = get_or_create()
    energy, _ = get_energy(spark, meas_time)
    spark.stop()
    del spark

    bad_bpms: list[str] | None = None
    if skip_reload:
        # Read the bad bpms from the file
        with bad_bpms_file.open("r") as f:
            bad_bpms = [line.strip() for line in f.readlines()]

    if not skip_reload:
        save_online_knobs(
            meas_time,
            beam=config.beam,
            tune_knobs_file=tune_knobs_file,
            corrector_knobs_file=corrector_knobs_file,
        )

    # Generate files from times
    files = [Path(f"{config.folder}/{config.name_prefix}{time}.sdds") for time in config.times]

    pzs_dict, bad_bpms, output_paths, _ = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        beam=config.beam,
        filename=None,
        bad_bpms=bad_bpms,
    )
    pzs = pzs_dict["combined"]
    ana_dir = output_paths["combined"]

    file_path = ana_dir / measurement_filename
    sequence_file = get_or_make_sequence(config.beam, Path(config.model_dir))

    # Compute weighted averages per BPM with proper variance of the mean
    rows = []
    for name, sub in pzs.groupby("name"):
        mu_x, vm_x = compute_weighted_mean_and_variance(sub, "x", "var_x")
        mu_y, vm_y = compute_weighted_mean_and_variance(sub, "y", "var_y")
        mu_px, vm_px = compute_weighted_mean_and_variance(sub, "px", "var_px")
        mu_py, vm_py = compute_weighted_mean_and_variance(sub, "py", "var_py")
        rows.append(
            {
                "name": name,
                "x": mu_x,
                "y": mu_y,
                "px": mu_px,
                "py": mu_py,
                # Store the variance of the weighted mean for downstream weighting
                "var_x": vm_x,
                "var_y": vm_y,
                "var_px": vm_px,
                "var_py": vm_py,
            }
        )

    averaged = pd.DataFrame(rows)
    print(
        averaged["var_x"].describe(),
        averaged["var_y"].describe(),
        averaged["var_px"].describe(),
        averaged["var_py"].describe(),
    )

    # Create new DataFrame with 3 turns, each with averaged values
    new_rows = []
    for turn in [1, 2, 3]:
        for _, row in averaged.iterrows():
            new_rows.append(
                {
                    "name": row["name"],
                    "turn": turn,
                    "x": row["x"],
                    "y": row["y"],
                    "px": row["px"],
                    "py": row["py"],
                    "var_x": row["var_x"],
                    "var_y": row["var_y"],
                    "var_px": row["var_px"],
                    "var_py": row["var_py"],
                    "kick_plane": "xy",
                }
            )
    new_df = pd.DataFrame(new_rows)
    new_df["name"] = new_df["name"].astype("category")
    new_df["turn"] = new_df["turn"].astype("int32")

    # Overwrite the measurement file
    new_df.to_parquet(file_path)

    if not skip_reload:
        # Save the bad bpms to a file
        with bad_bpms_file.open("w") as f:
            for bpm in bad_bpms:
                f.write(f"{bpm}\n")

    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=5,
        warmup_lr_start=5e-2,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-9,
        optimiser_type="lbfgs",
        expected_rel_error=0,
    )
    simulation_config = SimulationConfig(
        # For pre trimmed data
        tracks_per_worker=1,
        num_batches=1,
        num_workers=1,
        use_fixed_bpm=use_fixed_bpm,
        optimise_momenta=True,
    )

    results_arcs, uncs_arcs, fitted_deltaps = optimise_ranges(
        config.arc_config,
        "arc",
        config.beam,
        optimiser_config,
        simulation_config,
        sequence_file,
        corrector_knobs_file,
        tune_knobs_file,
        measurement_file,
        bad_bpms,
        config.title,
        energy,
    )

    logger.info(f"All arc optimisations complete for {config.title}.")
    logger.info("Final deltaps for each arc:")
    for i, dp in enumerate(results_arcs):
        logger.info(f"Arc {i + 1}: deltap = {dp}")

    if not results_arcs:
        logger.warning("No arc results produced; skipping summary output.")
        return

    try:
        mean_arcs = weighted_mean(results_arcs, uncs_arcs)
    except ValueError:
        mean_arcs = float(np.mean(results_arcs))
        logger.warning(
            "Falling back to unweighted mean for arcs due to non-positive uncertainties."
        )
    try:
        mean_fitted_deltap = weighted_mean(fitted_deltaps, uncs_arcs)
    except ValueError:
        mean_fitted_deltap = float(np.mean(fitted_deltaps))
        logger.warning(
            "Falling back to unweighted mean for fitted deltaps due to non-positive uncertainties."
        )
    logger.info(f"Weighted mean deltap arcs: {mean_arcs}")
    logger.info(f"Std dev of deltap arcs: {np.std(results_arcs)}")
    logger.info(f"Weighted mean fitted deltap: {mean_fitted_deltap}")

    # Write the results to a file
    results_file = results_dir / f"{config.title}.txt"
    with results_file.open("w") as f:
        f.write("range\tdeltap\n")

    with results_file.open("a") as f:
        for i, dp in enumerate(results_arcs):
            f.write(f"arc{i + 1}\t{dp}\n")
        f.write(f"MeanArcs\t{mean_arcs}\n")
        std_arcs = float(np.std(results_arcs))
        f.write(f"StdDevArcs\t{std_arcs}\n")
        stderr = std_arcs / np.sqrt(len(results_arcs)) if len(results_arcs) > 0 else 0.0
        f.write(f"StdErrArcs\t{stderr}\n")

    if not optimise_correctors:
        return

    corrector_optimiser_config = replace(
        optimiser_config,
        max_epochs=3000,
        warmup_epochs=500,
        min_lr=1e-1,
        warmup_lr_start=1e-4,
    )
    corrector_simulation_config = replace(
        simulation_config,
        optimise_energy=False,
        optimise_correctors=True,
    )
    corrector_results = optimise_corrector_ranges(
        config.arc_config,
        "arc",
        config.beam,
        corrector_optimiser_config,
        corrector_simulation_config,
        sequence_file,
        corrector_knobs_file,
        tune_knobs_file,
        measurement_file,
        bad_bpms,
        config.title,
        energy,
        mean_fitted_deltap,
    )

    combined_correctors: dict[str, float] = {}
    for arc_idx, arc_knobs in enumerate(corrector_results, start=1):
        for knob, value in arc_knobs.items():
            combined_correctors[knob] = value

    combined_correctors_file = results_dir / f"corrector_knobs_{config.title}_optimised.txt"
    save_knobs(combined_correctors, combined_correctors_file)
    logger.info("Saved combined corrector knobs to %s", combined_correctors_file)


def main():
    """Main function to run the measurement processing loop."""
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, choices=[1, 2], help="Beam number 1 or 2", default=2)
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Skip reloading strengths from LSA and redoing analysis",
    )
    parser.add_argument(
        "--no-fixed-bpm",
        action="store_true",
        help="Disable fixed BPM for start/end points, pair BPMs element-wise instead",
    )
    parser.add_argument(
        "--optimise-correctors",
        action="store_true",
        help="Optimise correctors after energy optimisation",
    )
    args = parser.parse_args()

    # Define date
    date = "2025-11-07"
    # folder = "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/11259/BPM"
    folder = "/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11259/BPM"
    name_prefix = f"Beam{args.beam}@BunchTurn@{date.replace('-', '_')}@"

    # Determine use_fixed_bpm from args
    use_fixed_bpm: bool = not args.no_fixed_bpm

    # Get configurations based on beam
    if args.beam == 1:
        configs = create_beam1_configs(folder, name_prefix, use_fixed_bpm)
    else:
        configs = create_beam2_configs(folder, name_prefix, use_fixed_bpm)

    # Temporary analysis directory
    temp_analysis_dir = PROJECT_ROOT / f"temp_analysis_co_{args.beam}"

    # Process each configuration
    for config in configs:
        process_single_config(
            config,
            temp_analysis_dir,
            date,
            args.skip_reload,
            args.optimise_correctors,
            use_fixed_bpm,
        )


if __name__ == "__main__":
    main()
