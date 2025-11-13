from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nxcals.spark_session_builder import get_or_create
from pylhc.nxcal_knobs import get_energy

from aba_optimiser.config import PROJECT_ROOT, OptSettings
from aba_optimiser.measurements.create_datafile import (
    process_measurements,
    save_online_knobs,
)
from aba_optimiser.training.controller import LHCController as Controller

logger = logging.getLogger(__name__)


@dataclass
class RangeConfig:
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


@dataclass
class MeasurementConfig:
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


def optimise_ranges(
    range_config: RangeConfig,
    range_type: str,
    beam: int,
    co_settings: OptSettings,
    corrector_knobs_file: Path,
    tune_knobs_file: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    energy: float,
) -> tuple[list[float], list[float]]:
    """Optimize for a given range configuration."""
    results = []
    uncertainties = []
    num_ranges = len(range_config.magnet_ranges)
    for i in range(num_ranges):
        logger.info(
            f"Starting optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )

        controller = Controller(
            beam=beam,
            opt_settings=co_settings,
            corrector_file=corrector_knobs_file,
            tune_knobs_file=tune_knobs_file,
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
            machine_deltap=0,
            magnet_range=range_config.magnet_ranges[i],
            bpm_start_points=range_config.bpm_starts[i],
            bpm_end_points=range_config.bpm_end_points[i],
            measurement_file=measurement_file,
            bad_bpms=bad_bpms,
            num_tracks=1,
            flattop_turns=3,
            beam_energy=energy,
        )
        final_knobs, uncs = controller.run()
        fitted_deltap = final_knobs["deltap"]
        # Convert to reference energy 6800 GeV (assume beta is 1 and in GeV)
        e_ref = 6800
        e_meas = energy * (1 + fitted_deltap)
        deltap_wrt_6800 = (e_meas - e_ref) / e_ref
        results.append(deltap_wrt_6800)
        uncertainties.append(uncs["deltap"])  # Assuming uncs is a dict with 'deltap'
        logger.info(f"{range_type.capitalize()} {i + 1}: deltap = {results[-1]}")
        logger.info(
            f"Finished optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )
    return results, uncertainties


def create_beam1_configs(folder: str, name_prefix: str) -> list[MeasurementConfig]:
    """Create measurement configurations for beam 1."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    arc_magnet_ranges_b1 = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    arc_bpm_starts_b1 = [
        [f"BPM.{i}R{s}.B1" for i in range(9, 35, 3)] for s in range(1, 9)
    ]
    arc_bpm_end_points_b1 = [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in range(9, 34, 3)] for s in range(1, 9)
    ]

    arc_config_b1 = RangeConfig(
        magnet_ranges=arc_magnet_ranges_b1,
        bpm_starts=arc_bpm_starts_b1,
        bpm_end_points=arc_bpm_end_points_b1,
    )

    return [
        MeasurementConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "07_53_05_820",
                "07_54_13_858",
            ],
            title="0",
        ),
        MeasurementConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_08_02_826",
                "08_09_11_940",
            ],
            title="0p2",
        ),
        MeasurementConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_11_13_745",
                "08_12_25_817",
            ],
            title="0p1",
        ),
        MeasurementConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_18_09_980",
                "08_19_16_847",
            ],
            title="m0p1",
        ),
        MeasurementConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_23_20_980",
                "08_24_32_020",
            ],
            title="m0p2",
        ),
    ]


def create_beam2_configs(folder: str, name_prefix: str) -> list[MeasurementConfig]:
    """Create measurement configurations for beam 2."""
    model_dir_b2 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    # Arc settings
    arc_magnet_ranges_b2 = [
        f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)
    ]
    arc_bpm_starts_b2 = [
        [f"BPM.{i}L{s}.B2" for i in range(9, 34, 3)] for s in range(8, 0, -1)
    ]
    arc_bpm_end_points_b2 = [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 35, 3)]
        for s in range(8, 0, -1)
    ]
    arc_config_b2 = RangeConfig(
        magnet_ranges=arc_magnet_ranges_b2,
        bpm_starts=arc_bpm_starts_b2,
        bpm_end_points=arc_bpm_end_points_b2,
    )

    return [
        MeasurementConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_35_27_940", "07_36_39_380", "07_38_44_035"],
            title="0",
        ),
        MeasurementConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_57_30_885", "08_00_44_900"],
            title="0p2",
        ),
        MeasurementConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_04_55_798", "08_06_06_900", "08_07_13_900"],
            title="0p1",
        ),
        MeasurementConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_15_06_860", "08_16_13_980"],
            title="m0p1",
        ),
        MeasurementConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_19_35_860", "08_22_57_752"],
            title="m0p2",
        ),
    ]


def process_single_config(
    config: MeasurementConfig, temp_analysis_dir: Path, date: str
) -> None:
    """Process a single measurement configuration."""
    results_dir = PROJECT_ROOT / f"b{config.beam}co_results"
    tune_knobs_file = results_dir / f"tune_knobs_{config.title}.txt"
    corrector_knobs_file = results_dir / f"corrector_knobs_{config.title}.txt"
    results_dir.mkdir(exist_ok=True)

    # Delete temp_analysis_dir if it exists
    temp_analysis_dir.mkdir(exist_ok=True)

    # Generate start_str from date and earliest time
    if not config.times:
        logger.warning(f"No times specified for config {config.title}, skipping.")
        return
    earliest_time = min(config.times)
    # e.g., "07_53_05_820" -> "07:53:05"
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{date} {time_str}"

    tz = ZoneInfo("UTC")
    meas_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    save_online_knobs(
        meas_time,
        beam=config.beam,
        tune_knobs_file=tune_knobs_file,
        corrector_knobs_file=corrector_knobs_file,
    )
    # Get beam energy from NXCALS
    spark = get_or_create()
    energy, _ = get_energy(spark, meas_time)
    measurement_filename = "pz_data.parquet"
    measurement_file = temp_analysis_dir / measurement_filename
    bad_bpms_file = temp_analysis_dir / "bad_bpms.txt"

    # Generate files from times
    files = [
        Path(f"{config.folder}/{config.name_prefix}{time}.sdds")
        for time in config.times
    ]

    pzs, bad_bpms, ana_dir = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        beam=config.beam,
        filename=None,
    )
    file_path = ana_dir / measurement_filename

    # Compute averages per BPM
    averaged = (
        pzs.groupby("name")[
            ["x", "px", "y", "py", "var_x", "var_y", "var_px", "var_py"]
        ]
        .mean()
        .reset_index()
    )
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

    # Save the bad bpms to a file
    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")

    # Read the bad bpms from the file
    with bad_bpms_file.open("r") as f:
        bad_bpms = [line.strip() for line in f.readlines()]

    co_settings = OptSettings(
        max_epochs=1000,
        # For pre trimmed data
        tracks_per_worker=1,
        num_batches=1,
        num_workers=1,
        warmup_epochs=3,
        warmup_lr_start=5e-7,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-6,
        optimiser_type="lbfgs",
        only_energy=True,
        use_off_energy_data=False,
    )

    results_arcs, uncs_arcs = optimise_ranges(
        config.arc_config,
        "arc",
        config.beam,
        co_settings,
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
    logger.info(f"Weighted mean deltap arcs: {mean_arcs}")
    logger.info(f"Std dev of deltap arcs: {np.std(results_arcs)}")

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


def main():
    """Main function to run the measurement processing loop."""
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam", type=int, choices=[1, 2], help="Beam number 1 or 2", default=2
    )
    args = parser.parse_args()

    # Define date
    date = "2025-11-07"
    # folder = "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/11259/BPM"
    folder = "/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11259/BPM"
    name_prefix = f"Beam{args.beam}@BunchTurn@{date.replace('-', '_')}@"

    # Get configurations based on beam
    if args.beam == 1:
        configs = create_beam1_configs(folder, name_prefix)
    else:
        configs = create_beam2_configs(folder, name_prefix)

    # Temporary analysis directory
    temp_analysis_dir = PROJECT_ROOT / f"temp_analysis_co_{args.beam}"

    # Process each configuration
    for config in configs:
        process_single_config(config, temp_analysis_dir, date)

    # Delete temp_analysis_dir
    shutil.rmtree(temp_analysis_dir)


if __name__ == "__main__":
    main()
