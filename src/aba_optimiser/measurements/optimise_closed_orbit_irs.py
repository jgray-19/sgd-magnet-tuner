"""Closed-orbit optimisation workflow across interaction-region segments."""

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
import tfs
from nxcals.spark_session_builder import get_or_create
from omc3.machine_data_extraction.nxcals_knobs import get_energy

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import (
    MEASUREMENTS_ARTIFACTS_ROOT,
    OptimiserConfig,
    SimulationConfig,
)
from aba_optimiser.measurements.create_datafile import (
    process_measurements,
    save_online_knobs,
)
from aba_optimiser.measurements.squeeze_helpers import get_or_make_sequence
from aba_optimiser.training.config.helpers import create_arc_measurement_config
from aba_optimiser.training.config.models import OutputConfig, SequenceConfig
from aba_optimiser.training.controller import Controller

logger = logging.getLogger(__name__)


@dataclass
class RangeConfig:
    """Grouped BPM ranges used for one IR optimisation sweep."""

    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


@dataclass
class IterationConfig:
    """One IR closed-orbit optimisation iteration with time metadata."""

    beam: int
    model_dir: str
    ir_config: RangeConfig
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
    sequence_file: Path,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
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
        logger.info(f"Starting optimisation for {range_type} {i + 1}/{num_ranges} for {title}")

        measurement_config = create_arc_measurement_config(
            measurement_file,
            corrector_strengths=corrector_knobs_file,
            tune_knobs_file=tune_knobs_file,
        )
        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
        )

        accelerator = LHC(
            beam=beam,
            sequence_file=sequence_file,
            kinetic_energy=energy,
            optimise_energy=True,
        )

        controller = Controller(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=range_config.bpm_starts[i],
            bpm_end_points=range_config.bpm_end_points[i],
            initial_knob_strengths=None,
            true_strengths=None,
            output_config=OutputConfig(show_plots=False),
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
        logger.info(f"Finished optimisation for {range_type} {i + 1}/{num_ranges} for {title}")
    return results, uncertainties


def get_bpm_ranges_from_model(
    model_dir: str, beam: int
) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Extract BPM ranges from twiss.dat file for IR optimisation.

    Args:
        model_dir: Path to the model directory containing twiss_elements.dat
        beam: Beam number (1 or 2)

    Returns:
        Tuple of (magnet_ranges, bpm_starts, bpm_end_points)
    """
    import re

    twiss_file = Path(model_dir) / "twiss_elements.dat"
    twiss_df = tfs.read(twiss_file, index="NAME")

    # Filter BPMs for this beam
    bpm_mask = twiss_df.index.str.startswith("BPM") & twiss_df.index.str.endswith(f".B{beam}")
    bpm_names = twiss_df.index[bpm_mask].tolist()

    # Regex to match BPM names: BPM.*.(IP)(L|R).*.B(beam)
    bpm_pattern = re.compile(r"BPM[A-Z]*\.(\d)([LR])(\d)\.B(\d+)")

    # Collect all matching BPMs with ip and side
    matches = [
        (bpm, int(match.group(3)), match.group(2), int(match.group(1)))
        for bpm in bpm_names
        if (match := bpm_pattern.match(bpm))
    ]

    magnet_ranges = []
    bpm_starts = []
    bpm_end_points = []

    ip_range = range(8, 0, -1) if beam == 2 else range(1, 9)
    before_side = "L" if beam == 1 else "R"
    after_side = "R" if beam == 1 else "L"
    for ip in ip_range:
        # Include BPMs from position 4 onwards to get more measurement points
        before_bpms = [
            bpm
            for bpm, ip_num, side, from_ip in matches
            if ip_num == ip and side == before_side and from_ip >= 4
        ]
        after_bpms = [
            bpm
            for bpm, ip_num, side, from_ip in matches
            if ip_num == ip and side == after_side and from_ip >= 4
        ]

        # Remove all bpms with W in their names
        # before_bpms = [bpm for bpm in before_bpms if "W" not in bpm]
        # after_bpms = [bpm for bpm in after_bpms if "W" not in bpm]

        bpm_starts.append(before_bpms)
        bpm_end_points.append(after_bpms)

        if beam == 1:
            magnet_ranges.append(f"BPM.9L{ip}.B1/BPM.9R{ip}.B1")
        else:
            magnet_ranges.append(f"BPM.9R{ip}.B2/BPM.9L{ip}.B2")
    return magnet_ranges, bpm_starts, bpm_end_points


def create_beam1_configs(folder: str, name_prefix: str) -> list[IterationConfig]:
    """Create measurement configurations for beam 1."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"

    # Get BPM ranges from model
    ir_magnet_ranges_b1, ir_bpm_starts_b1, ir_bpm_end_points_b1 = get_bpm_ranges_from_model(
        model_dir_b1, 1
    )

    ir_config_b1 = RangeConfig(
        magnet_ranges=ir_magnet_ranges_b1,
        bpm_starts=ir_bpm_starts_b1,
        bpm_end_points=ir_bpm_end_points_b1,
    )

    return [
        IterationConfig(
            beam=1,
            model_dir=model_dir_b1,
            ir_config=ir_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "07_53_05_820",
                "07_54_13_858",
            ],
            title="0",
        ),
        IterationConfig(
            beam=1,
            model_dir=model_dir_b1,
            ir_config=ir_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_08_02_826",
                "08_09_11_940",
            ],
            title="0p2",
        ),
        IterationConfig(
            beam=1,
            model_dir=model_dir_b1,
            ir_config=ir_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_11_13_745",
                "08_12_25_817",
            ],
            title="0p1",
        ),
        IterationConfig(
            beam=1,
            model_dir=model_dir_b1,
            ir_config=ir_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_18_09_980",
                "08_19_16_847",
            ],
            title="m0p1",
        ),
        IterationConfig(
            beam=1,
            model_dir=model_dir_b1,
            ir_config=ir_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_23_20_980",
                "08_24_32_020",
            ],
            title="m0p2",
        ),
    ]


def create_beam2_configs(folder: str, name_prefix: str) -> list[IterationConfig]:
    """Create measurement configurations for beam 2."""
    model_dir_b2 = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    )
    # Ir settings
    ir_magnet_ranges_b2, ir_bpm_starts_b2, ir_bpm_end_points_b2 = get_bpm_ranges_from_model(
        model_dir_b2, 2
    )

    ir_config_b2 = RangeConfig(
        magnet_ranges=ir_magnet_ranges_b2,
        bpm_starts=ir_bpm_starts_b2,
        bpm_end_points=ir_bpm_end_points_b2,
    )

    return [
        IterationConfig(
            beam=2,
            model_dir=model_dir_b2,
            ir_config=ir_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_35_27_940", "07_36_39_380", "07_38_44_035"],
            title="0",
        ),
        IterationConfig(
            beam=2,
            model_dir=model_dir_b2,
            ir_config=ir_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_57_30_885", "08_00_44_900"],
            title="0p2",
        ),
        IterationConfig(
            beam=2,
            model_dir=model_dir_b2,
            ir_config=ir_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_04_55_798", "08_06_06_900", "08_07_13_900"],
            title="0p1",
        ),
        IterationConfig(
            beam=2,
            model_dir=model_dir_b2,
            ir_config=ir_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_15_06_860", "08_16_13_980"],
            title="m0p1",
        ),
        IterationConfig(
            beam=2,
            model_dir=model_dir_b2,
            ir_config=ir_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_19_35_860", "08_22_57_752"],
            title="m0p2",
        ),
    ]


def process_single_config(
    config: IterationConfig,
    temp_analysis_dir: Path,
    date: str,
    skip_reload: bool,
    use_fixed_bpm: bool = False,
) -> None:
    """Process a single measurement configuration.

    Args:
        config: Measurement configuration for this run
        temp_analysis_dir: Temporary directory for analysis outputs
        date: Date string in YYYY-MM-DD format
        skip_reload: If True, skip reloading strengths from LSA and reuse existing analysis
        use_fixed_bpm: If True, use fixed reference BPM approach.
                       If False (default for IRs), create all combinations of start/end BPMs (Cartesian product)
                       to provide more measurement constraints.
    """
    results_dir = MEASUREMENTS_ARTIFACTS_ROOT / "results" / f"b{config.beam}ir_results"
    tune_knobs_file = results_dir / f"tune_knobs_{config.title}.txt"
    corrector_knobs_file = results_dir / f"corrector_knobs_{config.title}.txt"
    results_dir.mkdir(exist_ok=True)

    # Copy bad bpms from co results
    co_results_dir = MEASUREMENTS_ARTIFACTS_ROOT / "results" / f"b{config.beam}co_results"
    co_bad_bpms_file = co_results_dir / f"bad_bpms_{config.title}.txt"
    ir_bad_bpms_file = results_dir / f"bad_bpms_{config.title}.txt"
    if co_bad_bpms_file.exists():
        shutil.copy(co_bad_bpms_file, ir_bad_bpms_file)

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

    # Get beam energy from NXCALS
    spark = get_or_create()
    energy, _ = get_energy(spark, meas_time)
    measurement_filename = "pz_data.parquet"
    measurement_file = temp_analysis_dir / measurement_filename

    # Read bad bpms from copied file
    bad_bpms = []
    if ir_bad_bpms_file.exists():
        with ir_bad_bpms_file.open("r") as f:
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
    sequence_file = get_or_make_sequence(config.beam, Path(config.model_dir))
    accelerator = LHC(
        beam=config.beam,
        sequence_file=sequence_file,
        kinetic_energy=float(energy),
    )

    pzs_dict, _, output_paths, _ = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        accelerator=accelerator,
        filename=None,
        bad_bpms=bad_bpms,
        use_uniform_vars=False,
    )
    pzs = pzs_dict["combined"]
    ana_dir = output_paths["combined"]
    file_path = ana_dir / measurement_filename

    # Compute averages per BPM
    averaged = (
        pzs.groupby("name")[["x", "px", "y", "py", "var_x", "var_y", "var_px", "var_py"]]
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
                }
            )
    new_df = pd.DataFrame(new_rows)
    new_df["name"] = new_df["name"].astype("category")
    new_df["turn"] = new_df["turn"].astype("int32")

    # Overwrite the measurement file
    new_df.to_parquet(file_path)

    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=3,
        warmup_lr_start=5e-7,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-6,
        optimiser_type="lbfgs",
    )
    simulation_config = SimulationConfig(
        # For pre trimmed data
        tracks_per_worker=1,
        num_batches=1,
        num_workers=1,
        use_fixed_bpm=use_fixed_bpm,
    )

    results_irs, uncs_irs = optimise_ranges(
        config.ir_config,
        "ir",
        config.beam,
        sequence_file,
        optimiser_config,
        simulation_config,
        corrector_knobs_file,
        tune_knobs_file,
        measurement_file,
        bad_bpms,
        config.title,
        energy,
    )

    logger.info(f"All ir optimisations complete for {config.title}.")
    logger.info("Final deltaps for each ir:")
    for i, dp in enumerate(results_irs):
        logger.info(f"Ir {i + 1}: deltap = {dp}")

    if not results_irs:
        logger.warning("No ir results produced; skipping summary output.")
        return

    try:
        mean_irs = weighted_mean(results_irs, uncs_irs)
    except ValueError:
        mean_irs = float(np.mean(results_irs))
        logger.warning("Falling back to unweighted mean for irs due to non-positive uncertainties.")
    logger.info(f"Weighted mean deltap irs: {mean_irs}")
    logger.info(f"Std dev of deltap irs: {np.std(results_irs)}")

    # Write the results to a file
    results_file = results_dir / f"{config.title}.txt"
    with results_file.open("w") as f:
        f.write("range\tdeltap\tuncertainty\n")

    with results_file.open("a") as f:
        for i, (dp, unc) in enumerate(zip(results_irs, uncs_irs)):
            f.write(f"ir{i + 1}\t{dp}\t{unc}\n")
        f.write(f"MeanIrs\t{mean_irs}\t\n")
        std_irs = float(np.std(results_irs))
        f.write(f"StdDevIrs\t{std_irs}\t\n")
        stderr = std_irs / np.sqrt(len(results_irs)) if len(results_irs) > 0 else 0.0
        f.write(f"StdErrIrs\t{stderr}\t\n")
        # Compute weighted uncertainty on the mean
        if uncs_irs and all(u > 0 for u in uncs_irs):
            weights = [1 / (u**2) for u in uncs_irs]
            weighted_unc = np.sqrt(1 / sum(weights))
            f.write(f"WeightedUncIrs\t{weighted_unc}\t\n")


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
        "--fixed-bpm",
        action="store_true",
        help="Enable fixed BPM for start/end points (default: disabled)",
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
    temp_analysis_dir = MEASUREMENTS_ARTIFACTS_ROOT / "temp" / f"temp_analysis_co_{args.beam}"

    # Determine use_fixed_bpm from args (default False)
    use_fixed_bpm = args.fixed_bpm

    # Process each configuration
    for config in configs:
        process_single_config(config, temp_analysis_dir, date, args.skip_reload, use_fixed_bpm)

    # Delete temp_analysis_dir if not skipping reload
    if not args.skip_reload:
        shutil.rmtree(temp_analysis_dir)


if __name__ == "__main__":
    main()
