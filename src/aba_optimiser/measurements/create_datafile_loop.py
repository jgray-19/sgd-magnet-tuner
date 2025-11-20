from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from aba_optimiser.config import DPP_OPTIMISER_CONFIG, DPP_SIMULATION_CONFIG, PROJECT_ROOT
from aba_optimiser.measurements.create_datafile import process_measurements, save_online_knobs
from aba_optimiser.training.controller import LHCController as Controller
from aba_optimiser.training.controller_helpers import (
    create_arc_bpm_config,
    create_arc_measurement_config,
)

logger = logging.getLogger(__name__)


@dataclass
class MeasurementSetupConfig:
    beam: int
    model_dir: str
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]
    folder: str
    name_prefix: str
    times: list[str]
    title: str


def create_beam1_configs(folder: str, name_prefix: str) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 1."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    magnet_ranges_b1 = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    bpm_starts_b1 = [[f"BPM.{i}R{s}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]
    bpm_end_points_b1 = [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)
    ]

    return [
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            magnet_ranges=magnet_ranges_b1,
            bpm_starts=bpm_starts_b1,
            bpm_end_points=bpm_end_points_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "07_53_05_820",
                "07_54_13_858",
            ],
            title="0",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            magnet_ranges=magnet_ranges_b1,
            bpm_starts=bpm_starts_b1,
            bpm_end_points=bpm_end_points_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_08_02_826",
                "08_09_11_940",
            ],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            magnet_ranges=magnet_ranges_b1,
            bpm_starts=bpm_starts_b1,
            bpm_end_points=bpm_end_points_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_11_13_745",
                "08_12_25_817",
            ],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            magnet_ranges=magnet_ranges_b1,
            bpm_starts=bpm_starts_b1,
            bpm_end_points=bpm_end_points_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_18_09_980",
                "08_19_16_847",
            ],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            magnet_ranges=magnet_ranges_b1,
            bpm_starts=bpm_starts_b1,
            bpm_end_points=bpm_end_points_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=[
                "08_23_20_980",
                "08_24_32_020",
            ],
            title="m0p2",
        ),
    ]


def create_beam2_configs(folder: str, name_prefix: str) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 2."""
    model_dir_b2 = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    )
    magnet_ranges_b2 = [f"BPM.9L{i}.B2/BPM.9R{(i - 2) % 8 + 1}.B2" for i in range(8, 0, -1)]
    bpm_starts_b2 = [[f"BPM.{i}L{s}.B2" for i in range(9, 14)] for s in range(8, 0, -1)]
    bpm_end_points_b2 = [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 14)] for s in range(8, 0, -1)
    ]

    return [
        # MeasurementSetupConfig(
        #     beam=2,
        #     model_dir=model_dir_b2,
        #     magnet_ranges=magnet_ranges_b2,
        #     bpm_starts=bpm_starts_b2,
        #     bpm_end_points=bpm_end_points_b2,
        #     folder=folder,
        #     name_prefix=name_prefix,
        #     times=["07_35_27_940", "07_36_39_380", "07_38_44_035"],
        #     title="0",
        # ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            magnet_ranges=magnet_ranges_b2,
            bpm_starts=bpm_starts_b2,
            bpm_end_points=bpm_end_points_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_57_30_885", "08_00_44_900"],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            magnet_ranges=magnet_ranges_b2,
            bpm_starts=bpm_starts_b2,
            bpm_end_points=bpm_end_points_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_04_55_798", "08_06_06_900", "08_07_13_900"],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            magnet_ranges=magnet_ranges_b2,
            bpm_starts=bpm_starts_b2,
            bpm_end_points=bpm_end_points_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_15_06_860", "08_16_13_980"],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            magnet_ranges=magnet_ranges_b2,
            bpm_starts=bpm_starts_b2,
            bpm_end_points=bpm_end_points_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_19_35_860", "08_22_57_752"],
            title="m0p2",
        ),
    ]


def process_single_config(
    config: MeasurementSetupConfig, temp_analysis_dir: Path, date: str
) -> None:
    """Process a single measurement configuration."""
    results_dir = PROJECT_ROOT / f"b{config.beam}_results"
    results_dir.mkdir(exist_ok=True)

    # Delete temp_analysis_dir if it exists
    if temp_analysis_dir.exists():
        shutil.rmtree(temp_analysis_dir)
    temp_analysis_dir.mkdir()

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
    save_online_knobs(meas_time, beam=config.beam)
    measurement_filename = "pz_data.parquet"
    measurement_file = temp_analysis_dir / measurement_filename
    bad_bpms_file = temp_analysis_dir / "bad_bpms.txt"

    # Generate files from times
    files = [Path(f"{config.folder}/{config.name_prefix}{time}.sdds") for time in config.times]

    pzs, bad_bpms, _ = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        beam=config.beam,
        filename=measurement_filename,
    )

    # Save the bad bpms to a file
    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")

    # Read the bad bpms from the file
    with bad_bpms_file.open("r") as f:
        bad_bpms = [line.strip() for line in f.readlines()]

    # Write the results to a file
    results_file = results_dir / f"{config.title}.txt"
    with results_file.open("w") as f:
        f.write("Arc\tDeltap\n")

    results = []
    for arc in range(8):
        logger.info(f"Starting optimisation for arc {arc + 1}/8 for {config.title}")

        controller = Controller(
            beam=config.beam,
            measurement_config=create_arc_measurement_config(
                measurement_file,
                num_tracks=len(config.times),
                flattop_turns=6600,
            ),
            bpm_config=create_arc_bpm_config(config.bpm_starts[arc], config.bpm_end_points[arc]),
            magnet_range=config.magnet_ranges[arc],
            optimiser_config=DPP_OPTIMISER_CONFIG,
            simulation_config=DPP_SIMULATION_CONFIG,
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
            bad_bpms=bad_bpms,
        )
        final_knobs, uncs = controller.run()
        results.append(final_knobs["deltap"])
        with results_file.open("a") as f:
            f.write(f"{arc + 1}\t{results[-1]}\n")
        logger.info(f"Arc {arc + 1}: deltap = {results[-1]}")
        logger.info(f"Finished optimisation for arc {arc + 1}/8 for {config.title}")

    logger.info(f"All arc optimisations complete for {config.title}.")
    logger.info("Final deltaps for each arc:")
    for i, dp in enumerate(results):
        logger.info(f"Arc {i + 1}: deltap = {dp}")

    logger.info(f"Mean deltap: {np.mean(results)}")
    logger.info(f"Std dev of deltap: {np.std(results)}")

    with results_file.open("a") as f:
        f.write(f"Mean\t{np.mean(results)}\n")
        f.write(f"StdDev\t{np.std(results)}\n")

    # Delete temp_analysis_dir
    shutil.rmtree(temp_analysis_dir)


def main():
    """Main function to run the measurement processing loop."""
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("beam", type=int, choices=[1, 2], help="Beam number 1 or 2")
    args = parser.parse_args()

    # Define date
    date = "2025-11-07"
    folder = "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/11259/BPM"
    name_prefix = f"Beam{args.beam}@BunchTurn@{date.replace('-', '_')}@"

    # Get configurations based on beam
    if args.beam == 1:
        configs = create_beam1_configs(folder, name_prefix)
    else:
        configs = create_beam2_configs(folder, name_prefix)

    # Temporary analysis directory
    temp_analysis_dir = PROJECT_ROOT / "temp_analysis"

    # Process each configuration
    for config in configs:
        process_single_config(config, temp_analysis_dir, date)


if __name__ == "__main__":
    main()
