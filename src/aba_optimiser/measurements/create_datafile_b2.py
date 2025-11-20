from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aba_optimiser.config import DPP_OPTIMISER_CONFIG, DPP_SIMULATION_CONFIG, PROJECT_ROOT
from aba_optimiser.measurements.create_datafile import process_measurements, save_online_knobs
from aba_optimiser.training.controller import LHCController as Controller
from aba_optimiser.training.controller_helpers import (
    create_arc_bpm_config,
    create_arc_measurement_config,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # set logging level to debug
    logging.basicConfig(level=logging.INFO)
    analysis_dir = PROJECT_ROOT / "analysis_b2"
    # analysis_dir = PROJECT_ROOT / "analysis_trim_b2"
    model_dir = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB2/Models/2025_LHCB2_0p18m"
    )
    MAGNET_RANGES = [f"BPM.9L{i}.B2/BPM.9R{(i - 2) % 8 + 1}.B2" for i in range(8, 0, -1)]

    BPM_STARTS = [[f"BPM.{i}L{s}.B2" for i in range(9, 14)] for s in range(8, 0, -1)]
    BPM_END_POINTS = [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 14)] for s in range(8, 0, -1)
    ]

    folder = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB2/Measurements/")
    name_prefix = "Beam2@BunchTurn@2025_04_09@"
    times = [
        "18_48_02_383",
        "18_49_07_430",
        "18_50_10_785",
        # "19_05_17_036",
        # "19_06_21_160",
        # "19_07_25_349",
    ]

    files = [folder / f"{name_prefix}{t}/{name_prefix}{t}.sdds" for t in times]

    start_str = "2025-04-09 18:47:50"
    # start_str = "2025-04-09 19:04:50"
    tz = ZoneInfo("UTC")
    meas_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    save_online_knobs(meas_time, beam=2)
    measurement_filename = "pz_data.parquet"
    measurement_file = analysis_dir / measurement_filename
    bad_bpms_file = analysis_dir / "bad_bpms.txt"

    pzs, bad_bpms, _ = process_measurements(
        files,
        analysis_dir,
        model_dir,
        beam=2,
        filename=measurement_filename,
    )

    # save the bad bpms to a file
    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")

    # read the bad bpms from the file
    with bad_bpms_file.open("r") as f:
        bad_bpms = [line.strip() for line in f.readlines()]

    # Write the results to a file
    results_file = analysis_dir / "deltap_results.txt"
    with results_file.open("w") as f:
        f.write("Arc\tDeltap\n")

    results = []
    for arc in range(8):
        logger.info(f"Starting optimisation for arc {arc + 1}/8")

        controller = Controller(
            beam=2,
            measurement_config=create_arc_measurement_config(measurement_file),
            bpm_config=create_arc_bpm_config(BPM_STARTS[arc], BPM_END_POINTS[arc]),
            magnet_range=MAGNET_RANGES[arc],
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
        logger.info(f"Finished optimisation for arc {arc + 1}/8")

    logger.info("All arc optimisations complete.")
    logger.info("Final deltaps for each arc:")
    for i, dp in enumerate(results):
        logger.info(f"Arc {i + 1}: deltap = {dp}")

    import numpy as np

    logger.info(f"Mean deltap: {np.mean(results)}")
    logger.info(f"Std dev of deltap: {np.std(results)}")

    with results_file.open("a") as f:
        f.write(f"Mean\t{np.mean(results)}\n")
        f.write(f"StdDev\t{np.std(results)}\n")
