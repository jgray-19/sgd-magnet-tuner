from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import tfs
from nxcals.spark_session_builder import get_or_create
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.optics_measurements.constants import AMP_BETA_NAME, BETA, ERR, NAME
from pylhc import corrector_extraction, mqt_extraction
from turn_by_turn import TbtData, read_tbt

from aba_optimiser.config import (
    CORRECTOR_STRENGTHS,
    DPP_OPT_SETTINGS,
    SEQUENCE_FILE,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.momentum_recon.transverse import calculate_pz_from_measurements
from aba_optimiser.training.controller import Controller

if TYPE_CHECKING:
    from pylhc.nxcal_knobs import NXCalResult

logger = logging.getLogger(__name__)


def load_files(files: list[str | Path]) -> TbtData:
    """Load and concatenate multiple Parquet files into a single DataFrame."""
    measurements = []
    for file in files:
        logger.info("Loading data from %s", file)
        meas_tbt = read_tbt(file, datatype="lhc")
        measurements.append(meas_tbt)

    return measurements


def combine_data(measurements: list[TbtData]) -> pd.DataFrame:
    """Combine multiple TbtData objects into a single DataFrame.

    In each turn by turn object, there exists a list stored in the `matrices` attribute.
    Each element of this list is a TransverseData object, which contains the data per bunch.
    Each TransverseData object has an X and Y attribute, which are DataFrames containing the
    measurement data for each plane.
    The DataFrame has an index which is the BPM name, and columns which are the turn numbers.
    The turn numbers start from 0, so we need to offset them by the number of turns (+1) in previous
    TransverseData objects to ensure unique turn numbers across all bunches and all files.

    The final combined DataFrame will have no index, and columns: ['name', 'turn', 'x', 'y'] (x and y in metres).
    The name column should be a category dtype for efficiency.
    """
    all_data = []
    turn_offset = 1
    for meas in measurements:
        for bunch in meas.matrices:
            df_x = bunch.X.copy()
            df_y = bunch.Y.copy()
            df_x.index.name = "name"
            df_y.index.name = "name"
            df_x.columns = df_x.columns + turn_offset
            df_y.columns = df_y.columns + turn_offset
            df_combined = df_x.reset_index().melt(
                id_vars="name", var_name="turn", value_name="x"
            )
            df_combined["y"] = df_y.reset_index().melt(
                id_vars="name", var_name="turn", value_name="y"
            )["y"]
            df_combined["x"] = df_combined["x"] / 1000
            df_combined["y"] = df_combined["y"] / 1000
            all_data.append(df_combined)
            turn_offset += df_x.shape[1]  # Number of turns is number of columns

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["name"] = combined_df["name"].astype("category")
    combined_df["turn"] = combined_df["turn"].astype("int32")
    return combined_df


def compute_weights_from_beta_amplitude(
    combined: pd.DataFrame, analysis_dir: Path
) -> pd.DataFrame:
    """Compute weights for x and y planes based on relative errors from beta amplitude data.

    For each BPM, weight = 1 / relative_error if BPM exists in beta data, else 0.
    """
    beta_x_data = tfs.read(analysis_dir / (AMP_BETA_NAME + "x.tfs"), index=NAME)
    beta_y_data = tfs.read(analysis_dir / (AMP_BETA_NAME + "y.tfs"), index=NAME)

    beta_x = beta_x_data[BETA + "X"].to_dict()
    beta_y = beta_y_data[BETA + "Y"].to_dict()
    error_x = beta_x_data[ERR + BETA + "X"].to_dict()
    error_y = beta_y_data[ERR + BETA + "Y"].to_dict()

    def compute_weight(bpm, beta_dict, error_dict):
        if bpm not in beta_dict or bpm not in error_dict:
            return 0.0
        beta_val = beta_dict[bpm]
        err_val = error_dict[bpm]
        if beta_val == 0:
            return 0.0
        rel_err = abs(err_val) / abs(beta_val)
        if rel_err <= 0:
            return 0.0
        return 1.0 / rel_err

    combined["x_weight"] = combined["name"].apply(
        lambda bpm: compute_weight(bpm, beta_x, error_x)
    )
    combined["y_weight"] = combined["name"].apply(
        lambda bpm: compute_weight(bpm, beta_y, error_y)
    )
    return combined


def run_analysis(analysis_dir: str | Path, files: list[str | Path]) -> pd.DataFrame:
    """Load, combine, and process data from multiple files."""
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    hole_in_one_entrypoint(
        harpy=True,
        files=files,
        outputdir=analysis_dir / "lin_files",
        unit="mm",
        driven_excitation="acd",
        # nat_tunes=[0.28, 0.31],
        first_bpm="BPM.33L2.B1",
        is_free_kick=False,
        keep_exact_zeros=False,
        max_peak=0.02,
        nattunes=[0.28, 0.31, 0.0],
        num_svd_iterations=3,
        opposite_direction=False,
        output_bits=12,
        peak_to_peak=1e-08,
        resonances=4,
        sing_val=12,
        svd_dominance_limit=0.925,
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],
        tune_clean_limit=1e-05,
        tunes=[0.27, 0.322, 0.0],
        turn_bits=18,
        model_dir="/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Models/b1_flat_60_18cm",
        turns=[0, 50000],
        clean=True,
    )

    analysed_files = [analysis_dir / "lin_files" / f.name for f in files]

    hole_in_one_entrypoint(
        optics=True,
        files=analysed_files,
        outputdir=analysis_dir,
        analyse_dpp=0,
        calibrationdir="/afs/cern.ch/eng/sl/lintrack/LHC_commissioning2017/Calibration_factors_2017/Calibration_factors_2017_beam1",
        chromatic_beating=False,
        compensation="equation",
        coupling_method=2,
        coupling_pairing=0,
        isolation_forest=False,
        nonlinear=["rdt"],
        only_coupling=False,
        range_of_bpms=11,
        rdt_magnet_order=4,
        second_order_dispersion=False,
        three_bpm_method=False,
        three_d_excitation=False,
        union=False,
        accel="lhc",
        ats=False,
        beam=1,
        dpp=0.0,
        model_dir="/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Models/b1_flat_60_18cm",
        xing=False,
        year="2025",
    )


def build_dict_from_nxcal_result(result: list[NXCalResult]) -> dict[str, float]:
    """Convert NXCalResult to a dictionary of magnet strengths."""
    return {res.name: res.value for res in result}


# def write_datafile(data: pd.DataFrame, output_file: str | Path) -> None:
#     """Write the combined DataFrame to a Parquet file."""
#     data.to_parquet(output_file)


def save_online_knobs(meas_time: datetime) -> None:
    """Load and save knob data from NXCal."""
    spark = get_or_create()
    mqt_results = mqt_extraction.get_mqt_vals(spark, meas_time, 1)
    corrector_results = corrector_extraction.get_mcb_vals(spark, meas_time, 1)

    mqt_knobs = build_dict_from_nxcal_result(mqt_results)
    corrector_knobs = build_dict_from_nxcal_result(corrector_results)

    save_knobs(mqt_knobs, TUNE_KNOBS_FILE)
    save_knobs(corrector_knobs, CORRECTOR_STRENGTHS)


def process_measurements(
    files: list[Path], analysis_dir: Path, filename: str = "pz_data.parquet"
) -> tuple[pd.DataFrame, list[str], Path]:
    """Process measurement files to compute pz data and identify bad BPMs."""
    run_analysis(analysis_dir, files)

    data = load_files(files)
    combined = combine_data(data)
    combined["kick_plane"] = "xy"
    combined = compute_weights_from_beta_amplitude(combined, analysis_dir)
    cleaned = svd_clean_measurements(combined)

    pzs = calculate_pz_from_measurements(
        cleaned, analysis_dir, info=True, subtract_mean=True
    )
    mad_iface = OptimisationMadInterface(
        SEQUENCE_FILE,
        discard_mad_output=False,
    )
    all_bpms = set(mad_iface.all_bpms)
    del mad_iface

    pzs["name"] = pzs["name"].astype("category")

    # print all bpms and the corresponding plane where py and px are NaN
    bad_bpms_mask = pzs.groupby("name").apply(
        lambda g: g[["px", "py"]].isna().any().any()
    )
    bad_bpms = bad_bpms_mask[bad_bpms_mask].index.tolist()
    for bpm in bad_bpms:
        logger.info(f"BPM {bpm}: has_nan=True")

    missing_bpms = all_bpms - set(pzs["name"].unique())
    for bpm in missing_bpms:
        logger.info(f"BPM {bpm}: missing from data")
    bad_bpms.extend(missing_bpms)

    logger.info(f"Total bad BPMs: {len(bad_bpms)}")

    file_path = analysis_dir / filename
    pzs.to_parquet(file_path)

    return pzs, bad_bpms, file_path


if __name__ == "__main__":
    # set logging level to debug
    logging.basicConfig(level=logging.INFO)
    module_root = Path(__file__).absolute().parent.parent.parent.parent
    # analysis_dir = module_root / "analysis"
    analysis_dir = module_root / "analysis_trim"

    MAGNET_RANGES = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]

    BPM_STARTS = [[f"BPM.{i}R{s}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]
    BPM_END_POINTS = [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)
    ]

    files = [
        # Before the trim
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_52_18_410/Beam1@BunchTurn@2025_04_09@18_52_18_410.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_48_27_464/Beam1@BunchTurn@2025_04_09@18_48_27_464.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_51_14_983/Beam1@BunchTurn@2025_04_09@18_51_14_983.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_47_22_071/Beam1@BunchTurn@2025_04_09@18_47_22_071.sdds",
        # After the trim
        "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10423/BPM/Beam1@BunchTurn@2025_04_09@19_05_01_818.sdds",
        "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10423/BPM/Beam1@BunchTurn@2025_04_09@19_06_06_407.sdds",
        "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/10423/BPM/Beam1@BunchTurn@2025_04_09@19_07_10_507.sdds",
    ]
    files = [Path(f) for f in files]

    # start_str = "2025-04-09 18:46:50"
    start_str = "2025-04-09 19:04:50"
    tz = ZoneInfo("UTC")
    meas_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
    save_online_knobs(meas_time)
    measurement_filename = "pz_data.parquet"
    measurement_file = analysis_dir / measurement_filename
    bad_bpms_file = analysis_dir / "bad_bpms.txt"

    pzs, bad_bpms, _ = process_measurements(files, analysis_dir, measurement_filename)

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
            opt_settings=DPP_OPT_SETTINGS,
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths_file=None,
            machine_deltap=0,
            magnet_range=MAGNET_RANGES[arc],
            bpm_start_points=BPM_STARTS[arc],
            bpm_end_points=BPM_END_POINTS[arc],
            measurement_file=measurement_file,
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
