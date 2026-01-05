from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tfs
from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn import TbtData, read_tbt

from aba_optimiser.config import (
    CORRECTOR_STRENGTHS,
    DPP_OPTIMISER_CONFIG,
    DPP_SIMULATION_CONFIG,
    PROJECT_ROOT,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.io.utils import get_lhc_file_path, save_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

# from aba_optimiser.momentum_recon.transverse import calculate_pz
from aba_optimiser.momentum_recon.dispersive import calculate_pz
from aba_optimiser.training.controller import LHCController as Controller
from aba_optimiser.training.controller_helpers import (
    create_arc_bpm_config,
    create_arc_measurement_config,
)

if TYPE_CHECKING:
    try:
        from pylhc.nxcal_knobs import NXCalResult
    except ImportError:
        NXCalResult = None

logger = logging.getLogger(__name__)


def load_files(files: list[str | Path]) -> TbtData:
    """Load and concatenate multiple Parquet files into a single DataFrame."""
    measurements = []
    for file in files:
        logger.info("Loading data from %s", file)
        meas_tbt = read_tbt(file, datatype="lhc")
        measurements.append(meas_tbt)

    return measurements


def convert_measurements(
    measurements: list[TbtData], bad_bpms: list[str] = [], combine_measurements: bool = True
) -> list[pd.DataFrame]:
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
        if not combine_measurements:
            turn_offset = 1
        for bunch in meas.matrices:
            df_x = bunch.X.copy()
            df_y = bunch.Y.copy()
            df_x.index.name = "name"
            df_y.index.name = "name"
            df_x.columns = df_x.columns + turn_offset
            df_y.columns = df_y.columns + turn_offset
            df_combined = df_x.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
            df_combined["y"] = df_y.reset_index().melt(
                id_vars="name", var_name="turn", value_name="y"
            )["y"]
            df_combined["x"] = df_combined["x"] / 1000
            df_combined["y"] = df_combined["y"] / 1000
            df_combined["kick_plane"] = "xy"

            # Reorder the rows based on BPM names to match the original order
            original_order = df_x.index.tolist()
            assert df_y.index.tolist() == original_order, "BPM order mismatch between X and Y data"
            df_combined["name"] = pd.Categorical(df_combined["name"], categories=original_order)
            # Delete bad bpms from the combined dataframe
            if bad_bpms:
                df_combined = df_combined[~df_combined["name"].isin(bad_bpms)]
            df_combined = df_combined.sort_values(["turn", "name"]).reset_index(drop=True)

            all_data.append(df_combined)
            turn_offset += df_x.shape[1]  # Number of turns is number of columns
    return all_data


def compute_uniform_vars(
    combined: pd.DataFrame, bad_bpms: list[str], var_value: float = 1e-6
) -> pd.DataFrame:
    """Set uniform variances for all BPMs.

    Args:
        combined: DataFrame with BPM measurements
        bad_bpms: List of bad BPM names to set to infinite variance
        var_value: Uniform variance value to use (default: 1e-6)

    Returns:
        DataFrame with var_x, var_y, var_px, var_py columns added
    """
    combined["var_x"] = var_value
    combined["var_y"] = var_value
    combined["var_px"] = var_value
    combined["var_py"] = var_value
    combined.loc[combined["name"].isin(bad_bpms), "var_x"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_y"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_px"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_py"] = float("inf")
    return combined


def compute_vars_from_known_noise(combined: pd.DataFrame, bad_bpms: list[str]) -> pd.DataFrame:
    """Compute variances for x and y planes based on known noise levels."""
    # Read the file in the same directory as this script
    script_dir = Path(__file__).parent
    noise_file = script_dir / "bpm_std.txt"

    # The file has three columns: name, x_std, y_std, separated by a space
    # The units are mm, convert to meters
    noise_data = pd.read_csv(
        noise_file,
        sep=r"\s+",
        header=0,
        names=["name", "Horizontal_STD", "Vertical_STD"],
    )
    noise_data["Horizontal_STD"] /= 1000  # Convert mm to m
    noise_data["Vertical_STD"] /= 1000  # Convert mm to m
    noise_dict_x = noise_data.set_index("name")["Horizontal_STD"].to_dict()
    noise_dict_y = noise_data.set_index("name")["Vertical_STD"].to_dict()
    # Replace any 0 std values with infinity, as these should have zero weight (infinite variance)
    noise_dict_x = {k: (v if v != 0 else float("inf")) for k, v in noise_dict_x.items()}
    noise_dict_y = {k: (v if v != 0 else float("inf")) for k, v in noise_dict_y.items()}

    # Extract BPM types and compute mean std per type
    def get_bpm_type(name):
        if not name.startswith("BPM"):
            raise ValueError(f"Invalid BPM name: {name}")
        parts = name.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid BPM name: {name}")
        type_ = parts[0].strip("BPM")
        # Special handling for BPMW variations: treat all as type 'W'
        if type_.startswith("W") and len(type_) >= 2:
            return "W"
        return type_

    noise_data["type"] = noise_data["name"].apply(get_bpm_type)

    type_means_x = {}
    type_means_y = {}
    for type_, group in noise_data.groupby("type"):
        x_vals = group["Horizontal_STD"]
        y_vals = group["Vertical_STD"]
        non_zero_x = x_vals[x_vals != 0]
        non_zero_y = y_vals[y_vals != 0]
        type_means_x[type_] = (non_zero_x**2).mean() if len(non_zero_x) > 0 else float("inf")
        type_means_y[type_] = (non_zero_y**2).mean() if len(non_zero_y) > 0 else float("inf")

    def get_variance(bpm, noise_dict, type_means):
        if bpm in noise_dict:
            val = noise_dict[bpm] ** 2
        else:
            type_ = get_bpm_type(bpm)
            if type_ not in type_means:
                raise ValueError(f"Unknown BPM type '{type_}' for BPM {bpm}")
            val = type_means[type_]
        if val == float("inf"):
            return float("inf")
        return val

    combined["var_x"] = combined["name"].apply(
        lambda bpm: get_variance(bpm, noise_dict_x, type_means_x)
    )
    combined["var_y"] = combined["name"].apply(
        lambda bpm: get_variance(bpm, noise_dict_y, type_means_y)
    )
    combined.loc[combined["name"].isin(bad_bpms), "var_x"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_y"] = float("inf")
    return combined


def run_analysis(
    analysis_dir: str | Path, model_dir: str | Path, files: list[str | Path], beam: int
) -> list[str]:
    """Load, combine, and process data from multiple files."""
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    bunches = [32, 1228, 2000] if beam == 2 else [0, 1138, 1968]
    hole_in_one_entrypoint(
        harpy=True,
        files=files,
        outputdir=analysis_dir / "lin_files",
        unit="mm",
        driven_excitation="acd",
        # nat_tunes=[0.28, 0.31],
        first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        is_free_kick=False,
        keep_exact_zeros=False,
        max_peak=0.02,
        nattunes=[0.28, 0.31, 0.0],
        num_svd_iterations=3,
        opposite_direction=beam == 2,
        output_bits=12,
        peak_to_peak=1e-08,
        resonances=4,
        sing_val=12,
        svd_dominance_limit=0.925,
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],
        tune_clean_limit=1e-05,
        tunes=[0.27, 0.322, 0.0],
        turn_bits=18,
        model_dir=model_dir,
        turns=[0, 50000],
        clean=True,
    )

    analysed_files = [
        analysis_dir / "lin_files" / f"{f.name}_bunchID{bunch_id}"
        for f in files
        for bunch_id in bunches
    ]

    hole_in_one_entrypoint(
        optics=True,
        files=analysed_files,
        outputdir=analysis_dir,
        analyse_dpp=0,
        # calibrationdir="/afs/cern.ch/eng/sl/lintrack/LHC_commissioning2017/Calibration_factors_2017/Calibration_factors_2017_beam1",
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
        beam=beam,
        dpp=0.0,
        model_dir=model_dir,
        xing=False,
        year="2025",
    )

    bad_bpms = []
    for file in analysed_files:
        bpm_summary_file_x = file.parent / (file.name + ".bad_bpms_x")
        bpm_summary_file_y = file.parent / (file.name + ".bad_bpms_y")
        if bpm_summary_file_x.exists():
            with bpm_summary_file_x.open("r") as f:
                bad_bpms.extend([line.split(" ")[0] for line in f.readlines()])
        if bpm_summary_file_y.exists():
            with bpm_summary_file_y.open("r") as f:
                bad_bpms.extend([line.split(" ")[0] for line in f.readlines()])
    bad_bpms = list(set(bad_bpms))
    logger.info(f"Identified {len(bad_bpms)} bad BPMs from analysis.")
    return bad_bpms


def build_dict_from_nxcal_result(result: list[NXCalResult]) -> dict[str, float]:  # type: ignore
    """Convert NXCalResult to a dictionary of magnet strengths."""
    return {res.name: res.value for res in result}


# def write_datafile(data: pd.DataFrame, output_file: str | Path) -> None:
#     """Write the combined DataFrame to a Parquet file."""
#     data.to_parquet(output_file)


def _process_single_dataframe(
    df_with_index: tuple[int, pd.DataFrame],
    tws: pd.DataFrame,
    bad_bpms: list[str],
    use_uniform_vars: bool,
) -> tuple[int, pd.DataFrame]:
    """Process a single DataFrame: clean, compute vars, and calculate pz.

    Args:
        df_with_index: Tuple of (index, dataframe)
        tws: Twiss DataFrame with optics parameters
        bad_bpms: List of bad BPM names
        use_uniform_vars: Whether to use uniform variances

    Returns:
        Tuple of (original_index, processed_dataframe)
    """
    i, df = df_with_index

    # SVD clean
    df = svd_clean_measurements(df)

    # Remove BPMs not in twiss data
    df = df[df["name"].isin(tws.index)]

    # Compute variances
    if use_uniform_vars:
        df = compute_uniform_vars(df, bad_bpms)
    else:
        df = compute_vars_from_known_noise(df, bad_bpms)

    # Subtract closed orbit
    x_co = tws["x"].to_dict()
    y_co = tws["y"].to_dict()
    df["x"] = df["x"] - df["name"].map(x_co)
    df["y"] = df["y"] - df["name"].map(y_co)

    # Calculate pz
    df = calculate_pz(df, tws=tws, inject_noise=False)

    # Add back closed orbit
    df["x"] = df["x"] + df["name"].map(x_co)
    df["y"] = df["y"] + df["name"].map(y_co)

    # Handle NaN values
    if df["px"].isna().any() or df["py"].isna().any():
        logger.warning(f"NaN values found in px or py for dataframe {i}, dropping rows.")
        df = df.dropna(subset=["px", "py"])

    return i, df


def save_online_knobs(
    meas_time: datetime,
    beam: int,
    tune_knobs_file: Path | None = None,
    corrector_knobs_file: Path | None = None,
) -> None:
    """Load and save knob data from NXCal."""
    try:
        from nxcals.spark_session_builder import get_or_create
        from pylhc import corrector_extraction, mqt_extraction
    except ImportError as e:
        raise ImportError(
            "nxcals and pylhc are required for save_online_knobs but are not installed."
        ) from e

    spark = get_or_create()
    mq_results = mqt_extraction.get_mq_vals(spark, meas_time, beam)
    mqt_results = mqt_extraction.get_mqt_vals(spark, meas_time, beam)
    ms_results = mqt_extraction.get_ms_vals(spark, meas_time, beam)
    mb_results = mqt_extraction.get_mb_vals(spark, meas_time, beam)
    corrector_results = corrector_extraction.get_mcb_vals(spark, meas_time, beam)

    # Stop Spark context to avoid conflicts with multiprocessing
    # Spark signal handlers interfere with ProcessPoolExecutor shutdown
    spark.stop()

    mqt_knobs = build_dict_from_nxcal_result(mqt_results)
    ms_knobs = build_dict_from_nxcal_result(ms_results)
    mb_knobs = build_dict_from_nxcal_result(mb_results)
    mq_knobs = build_dict_from_nxcal_result(mq_results)
    corrector_knobs = build_dict_from_nxcal_result(corrector_results)

    if tune_knobs_file is None:
        tune_knobs_file = TUNE_KNOBS_FILE
    if corrector_knobs_file is None:
        corrector_knobs_file = CORRECTOR_STRENGTHS

    main_magnet_knobs = {**mqt_knobs, **ms_knobs, **mb_knobs, **mq_knobs}
    save_knobs(main_magnet_knobs, tune_knobs_file)
    save_knobs(corrector_knobs, corrector_knobs_file)


def detect_bad_bpms(
    pzs: pd.DataFrame | list[pd.DataFrame],
    all_bpms: set[str],
    bad_bpms: list[str],
    log_individual: bool = True,
) -> None:
    """Detect and add bad BPMs to the list.

    Args:
        pzs: DataFrame or list of DataFrames with processed data
        all_bpms: Set of all expected BPM names
        bad_bpms: List to extend with bad BPMs
        log_individual: Whether to log individual bad BPMs
    """
    if isinstance(pzs, pd.DataFrame):
        pzs = [pzs]

    for pz in pzs:
        # BPMs with NaN in px or py
        mask = pz["px"].isna() | pz["py"].isna()
        bad_bpms_mask = mask.groupby(pz["name"]).any()
        new_bad = bad_bpms_mask[bad_bpms_mask].index.tolist()
        bad_bpms.extend(new_bad)
        if log_individual:
            for bpm in new_bad:
                logger.info(f"BPM {bpm}: has_nan=True")

        # BPMs with infinite variance in both planes (x/y)
        zero_mask = ((np.isinf(pz["var_x"])) | (np.isinf(pz["var_px"]))) & (
            (np.isinf(pz["var_y"])) | (np.isinf(pz["var_py"]))
        )
        bad_bpms_zero = zero_mask.groupby(pz["name"]).any()
        new_bad_zero = bad_bpms_zero[bad_bpms_zero].index.tolist()
        bad_bpms.extend(new_bad_zero)
        if log_individual:
            for bpm in new_bad_zero:
                logger.info(f"BPM {bpm}: zero_weight=True")

    # Missing BPMs
    all_unique_bpms = set.union(*(set(pz["name"].unique()) for pz in pzs))
    missing_bpms = all_bpms - all_unique_bpms
    bad_bpms.extend(missing_bpms)
    if log_individual:
        for bpm in missing_bpms:
            logger.info(f"BPM {bpm}: missing from data")

    # Remove duplicates
    bad_bpms[:] = list(set(bad_bpms))


def process_measurements(
    files: list[Path],
    analysis_dir: Path,
    model_dir: str,
    beam: int,
    filename: str | None = "pz_data.parquet",
    bad_bpms: list[str] | None = None,
    use_uniform_vars: bool = False,
    num_workers: int | None = None,
    sequence_path: Path | None = None,
    combine_files: bool = True,
) -> tuple[pd.DataFrame | list[pd.DataFrame], list[str], Path | list[Path]]:
    """Process measurement files to compute pz data and identify bad BPMs.

    Args:
        files: List of measurement file paths
        analysis_dir: Directory for analysis outputs
        model_dir: Directory containing model files
        beam: Beam number (1 or 2)
        filename: Output filename for parquet file (None to skip saving)
        bad_bpms: List of bad BPM names (None to run analysis)
        use_uniform_vars: If True, use uniform variances instead of noise-based
        num_workers: Number of parallel workers (None for auto)
        sequence_path: Path to the MAD-X sequence file
        combine_files: If True, combine all processed dataframes into one; if False, return list of dataframes

    Returns:
        Tuple of (pzs_dataframe or list of dataframes, bad_bpms_list, output_path or list of paths)
    """
    if bad_bpms is None:
        bad_bpms = run_analysis(analysis_dir, model_dir, files, beam)

    data = load_files(files)
    combined = convert_measurements(data, bad_bpms, combine_measurements=combine_files)
    logger.info(f"Combined data has {len(combined)} DataFrames from different files/bunches.")
    tws = tfs.read(Path(model_dir) / "twiss_ac.dat")
    tws.columns = [col.lower() for col in tws.columns]
    tws = tws.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws.headers = {k.lower(): v for k, v in tws.headers.items()}
    tws = tws.set_index("name")

    # Process DataFrames in parallel using threads to avoid Spark context inheritance issues
    # ThreadPoolExecutor shares memory space and doesn't inherit problematic global state like ProcessPoolExecutor
    logger.info(f"Processing {len(combined)} DataFrames in parallel with threads...")
    processed_results = [None] * len(combined)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid Spark context conflicts
    # Limit to max 9 threads to avoid overloading the system
    effective_workers = min(num_workers or len(combined), 9)
    if effective_workers > 0:
        try:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(
                        _process_single_dataframe, (i, df), tws, bad_bpms, use_uniform_vars
                    ): i
                    for i, df in enumerate(combined)
                }

                for future in as_completed(futures):
                    try:
                        idx, processed_df = future.result(timeout=600)  # 10 minute timeout per task
                        processed_results[idx] = processed_df
                        logger.info(f"Completed processing dataframe {idx + 1}/{len(combined)}")
                    except Exception as e:
                        idx = futures[future]
                        logger.error(f"Error processing dataframe {idx}: {e}")
                        raise
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt received, shutting down gracefully...")
            # The ThreadPoolExecutor context manager will handle cleanup
            raise
    else:
        for i, df in enumerate(combined):
            idx, processed_df = _process_single_dataframe(
                df_with_index=(i, df), tws=tws, bad_bpms=bad_bpms, use_uniform_vars=use_uniform_vars
            )
            processed_results[idx] = processed_df
            logger.info(f"Completed processing dataframe {idx + 1}/{len(combined)}")

    combined = processed_results
    if combine_files:
        pzs = pd.concat(combined, ignore_index=True)
        pzs["name"] = pzs["name"].astype("category")
        pzs["turn"] = pzs["turn"].astype("int32")
        # Add the average dpp estimate to the headers
        dpp_est = sum(proc_res.attrs["DPP_EST"] for proc_res in combined) / len(combined)
        pzs.attrs["DPP_EST"] = dpp_est
    else:
        # Group by file: each file has multiple bunches combined
        num_files = len(files)
        num_bunches_per_file = len(combined) // num_files
        pzs = []
        for i in range(num_files):
            start = i * num_bunches_per_file
            end = (i + 1) * num_bunches_per_file
            file_dfs = combined[start:end]
            file_pzs = pd.concat(file_dfs, ignore_index=True)
            file_pzs["name"] = file_pzs["name"].astype("category")
            file_pzs["turn"] = file_pzs["turn"].astype("int32")
            file_pzs.attrs["DPP_EST"] = sum(df.attrs["DPP_EST"] for df in file_dfs) / len(file_dfs)
            pzs.append(file_pzs)

    sequence_path = sequence_path or get_lhc_file_path(beam)

    mad_iface = OptimisationMadInterface(
        sequence_path,
        seq_name=f"LHCB{beam}",
    )
    all_bpms = set(mad_iface.all_bpms)
    del mad_iface

    if combine_files:
        pzs["name"] = pzs["name"].astype("category")

        detect_bad_bpms(pzs, all_bpms, bad_bpms, log_individual=True)

        logger.info(f"Total bad BPMs: {len(bad_bpms)}")

        if filename:
            file_path = analysis_dir / filename
            pzs.to_parquet(file_path)

            return pzs, bad_bpms, file_path
        return pzs, bad_bpms, analysis_dir

    detect_bad_bpms(pzs, all_bpms, bad_bpms, log_individual=False)

    logger.info(f"Total bad BPMs: {len(bad_bpms)}")

    if filename:
        file_paths = []
        for i, pz in enumerate(pzs):
            file_path = analysis_dir / f"{Path(filename).stem}_{i}.parquet"
            pz.to_parquet(file_path)
            file_paths.append(file_path)
        return pzs, bad_bpms, file_paths
    return pzs, bad_bpms, analysis_dir


if __name__ == "__main__":
    # set logging level to debug
    logging.basicConfig(level=logging.INFO)
    # analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir = PROJECT_ROOT / "analysis_trim"
    model_dir = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Models/b1_flat_60_18cm"

    MAGNET_RANGES = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]

    BPM_STARTS = [[f"BPM.{i}R{s}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]
    BPM_END_POINTS = [[f"BPM.{i}L{s % 8 + 1}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]

    files = [
        # Before the trim
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_47_22_071/Beam1@BunchTurn@2025_04_09@18_47_22_071.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_48_27_464/Beam1@BunchTurn@2025_04_09@18_48_27_464.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_51_14_983/Beam1@BunchTurn@2025_04_09@18_51_14_983.sdds",
        # "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_52_18_410/Beam1@BunchTurn@2025_04_09@18_52_18_410.sdds",
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
    save_online_knobs(meas_time, beam=1)
    measurement_filename = "pz_data.parquet"
    measurement_file = analysis_dir / measurement_filename
    bad_bpms_file = analysis_dir / "bad_bpms.txt"

    pzs, bad_bpms, _ = process_measurements(
        files, analysis_dir, model_dir, beam=1, filename=measurement_filename
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
            beam=1,
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

    logger.info(f"Mean deltap: {np.mean(results)}")
    logger.info(f"Std dev of deltap: {np.std(results)}")

    with results_file.open("a") as f:
        f.write(f"Mean\t{np.mean(results)}\n")
        f.write(f"StdDev\t{np.std(results)}\n")
