from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.measurements.create_datafile import (
    ACDipoleReconstructionConfig,
    process_measurements,
    save_online_knobs,
)
from aba_optimiser.measurements.squeeze_helpers import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    extract_tunes_from_job_file,
    get_ir_bpm_ranges_from_model,
    get_measurement_date,
    get_or_make_sequence,
)
from aba_optimiser.measurements.utils import find_all_bad_bpms
from aba_optimiser.model_creator.config import AC_MARKER_PATTERN
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import (
    CheckpointConfig,
    MeasurementConfig,
    OutputConfig,
    SequenceConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ACDipoleOptimisationWindow:
    """Window definition for full-turn arc-by-arc tracking around the AC dipole."""

    bpm_upstream: str
    bpm_downstream: str


# ==================== CONSTANTS ====================
FILL_NUMBER = 10533
SQUEEZE_STEPS = [
    "1.2m",
    "1.2m_agc",
    "1.05m",
    "0.93m",
    "0.725m",
    "0.6m",
    "0.45m",
    "0.3m",
    "0.25m",
    "0.18m",
]

ZEROHZ = "0Hz"
PLUS_50HZ = "+50Hz"
MINUS_50HZ = "-50Hz"
PLUS_100HZ = "+100Hz"
PLUS_150HZ = "+150Hz"
PLUS_200HZ = "+200Hz"
PLUS_250HZ = "+250Hz"
PLUS_300HZ = "+300Hz"
PLUS_350HZ = "+350Hz"
MINUS_100HZ = "-100Hz"
MINUS_150HZ = "-150Hz"
MINUS_200HZ = "-200Hz"
MINUS_250HZ = "-250Hz"
MINUS_300HZ = "-300Hz"
MINUS_350HZ = "-350Hz"
CLOSED_ORBIT_FILE = "closed_orbit_0Hz.parquet"
CLOSED_ORBIT_TWISS_FILE = "closed_orbit_twiss_reference.parquet"
CLOSED_ORBIT_COLUMNS = ("x", "px", "y", "py")

# ==================== MEASUREMENT TIMES ====================
MEAS_TIMES = {
    1: {
        "1.2m": {
            ZEROHZ: ["06_17_47_405", "06_19_27_443", "06_20_39_422"],
            PLUS_50HZ: ["06_24_41_350"],
            MINUS_50HZ: ["06_26_09_426"],
        },
        # Please also check the 1.2m after a global correction
        "1.2m_agc": {
            ZEROHZ: ["07_59_50_367", "08_01_04_464", "08_02_31_317"],
            # ZEROHZ: ["07_59_50_367"],
            # ZEROHZ: ["08_02_31_317"],
            # PLUS_50HZ: ["08_05_57_495"],
            # MINUS_50HZ: ["08_07_03_451"],
        },
        "1.05m": {
            ZEROHZ: ["09_19_42_460", "09_20_58_100", "09_23_22_506"],
            PLUS_50HZ: ["09_26_23_443", "09_27_41_385"],
            MINUS_50HZ: ["09_29_36_402", "09_30_46_492"],
        },
        "0.93m": {
            ZEROHZ: ["10_56_44_423", "10_57_53_330"],
            MINUS_50HZ: ["11_00_38_320"],
            PLUS_50HZ: ["11_02_35_390"],
        },
        "0.725m": {
            ZEROHZ: ["12_56_35_478", "12_58_04_387", "12_59_41_358"],
            MINUS_50HZ: ["13_01_45_317"],
            PLUS_50HZ: ["13_03_09_448"],
        },
        "0.6m": {
            ZEROHZ: ["13_50_14_431", "13_51_22_434", "13_52_31_321"],
            MINUS_50HZ: ["13_54_02_456"],
            PLUS_50HZ: ["13_55_11_500"],
        },
        "0.45m": {
            ZEROHZ: ["14_48_46_464", "14_49_53_484"],
            MINUS_50HZ: ["14_51_35_448"],
            PLUS_50HZ: ["14_52_41_380"],
        },
        "0.3m": {
            ZEROHZ: ["15_35_51_360", "15_36_59_413"],
            MINUS_50HZ: ["15_38_30_335"],
            PLUS_50HZ: ["15_39_41_491"],
        },
        "0.25m": {
            ZEROHZ: ["16_22_59_435", "16_24_06_387"],
            MINUS_50HZ: ["16_25_33_427"],
            PLUS_50HZ: ["16_27_05_421"],
        },
        "0.18m": {
            ZEROHZ: [
                "17_34_50_472",
                "17_36_06_419",
                "17_39_46_502",
                "17_41_31_474",
                "17_42_37_338",
            ],
            MINUS_50HZ: ["17_44_16_396"],
            PLUS_50HZ: ["17_45_33_318"],
        },
        "inj": {  # This has a special date in the squeeze_helpers.py
            ZEROHZ: ["16_49_56_490", "16_51_01_464", "16_52_08_398"],
            # PLUS_50HZ: ["16_58_06_419", "16_59_12_366", "17_00_27_504"],
            # PLUS_100HZ: ["17_03_16_394", "17_04_25_452", "17_05_53_444"],
            # PLUS_150HZ: ["17_10_45_523", "17_11_52_941", "17_12_57_191"],
            # PLUS_200HZ: ["17_20_55_315", "17_22_05_108", "17_23_09_370"],
            # PLUS_250HZ: ["17_26_28_349", "17_27_50_402", "17_29_16_374"],
            # PLUS_300HZ: ["17_33_14_338", "17_34_20_356", "17_35_28_317"],
            # PLUS_350HZ: ["17_40_25_475", "17_41_36_368", "17_42_54_434"],
            # MINUS_50HZ: ["17_51_38_387", "17_52_44_501", "17_53_50_407"],
            # MINUS_100HZ: ["17_57_13_340", "17_58_24_385", "17_59_32_491"],
            # MINUS_150HZ: ["18_02_51_331", "18_03_57_483", "18_05_05_332"],
            # MINUS_200HZ: ["18_06_53_392", "18_07_59_340", "18_09_04_349"],
            # MINUS_250HZ: ["18_11_32_374", "18_12_51_327", "18_14_23_472"],
            # MINUS_300HZ: ["18_16_36_372", "18_17_44_322", "18_18_52_182"],
            # MINUS_350HZ: ["18_20_38_431", "18_21_46_310", "18_22_51_320"],
        },
        "inj_rdt": {
            ZEROHZ: ["15_22_50_333", "15_23_58_444"]  # , "15_25_19_330"],
        },
    },
    2: {
        "1.2m": {
            ZEROHZ: ["06_18_14_332", "06_19_48_490", "06_20_57_500"],
            PLUS_50HZ: ["06_25_20_342"],
            MINUS_50HZ: ["06_26_47_456"],
        },
        "1.05m": {
            ZEROHZ: ["09_19_49_333", "09_21_08_376", "09_23_29_348"],
            PLUS_50HZ: ["09_26_31_404", "09_28_03_384"],
            MINUS_50HZ: ["09_29_55_397", "09_31_07_320"],
        },
        "0.93m": {
            ZEROHZ: ["10_57_01_409", "10_58_15_318"],
            MINUS_50HZ: ["11_01_01_305"],
            PLUS_50HZ: ["11_02_56_432"],
        },
        "0.725m": {
            ZEROHZ: ["12_58_53_397", "13_00_13_435"],
            MINUS_50HZ: ["13_02_01_455"],
            PLUS_50HZ: ["13_03_26_368"],
        },
        "0.6m": {
            ZEROHZ: ["13_50_31_386", "13_51_38_386", "13_52_47_410"],
            MINUS_50HZ: ["13_54_20_403"],
            PLUS_50HZ: ["13_55_28_330"],
        },
        "0.45m": {
            ZEROHZ: ["14_49_16_409", "14_50_23_338"],
            MINUS_50HZ: ["14_51_51_461"],
            PLUS_50HZ: ["14_52_58_336"],
        },
        "0.3m": {
            ZEROHZ: ["15_36_09_480", "15_37_15_426"],
            MINUS_50HZ: ["15_38_45_453"],
            PLUS_50HZ: ["15_39_57_354"],
        },
        "0.25m": {
            ZEROHZ: ["16_23_14_482", "16_24_20_310"],
            MINUS_50HZ: ["16_25_49_324"],
            PLUS_50HZ: ["16_27_20_308"],
        },
        "0.18m": {
            ZEROHZ: ["17_35_05_355", "17_41_46_322", "17_42_55_359"],
            MINUS_50HZ: ["17_44_33_344"],
            PLUS_50HZ: ["17_45_48_406"],
        },
        "inj": {  # This has a special date in the squeeze_helpers.py
            ZEROHZ: ["16_50_23_408", "16_51_29_457", "16_52_34_444"],
            # PLUS_50HZ: ["16_58_39_378", "16_59_45_316", "17_00_51_406"],
            # PLUS_100HZ: ["17_03_42_327", "17_04_51_372", "17_06_00_423"],
            # PLUS_150HZ: ["17_09_53_327", "17_10_58_495", "17_12_09_324"],
            # PLUS_200HZ: ["17_23_20_460", "17_19_54_443", "17_21_03_313"],
            # PLUS_250HZ: ["17_28_11_396", "17_29_22_415", "17_30_40_479"],
            # PLUS_300HZ: ["17_34_11_504", "17_35_21_234", "17_36_28_939"],
            # PLUS_350HZ: ["17_40_14_958", "17_41_28_883", "17_42_46_833"],
            # MINUS_50HZ: ["17_51_17_922", "17_52_22_657", "17_53_28_125"],
            # MINUS_100HZ: ["17_57_57_917", "17_59_08_915", "18_00_26_732"],
            # MINUS_150HZ: ["18_03_31_335", "18_04_37_370", "18_05_49_463"],
            # MINUS_200HZ: ["18_07_24_413", "18_08_30_429", "18_09_36_407"],
            # MINUS_250HZ: ["18_11_42_328", "18_13_31_372", "18_14_45_489"],
            # MINUS_300HZ: ["18_16_48_300", "18_17_58_399", "18_19_04_402"],
            # MINUS_350HZ: ["18_20_15_411", "18_21_21_418", "18_22_26_431"],
        },
        "inj_rdt": {
            ZEROHZ: ["15_16_32_449"],  # , "15_15_23_387", "15_17_43_444"],
        },
    },
}


# ==================== HELPER FUNCTIONS ====================
def get_beam_paths(beam: int, squeeze_step: str) -> tuple[Path, Path]:
    """Get measurement and model base directories for a beam."""
    beam_path = BETABEAT_DIR / get_measurement_date(squeeze_step) / f"LHCB{beam}/"
    meas_dir = beam_path / "Measurements/"
    model_dir = beam_path / "Models/"
    return meas_dir, model_dir


def get_analysis_dir(beam: int, squeeze_step: str) -> Path:
    """Get analysis directory for a beam and squeeze step."""
    beam_path = BETABEAT_DIR / get_measurement_date(squeeze_step) / f"LHCB{beam}/"
    return beam_path / "Results" / ANALYSIS_DIRS[beam][squeeze_step]


def get_bpm_points(arc_num: int, beam: int) -> tuple[str, list[str], list[str]]:
    """Get magnet range and BPM points for an arc."""
    next_arc = arc_num % 8 + 1
    suffix = f".B{beam}"
    start_bpm = f"BPM.14R{arc_num}{suffix}"
    # end_bpm = f"BPM.15R{arc_num}{suffix}" # Testing with BPM.15R
    end_bpm = f"BPM.13L{next_arc}{suffix}"
    magnet_range = f"{start_bpm}/{end_bpm}"
    first_i = 14
    last_i = 13
    bpm_start_points = [f"BPM.{i}R{arc_num}{suffix}" for i in range(first_i, 17, 1)]
    bpm_end_points = [f"BPM.{i}L{next_arc}{suffix}" for i in range(last_i, 16, 1)]

    # Testing these fixed points
    # bpm_start_points = [start_bpm]
    # bpm_end_points = [end_bpm]

    return magnet_range, bpm_start_points, bpm_end_points


def get_full_ring_bpm_points(beam: int) -> tuple[str, list[str], list[str]]:
    """Get full-ring BPM points for non-arc-by-arc (multi-turn) running."""
    if beam == 1:
        bpm_start_points = [f"BPM.9{lr}{s}.B1" for s in range(1, 9) for lr in ["R"]]
        bpm_end_points = []
    else:
        bpm_start_points = [f"BPM.9L{s}.B2" for s in range(8, 0, -1)]
        bpm_end_points = []
    return "$start/$end", bpm_start_points, bpm_end_points


def get_ac_dipole_bpm_points(
    beam: int, window: ACDipoleOptimisationWindow
) -> tuple[str, list[str], list[str]]:
    """Get full-turn arc-by-arc points anchored around the AC dipole."""
    suffix = f".B{beam}"
    bpm_upstream = window.bpm_upstream
    bpm_downstream = window.bpm_downstream
    if not bpm_upstream.endswith(suffix):
        raise ValueError(f"Upstream BPM {bpm_upstream} does not match beam suffix {suffix}")
    if not bpm_downstream.endswith(suffix):
        raise ValueError(f"Downstream BPM {bpm_downstream} does not match beam suffix {suffix}")
    return f"{bpm_downstream}/{bpm_upstream}", [bpm_downstream], [bpm_upstream]


def window_from_attrs(attrs: dict) -> ACDipoleOptimisationWindow | None:
    """Build AC-dipole optimisation window from dataframe/metadata attrs."""
    upstream = attrs.get("ac_dipole_bpm_upstream")
    downstream = attrs.get("ac_dipole_bpm_downstream")
    if upstream and downstream:
        return ACDipoleOptimisationWindow(
            bpm_upstream=str(upstream),
            bpm_downstream=str(downstream),
        )
    return None


def validate_processed_files(temp_analysis_dir: Path, freq: str, num_files: int) -> None:
    """Validate that all processed measurement files exist."""
    expected_files = [temp_analysis_dir / f"pz_data_{freq}_{i}.parquet" for i in range(num_files)]
    missing_files = [f for f in expected_files if not f.exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} processed measurement files for {freq}. "
            f"First missing: {missing_files[0]}. Run without --skip-reload to regenerate."
        )

    logger.info(f"Verified {num_files} processed files exist for {freq}")


def load_bad_bpms(bad_bpms_file: Path) -> set[str]:
    """Load bad BPMs from file."""
    if not bad_bpms_file.exists():
        raise FileNotFoundError(
            f"Bad BPMs file {bad_bpms_file} not found. Run without --skip-reload first."
        )

    with bad_bpms_file.open("r") as f:
        bad_bpms = {line.strip() for line in f if line.strip()}

    logger.info(f"Loaded {len(bad_bpms)} bad BPMs from {bad_bpms_file}")
    return bad_bpms


def save_bad_bpms(bad_bpms_file: Path, bad_bpms: set[str]) -> None:
    """Save bad BPMs to file."""
    if not bad_bpms:
        return

    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")


def get_measurement_time(earliest_time: str, squeeze_step: str) -> datetime:
    """Convert timestamp string to datetime object."""
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{get_measurement_date(squeeze_step).replace('_', '-')} {time_str}"
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))


def get_sequence_creation_time(meas_times_for_step: dict[str, list[str]], squeeze_step: str) -> str:
    """Return an explicit machine-settings time for sequence creation."""
    all_times = [time for times in meas_times_for_step.values() for time in times]
    if not all_times:
        raise ValueError(f"No measurement times configured for squeeze step {squeeze_step}.")
    return get_measurement_time(min(all_times), squeeze_step).isoformat()


def collect_bad_bpms_from_folders(analysed_folders: list[Path]) -> set[str]:
    """Collect bad BPMs from analysis folders."""
    bad_bpms = set()
    for folder in analysed_folders:
        bad_bpms.update(find_all_bad_bpms(folder))
    return bad_bpms


def validate_folders_exist(analysed_folders: list[Path], squeeze_step: str, freq: str) -> None:
    """Validate that all analysis folders exist."""
    for folder in analysed_folders:
        if not folder.exists():
            raise FileNotFoundError(
                f"Analysed folder {folder} does not exist for squeeze step {squeeze_step} and frequency {freq}."
            )


def setup_temp_directory(temp_analysis_dir: Path, skip_reload: bool) -> None:
    """Setup or validate temp analysis directory."""
    if skip_reload:
        if not temp_analysis_dir.exists():
            raise FileNotFoundError(
                f"Temp analysis directory {temp_analysis_dir} not found. "
                "Run without --skip-reload first to generate processed data."
            )
        logger.info(f"Using existing temp directory: {temp_analysis_dir}")
    else:
        temp_analysis_dir.mkdir(exist_ok=True)


def load_metadata(temp_analysis_dir: Path) -> dict:
    """Load metadata from unified metadata file."""
    metadata_file = temp_analysis_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
    with metadata_file.open("r") as f:
        return json.load(f)


def save_metadata(temp_analysis_dir: Path, metadata: dict) -> None:
    """Save metadata to unified metadata file."""
    metadata_file = temp_analysis_dir / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)


def update_metadata(temp_analysis_dir: Path, **kwargs) -> None:
    """Update specific keys in metadata file."""
    metadata = load_metadata(temp_analysis_dir)
    metadata.update(kwargs)
    save_metadata(temp_analysis_dir, metadata)


def get_closed_orbit_file(temp_analysis_dir: Path) -> Path:
    """Return the stored 0 Hz closed-orbit reference path."""
    return temp_analysis_dir / CLOSED_ORBIT_FILE


def get_closed_orbit_twiss_file(temp_analysis_dir: Path) -> Path:
    """Return the stored Twiss orbit reference path used for closed-orbit building."""
    return temp_analysis_dir / CLOSED_ORBIT_TWISS_FILE


def compute_weighted_mean_and_variance(
    sub: pd.DataFrame, value_col: str, var_col: str
) -> tuple[float, float]:
    """Compute an inverse-variance weighted mean and its variance."""
    values = sub[value_col].to_numpy()
    variances = sub[var_col].to_numpy()
    mask = np.isfinite(values) & np.isfinite(variances) & (variances > 0)
    values = values[mask]
    variances = variances[mask]

    if values.size == 0:
        mean = float(sub[value_col].mean())
        count = sub[value_col].count()
        if count >= 2:
            sample_var = float(np.var(sub[value_col].to_numpy(), ddof=1))
            return mean, sample_var / count
        return mean, np.nan

    weights = 1.0 / variances
    sum_weights = float(np.sum(weights))
    mean = float(np.sum(weights * values) / sum_weights)
    return mean, 1.0 / sum_weights


def calculate_reference_closed_orbit(
    pzs_list: list[pd.DataFrame], twiss: pd.DataFrame
) -> pd.DataFrame:
    """Build the BPM-by-BPM additive closed-orbit reference from 0 Hz measurements.

    The stored reference is the extra orbit on top of the model expectation, i.e.
    ``closed_orbit = measurement_mean - twiss_expected`` for each BPM and each
    coordinate in :data:`CLOSED_ORBIT_COLUMNS`.

    """
    reference_input = pd.concat(pzs_list, ignore_index=True)
    rows = []
    for name, sub in reference_input.groupby("name", observed=False):
        x, var_x = compute_weighted_mean_and_variance(sub, "x", "var_x")
        y, var_y = compute_weighted_mean_and_variance(sub, "y", "var_y")
        px, var_px = compute_weighted_mean_and_variance(sub, "px", "var_px")
        py, var_py = compute_weighted_mean_and_variance(sub, "py", "var_py")
        # Store only the additive orbit beyond the nominal Twiss expectation.
        twiss_row = twiss.loc[name]
        rows.append(
            {
                "name": name,
                "x": x - twiss_row["x"],
                "y": y - twiss_row["y"],
                "px": px - twiss_row["px"],
                "py": py - twiss_row["py"],
                "var_x": var_x,
                "var_y": var_y,
                "var_px": var_px,
                "var_py": var_py,
            }
        )

    closed_orbit = pd.DataFrame(rows)
    closed_orbit["name"] = closed_orbit["name"].astype("category")
    return closed_orbit


def save_reference_closed_orbit(temp_analysis_dir: Path, closed_orbit: pd.DataFrame) -> None:
    """Persist the 0 Hz closed-orbit reference."""
    closed_orbit.to_parquet(get_closed_orbit_file(temp_analysis_dir))


def save_closed_orbit_twiss_reference(temp_analysis_dir: Path, twiss: pd.DataFrame) -> None:
    """Persist the Twiss orbit reference used to build additive closed orbit."""
    required_cols = ["x", "px", "y", "py"]
    missing_cols = [c for c in required_cols if c not in twiss.columns]
    if missing_cols:
        raise KeyError(f"Twiss orbit reference missing required columns: {missing_cols}")

    twiss_orbit = twiss[required_cols].copy()
    if "name" in twiss_orbit.columns:
        twiss_orbit = twiss_orbit.set_index("name")
    if twiss_orbit.index.name is None:
        twiss_orbit.index.name = "name"
    twiss_orbit.to_parquet(get_closed_orbit_twiss_file(temp_analysis_dir))


def load_closed_orbit_twiss_reference(temp_analysis_dir: Path) -> pd.DataFrame:
    """Load the stored Twiss orbit reference used for closed-orbit building."""
    twiss_file = get_closed_orbit_twiss_file(temp_analysis_dir)
    if not twiss_file.exists():
        raise FileNotFoundError(
            f"Twiss orbit reference {twiss_file} not found. "
            "Run without --skip-reload to regenerate."
        )
    twiss_orbit = pd.read_parquet(twiss_file)
    if "name" in twiss_orbit.columns:
        twiss_orbit = twiss_orbit.set_index("name")
    return twiss_orbit


def load_reference_closed_orbit(temp_analysis_dir: Path) -> pd.DataFrame:
    """Load the stored 0 Hz closed-orbit reference."""
    closed_orbit_file = get_closed_orbit_file(temp_analysis_dir)
    if not closed_orbit_file.exists():
        raise FileNotFoundError(
            f"Closed-orbit reference {closed_orbit_file} not found. "
            "Run without --skip-reload to regenerate."
        )
    return pd.read_parquet(closed_orbit_file)


def find_missing_closed_orbit_bpms(
    pzs_list: list[pd.DataFrame], closed_orbit: pd.DataFrame
) -> set[str]:
    """Return BPMs present in measurements but absent from the 0 Hz closed-orbit reference."""
    reference_bpms = set(closed_orbit["name"].astype(str))
    missing_bpms = set()
    for pzs in pzs_list:
        missing_bpms.update(set(pzs["name"].astype(str).unique()) - reference_bpms)

    if missing_bpms:
        logger.warning(
            "Adding %d BPMs missing from the 0 Hz closed orbit to the global bad BPM list.",
            len(missing_bpms),
        )
    return missing_bpms


def subtract_reference_closed_orbit(
    pzs: pd.DataFrame, closed_orbit: pd.DataFrame, freq: str
) -> pd.DataFrame:
    """Remove the additive closed-orbit component from a measurement.

    This applies to all orbit coordinates in :data:`CLOSED_ORBIT_COLUMNS` so that
    each corrected value is approximately ``measurement - (measurement_0Hz - twiss)``.
    The horizontal channels are then zero-weighted so only the vertical plane
    contributes to the downstream squeeze fit.
    """
    corrected = pzs.copy()
    # reference = closed_orbit.set_index("name")
    missing_bpms = set()

    # Subtract additive closed-orbit offsets for all tracked orbit coordinates.
    # for column in CLOSED_ORBIT_COLUMNS:
    #     mapped_values = corrected["name"].map(reference[column])
    #     reference_values = pd.to_numeric(mapped_values.astype(object), errors="coerce")
    #     missing_mask = reference_values.isna()
    #     if missing_mask.any():
    #         missing_bpms.update(corrected.loc[missing_mask, "name"].astype(str).unique())
    # corrected[column] = corrected[column] - reference_values.fillna(0.0)

    if missing_bpms:
        logger.warning(
            "Missing %d BPMs in 0 Hz closed-orbit reference for %s; leaving those rows unchanged and expecting them in the global bad BPM list.",
            len(missing_bpms),
            freq,
        )

    corrected.attrs = pzs.attrs.copy()
    return corrected


def migrate_saved_measurements_to_reference_orbit(
    temp_analysis_dir: Path, freq_file_counts: dict[str, int], closed_orbit: pd.DataFrame
) -> None:
    """Apply the 0 Hz closed-orbit subtraction to previously saved parquet files."""
    for freq, num_files in freq_file_counts.items():
        if num_files == 0:
            continue
        validate_processed_files(temp_analysis_dir, freq, num_files)
        for i in range(num_files):
            meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
            corrected = subtract_reference_closed_orbit(
                pd.read_parquet(meas_save_path), closed_orbit, freq
            )
            corrected.to_parquet(meas_save_path)


def prepare_reload_reference_data(
    temp_analysis_dir: Path, freq_file_counts: dict[str, int]
) -> tuple[float, pd.DataFrame]:
    """Load or reconstruct the 0 Hz reference deltap and closed orbit for reloads."""
    metadata = load_metadata(temp_analysis_dir)
    if "dpp_values" not in metadata or ZEROHZ not in metadata["dpp_values"]:
        raise ValueError(
            "Missing 0 Hz DPP metadata in the temp analysis directory. "
            "Run without --skip-reload to regenerate."
        )

    zero_dpp_values = metadata["dpp_values"][ZEROHZ]
    reference_deltap = float(np.mean(zero_dpp_values)) if zero_dpp_values else 0.0

    if (
        metadata.get("closed_orbit_removed") is True
        and get_closed_orbit_file(temp_analysis_dir).exists()
    ):
        closed_orbit = load_reference_closed_orbit(temp_analysis_dir)
    else:
        zero_num_files = freq_file_counts.get(ZEROHZ, 0)
        validate_processed_files(temp_analysis_dir, ZEROHZ, zero_num_files)
        zero_pzs_list = [
            pd.read_parquet(temp_analysis_dir / f"pz_data_{ZEROHZ}_{i}.parquet")
            for i in range(zero_num_files)
        ]
        twiss_orbit = load_closed_orbit_twiss_reference(temp_analysis_dir)
        closed_orbit = calculate_reference_closed_orbit(zero_pzs_list, twiss=twiss_orbit)
        save_reference_closed_orbit(temp_analysis_dir, closed_orbit)
        migrate_saved_measurements_to_reference_orbit(
            temp_analysis_dir, freq_file_counts, closed_orbit
        )

    update_metadata(
        temp_analysis_dir,
        reference_deltap=reference_deltap,
        machine_deltap=0.0,
        closed_orbit_removed=True,
        closed_orbit_reference_definition="measurement_0Hz - twiss_expected",
    )
    return reference_deltap, closed_orbit


def load_saved_pzs_for_frequencies(
    temp_analysis_dir: Path, freq_file_counts: dict[str, int]
) -> list[pd.DataFrame]:
    """Load all saved parquet measurements for the provided frequencies."""
    pzs_list = []
    for freq, num_files in freq_file_counts.items():
        validate_processed_files(temp_analysis_dir, freq, num_files)
        pzs_list.extend(
            pd.read_parquet(temp_analysis_dir / f"pz_data_{freq}_{i}.parquet")
            for i in range(num_files)
        )
    return pzs_list


def get_analysis_folders(
    times: list[str], beam: int, meas_base_dir: Path, squeeze_step: str
) -> list[Path]:
    """Get analysis folder paths for measurement times."""
    name_prefix = f"Beam{beam}@BunchTurn@{get_measurement_date(squeeze_step)}@".replace("-", "_")
    return [meas_base_dir / f"{name_prefix}{time}" for time in times]


def get_measurement_files(
    times: list[str], analysed_folders: list[Path], beam: int, squeeze_step: str
) -> list[Path]:
    """Get measurement file paths."""
    name_prefix = f"Beam{beam}@BunchTurn@{get_measurement_date(squeeze_step)}@".replace("-", "_")
    return [analysed_folders[i] / f"{name_prefix}{times[i]}.sdds" for i in range(len(times))]


def get_knob_files(results_dir: Path, squeeze_step: str, freq: str) -> tuple[Path, Path]:
    """Return tune/corrector knobs file paths for a squeeze step and frequency."""
    return (
        results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt",
        results_dir / f"corrector_strengths_{squeeze_step}_{freq}.txt",
    )


def build_stage_checkpoint_configs(
    checkpoint_dir: Path,
    beam: int,
    squeeze_step: str,
    arc_num: int,
    checkpoint_every_n_epochs: int,
    restore_bends_opt: bool,
    restore_quads_opt: bool,
) -> tuple[CheckpointConfig, CheckpointConfig]:
    """Build stage-specific checkpoint configs for one arc."""
    squeeze_step_id = squeeze_step.replace(".", "_")
    bend_checkpoint = (
        checkpoint_dir / f"checkpoint_b{beam}_{squeeze_step_id}_arc{arc_num}_bends.json"
    )
    quad_checkpoint = (
        checkpoint_dir / f"checkpoint_b{beam}_{squeeze_step_id}_arc{arc_num}_quads.json"
    )
    return (
        CheckpointConfig(
            checkpoint_path=bend_checkpoint,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            restore_from_checkpoint=restore_bends_opt,
        ),
        CheckpointConfig(
            checkpoint_path=quad_checkpoint,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            restore_from_checkpoint=restore_quads_opt,
        ),
    )


def resolve_restore_resume(
    arc_numbers: list[int],
    checkpoint_config: CheckpointConfig,
    beam: int,
    squeeze_step: str,
    restore_bends_opt: bool,
    restore_quads_opt: bool,
) -> tuple[list[int], bool, bool, int | None]:
    """Resolve the resume arc from checkpoints and trim arc list accordingly."""
    if not (restore_bends_opt or restore_quads_opt):
        return arc_numbers, restore_bends_opt, restore_quads_opt, None

    restore_stage = "bends" if restore_bends_opt else "quads"
    squeeze_step_id = squeeze_step.replace(".", "_")
    prefix = f"checkpoint_b{beam}_{squeeze_step_id}_arc"
    suffix = f"_{restore_stage}.json"
    candidates = [
        p
        for p in checkpoint_config.checkpoint_path.glob("*.json")
        if p.name.startswith(prefix) and p.name.endswith(suffix)
    ]
    if not candidates:
        logger.warning(
            "No %s checkpoint files found in %s for beam %d, squeeze step %s. Running without restore.",
            restore_stage,
            checkpoint_config.checkpoint_path,
            beam,
            squeeze_step,
        )
        return arc_numbers, False, False, None

    checkpoint_file = max(candidates, key=lambda p: p.stat().st_mtime)
    arc_token = checkpoint_file.stem.rsplit("_", 2)[1]
    restore_arc = int(arc_token.replace("arc", ""))
    logger.info(
        "Resuming from most recent %s checkpoint: arc %d (%s)",
        restore_stage,
        restore_arc,
        checkpoint_file,
    )

    allowed_arc_set = set(arc_numbers)
    if restore_arc not in allowed_arc_set:
        logger.warning(
            "Most recent restore arc %d is incompatible with current run mode (allowed: %s). Running without restore.",
            restore_arc,
            sorted(allowed_arc_set),
        )
        return arc_numbers, False, False, None

    resumed_arcs = [arc for arc in arc_numbers if arc >= restore_arc]
    logger.info("Continuing arc loop from restored arc %d: %s", restore_arc, resumed_arcs)
    return resumed_arcs, restore_bends_opt, restore_quads_opt, restore_arc


def prepare_frequency_metadata(
    freq: str,
    times: list[str],
    beam: int,
    meas_base_dir: Path,
    results_dir: Path,
    squeeze_step: str,
) -> tuple[list[Path], list[Path], Path | None, Path | None, set[str], float | int]:
    """Prepare metadata for a frequency without processing measurements."""
    if not times:
        return [], [], None, None, set(), 0.0

    analysed_folders = get_analysis_folders(times, beam, meas_base_dir, squeeze_step)
    validate_folders_exist(analysed_folders, squeeze_step, freq)

    bad_bpms = collect_bad_bpms_from_folders(analysed_folders)
    if not bad_bpms:
        raise ValueError("No bad BPMs found, something is wrong.")

    files = get_measurement_files(times, analysed_folders, beam, squeeze_step)
    tune_knobs_file, corrector_knobs_file = get_knob_files(results_dir, squeeze_step, freq)

    # Save knobs for the earliest measurement time
    meas_time = get_measurement_time(min(times), squeeze_step)
    energy = save_online_knobs(
        meas_time,
        beam=beam,
        tune_knobs_file=tune_knobs_file,
        corrector_knobs_file=corrector_knobs_file,
    )

    return files, analysed_folders, tune_knobs_file, corrector_knobs_file, bad_bpms, energy


def process_frequency_results(
    freq: str,
    file_keys: list[str],
    pzs_dict: dict[str, pd.DataFrame],
    tune_knobs_file: Path,
    corrector_knobs_file: Path,
    temp_analysis_dir: Path,
    closed_orbit: pd.DataFrame,
    reference_deltap: float,
) -> list[dict]:
    """Process results for a single frequency from the pzs_dict."""
    pzs_list = [pzs_dict[key] for key in file_keys]
    measurements = []

    dpp_values = []
    for i, pzs in enumerate(pzs_list):
        dpp_est = pzs.attrs["DPP_EST"]
        dpp_values.append(dpp_est)
        deltap = 0.0 if freq == ZEROHZ else dpp_est - reference_deltap
        corrected_pzs = subtract_reference_closed_orbit(pzs, closed_orbit, freq)

        meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        corrected_pzs.to_parquet(meas_save_path)

        measurements.append(
            {
                "file": meas_save_path,
                "tune_knobs_file": tune_knobs_file,
                "corrector_file": corrector_knobs_file,
                "machine_deltap": deltap,
            }
        )

    # Save dpp values for this frequency
    metadata = load_metadata(temp_analysis_dir)
    if "dpp_values" not in metadata:
        metadata["dpp_values"] = {}
    metadata["dpp_values"][freq] = dpp_values
    save_metadata(temp_analysis_dir, metadata)

    return measurements


def load_frequency_results(
    freq: str,
    num_files: int,
    tune_knobs_file: Path,
    corrector_knobs_file: Path,
    temp_analysis_dir: Path,
    reference_deltap: float,
) -> list[dict]:
    """Load previously processed frequency results."""
    validate_processed_files(temp_analysis_dir, freq, num_files)
    metadata = load_metadata(temp_analysis_dir)
    dpp_values = metadata["dpp_values"][freq]

    measurements = []
    for i, dpp_est in enumerate(dpp_values):
        meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        deltap = 0.0 if freq == ZEROHZ else dpp_est - reference_deltap
        measurements.append(
            {
                "file": meas_save_path,
                "tune_knobs_file": tune_knobs_file,
                "corrector_file": corrector_knobs_file,
                "machine_deltap": deltap,
            }
        )

    logger.info(f"Loaded {num_files} processed files for {freq}")
    return measurements


def create_configs(
    beam: int,
    model_dir: Path,
    all_bad_bpms: set[str],
    arc_num: int,
    measurements: list[dict],
    energy: float,
    run_in_irs: bool,
    run_arc_by_arc: bool,
    ac_dipole_window: ACDipoleOptimisationWindow | None = None,
) -> tuple[SequenceConfig, list[str], list[str], MeasurementConfig]:
    """Create configuration objects for optimisation.

    Returns:
        Tuple of (sequence_config, bpm_start_points, bpm_end_points, measurement_config)
    """
    if ac_dipole_window is not None:
        magnet_range, bpm_start_points, bpm_end_points = get_ac_dipole_bpm_points(
            beam, ac_dipole_window
        )
    elif run_in_irs:
        magnet_range, bpm_start_points, bpm_end_points = get_ir_bpm_ranges_from_model(
            model_dir, beam, arc_num
        )
    elif run_arc_by_arc:
        magnet_range, bpm_start_points, bpm_end_points = get_bpm_points(arc_num, beam)
    else:
        magnet_range, bpm_start_points, bpm_end_points = get_full_ring_bpm_points(beam)

    sequence_config = SequenceConfig(
        magnet_range=magnet_range,
        bad_bpms=list(all_bad_bpms),
        first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
    )

    measurement_config = MeasurementConfig(
        measurement_files=[m["file"] for m in measurements],
        corrector_files=[m["corrector_file"] for m in measurements],
        tune_knobs_files=[m["tune_knobs_file"] for m in measurements],
        flattop_turns=6600,
        machine_deltaps=[m["machine_deltap"] for m in measurements],
        bunches_per_file=3,
    )

    return sequence_config, bpm_start_points, bpm_end_points, measurement_config


def get_default_simulation_config(
    run_arc_by_arc: bool = True, using_lbfgs: bool = False
) -> SimulationConfig:
    """Get default simulation configuration."""
    return SimulationConfig(
        # tracks_per_worker=824,
        tracks_per_worker=10 if run_arc_by_arc else 250,
        # tracks_per_worker=9_890,
        # tracks_per_worker=int(19700.0 / 2),
        num_batches=5 if not using_lbfgs else 1,
        num_workers=60,
        use_fixed_bpm=True,
        run_arc_by_arc=run_arc_by_arc,
        n_run_turns=1 if run_arc_by_arc else 3,
        optimise_momenta=False,  # Enable momentum optimisation (x, px, y, py) not just (x, y)
        bpm_loss_outlier_sigma=6,
    )


def save_arc_estimates(
    results_dir: Path, squeeze_step: str, arc_num: int, estimate: dict, rewrite_file: bool = False
) -> None:
    """Save arc optimisation estimates to file."""
    outfile = results_dir / f"quad_estimates_{squeeze_step}.txt"
    write_mode = "a" if not rewrite_file else "w"
    with outfile.open(write_mode) as f:
        f.write(f"Arc {arc_num}:\n")
        for magnet, value in estimate.items():
            f.write(f"{magnet}\t{value}\n")
        f.write("\n")


def optimise_arc(
    arc_num: int,
    beam: int,
    sequence_path: Path,
    model_dir: Path,
    measurements: list[dict],
    temp_analysis_dir: Path,
    results_dir: Path,
    squeeze_step: str,
    all_bad_bpms: set[str],
    energy: float,
    base_checkpoint_config: CheckpointConfig,
    rewrite_file: bool = False,
    run_in_irs: bool = False,
    run_arc_by_arc: bool = True,
    ac_dipole_window: ACDipoleOptimisationWindow | None = None,
    restore_bends_opt: bool = False,
    restore_quads_opt: bool = False,
) -> None:
    """Optimise quadrupoles for a single arc."""
    logger.info(f"Optimising arc {arc_num} for {squeeze_step}")

    sequence_config, bpm_start_points, bpm_end_points, measurement_config = create_configs(
        beam,
        model_dir,
        all_bad_bpms,
        arc_num,
        measurements,
        energy,
        run_in_irs,
        run_arc_by_arc,
        ac_dipole_window,
    )

    checkpoint_dir = base_checkpoint_config.checkpoint_path
    output_cfg = OutputConfig(
        include_uncertainty=False,
        plot_real_values=True,
        show_plots=False,
        plots_dir=temp_analysis_dir,
        mad_logfile=temp_analysis_dir / "mad_log.txt",
        python_logfile=temp_analysis_dir / "python_worker_log.txt",
    )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bend_checkpoint_cfg, quad_checkpoint_cfg = build_stage_checkpoint_configs(
        checkpoint_dir=checkpoint_dir,
        beam=beam,
        squeeze_step=squeeze_step,
        arc_num=arc_num,
        checkpoint_every_n_epochs=base_checkpoint_config.checkpoint_every_n_epochs,
        restore_bends_opt=restore_bends_opt,
        restore_quads_opt=restore_quads_opt,
    )

    bend_estimates: dict[str, float] | None = None
    if restore_quads_opt:
        logger.info(
            "Skipping bend stage for arc %d because quadrupole checkpoint restore is enabled.",
            arc_num,
        )
    else:
        bend_optimiser_config = OptimiserConfig(
            max_epochs=200,
            warmup_epochs=10,
            warmup_lr_start=1e-10,
            max_lr=1e0,
            min_lr=1e0,
            gradient_converged_value=1e-6,
            optimiser_type="lbfgs",  # 'adam' or 'lbfgs'
            expected_rel_error=1,
        )

        accelerator_bends = LHC(
            beam=beam,
            beam_energy=energy,
            sequence_file=sequence_path,
            optimise_quadrupoles=False,
            optimise_energy=False,
            optimise_bends=True,
            optimise_correctors=False,
            optimise_other_quadrupoles=False,
            optimise_quad_dx=False,
            optimise_quad_dy=True,
        )

        bend_ctrl = Controller(
            accelerator_bends,
            bend_optimiser_config,
            get_default_simulation_config(run_arc_by_arc=run_arc_by_arc, using_lbfgs=True),
            sequence_config,
            measurement_config,
            bpm_start_points,
            bpm_end_points,
            initial_knob_strengths=None,
            true_strengths=None,
            output_config=output_cfg,
            checkpoint_config=bend_checkpoint_cfg,
        )
        bend_estimates, _ = bend_ctrl.run()

    # Increase the OptimiserConfig max_epochs for quadrupole optimisation
    quadrupole_optimiser_config = OptimiserConfig(
        max_epochs=50,
        warmup_epochs=5,
        warmup_lr_start=1e-10,
        max_lr=1e-7,
        min_lr=5e-8,
        # max_lr=1,
        # min_lr=1,
        gradient_converged_value=1e-7,
        optimiser_type="adam",  # 'adam' or 'lbfgs'
        expected_rel_error=18e-4,
    )

    opt_quads = LHC(
        beam=beam,
        beam_energy=energy,
        sequence_file=sequence_path,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=True,
        optimise_correctors=False,
        optimise_other_quadrupoles=True,
        optimise_quad_dx=False,
        optimise_quad_dy=True,
    )

    ctrl = Controller(
        opt_quads,
        quadrupole_optimiser_config,
        get_default_simulation_config(run_arc_by_arc=run_arc_by_arc),
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        initial_knob_strengths=bend_estimates,
        true_strengths=None,
        output_config=output_cfg,
        checkpoint_config=quad_checkpoint_cfg,
    )
    estimates, _ = ctrl.run()
    save_arc_estimates(results_dir, squeeze_step, arc_num, estimates, rewrite_file=rewrite_file)


def load_measurements_from_reload(
    temp_analysis_dir: Path,
    results_dir: Path,
    squeeze_step: str,
    meas_times_for_step: dict[str, list[str]],
    ac_dipole_reconstruction_config: ACDipoleReconstructionConfig | None,
    ac_dipole_window: ACDipoleOptimisationWindow | None,
) -> tuple[list[dict], set[str], float, ACDipoleOptimisationWindow | None]:
    """Load already-processed measurements from temp parquet files."""
    metadata = load_metadata(temp_analysis_dir)
    energy = metadata["energy"]
    if ac_dipole_reconstruction_config is not None and ac_dipole_window is None:
        resolved_window = window_from_attrs(metadata)
        if resolved_window is None:
            raise ValueError(
                "AC-dipole window missing in metadata for --skip-reload. "
                "Run once without --skip-reload to generate it."
            )
        ac_dipole_window = resolved_window

    freq_file_counts = {freq: len(times) for freq, times in meas_times_for_step.items() if times}
    reference_deltap, closed_orbit = prepare_reload_reference_data(
        temp_analysis_dir, freq_file_counts
    )

    all_bad_bpms = find_missing_closed_orbit_bpms(
        load_saved_pzs_for_frequencies(temp_analysis_dir, freq_file_counts),
        closed_orbit,
    )
    all_measurements: list[dict] = []
    for freq, times in meas_times_for_step.items():
        if not times:
            continue
        logger.info(f"  Frequency {freq}: {len(times)} measurements (loading)")
        tune_knobs_file, corrector_knobs_file = get_knob_files(results_dir, squeeze_step, freq)
        all_measurements.extend(
            load_frequency_results(
                freq,
                len(times),
                tune_knobs_file,
                corrector_knobs_file,
                temp_analysis_dir,
                reference_deltap,
            )
        )

    return all_measurements, all_bad_bpms, energy, ac_dipole_window


def process_measurements_fresh(
    beam: int,
    squeeze_step: str,
    meas_times_for_step: dict[str, list[str]],
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    temp_analysis_dir: Path,
    sequence_path: Path,
    nattunes: list[float],
    tunes: list[float],
    ac_dipole_reconstruction_config: ACDipoleReconstructionConfig | None,
    ac_dipole_window: ACDipoleOptimisationWindow | None,
) -> tuple[list[dict], set[str], float, ACDipoleOptimisationWindow | None]:
    """Process raw measurement files and persist normalised parquet outputs."""
    freq_metadata: dict[str, tuple[list[Path], Path, Path]] = {}
    all_files: list[Path] = []
    all_bad_bpms: set[str] = set()
    energy = 0.0

    for freq, times in meas_times_for_step.items():
        if not times:
            raise ValueError(f"No measurement times found for frequency {freq}")
        logger.info(f"  Frequency {freq}: {len(times)} measurements")
        files, _, tune_knobs_file, corrector_knobs_file, bad_bpms, freq_energy = (
            prepare_frequency_metadata(freq, times, beam, meas_base_dir, results_dir, squeeze_step)
        )
        assert tune_knobs_file is not None and corrector_knobs_file is not None
        if freq == ZEROHZ:
            energy = float(freq_energy)
        freq_metadata[freq] = (files, tune_knobs_file, corrector_knobs_file)
        all_files.extend(files)
        all_bad_bpms.update(bad_bpms)

    update_metadata(temp_analysis_dir, energy=energy)

    logger.info(f"Processing {len(all_files)} total measurement files...")
    analysis_dir = get_analysis_dir(beam, squeeze_step)
    accelerator = LHC(
        beam=beam,
        beam_energy=energy,
        sequence_file=sequence_path,
    )
    pzs_dict, bad_bpms_out, _, twiss = process_measurements(
        all_files,
        temp_analysis_dir,
        model_dir,
        accelerator=accelerator,
        filename=None,
        bad_bpms=list(all_bad_bpms),
        previous_analysis_dir=analysis_dir,
        use_uniform_vars=True,
        num_workers=8,
        combine_files=False,
        nattunes=nattunes,
        tunes=tunes,
        ac_dipole_reconstruction_config=ac_dipole_reconstruction_config,
    )
    all_bad_bpms.update(bad_bpms_out)

    if ac_dipole_reconstruction_config is not None:
        first_pzs = next(iter(pzs_dict.values()))
        resolved_window = window_from_attrs(first_pzs.attrs)
        if resolved_window is None and ac_dipole_window is None:
            raise ValueError(
                "AC-dipole mode enabled but no resolved BPM window was returned by reconstruction."
            )
        if resolved_window is not None:
            ac_dipole_window = resolved_window
            update_metadata(
                temp_analysis_dir,
                ac_dipole_marker=first_pzs.attrs.get("ac_dipole_marker"),
                ac_dipole_bpm_upstream=resolved_window.bpm_upstream,
                ac_dipole_bpm_downstream=resolved_window.bpm_downstream,
                ac_dipole_n_bpms_each_side=first_pzs.attrs.get("ac_dipole_n_bpms_each_side"),
                ac_dipole_smooth_lambda=first_pzs.attrs.get("ac_dipole_smooth_lambda"),
            )

    files_0hz, _, _ = freq_metadata[ZEROHZ]
    pzs_0hz = [pzs_dict[str(f)] for f in files_0hz]
    reference_deltap = float(np.mean([pzs.attrs["DPP_EST"] for pzs in pzs_0hz]))
    save_closed_orbit_twiss_reference(temp_analysis_dir, twiss)
    closed_orbit = calculate_reference_closed_orbit(pzs_0hz, twiss=twiss)
    all_bad_bpms.update(find_missing_closed_orbit_bpms(list(pzs_dict.values()), closed_orbit))
    save_reference_closed_orbit(temp_analysis_dir, closed_orbit)
    update_metadata(
        temp_analysis_dir,
        reference_deltap=reference_deltap,
        machine_deltap=0.0,
        closed_orbit_removed=True,
        closed_orbit_reference_definition="measurement_0Hz - twiss_expected",
    )

    all_measurements: list[dict] = []
    for freq in freq_metadata:
        files_freq, tune_knobs_freq, corrector_knobs_freq = freq_metadata[freq]
        file_keys_freq = [str(f) for f in files_freq]
        all_measurements.extend(
            process_frequency_results(
                freq,
                file_keys_freq,
                pzs_dict,
                tune_knobs_freq,
                corrector_knobs_freq,
                temp_analysis_dir,
                closed_orbit,
                reference_deltap,
            )
        )

    return all_measurements, all_bad_bpms, energy, ac_dipole_window


def get_arc_numbers(
    run_arc_by_arc: bool,
    ac_dipole_window: ACDipoleOptimisationWindow | None,
) -> list[int]:
    """Return the arc iteration sequence for the selected run mode."""
    if ac_dipole_window is not None:
        return [1]
    return list(range(1, 9)) if run_arc_by_arc else [1]


def process_squeeze_step(
    beam: int,
    squeeze_step: str,
    meas_times: dict,
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    checkpoint_every_n_epochs: int,
    skip_reload: bool = False,
    cleanup_temp: bool = False,
    run_in_irs: bool = False,
    run_arc_by_arc: bool = True,
    ac_dipole_reconstruction_config: ACDipoleReconstructionConfig | None = None,
    ac_dipole_window: ACDipoleOptimisationWindow | None = None,
    restore_bends_opt: bool = False,
    restore_quads_opt: bool = False,
) -> None:
    """Process a single squeeze step for quadrupole optimisation."""
    logger.info(f"Processing squeeze step {squeeze_step} for beam {beam}")

    # Setup directories
    results_dir.mkdir(exist_ok=True)
    temp_analysis_dir = (
        PROJECT_ROOT / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
    )
    bad_bpms_file = results_dir / f"bad_bpms_{squeeze_step}.txt"
    meas_times_for_step = meas_times[squeeze_step]
    sequence_time = get_sequence_creation_time(meas_times_for_step, squeeze_step)
    setup_temp_directory(temp_analysis_dir, skip_reload)
    sequence_path = get_or_make_sequence(beam, model_dir, time=sequence_time)

    checkpoint_config = CheckpointConfig(
        checkpoint_path=temp_analysis_dir / "checkpoints",
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        restore_from_checkpoint=restore_quads_opt or restore_bends_opt,
    )

    # Extract tunes from model job file
    job_file = model_dir / "job.create_model_nominal.madx"
    nat_x, nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
    nattunes = [nat_x, nat_y, 0.0]
    tunes = [drv_x, drv_y, 0.0]
    logger.info(f"Extracted tunes for {squeeze_step}: nattunes={nattunes}, tunes={tunes}")

    # Process all frequencies
    if ZEROHZ not in meas_times[squeeze_step]:
        raise NotImplementedError(
            "Please include 0Hz measurements to build the closed-orbit reference."
        )

    meas_times_for_step = meas_times[squeeze_step]
    if skip_reload:
        all_measurements, all_bad_bpms, energy, ac_dipole_window = load_measurements_from_reload(
            temp_analysis_dir=temp_analysis_dir,
            results_dir=results_dir,
            squeeze_step=squeeze_step,
            meas_times_for_step=meas_times_for_step,
            ac_dipole_reconstruction_config=ac_dipole_reconstruction_config,
            ac_dipole_window=ac_dipole_window,
        )
    else:
        all_measurements, all_bad_bpms, energy, ac_dipole_window = process_measurements_fresh(
            beam=beam,
            squeeze_step=squeeze_step,
            meas_times_for_step=meas_times_for_step,
            meas_base_dir=meas_base_dir,
            model_dir=model_dir,
            results_dir=results_dir,
            temp_analysis_dir=temp_analysis_dir,
            sequence_path=sequence_path,
            nattunes=nattunes,
            tunes=tunes,
            ac_dipole_reconstruction_config=ac_dipole_reconstruction_config,
            ac_dipole_window=ac_dipole_window,
        )

    # Load or save bad BPMs
    all_bad_bpms = load_bad_bpms(bad_bpms_file) if skip_reload else all_bad_bpms
    save_bad_bpms(bad_bpms_file, all_bad_bpms)

    logger.info(f"Total bad BPMs: {len(all_bad_bpms)}")
    logger.info(f"Total measurements: {len(all_measurements)}")
    logger.info(f"Machine deltaps: {[m['machine_deltap'] for m in all_measurements]}")

    # Optimise each arc around the full ring
    rewrite_file = True
    arc_numbers = get_arc_numbers(run_arc_by_arc, ac_dipole_window)

    arc_numbers, restore_bends_opt, restore_quads_opt, restore_arc = resolve_restore_resume(
        arc_numbers=arc_numbers,
        checkpoint_config=checkpoint_config,
        beam=beam,
        squeeze_step=squeeze_step,
        restore_bends_opt=restore_bends_opt,
        restore_quads_opt=restore_quads_opt,
    )

    for arc_num in arc_numbers:
        optimise_arc(
            arc_num,
            beam,
            sequence_path,
            model_dir,
            all_measurements,
            temp_analysis_dir,
            results_dir,
            squeeze_step,
            all_bad_bpms,
            energy,
            rewrite_file=rewrite_file,
            run_in_irs=run_in_irs,
            run_arc_by_arc=run_arc_by_arc,
            ac_dipole_window=ac_dipole_window,
            base_checkpoint_config=checkpoint_config,
            restore_bends_opt=restore_bends_opt and restore_arc == arc_num,
            restore_quads_opt=restore_quads_opt and restore_arc == arc_num,
        )
        rewrite_file = False  # Only rewrite for the first arc

    # Cleanup
    if cleanup_temp:
        import shutil

        logger.info(f"Cleaning up temp directory: {temp_analysis_dir}")
        shutil.rmtree(temp_analysis_dir)


def main():
    """Main entry point for squeeze quadrupole optimisation."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Optimise LHC squeeze quadrupoles using measurement data"
    )
    parser.add_argument(
        "--beam", type=int, choices=[1, 2], required=True, help="Beam number (1 or 2)"
    )
    parser.add_argument(
        "--squeeze-step", type=str, required=True, help="Squeeze step (e.g., '1.2m', '0.6m')"
    )
    parser.add_argument("--skip-reload", action="store_true", help="Reuse existing processed data")
    parser.add_argument(
        "--irs", action="store_true", help="Optimise within the IRs instead of arcs"
    )
    parser.add_argument(
        "--fullring",
        action="store_true",
        help="Disable arc-by-arc worker mode and use multi-turn running",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Delete temporary analysis directory after completion",
    )
    parser.add_argument(
        "--acdipole-mode",
        action="store_true",
        help="Use AC-dipole assisted momentum reconstruction and full-turn arc-by-arc tracking",
    )
    parser.add_argument(
        "--acdipole-n-bpms-each-side",
        type=int,
        default=1,
        help="Number of BPMs per side for AC-dipole reconstruction",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=3,
        help="Save optimisation checkpoints every N epochs (0 disables checkpointing)",
    )
    parser.add_argument(
        "--restore-bends-opt",
        action="store_true",
        help="Restore bend optimisation state from bend checkpoint files",
    )
    parser.add_argument(
        "--restore-quads-opt",
        action="store_true",
        help="Restore quadrupole optimisation state from quadrupole checkpoint files",
    )
    args = parser.parse_args()

    # Validate squeeze step
    if args.squeeze_step not in MEAS_TIMES[args.beam]:
        raise ValueError(f"Unknown squeeze step '{args.squeeze_step}' for beam {args.beam}")

    if args.acdipole_mode and args.fullring:
        raise ValueError("--acdipole-mode requires arc-by-arc worker mode; do not pass --fullring")

    if args.checkpoint_every_epochs < 0:
        raise ValueError("--checkpoint-every-epochs must be >= 0")

    if args.restore_bends_opt and args.restore_quads_opt:
        raise ValueError(
            "Choose only one restore option: --restore-bends-opt or --restore-quads-opt"
        )

    if (args.restore_bends_opt or args.restore_quads_opt) and args.checkpoint_every_epochs <= 0:
        logger.info(
            "Restoring from checkpoint with checkpoint saving disabled for this run "
            "(--checkpoint-every-epochs=0)."
        )

    ac_dipole_window = None
    ac_dipole_reconstruction_config = None
    if args.acdipole_mode:
        ac_dipole_reconstruction_config = ACDipoleReconstructionConfig(
            ac_dipole_marker=AC_MARKER_PATTERN.format(beam=args.beam),
            beam_energy=6800.0,
            n_bpms_each_side=args.acdipole_n_bpms_each_side,
        )

    # Get paths
    from aba_optimiser.measurements.squeeze_helpers import get_results_dir

    meas_base_dir, model_base_dir = get_beam_paths(args.beam, args.squeeze_step)
    model_dir = model_base_dir / MODEL_DIRS[args.beam][args.squeeze_step]
    results_dir = get_results_dir(args.beam)

    process_squeeze_step(
        args.beam,
        args.squeeze_step,
        MEAS_TIMES[args.beam],
        meas_base_dir,
        model_dir,
        results_dir,
        args.checkpoint_every_epochs,
        args.skip_reload,
        args.cleanup_temp,
        args.irs,
        (not args.fullring) or args.acdipole_mode,
        ac_dipole_reconstruction_config,
        ac_dipole_window,
        args.restore_bends_opt,
        args.restore_quads_opt,
    )


if __name__ == "__main__":
    main()
