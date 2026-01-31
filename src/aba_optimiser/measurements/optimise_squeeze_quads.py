from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd

from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.measurements.create_datafile import process_measurements, save_online_knobs
from aba_optimiser.measurements.squeeze_helpers import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
    get_or_make_sequence,
)
from aba_optimiser.measurements.utils import find_all_bad_bpms
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

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
            PLUS_50HZ: ["16_58_06_419", "16_59_12_366", "17_00_27_504"],
            PLUS_100HZ: ["17_03_16_394", "17_04_25_452", "17_05_53_444"],
            PLUS_150HZ: ["17_10_45_523", "17_11_52_941", "17_12_57_191"],
            PLUS_200HZ: ["17_20_55_315", "17_22_05_108", "17_23_09_370"],
            PLUS_250HZ: ["17_26_28_349", "17_27_50_402", "17_29_16_374"],
            PLUS_300HZ: ["17_33_14_338", "17_34_20_356", "17_35_28_317"],
            PLUS_350HZ: ["17_40_25_475", "17_41_36_368", "17_42_54_434"],
            MINUS_50HZ: ["17_51_38_387", "17_52_44_501", "17_53_50_407"],
            MINUS_100HZ: ["17_57_13_340", "17_58_24_385", "17_59_32_491"],
            MINUS_150HZ: ["18_02_51_331", "18_03_57_483", "18_05_05_332"],
            MINUS_200HZ: ["18_06_53_392", "18_07_59_340", "18_09_04_349"],
            MINUS_250HZ: ["18_11_32_374", "18_12_51_327", "18_14_23_472"],
            MINUS_300HZ: ["18_16_36_372", "18_17_44_322", "18_18_52_182"],
            MINUS_350HZ: ["18_20_38_431", "18_21_46_310", "18_22_51_320"],
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
            PLUS_50HZ: ["16_58_39_378", "16_59_45_316", "17_00_51_406"],
            PLUS_100HZ: ["17_03_42_327", "17_04_51_372", "17_06_00_423"],
            PLUS_150HZ: ["17_09_53_327", "17_10_58_495", "17_12_09_324"],
            PLUS_200HZ: ["17_23_20_460", "17_19_54_443", "17_21_03_313"],
            PLUS_250HZ: ["17_28_11_396", "17_29_22_415", "17_30_40_479"],
            PLUS_300HZ: ["17_34_11_504", "17_35_21_234", "17_36_28_939"],
            PLUS_350HZ: ["17_40_14_958", "17_41_28_883", "17_42_46_833"],
            MINUS_50HZ: ["17_51_17_922", "17_52_22_657", "17_53_28_125"],
            MINUS_100HZ: ["17_57_57_917", "17_59_08_915", "18_00_26_732"],
            MINUS_150HZ: ["18_03_31_335", "18_04_37_370", "18_05_49_463"],
            MINUS_200HZ: ["18_07_24_413", "18_08_30_429", "18_09_36_407"],
            MINUS_250HZ: ["18_11_42_328", "18_13_31_372", "18_14_45_489"],
            MINUS_300HZ: ["18_16_48_300", "18_17_58_399", "18_19_04_402"],
            MINUS_350HZ: ["18_20_15_411", "18_21_21_418", "18_22_26_431"],
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


def get_bpm_points(arc_num: int, beam: int) -> tuple[str, list[str], list[str], str]:
    """Get magnet range and BPM points for an arc."""
    next_arc = arc_num % 8 + 1
    suffix = f".B{beam}"
    start_bpm = f"BPM.13R{arc_num}{suffix}"
    # end_bpm = f"BPM.15R{arc_num}{suffix}" # Testing with BPM.15R
    end_bpm = f"BPM.12L{next_arc}{suffix}"
    magnet_range = f"{start_bpm}/{end_bpm}"
    first_i = 11
    last_i = 11
    bpm_start_points = [f"BPM.{i}R{arc_num}{suffix}" for i in range(first_i, 14, 1)]
    bpm_end_points = [f"BPM.{i}L{next_arc}{suffix}" for i in range(last_i, 14, 1)]
    bpm_range = f"BPM.{first_i}R{arc_num}{suffix}/BPM.{last_i}L{next_arc}{suffix}"

    # Testing these fixed points
    # bpm_start_points = [start_bpm]
    # bpm_end_points = [end_bpm]

    return magnet_range, bpm_start_points, bpm_end_points, bpm_range


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


def save_dpp_metadata(temp_analysis_dir: Path, freq: str, dpp_values: list[float]) -> None:
    """Save dpp_est values to metadata file."""
    metadata_file = temp_analysis_dir / f"dpp_metadata_{freq}.json"
    with metadata_file.open("w") as f:
        json.dump({"dpp_est_values": dpp_values}, f)


def load_dpp_metadata(temp_analysis_dir: Path, freq: str) -> list[float]:
    """Load dpp_est values from metadata file."""
    metadata_file = temp_analysis_dir / f"dpp_metadata_{freq}.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"DPP metadata file {metadata_file} not found. Run without --skip-reload to regenerate."
        )
    with metadata_file.open("r") as f:
        data = json.load(f)
    return data["dpp_est_values"]


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


def prepare_frequency_metadata(
    freq: str,
    times: list[str],
    beam: int,
    meas_base_dir: Path,
    results_dir: Path,
    squeeze_step: str,
) -> tuple[list[Path], list[Path], Path | None, set[str], float | int]:
    """Prepare metadata for a frequency without processing measurements."""
    if not times:
        return [], [], None, set(), 0.0

    analysed_folders = get_analysis_folders(times, beam, meas_base_dir, squeeze_step)
    validate_folders_exist(analysed_folders, squeeze_step, freq)

    bad_bpms = collect_bad_bpms_from_folders(analysed_folders)
    if not bad_bpms:
        raise ValueError("No bad BPMs found, something is wrong.")

    files = get_measurement_files(times, analysed_folders, beam, squeeze_step)
    tune_knobs_file = results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt"

    # Save knobs for the earliest measurement time
    meas_time = get_measurement_time(min(times), squeeze_step)
    corrector_knobs_file = results_dir / "null.txt"
    energy = save_online_knobs(
        meas_time,
        beam=beam,
        tune_knobs_file=tune_knobs_file,
        corrector_knobs_file=corrector_knobs_file,
    )

    return files, analysed_folders, tune_knobs_file, bad_bpms, energy


def process_frequency_results(
    freq: str,
    file_keys: list[str],
    pzs_dict: dict[str, pd.DataFrame],
    tune_knobs_file: Path,
    temp_analysis_dir: Path,
    sequence_path: Path,
    beam: int,
    machine_deltap: float | None,
    energy: float,
) -> tuple[list[dict], float | None]:
    """Process results for a single frequency from the pzs_dict."""
    pzs_list = [pzs_dict[key] for key in file_keys]
    measurements = []

    # Calculate machine deltap from 0hz measurements
    all_0hz_pzs = pd.concat(pzs_list, ignore_index=True)
    means = all_0hz_pzs.groupby("name", observed=False, sort=False)[["x", "px", "y", "py"]].mean()
    if freq == ZEROHZ and machine_deltap is None:
        # Order is preserved: concat maintains order within each dataframe, ignore_index resets index but preserves row order
        machine_deltap = sum(pz.attrs["DPP_EST"] for pz in pzs_list) / len(pzs_list)
        if machine_deltap == 0.0:
            logger.warning("No DPP_EST found in processed data headers, defaulting to 0.0")

    dpp_values = []
    for i, pzs in enumerate(pzs_list):
        dpp_est = pzs.attrs["DPP_EST"]
        dpp_values.append(dpp_est)
        deltap = 0.0 if freq == ZEROHZ else dpp_est - (machine_deltap or 0.0)

        mad_iface = OptimisationMadInterface(
            sequence_file=sequence_path,
            seq_name=f"lhcb{beam}",
            beam_energy=energy,
            bpm_pattern="BPM",
        )
        tws = mad_iface.run_twiss(deltap=deltap, observe=1)
        closed_orbit_means = means - tws.loc[means.index, ["x", "px", "y", "py"]].values

        # Subtract closed orbit to focus optimisation on betatron oscillations
        # Order is preserved: merge with sort=False maintains order from left DataFrame (pzs)
        pzs = pzs.merge(closed_orbit_means, on="name", suffixes=("", "_mean"), sort=False)
        pzs[["x", "px", "y", "py"]] = pzs[["x", "px", "y", "py"]].sub(
            pzs[["x_mean", "px_mean", "y_mean", "py_mean"]].values
        )
        pzs = pzs.drop(columns=["x_mean", "px_mean", "y_mean", "py_mean"])
        pzs.attrs = pzs_list[i].attrs

        meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        pzs.to_parquet(meas_save_path)

        measurements.append(
            {
                "file": meas_save_path,
                "tune_knobs_file": tune_knobs_file,
                "machine_deltap": deltap,
            }
        )

    save_dpp_metadata(temp_analysis_dir, freq, dpp_values)
    return measurements, machine_deltap


def load_frequency_results(
    freq: str,
    num_files: int,
    tune_knobs_file: Path,
    temp_analysis_dir: Path,
    machine_deltap: float | None,
) -> tuple[list[dict], float | None]:
    """Load previously processed frequency results."""
    validate_processed_files(temp_analysis_dir, freq, num_files)
    dpp_values = load_dpp_metadata(temp_analysis_dir, freq)

    if freq == ZEROHZ and machine_deltap is None:
        machine_deltap = dpp_values[0]

    measurements = []
    for i, dpp_est in enumerate(dpp_values):
        meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        deltap = 0.0 if freq == ZEROHZ else dpp_est - (machine_deltap or 0.0)
        measurements.append(
            {
                "file": meas_save_path,
                "tune_knobs_file": tune_knobs_file,
                "machine_deltap": deltap,
            }
        )

    logger.info(f"Loaded {num_files} processed files for {freq}")
    return measurements, machine_deltap


def create_configs(
    beam: int,
    sequence_path: Path,
    all_bad_bpms: set[str],
    arc_num: int,
    measurements: list[dict],
    energy: float,
) -> tuple[SequenceConfig, BPMConfig, MeasurementConfig]:
    """Create configuration objects for optimisation."""
    magnet_range, bpm_start_points, bpm_end_points, bpm_range = get_bpm_points(arc_num, beam)

    sequence_config = SequenceConfig(
        sequence_file_path=sequence_path,
        magnet_range=magnet_range,
        bpm_range=bpm_range,
        beam_energy=energy,
        seq_name=f"lhcb{beam}",
        bad_bpms=list(all_bad_bpms),
        first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
    )

    bpm_config = BPMConfig(start_points=bpm_start_points, end_points=bpm_end_points)

    measurement_config = MeasurementConfig(
        measurement_files=[m["file"] for m in measurements],
        corrector_files=None,
        tune_knobs_files=[m["tune_knobs_file"] for m in measurements],
        flattop_turns=6600,
        machine_deltaps=[m["machine_deltap"] for m in measurements],
        bunches_per_file=3,
    )

    return sequence_config, bpm_config, measurement_config


def get_default_optimiser_config() -> OptimiserConfig:
    """Get default optimiser configuration."""
    return OptimiserConfig(
        max_epochs=30,
        warmup_epochs=3,
        warmup_lr_start=1e-7,
        max_lr=1e-5,
        min_lr=1e-5,
        # max_lr=1,
        # min_lr=1,
        gradient_converged_value=1e-8,
        optimiser_type="adam",  # 'adam' or 'lbfgs'
        expected_rel_error=1e-2,
    )


def get_default_simulation_config() -> SimulationConfig:
    """Get default simulation configuration."""
    return SimulationConfig(
        tracks_per_worker=9_890,
        # tracks_per_worker=int(19700.0 / 2),
        num_batches=200,
        num_workers=60,
        use_fixed_bpm=False,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
        optimise_momenta=True,  # Enable momentum optimisation (x, px, y, py) not just (x, y)
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
    measurements: list[dict],
    temp_analysis_dir: Path,
    results_dir: Path,
    squeeze_step: str,
    all_bad_bpms: set[str],
    energy: float,
    rewrite_file: bool = False,
) -> None:
    """Optimise quadrupoles for a single arc."""
    logger.info(f"Optimising arc {arc_num} for {squeeze_step}")

    sequence_config, bpm_config, measurement_config = create_configs(
        beam, sequence_path, all_bad_bpms, arc_num, measurements, energy
    )

    final_knobs_arc = None  # Start from sequence values or previous arc results

    ctrl = Controller(
        get_default_optimiser_config(),
        get_default_simulation_config(),
        sequence_config,
        measurement_config,
        bpm_config,
        show_plots=False,
        initial_knob_strengths=final_knobs_arc,
        true_strengths=None,
        plots_dir=temp_analysis_dir,
        mad_logfile=temp_analysis_dir / "mad_log.txt",
    )
    estimate, _ = ctrl.run()
    save_arc_estimates(results_dir, squeeze_step, arc_num, estimate, rewrite_file=rewrite_file)


def process_squeeze_step(
    beam: int,
    squeeze_step: str,
    meas_times: dict,
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    skip_reload: bool = False,
    cleanup_temp: bool = False,
) -> None:
    """Process a single squeeze step for quadrupole optimisation."""
    logger.info(f"Processing squeeze step {squeeze_step} for beam {beam}")

    # Setup directories
    results_dir.mkdir(exist_ok=True)
    temp_analysis_dir = (
        PROJECT_ROOT / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
    )
    bad_bpms_file = results_dir / f"bad_bpms_{squeeze_step}.txt"
    setup_temp_directory(temp_analysis_dir, skip_reload)
    sequence_path = get_or_make_sequence(beam, model_dir)

    # Process all frequencies
    all_measurements = []
    all_bad_bpms = set()
    machine_deltap = None

    if skip_reload:
        # Load previously processed data
        for freq, times in meas_times[squeeze_step].items():
            if not times:
                continue
            logger.info(f"  Frequency {freq}: {len(times)} measurements (loading)")
            tune_knobs_file = results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt"
            freq_measurements, machine_deltap = load_frequency_results(
                freq, len(times), tune_knobs_file, temp_analysis_dir, machine_deltap
            )
            all_measurements.extend(freq_measurements)
    else:
        # Prepare metadata for all frequencies
        freq_metadata = {}
        all_files = []
        file_to_freq = {}  # Map file path to frequency

        for freq, times in meas_times[squeeze_step].items():
            if not times:
                continue
            logger.info(f"  Frequency {freq}: {len(times)} measurements")
            files, folders, tune_knobs_file, bad_bpms, energy = prepare_frequency_metadata(
                freq, times, beam, meas_base_dir, results_dir, squeeze_step
            )
            freq_metadata[freq] = (files, tune_knobs_file)
            all_files.extend(files)
            for f in files:
                file_to_freq[str(f)] = freq
            all_bad_bpms.update(bad_bpms)

        # Single call to process_measurements for all files
        logger.info(f"Processing {len(all_files)} total measurement files...")
        analysis_dir = get_analysis_dir(beam, squeeze_step)
        pzs_dict, bad_bpms_out, _, _ = process_measurements(
            all_files,
            temp_analysis_dir,
            model_dir,
            beam=beam,
            filename=None,
            bad_bpms=list(all_bad_bpms),
            previous_analysis_dir=analysis_dir,
            sequence_path=sequence_path,
            use_uniform_vars=True,
            num_workers=8,
            combine_files=False,
        )
        all_bad_bpms.update(bad_bpms_out)

        # Process results for each frequency
        machine_deltap = None

        # Process 0hz first to get closed orbit and machine deltap
        if ZEROHZ in freq_metadata:
            files_0hz, tune_knobs_0hz = freq_metadata[ZEROHZ]
            file_keys_0hz = [str(f) for f in files_0hz]
            freq_measurements, machine_deltap = process_frequency_results(
                ZEROHZ,
                file_keys_0hz,
                pzs_dict,
                tune_knobs_0hz,
                temp_analysis_dir,
                sequence_path,
                beam,
                None,
                energy,
            )
            all_measurements.extend(freq_measurements)

        # Process other frequencies
        for freq in freq_metadata:
            if freq == ZEROHZ:
                continue
            files_freq, tune_knobs_freq = freq_metadata[freq]
            file_keys_freq = [str(f) for f in files_freq]
            freq_measurements, _ = process_frequency_results(
                freq,
                file_keys_freq,
                pzs_dict,
                tune_knobs_freq,
                temp_analysis_dir,
                sequence_path,
                beam,
                machine_deltap,
                energy,
            )
            all_measurements.extend(freq_measurements)

    # Load or save bad BPMs
    all_bad_bpms = load_bad_bpms(bad_bpms_file) if skip_reload else all_bad_bpms
    if not skip_reload:
        save_bad_bpms(bad_bpms_file, all_bad_bpms)

    logger.info(f"Total bad BPMs: {len(all_bad_bpms)}")
    logger.info(f"Total measurements: {len(all_measurements)}")
    logger.info(f"Machine deltaps: {[m['machine_deltap'] for m in all_measurements]}")

    # Optimise each arc
    rewrite_file = True
    for arc_num in [3, 6]:  # Just try arc34 for now as there are no orbit bumps and arc 67
        optimise_arc(
            arc_num,
            beam,
            sequence_path,
            all_measurements,
            temp_analysis_dir,
            results_dir,
            squeeze_step,
            all_bad_bpms,
            energy,
            rewrite_file=rewrite_file,
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
        "--cleanup-temp",
        action="store_true",
        help="Delete temporary analysis directory after completion",
    )
    args = parser.parse_args()

    # Validate squeeze step
    if args.squeeze_step not in MEAS_TIMES[args.beam]:
        raise ValueError(f"Unknown squeeze step '{args.squeeze_step}' for beam {args.beam}")

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
        args.skip_reload,
        args.cleanup_temp,
    )


if __name__ == "__main__":
    main()
