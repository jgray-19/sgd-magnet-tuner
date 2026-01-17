from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.mad import OptimisationMadInterface
from aba_optimiser.measurements.create_datafile import process_measurements, save_online_knobs
from aba_optimiser.measurements.utils import find_all_bad_bpms
from aba_optimiser.model_creator.madx_utils import make_madx_sequence
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
MEASUREMENT_DATE = "2025_04_27"
FILL_NUMBER = 10533
BETABEAT_DIR = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-27/")
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

BEAM_ENERGY = 6800.0  # GeV

# ==================== MEASUREMENT TIMES ====================
MEAS_TIMES = {
    1: {
        "1.2m": {
            "0hz": ["06_17_47_405", "06_19_27_443", "06_20_39_422"],
            # "0hz": ["06_20_39_422"],
            # "+50hz": ["06_24_41_350"],
            # "-50hz": ["06_26_09_426"],
        },
        # Please also check the 1.2m after a global correction
        "1.2m_agc": {
            "0hz": ["07_59_50_367", "07_59_50_367", "07_59_50_367"],
            "+50Hz": ["08_05_57_495"],
            "-50Hz": ["08_07_03_451"],
        },
        "1.05m": {
            "0hz": ["09_19_42_460", "09_20_58_100", "09_23_22_506"],
            "+50hz": ["09_26_23_443", "09_27_41_385"],
            "-50hz": ["09_29_36_402", "09_30_46_492"],
        },
        "0.93m": {
            "0hz": ["10_56_44_423", "10_57_53_330"],
            "-50hz": ["11_00_38_320"],
            "+50hz": ["11_02_35_390"],
        },
        "0.725m": {
            "0hz": ["12_56_35_478", "12_58_04_387", "12_59_41_358"],
            "-50hz": ["13_01_45_317"],
            "+50hz": ["13_03_09_448"],
        },
        "0.6m": {
            "0hz": ["13_50_14_431", "13_51_22_434", "13_52_31_321"],
            "-50hz": ["13_54_02_456"],
            "+50hz": ["13_55_11_500"],
        },
        "0.45m": {
            "0hz": ["14_48_46_464", "14_49_53_484"],
            "-50hz": ["14_51_35_448"],
            "+50hz": ["14_52_41_380"],
        },
        "0.3m": {
            "0hz": ["15_35_51_360", "15_36_59_413"],
            "-50hz": ["15_38_30_335"],
            "+50hz": ["15_39_41_491"],
        },
        "0.25m": {
            "0hz": ["16_22_59_435", "16_24_06_387"],
            "-50hz": ["16_25_33_427"],
            "+50hz": ["16_27_05_421"],
        },
        "0.18m": {
            "0hz": ["17_34_50_472", "17_36_06_419", "17_39_46_502", "17_41_31_474", "17_42_37_338"],
            "-50hz": ["17_44_16_396"],
            "+50hz": ["17_45_33_318"],
        },
    },
    2: {
        "1.2m": {
            "0hz": ["06_18_14_332", "06_19_48_490", "06_20_57_500"],
            "+50hz": ["06_25_20_342"],
            "-50hz": ["06_26_47_456"],
        },
        "1.05m": {
            "0hz": ["09_19_49_333", "09_21_08_376", "09_23_29_348"],
            "+50hz": ["09_26_31_404", "09_28_03_384"],
            "-50hz": ["09_29_55_397", "09_31_07_320"],
        },
        "0.93m": {
            "0hz": ["10_57_01_409", "10_58_15_318"],
            "-50hz": ["11_01_01_305"],
            "+50hz": ["11_02_56_432"],
        },
        "0.725m": {
            "0hz": ["12_58_53_397", "13_00_13_435"],
            "-50hz": ["13_02_01_455"],
            "+50hz": ["13_03_26_368"],
        },
        "0.6m": {
            "0hz": ["13_50_31_386", "13_51_38_386", "13_52_47_410"],
            "-50hz": ["13_54_20_403"],
            "+50hz": ["13_55_28_330"],
        },
        "0.45m": {
            "0hz": ["14_49_16_409", "14_50_23_338"],
            "-50hz": ["14_51_51_461"],
            "+50hz": ["14_52_58_336"],
        },
        "0.3m": {
            "0hz": ["15_36_09_480", "15_37_15_426"],
            "-50hz": ["15_38_45_453"],
            "+50hz": ["15_39_57_354"],
        },
        "0.25m": {
            "0hz": ["16_23_14_482", "16_24_20_310"],
            "-50hz": ["16_25_49_324"],
            "+50hz": ["16_27_20_308"],
        },
        "0.18m": {
            "0hz": ["17_35_05_355", "17_41_46_322", "17_42_55_359"],
            "-50hz": ["17_44_33_344"],
            "+50hz": ["17_45_48_406"],
        },
    },
}

# ==================== MODEL DIRECTORIES ====================
MODEL_DIRS = {
    1: {
        "1.2m": "b1_120cm_injTunes",
        "1.05m": "b1_105cm_injTunes",
        "0.93m": "b2_93cm_injTunes",  # Double checked - this is correct (they accidentally wrote b2 in the folder name)
        "0.725m": "b1_72cm_injTunes",
        "0.6m": "b1_60cm_injTunes",
        "0.45m": "b1_44cm_flat_injTunes",
        "0.3m": "b1_30cm_flat_injTunes",
        "0.25m": "b1_24cm_flat_injTunes",
        "0.18m": "b1_18cm_flat_injTunes",
    },
    2: {
        "1.2m": "b2_120cm_injTunes",
        "1.05m": "OMC3_LHCB2_105cm",
        "0.93m": "b2_93cm_injTunes",
        "0.725m": "b2_72cm_injTunes",
        "0.6m": "b2_60cm_injTunes",
        "0.45m": "b2_44cm_flat_injTunes",
        "0.3m": "b2_30cm_flat_injTunes",
        "0.25m": "b2_24cm_flat_injTunes",
        "0.18m": "b2_18cm_flat_injTunes",
    },
}


ANALYSIS_DIRS = {
    1: {
        "1.2m": "2025-04-27_B1_120cm_injTunes_onOffMom",
        # Add other squeeze steps as needed
    },
    2: {
        "1.2m": "2025-04-27_B2_120cm_injTunes_onOffMom",
        # Add other squeeze steps as needed
    },
}


# ==================== HELPER FUNCTIONS ====================
def get_beam_paths(beam: int) -> tuple[Path, Path]:
    """Get measurement and model base directories for a beam."""
    meas_dir = BETABEAT_DIR / f"LHCB{beam}/Measurements/"
    model_dir = BETABEAT_DIR / f"LHCB{beam}/Models/"
    return meas_dir, model_dir


def get_bpm_points(arc_num: int, beam: int) -> tuple[str, list[str], list[str]]:
    """Get magnet range and BPM points for an arc."""
    next_arc = arc_num % 8 + 1
    suffix = f".B{beam}"
    start_bpm = f"BPM.13R{arc_num}{suffix}"
    # end_bpm = f"BPM.15R{arc_num}{suffix}" # Testing with BPM.15R
    end_bpm = f"BPM.12L{next_arc}{suffix}"
    magnet_range = f"{start_bpm}/{end_bpm}"
    bpm_start_points = [f"BPM.{i}R{arc_num}{suffix}" for i in range(13, 17, 1)]
    bpm_end_points = [f"BPM.{i}L{next_arc}{suffix}" for i in range(12, 16, 1)]

    # Testing these fixed points
    # bpm_start_points = [start_bpm]
    # bpm_end_points = [end_bpm]

    return magnet_range, bpm_start_points, bpm_end_points


def get_or_make_sequence(beam: int, model_dir: Path) -> Path:
    """Get cached sequence or generate new one."""
    sequences_dir = PROJECT_ROOT / "sequences_from_models"
    sequences_dir.mkdir(exist_ok=True)
    seq_name = f"{model_dir.name}.seq"
    seq_path = sequences_dir / seq_name
    if seq_path.exists():
        logging.info(f"Using cached sequence: {seq_path}")
        return seq_path

    logging.info(f"Generating new sequence: {seq_path}")
    make_madx_sequence(beam, model_dir, seq_outdir=sequences_dir)
    generated = sequences_dir / f"lhcb{beam}_saved.seq"
    generated.rename(seq_path)
    return seq_path


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


def get_measurement_time(earliest_time: str) -> datetime:
    """Convert timestamp string to datetime object."""
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{MEASUREMENT_DATE.replace('_', '-')} {time_str}"
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


def get_analysis_folders(times: list[str], beam: int, meas_base_dir: Path) -> list[Path]:
    """Get analysis folder paths for measurement times."""
    name_prefix = f"Beam{beam}@BunchTurn@{MEASUREMENT_DATE}@"
    return [meas_base_dir / f"{name_prefix}{time}" for time in times]


def get_measurement_files(times: list[str], analysed_folders: list[Path], beam: int) -> list[Path]:
    """Get measurement file paths."""
    name_prefix = f"Beam{beam}@BunchTurn@{MEASUREMENT_DATE}@"
    return [analysed_folders[i] / f"{name_prefix}{times[i]}.sdds" for i in range(len(times))]


def process_frequency_data(
    freq: str,
    times: list[str],
    beam: int,
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    temp_analysis_dir: Path,
    squeeze_step: str,
    sequence_path: Path,
    skip_reload: bool,
    closed_orbit_means: pd.DataFrame | None,
    machine_deltap: float | None,
) -> tuple[list[dict], set[str], pd.DataFrame | None, float | None]:
    """Process measurement data for a single frequency."""
    if not times:
        logger.warning(f"No times for {squeeze_step} at frequency {freq}, skipping.")
        return [], set(), closed_orbit_means, machine_deltap

    analysed_folders = get_analysis_folders(times, beam, meas_base_dir)
    validate_folders_exist(analysed_folders, squeeze_step, freq)

    bad_bpms = collect_bad_bpms_from_folders(analysed_folders)
    if not bad_bpms:
        raise ValueError("No bad BPMs found, something is wrong.")
    logger.info(f"Found {len(bad_bpms)} bad BPMs for squeeze step {squeeze_step}")

    meas_time = get_measurement_time(min(times))
    measurements = []

    if skip_reload:
        validate_processed_files(temp_analysis_dir, freq, len(times))
        tune_knobs_file = results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt"
        dpp_values = load_dpp_metadata(temp_analysis_dir, freq)

        if freq == "0hz" and machine_deltap is None:
            machine_deltap = dpp_values[0]

        for i, dpp_est in enumerate(dpp_values):
            meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
            deltap = 0.0 if freq == "0hz" else dpp_est - (machine_deltap or 0.0)
            measurements.append(
                {
                    "file": meas_save_path,
                    "tune_knobs_file": tune_knobs_file,
                    "machine_deltap": deltap,
                }
            )

        logger.info(f"Loaded {len(times)} processed files for {freq}")
    else:
        tune_knobs_file = results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt"
        corrector_knobs_file = results_dir / "null.txt"
        save_online_knobs(
            meas_time,
            beam=beam,
            tune_knobs_file=tune_knobs_file,
            corrector_knobs_file=corrector_knobs_file,
            energy=BEAM_ENERGY,
        )
        files = get_measurement_files(times, analysed_folders, beam)

        pzs_list, bad_bpms_out, _ = process_measurements(
            files,
            temp_analysis_dir,
            model_dir,
            beam=beam,
            filename=None,
            bad_bpms=list(bad_bpms),
            sequence_path=sequence_path,
            use_uniform_vars=True,
            num_workers=8,
            combine_files=False,
        )

        if freq == "0hz":
            all_0hz_pzs = pd.concat(pzs_list, ignore_index=True)
            machine_deltap = sum(pz.attrs["DPP_EST"] for pz in pzs_list) / len(pzs_list)
            if machine_deltap == 0.0:
                logger.warning("No DPP_EST found in processed data headers, defaulting to 0.0")

            # Subtract closed orbit mean to focus on betatron oscillations
            # This removes orbit effects (which dipoles/correctors handle)
            # and lets us optimize the shape of phase space ellipses (quadrupole strengths)
            closed_orbit_means = all_0hz_pzs.groupby(all_0hz_pzs["name"], observed=False)[
                ["x", "px", "y", "py"]
            ].mean()

            # Take the closed orbit from a twiss from the sequence file
            # The twiss in model dir does not contain the px and py offsets
            # This is so that when we subtract the bend + corrector contribution,
            # we get the closed orbit in the absence of these elements
            mad_iface = OptimisationMadInterface(
                sequence_file=sequence_path,
                seq_name=f"lhcb{beam}",
                corrector_strengths=None,
                tune_knobs_file=None,
            )
            twiss_df = mad_iface.run_twiss()
            twiss_df = twiss_df.loc[closed_orbit_means.index]
            closed_orbit_means[["x", "px", "y", "py"]] -= twiss_df[["x", "px", "y", "py"]].values

        dpp_values = []
        for i, pzs in enumerate(pzs_list):
            # Subtract closed orbit to focus optimisation on betatron oscillations
            pzs = pzs.merge(closed_orbit_means, on="name", suffixes=("", "_mean"))
            if freq == "0hz":
                pzs[["x", "px", "y", "py"]] = pzs[["x", "px", "y", "py"]].sub(
                    pzs[["x_mean", "px_mean", "y_mean", "py_mean"]].values
                )
            else:
                pzs[["x", "y"]] = pzs[["x", "y"]].sub(pzs[["x_mean", "y_mean"]].values)
            pzs = pzs.drop(columns=["x_mean", "px_mean", "y_mean", "py_mean"])
            pzs.attrs = pzs_list[i].attrs

            meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
            pzs.to_parquet(meas_save_path)

            dpp_est = pzs.attrs["DPP_EST"]
            dpp_values.append(dpp_est)
            measurements.append(
                {
                    "file": meas_save_path,
                    "tune_knobs_file": tune_knobs_file,
                    "machine_deltap": dpp_est - machine_deltap,
                }
            )

        save_dpp_metadata(temp_analysis_dir, freq, dpp_values)
        bad_bpms.update(bad_bpms_out)

    return measurements, set(bad_bpms), closed_orbit_means, machine_deltap


def create_configs(
    beam: int,
    sequence_path: Path,
    all_bad_bpms: set[str],
    arc_num: int,
    measurements: list[dict],
) -> tuple[SequenceConfig, BPMConfig, MeasurementConfig]:
    """Create configuration objects for optimisation."""
    magnet_range, bpm_start_points, bpm_end_points = get_bpm_points(arc_num, beam)

    sequence_config = SequenceConfig(
        sequence_file_path=sequence_path,
        magnet_range=magnet_range,
        beam_energy=BEAM_ENERGY,
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
        max_epochs=5,
        warmup_epochs=3,
        warmup_lr_start=1e-4,
        max_lr=1e-5,
        min_lr=1e-5,
        # max_lr=1,
        # min_lr=1,
        gradient_converged_value=1e-9,
        optimiser_type="adam",  # 'adam' or 'lbfgs'
    )


def get_default_simulation_config() -> SimulationConfig:
    """Get default simulation configuration."""
    return SimulationConfig(
        tracks_per_worker=int(19700 / 3),
        num_batches=100,
        num_workers=60,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
        optimise_momenta=True,  # Enable momentum optimisation (x, px, y, py) not just (x, y)
    )


def save_arc_estimates(results_dir: Path, squeeze_step: str, arc_num: int, estimate: dict) -> None:
    """Save arc optimisation estimates to file."""
    outfile = results_dir / f"quad_estimates_{squeeze_step}.txt"
    write_mode = "a" if arc_num > 1 else "w"
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
) -> None:
    """Optimise quadrupoles for a single arc."""
    logger.info(f"Optimising arc {arc_num} for {squeeze_step}")

    sequence_config, bpm_config, measurement_config = create_configs(
        beam, sequence_path, all_bad_bpms, arc_num, measurements
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
    )
    estimate, _ = ctrl.run()
    save_arc_estimates(results_dir, squeeze_step, arc_num, estimate)


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
    closed_orbit_means = None
    machine_deltap = None

    for freq, times in meas_times[squeeze_step].items():
        logger.info(f"  Frequency {freq}: {len(times)} measurements")
        freq_measurements, freq_bad_bpms, closed_orbit_means, machine_deltap = (
            process_frequency_data(
                freq,
                times,
                beam,
                meas_base_dir,
                model_dir,
                results_dir,
                temp_analysis_dir,
                squeeze_step,
                sequence_path,
                skip_reload,
                closed_orbit_means,
                machine_deltap,
            )
        )
        all_measurements.extend(freq_measurements)
        all_bad_bpms.update(freq_bad_bpms)

    # Load or save bad BPMs
    all_bad_bpms = load_bad_bpms(bad_bpms_file) if skip_reload else all_bad_bpms
    if not skip_reload:
        save_bad_bpms(bad_bpms_file, all_bad_bpms)

    logger.info(f"Total bad BPMs: {len(all_bad_bpms)}")
    logger.info(f"Total measurements: {len(all_measurements)}")
    logger.info(f"Machine deltaps: {[m['machine_deltap'] for m in all_measurements]}")

    # Optimise each arc
    for arc_num in range(1, 9):
        optimise_arc(
            arc_num,
            beam,
            sequence_path,
            all_measurements,
            temp_analysis_dir,
            results_dir,
            squeeze_step,
            all_bad_bpms,
        )

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
    meas_base_dir, model_base_dir = get_beam_paths(args.beam)
    model_dir = model_base_dir / MODEL_DIRS[args.beam][args.squeeze_step]
    results_dir = PROJECT_ROOT / f"b{args.beam}_squeeze_results"

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
