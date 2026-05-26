"""File I/O helpers for squeeze quadrupole optimisation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aba_optimiser.measurements.online_knobs import save_online_knobs
from aba_optimiser.measurements.squeeze_helpers import get_measurement_date
from aba_optimiser.measurements.utils import find_all_bad_bpms

logger = logging.getLogger(__name__)


def get_measurement_time(earliest_time: str, squeeze_step: str) -> datetime:
    """Convert a timestamp string to a UTC datetime."""
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{get_measurement_date(squeeze_step)} {time_str}"
    return datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("UTC"))


def get_sequence_creation_time(meas_times_for_step: dict[str, list[str]], squeeze_step: str) -> str:
    """Return the ISO timestamp of the earliest measurement for sequence creation."""
    all_times = [t for times in meas_times_for_step.values() for t in times]
    if not all_times:
        raise ValueError(f"No measurement times configured for squeeze step {squeeze_step}.")
    return get_measurement_time(min(all_times), squeeze_step).isoformat()


def get_knob_files(results_dir: Path, squeeze_step: str, freq: str) -> tuple[Path, Path]:
    """Return (tune_knobs_file, corrector_strengths_file) for a squeeze step and frequency."""
    return (
        results_dir / f"tune_knobs_{squeeze_step}_{freq}.txt",
        results_dir / f"corrector_strengths_{squeeze_step}_{freq}.txt",
    )


def load_bad_bpms(bad_bpms_file: Path) -> set[str]:
    """Load bad BPMs from a text file (one per line)."""
    if not bad_bpms_file.exists():
        raise FileNotFoundError(
            f"Bad BPMs file {bad_bpms_file} not found. Run without --skip-reload first."
        )
    with bad_bpms_file.open() as f:
        bad_bpms = {line.strip() for line in f if line.strip()}
    logger.info("Loaded %d bad BPMs from %s", len(bad_bpms), bad_bpms_file)
    return bad_bpms


def save_bad_bpms(bad_bpms_file: Path, bad_bpms: set[str]) -> None:
    """Write bad BPMs to a text file (one per line)."""
    if not bad_bpms:
        return
    with bad_bpms_file.open("w") as f:
        f.writelines(f"{bpm}\n" for bpm in bad_bpms)


def load_metadata(temp_analysis_dir: Path) -> dict:
    """Load metadata.json from the temp directory, returning {} if absent."""
    metadata_file = temp_analysis_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
    with metadata_file.open() as f:
        return json.load(f)


def save_metadata(temp_analysis_dir: Path, metadata: dict) -> None:
    """Write metadata dict to metadata.json in the temp directory."""
    with (temp_analysis_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def update_metadata(temp_analysis_dir: Path, **kwargs) -> None:
    """Merge kwargs into the temp directory metadata.json."""
    metadata = load_metadata(temp_analysis_dir)
    metadata.update(kwargs)
    save_metadata(temp_analysis_dir, metadata)


def validate_processed_files(temp_analysis_dir: Path, freq: str, num_files: int) -> None:
    """Raise if any expected parquet cache files for freq are missing."""
    missing = [
        temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        for i in range(num_files)
        if not (temp_analysis_dir / f"pz_data_{freq}_{i}.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} processed measurement files for {freq}. "
            f"First missing: {missing[0]}. Run without --skip-reload to regenerate."
        )
    logger.info("Verified %d processed files exist for %s", num_files, freq)


def save_arc_estimates(
    results_dir: Path,
    squeeze_step: str,
    arc_num: int,
    estimate: dict[str, float],
    uncertainties: dict[str, float],
    rewrite_file: bool = False,
) -> None:
    """Save arc optimisation estimates and uncertainties to the JSON results file."""
    outfile = results_dir / f"quad_estimates_{squeeze_step}.json"
    payload: dict[str, dict[str, dict[str, float]]] = {}
    if outfile.exists() and not rewrite_file:
        with outfile.open() as f:
            payload = json.load(f)
    payload[f"Arc {arc_num}"] = {
        magnet: {"value": float(value), "uncertainty": float(uncertainties.get(magnet, 0.0))}
        for magnet, value in estimate.items()
    }
    with outfile.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def prepare_frequency_metadata(
    freq: str,
    times: list[str],
    beam: int,
    meas_base_dir: Path,
    results_dir: Path,
    squeeze_step: str,
) -> tuple[list[Path], Path, Path, set[str], float]:
    """Resolve file paths, download knobs, and collect bad BPMs for one frequency."""
    meas_date = get_measurement_date(squeeze_step)
    name_prefix = f"Beam{beam}@BunchTurn@{meas_date}@".replace("-", "_")
    analysed_folders = [meas_base_dir / f"{name_prefix}{t}" for t in times]

    missing_folders = [f for f in analysed_folders if not f.exists()]
    if missing_folders:
        raise FileNotFoundError(
            f"Analysis folders missing for squeeze step {squeeze_step!r}, freq {freq!r}: "
            + ", ".join(str(f) for f in missing_folders)
        )

    bad_bpms: set[str] = set()
    for folder in analysed_folders:
        bad_bpms.update(find_all_bad_bpms(folder))
    if not bad_bpms:
        raise ValueError(f"No bad BPMs found for {squeeze_step!r} {freq!r} — something is wrong.")

    files = [analysed_folders[i] / f"{name_prefix}{times[i]}.sdds" for i in range(len(times))]
    tune_knobs_file, corrector_knobs_file = get_knob_files(results_dir, squeeze_step, freq)

    meas_time = get_measurement_time(min(times), squeeze_step)
    energy = save_online_knobs(
        meas_time,
        beam=beam,
        tune_knobs_file=tune_knobs_file,
        corrector_knobs_file=corrector_knobs_file,
    )

    return files, tune_knobs_file, corrector_knobs_file, bad_bpms, float(energy)


def process_frequency_results(
    freq: str,
    file_keys: list[str],
    pzs_dict: dict,
    tune_knobs_file: Path,
    corrector_knobs_file: Path,
    temp_analysis_dir: Path,
) -> list[dict]:
    """Persist parquet files for a frequency and return measurement descriptor dicts."""
    measurements = []
    dpp_values = []
    for i, key in enumerate(file_keys):
        stem = Path(key).stem
        pzs = pzs_dict[stem]
        dpp_est = float(pzs.attrs["DPP_EST"])
        dpp_values.append(dpp_est)
        meas_save_path = temp_analysis_dir / f"pz_data_{freq}_{i}.parquet"
        pzs.to_parquet(meas_save_path)
        measurements.append(
            {
                "file": meas_save_path,
                "tune_knobs_file": tune_knobs_file,
                "corrector_file": corrector_knobs_file,
                "machine_deltap": dpp_est,
            }
        )

    metadata = load_metadata(temp_analysis_dir)
    metadata.setdefault("dpp_values", {})[freq] = dpp_values
    save_metadata(temp_analysis_dir, metadata)
    return measurements


def load_frequency_results(
    freq: str,
    num_files: int,
    tune_knobs_file: Path,
    corrector_knobs_file: Path,
    temp_analysis_dir: Path,
) -> list[dict]:
    """Load previously persisted parquet files and return measurement descriptor dicts."""
    validate_processed_files(temp_analysis_dir, freq, num_files)
    dpp_values = load_metadata(temp_analysis_dir)["dpp_values"][freq]
    measurements = []
    for i, dpp_est in enumerate(dpp_values):
        measurements.append(
            {
                "file": temp_analysis_dir / f"pz_data_{freq}_{i}.parquet",
                "tune_knobs_file": tune_knobs_file,
                "corrector_file": corrector_knobs_file,
                "machine_deltap": float(dpp_est),
            }
        )
    logger.info("Loaded %d processed files for %s", num_files, freq)
    return measurements
