"""MAD model and sequence preparation for LHC measurement workflows.

Builds (or reuses cached) MAD-X sequences from a measurement model directory,
extracts machine-settings knobs from NXCALS via omc3, and reads the natural and
driven tunes out of a model's ``job.create_model_nominal.madx``.
"""

from __future__ import annotations

import logging
import re
import warnings
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version
from pandas.errors import Pandas4Warning
from pymadng_utils.madx import make_madx_sequence

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_omc3_pandas_copy_warning() -> Iterator[None]:
    """Suppress the known omc3/pandas copy warning for the legacy NXCALS path."""
    if not _should_suppress_omc3_pandas_copy_warning():
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*",
            category=Pandas4Warning,
        )
        yield


def _should_suppress_omc3_pandas_copy_warning() -> bool:
    """Return True for the known incompatible omc3/pandas version combination."""
    try:
        omc3_version = Version(version("omc3"))
        pandas_version = Version(version("pandas"))
    except (PackageNotFoundError, InvalidVersion):
        return False

    return omc3_version == Version("0.28.0") and pandas_version > Version("3.0.0")


def make_machine_settings_knobs_file(output_file: Path, time: str) -> Path:
    """Extract a single MAD-X knobs file for the requested machine-settings time."""
    from omc3.knob_extractor import main as extract_knobs

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with _suppress_omc3_pandas_copy_warning():
        extract_knobs({"time": time, "output": output_file})
    if not output_file.exists():
        raise FileNotFoundError(
            f"Expected machine-settings knobs file was not created: {output_file}"
        )
    logger.info("Saved machine-settings knobs for %s to %s", time, output_file)
    return output_file


def get_or_make_sequence(beam: int, madng_model_dir: Path, time: str | None = None) -> Path:
    """Get cached sequence or generate a new one.

    Args:
        beam: Beam number (1 or 2)
        madng_model_dir: Path to MAD-NG model directory
        time: Optional machine-settings extraction time. When provided, a
            machine-settings knobs file is generated and applied after the
            optics modifiers during sequence creation.

    Returns:
        Path to sequence file
    """
    if time is None:
        knobs_file = None
    else:
        knobs_file = madng_model_dir / f"machine_settings_{time}.madx"
        if not knobs_file.exists():
            make_machine_settings_knobs_file(knobs_file, time)
        knobs_file = [Path(knobs_file)]
    expected_sequence_file = madng_model_dir / f"lhcb{beam}_saved.seq"
    if expected_sequence_file.exists():
        logger.info(f"Found existing sequence file for beam {beam} at {expected_sequence_file}")
        return expected_sequence_file
    return make_madx_sequence(madng_model_dir, post_optics_madx_files=knobs_file)


def extract_tunes_from_job_file(job_file_path: Path) -> tuple[float, float, float, float]:
    """Extract natural and driven tunes from the MAD-X job file.

    Args:
        job_file_path: Path to the job.create_model_nominal.madx file

    Returns:
        Tuple of (nat_x, nat_y, drv_x, drv_y)
    """
    with job_file_path.open("r") as f:
        content = f.read()

    # Regex to match twiss_ac_dipole(nat_x, nat_y, drv_x, drv_y, ...)
    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.search(
        rf"twiss_ac_dipole\(\s*({number_pattern})\s*,\s*({number_pattern})\s*,\s*({number_pattern})\s*,\s*({number_pattern})\s*,",
        content,
    )
    if not match:
        raise ValueError(f"Could not find twiss_ac_dipole call in {job_file_path}")

    nat_x = float(match.group(1))
    nat_y = float(match.group(2))
    drv_x = float(match.group(3))
    drv_y = float(match.group(4))

    logger.info(
        f"Extracted tunes from {job_file_path}: nat_x={nat_x}, nat_y={nat_y}, drv_x={drv_x}, drv_y={drv_y}"
    )
    return nat_x, nat_y, drv_x, drv_y
