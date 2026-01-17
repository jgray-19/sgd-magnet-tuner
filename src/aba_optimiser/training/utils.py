"""Shared utilities for training controllers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def filter_bad_bpms(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    bad_bpms: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Remove bad BPMs from start and end point lists.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        bad_bpms: Optional list of BPM names to remove

    Returns:
        Tuple of (filtered_start_points, filtered_end_points)
    """
    if bad_bpms is None:
        return bpm_start_points, bpm_end_points

    filtered_start = bpm_start_points.copy()
    filtered_end = bpm_end_points.copy()

    for bpm in bad_bpms:
        if bpm in filtered_start:
            filtered_start.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from start points")
        if bpm in filtered_end:
            filtered_end.remove(bpm)
            LOGGER.warning(f"Removed bad BPM {bpm} from end points")

    return filtered_start, filtered_end


def normalize_true_strengths(
    true_strengths: Path | dict[str, float] | None,
) -> dict[str, float]:
    """Normalize true strengths to a dictionary format.

    Args:
        true_strengths: Can be None, a Path to a file, or a dict

    Returns:
        Dictionary of true strengths (empty if None was provided)
    """
    from aba_optimiser.io.utils import read_knobs

    if true_strengths is None:
        return {}
    if isinstance(true_strengths, Path):
        return read_knobs(true_strengths)
    if isinstance(true_strengths, dict):
        return true_strengths.copy()
    raise TypeError(f"Unexpected type for true_strengths: {type(true_strengths)}")


def extract_bpm_range_names(
    all_bpms: list[str],
    start_bpm: str,
    end_bpm: str,
    sdir: int,
) -> list[str]:
    """Extract BPM names between start and end BPMs, handling circular wrapping.

    Args:
        all_bpms: List of all BPM names
        start_bpm: Starting BPM name
        end_bpm: Ending BPM name
        sdir: Direction (1 for forward, -1 for reverse)

    Returns:
        List of BPM names in the range
    """
    start_pos = all_bpms.index(start_bpm)
    end_pos = all_bpms.index(end_bpm) + 1

    if end_pos <= start_pos:
        # Circular wrapping
        extracted = all_bpms[start_pos:] + all_bpms[:end_pos]
    else:
        extracted = all_bpms[start_pos:end_pos]
    # Reverse for negative direction
    if sdir == -1:
        extracted = extracted[::-1]

    return extracted


def find_common_bpms(*dataframes: pd.DataFrame) -> list[str]:
    """Find common BPMs across multiple dataframes.

    Args:
        dataframes: Variable number of DataFrames with BPM names as index

    Returns:
        List of common BPM names in the order they appear in the first dataframe
    """
    if not dataframes:
        return []

    # Get common BPMs across all dataframes
    common = set(dataframes[0].index)
    for df in dataframes[1:]:
        common &= set(df.index)

    # Return in order of first dataframe
    return [bpm for bpm in dataframes[0].index if bpm in common]


def load_tfs_files(
    directory: Path,
    file_specs: dict[str, tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    """Load multiple TFS files and return as a dictionary.

    Args:
        directory: Directory containing the TFS files
        file_specs: Dict mapping keys to (prefix, suffix) tuples.
            Example: ``{"beta_x": ("beta_phase_", "x")}`` will load ``beta_phase_x.tfs``

    Returns:
        Dictionary mapping keys to loaded DataFrames

    Raises:
        FileNotFoundError: If any required file is missing
    """
    import tfs
    from omc3.optics_measurements.constants import EXT

    loaded = {}
    for key, (prefix, suffix) in file_specs.items():
        file_path = directory / f"{prefix}{suffix}{EXT}"
        if not file_path.exists():
            raise FileNotFoundError(f"Required TFS file not found: {file_path}")
        # Phase files have NAME/NAME2 pairs, so don't index by NAME
        # Other files have single BPM per row, so index by NAME
        if "phase" in key and "beta" not in key:
            loaded[key] = tfs.read(file_path)
        else:
            loaded[key] = tfs.read(file_path, index="NAME")

    return loaded


def create_bpm_range_specs(
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    use_fixed_bpm: bool,
    fixed_start: str | None = None,
    fixed_end: str | None = None,
) -> list[tuple[str, str, int]]:
    """Create BPM range specifications for optimisation workers.

    Args:
        bpm_start_points: List of starting BPM names
        bpm_end_points: List of ending BPM names
        use_fixed_bpm: If True, use fixed BPM pairs; if False, create cartesian product
        fixed_start: Fixed start BPM for backward tracking (only used if use_fixed_bpm=True)
        fixed_end: Fixed end BPM for forward tracking (only used if use_fixed_bpm=True)

    Returns:
        List of (start_bpm, end_bpm, sdir) tuples where sdir is 1 for forward, -1 for reverse
    """
    if use_fixed_bpm:
        if fixed_start is None or fixed_end is None:
            raise ValueError("fixed_start and fixed_end must be provided when use_fixed_bpm=True")
        # Forward: start -> fixed_end; Backward: fixed_start -> end
        range_specs = [(s, fixed_end, 1) for s in bpm_start_points] + [
            (fixed_start, e, -1) for e in bpm_end_points
        ]
    else:
        # Cartesian product: every start with every end, forward direction only
        range_specs = [(s, e, 1) for s in bpm_start_points for e in bpm_end_points]

    return range_specs
