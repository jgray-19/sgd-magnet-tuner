"""Shared utilities for training controllers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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


def extract_bpm_range_data(
    df: pd.DataFrame,
    start_bpm: str,
    end_bpm: str,
    sdir: int,
    column_names: list[str],
) -> np.ndarray:
    """Extract data between start and end BPMs, handling circular wrapping.

    Args:
        df: DataFrame with BPM index
        start_bpm: Starting BPM name
        end_bpm: Ending BPM name
        sdir: Direction (1 for forward, -1 for reverse)
        column_names: List of column names to extract (will be stacked horizontally)

    Returns:
        Array of shape (n_bpms, len(column_names)) with extracted data
    """
    start_pos = df.index.get_loc(start_bpm)
    end_pos = df.index.get_loc(end_bpm) + 1

    # Extract each column
    column_arrays = []
    for col_name in column_names:
        values = df[col_name].to_numpy()
        if end_pos <= start_pos:
            # Circular wrapping
            extracted = np.concatenate((values[start_pos:], values[:end_pos]))
        else:
            extracted = values[start_pos:end_pos]

        # Reverse for negative direction
        if sdir == -1:
            extracted = extracted[::-1]

        column_arrays.append(extracted)

    # Stack into (n_bpms, n_columns)
    return np.vstack(column_arrays).T


def find_common_bpms(*dataframes: pd.DataFrame) -> list[str]:
    """Find common BPMs across multiple dataframes.

    Args:
        *dataframes: Variable number of DataFrames with BPM names as index

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
        file_specs: Dict mapping keys to (prefix, suffix) tuples
            Example: {"beta_x": ("beta_phase_", "x")} -> beta_phase_x.tfs

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
        loaded[key] = tfs.read(file_path, index="NAME")

    return loaded
