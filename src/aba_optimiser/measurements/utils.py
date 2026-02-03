"""Utilities for measurements."""

from __future__ import annotations

import ast
import configparser
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def find_all_bad_bpms_from_analysis(optics_folder: Path) -> set[str]:
    """Find all bad BPMs from the analysis ini file and associated measurement folders.

    Args:
        optics_folder: Path to the optics/analysis directory

    Returns:
        Set of bad BPM names
    """
    # Find the analysis ini file
    ini_files = list(optics_folder.glob("analysis*.ini"))
    if not ini_files:
        LOGGER.warning(f"No analysis*.ini file found in {optics_folder}, using empty bad BPMs list")
        return set()

    ini_file = ini_files[0]  # Take the first one if multiple
    LOGGER.info(f"Found analysis ini file: {ini_file}")

    # Parse the ini file
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Get the files list
    if "DEFAULT" not in config or "files" not in config["DEFAULT"]:
        LOGGER.warning(f"No 'files' entry in {ini_file}, using empty bad BPMs list")
        return set()

    # Parse the files list using ast.literal_eval (safe evaluation of Python literals)
    files_str = config["DEFAULT"]["files"]
    try:
        file_paths = ast.literal_eval(files_str)
    except (ValueError, SyntaxError) as e:
        LOGGER.warning(f"Failed to parse files list in {ini_file}: {e}, using empty bad BPMs list")
        return set()

    # Extract unique folders from file paths
    measurement_folders = set()
    for file_path in file_paths:
        folder = Path(file_path).parent
        measurement_folders.add(folder)

    LOGGER.info(f"Found {len(measurement_folders)} unique measurement folders")

    # Find bad BPMs in each folder
    all_bad_bpms = set()
    for folder in measurement_folders:
        if folder.exists():
            folder_bad_bpms = find_all_bad_bpms(folder)
            all_bad_bpms.update(folder_bad_bpms)
            LOGGER.debug(f"Found {len(folder_bad_bpms)} bad BPMs in {folder}")
        else:
            LOGGER.warning(f"Measurement folder does not exist: {folder}")

    LOGGER.info(f"Total unique bad BPMs found: {len(all_bad_bpms)}")
    return all_bad_bpms


def find_all_bad_bpms(measurement_dir: Path) -> set[str]:
    """Find all bad BPMs from .bad_bpms_x|y files in the measurement directory."""
    bad_bpms = set()
    for filepath in measurement_dir.glob("*.bad_bpms_*"):
        with filepath.open("r") as file:
            bad_bpms.update(line.split(" ")[0] for line in file.readlines())
    return bad_bpms


def merge_horizontal_vertical_bpms(
    df_horizontal: pd.DataFrame,
    df_vertical: pd.DataFrame,
) -> pd.DataFrame:
    """Merge horizontal and vertical BPM dataframes, filling missing planes with NaN.

    This function handles the case where not all BPMs in an accelerator are dual-plane.
    For single-plane BPMs, the missing plane is filled with NaN to indicate no measurement
    is available.

    Args:
        df_horizontal: DataFrame with horizontal BPM measurements
            Expected columns: ['name', 'turn', 'x', 'var_x', ...] (x in meters)
        df_vertical: DataFrame with vertical BPM measurements
            Expected columns: ['name', 'turn', 'y', 'var_y', ...] (y in meters)

    Returns:
        Merged DataFrame with both planes for all BPMs. Single-plane BPMs will have
        NaN values in the plane they don't measure. Columns: ['name', 'turn', 'x', 'y',
        'var_x', 'var_y', 'px', 'py', 'var_px', 'var_py', 'kick_plane']

    Note:
        - BPMs present only in horizontal data will have NaN for y, var_y, py, var_py
        - BPMs present only in vertical data will have NaN for x, var_x, px, var_px
        - Dual-plane BPMs will have measurements for both planes
    """
    import pandas as pd

    # Get all unique BPMs across both dataframes
    all_bpms = pd.Index(df_horizontal["name"].unique()).union(df_vertical["name"].unique())

    # Get all turns across both dataframes
    all_turns = pd.Index(df_horizontal["turn"].unique()).union(df_vertical["turn"].unique())

    # Create a complete index with all BPM-turn combinations
    complete_index = pd.MultiIndex.from_product(
        [all_bpms, all_turns], names=["name", "turn"]
    ).to_frame(index=False)

    # Merge horizontal data
    h_cols = [col for col in df_horizontal.columns if col not in ["name", "turn"]]
    merged = complete_index.merge(
        df_horizontal[["name", "turn"] + h_cols], on=["name", "turn"], how="left"
    )

    # Merge vertical data
    v_cols = [col for col in df_vertical.columns if col not in ["name", "turn"]]
    merged = merged.merge(df_vertical[["name", "turn"] + v_cols], on=["name", "turn"], how="left")

    # Set kick_plane based on which measurements are present
    # Dual-plane: both x and y are non-NaN -> "xy"
    # H-only: x is non-NaN, y is NaN -> "x"
    # V-only: y is non-NaN, x is NaN -> "y"
    has_x = merged["x"].notna()
    has_y = merged["y"].notna()

    merged["kick_plane"] = "xy"
    merged.loc[has_x & ~has_y, "kick_plane"] = "x"
    merged.loc[~has_x & has_y, "kick_plane"] = "y"

    # Convert name to category for efficiency
    merged["name"] = merged["name"].astype("category")
    merged["turn"] = merged["turn"].astype("int32")

    # Sort by turn then name to maintain consistent ordering
    merged = merged.sort_values(["turn", "name"]).reset_index(drop=True)

    # Log statistics about single vs dual plane BPMs
    dual_plane_bpms = merged[has_x & has_y]["name"].nunique()
    h_only_bpms = merged[has_x & ~has_y]["name"].nunique()
    v_only_bpms = merged[~has_x & has_y]["name"].nunique()

    LOGGER.info("BPM plane configuration:")
    LOGGER.info(f"  Dual-plane BPMs: {dual_plane_bpms}")
    LOGGER.info(f"  Horizontal-only BPMs: {h_only_bpms}")
    LOGGER.info(f"  Vertical-only BPMs: {v_only_bpms}")
    LOGGER.info(f"  Total BPMs: {all_bpms.size}")

    return merged
