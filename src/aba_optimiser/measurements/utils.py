"""Utilities for measurements."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aba_optimiser.measurements.bad_bpms import find_all_bad_bpms, find_all_bad_bpms_from_analysis

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)

__all__ = [
    "find_all_bad_bpms",
    "find_all_bad_bpms_from_analysis",
    "merge_horizontal_vertical_bpms",
]


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
        'var_x', 'var_y', 'px', 'py', 'var_px', 'var_py']

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

    # Dual-plane: both x and y are non-NaN -> "xy"
    # H-only: x is non-NaN, y is NaN -> "x"
    # V-only: y is non-NaN, x is NaN -> "y"
    has_x = merged["x"].notna()
    has_y = merged["y"].notna()

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
