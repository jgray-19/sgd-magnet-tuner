from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import tfs

LOGGER = logging.getLogger(__name__)


def filter_out_marker(tbl: tfs.TfsDataFrame, marker_name: str) -> tfs.TfsDataFrame:
    """
    Filter out markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.

    Returns:
        A TFS DataFrame without markers.
    """
    LOGGER.debug(f"Filtering out marker {marker_name}")
    if tbl.index.name != "name":
        LOGGER.debug("Index is not 'name', using 'name' column for filtering")
        return tbl[tbl["name"] != marker_name]
    return tbl[tbl.index != marker_name]


def filter_out_markers(
    tbl: tfs.TfsDataFrame, marker_names: list[str]
) -> tfs.TfsDataFrame:
    """
    Filter out markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.
        marker_names: A list of marker names to filter out.

    Returns:
        A TFS DataFrame without the specified markers.
    """
    LOGGER.debug(f"Filtering out markers {marker_names}")
    if tbl.index.name != "name":
        LOGGER.debug("Index is not 'name', using 'name' column for filtering")
        return tbl[~tbl["name"].isin(marker_names)]
    return tbl[~tbl.index.isin(marker_names)]


def select_markers(tbl: pd.DataFrame, marker_name: str | list[str]) -> pd.DataFrame:
    """
    Select markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.

    Returns:
        A TFS DataFrame with only the specified markers.
    """
    LOGGER.debug(f"Selecting markers {marker_name}")
    if isinstance(marker_name, str):
        marker_name = [marker_name]
    if tbl.index.name != "name":
        LOGGER.debug("Index is not 'name', using 'name' column for filtering")
        return tbl[tbl["name"].isin(marker_name)]
    return tbl[tbl.index.isin(marker_name)]
