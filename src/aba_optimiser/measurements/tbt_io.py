"""Turn-by-turn data loading and conversion utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from turn_by_turn import TbtData, read_tbt

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _build_dataframe_file_indices(measurements: list[TbtData]) -> list[int]:
    """Map each converted dataframe back to the source measurement-file index."""
    file_indices: list[int] = []
    for file_idx, meas in enumerate(measurements):
        file_indices.extend([file_idx] * len(meas.matrices))
    return file_indices


def load_files(files: list[Path]) -> list[TbtData]:
    """Load multiple SDDS turn-by-turn files into TbtData objects."""
    measurements: list[TbtData] = []
    for file in files:
        LOGGER.info("Loading data from %s", file)
        measurements.append(read_tbt(file, datatype="lhc"))
    return measurements


def convert_measurements(
    measurements: list[TbtData],
    bad_bpms: list[str] | None = None,
    combine_measurements: bool = True,
) -> list[pd.DataFrame]:
    """Convert TbtData objects into long-form DataFrames with columns [name, turn, x, y].

    Each row is one BPM / turn sample. x and y are converted from mm to metres.
    Turn numbers are made globally unique across all bunches and files when
    combine_measurements is True.
    """
    if bad_bpms is None:
        bad_bpms = []

    all_data: list[pd.DataFrame] = []
    turn_offset = 1
    for meas in measurements:
        if not combine_measurements:
            turn_offset = 1
        for bunch in meas.matrices:
            df_x = bunch.X.copy()
            df_y = bunch.Y.copy()
            df_x.index.name = "name"
            df_y.index.name = "name"
            df_x.columns = df_x.columns + turn_offset
            df_y.columns = df_y.columns + turn_offset

            df_combined = df_x.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
            df_combined["y"] = df_y.reset_index().melt(id_vars="name", var_name="turn", value_name="y")["y"]
            df_combined["x"] = df_combined["x"] / 1000
            df_combined["y"] = df_combined["y"] / 1000

            original_order = df_x.index.tolist()
            assert df_y.index.tolist() == original_order, "BPM order mismatch between X and Y data"
            df_combined["name"] = pd.Categorical(df_combined["name"], categories=original_order)

            if bad_bpms:
                df_combined = df_combined[~df_combined["name"].isin(bad_bpms)]
            df_combined = df_combined.sort_values(["turn", "name"]).reset_index(drop=True)

            all_data.append(df_combined)
            turn_offset += df_x.shape[1]
    return all_data
