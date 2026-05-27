"""Raw measurement loading helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from turn_by_turn import TbtData, read_tbt

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


def build_dataframe_file_indices(measurements: list[TbtData]) -> list[int]:
    """Map each converted dataframe back to the source measurement-file index."""
    file_indices: list[int] = []
    for file_idx, measurement in enumerate(measurements):
        file_indices.extend([file_idx] * len(measurement.matrices))
    return file_indices


def load_measurement_files(files: list[Path]) -> list[TbtData]:
    """Load multiple SDDS turn-by-turn files into TbtData objects."""
    measurements: list[TbtData] = []
    for file in files:
        LOGGER.info("Loading data from %s", file)
        measurements.append(read_tbt(file, datatype="lhc"))
    return measurements


def convert_tbt_to_dataframes(
    measurements: list[TbtData],
    bad_bpms: list[str] | None = None,
    *,
    combine_measurements: bool = True,
) -> list[pd.DataFrame]:
    """Convert turn-by-turn matrices into long-form dataframes."""
    excluded_bpms = set(bad_bpms or [])
    converted: list[pd.DataFrame] = []
    turn_offset = 1

    for measurement in measurements:
        if not combine_measurements:
            turn_offset = 1

        for bunch in measurement.matrices:
            df_x = bunch.X.copy()
            df_y = bunch.Y.copy()
            df_x.index.name = "name"
            df_y.index.name = "name"
            df_x.columns = df_x.columns + turn_offset
            df_y.columns = df_y.columns + turn_offset

            combined = df_x.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
            combined["y"] = df_y.reset_index().melt(id_vars="name", var_name="turn", value_name="y")["y"]
            combined["x"] = combined["x"] / 1000.0
            combined["y"] = combined["y"] / 1000.0

            original_order = df_x.index.tolist()
            assert df_y.index.tolist() == original_order, "BPM order mismatch between X and Y data"
            combined["name"] = pd.Categorical(combined["name"], categories=original_order)

            if excluded_bpms:
                combined = combined[~combined["name"].isin(excluded_bpms)]

            converted.append(combined.sort_values(["turn", "name"]).reset_index(drop=True))
            turn_offset += df_x.shape[1]

    return converted
