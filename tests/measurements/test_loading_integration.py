from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.measurements.loading import (
    build_dataframe_file_indices,
    convert_tbt_to_dataframes,
    load_measurement_files,
)
from tests.measurements.helpers import write_lhc_sdds_from_long_dataframe

if TYPE_CHECKING:
    from pathlib import Path


def test_loading_round_trip_from_generated_lhc_sdds(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {
            "turn": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "name": [
                "BPM.1",
                "BPM.2",
                "BPM.3",
                "BPM.1",
                "BPM.2",
                "BPM.3",
                "BPM.1",
                "BPM.2",
                "BPM.3",
            ],
            "x": [1e-6, 2e-6, 3e-6, 1.5e-6, 2.5e-6, 3.5e-6, 2e-6, 3e-6, 4e-6],
            "y": [4e-6, 5e-6, 6e-6, 4.5e-6, 5.5e-6, 6.5e-6, 5e-6, 6e-6, 7e-6],
        }
    )
    source["name"] = source["name"].astype(str)
    source = source.sort_values(["turn", "name"]).reset_index(drop=True)

    sdds_path = write_lhc_sdds_from_long_dataframe(source, tmp_path / "synthetic_measurement.sdds")

    measurements = load_measurement_files([sdds_path])
    assert len(measurements) == 1
    assert measurements[0].nturns == int(source["turn"].max())
    assert build_dataframe_file_indices(measurements) == [0]

    converted = convert_tbt_to_dataframes(measurements, combine_measurements=True)
    assert len(converted) == 1

    round_tripped = converted[0].copy()
    round_tripped["name"] = round_tripped["name"].astype(str)
    round_tripped = round_tripped.sort_values(["turn", "name"]).reset_index(drop=True)

    assert set(round_tripped["bunch_number"]) == {0}
    assert round_tripped[["turn", "name", "x", "y"]].shape == source.shape
    assert (round_tripped["x"] - source["x"]).abs().max() < 1e-12
    assert (round_tripped["y"] - source["y"]).abs().max() < 1e-12
