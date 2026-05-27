"""Helpers for measurement integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from turn_by_turn import write_tbt
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def write_lhc_sdds_from_long_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    *,
    bunch_id: int = 0,
) -> Path:
    """Write one long-form measurement dataframe to an LHC-style SDDS file.

    Expected input columns:

    - ``name``
    - ``turn`` using 1-based turn numbering
    - ``x`` and ``y`` in metres
    """
    required_columns = {"name", "turn", "x", "y"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required measurement columns: {sorted(missing_columns)}")

    measurement = df.loc[:, ["name", "turn", "x", "y"]].copy()
    measurement["name"] = measurement["name"].astype(str)
    measurement["turn0"] = measurement["turn"].astype(int) - 1

    bpm_names = measurement["name"].drop_duplicates().tolist()
    x_mm = (
        measurement.pivot(index="name", columns="turn0", values="x")
        .reindex(index=bpm_names)
        .astype(float)
        * 1e3
    )
    y_mm = (
        measurement.pivot(index="name", columns="turn0", values="y")
        .reindex(index=bpm_names)
        .astype(float)
        * 1e3
    )

    tbt_data = TbtData(
        matrices=[TransverseData(X=x_mm, Y=y_mm)],
        nturns=x_mm.shape[1],
        bunch_ids=[bunch_id],
        meta={"source_datatype": "synthetic_long_dataframe"},
    )
    write_tbt(output_path, tbt_data, datatype="lhc")
    return output_path


def generate_fake_analysis_dir_from_twiss(
    output_dir: Path,
    *,
    twiss_path: Path,
    parameters: list[str] | None = None,
) -> Path:
    """Generate a minimal fake optics-measurement folder for reconstruction tests."""
    measurement_twiss = tfs.read(twiss_path, index="NAME")
    measurement_twiss.columns = [column.upper() for column in measurement_twiss.columns]
    measurement_twiss = measurement_twiss.rename(columns={"MU1": "MUX", "MU2": "MUY"})
    measurement_twiss.headers = {
        str(key).upper(): value for key, value in measurement_twiss.headers.items()
    }

    fake_measurement(
        twiss=measurement_twiss,
        outputdir=output_dir,
        parameters=parameters or ["BETX", "BETY", "PHASEX", "PHASEY", "X", "Y"],
    )
    return output_dir
