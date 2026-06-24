from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import tfs

from aba_optimiser.measurements.create_datafile import (
    build_madng_twiss_table,
    copy_ac_dipole_attrs,
    detect_bad_bpms,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_copy_ac_dipole_attrs_copies_only_known_metadata() -> None:
    source = pd.DataFrame({"x": [1.0]})
    source.attrs["ac_dipole_marker"] = "MKQA"
    source.attrs["ac_dipole_bpm_upstream"] = "BPM.UP"
    source.attrs["custom_key"] = "leave-behind"
    target = pd.DataFrame({"x": [2.0]})

    copy_ac_dipole_attrs(source, target)

    assert target.attrs == {
        "ac_dipole_marker": "MKQA",
        "ac_dipole_bpm_upstream": "BPM.UP",
    }


def test_detect_bad_bpms_marks_nan_inf_and_missing_bpms() -> None:
    pzs = pd.DataFrame(
        {
            "name": ["BPM1", "BPM1", "BPM2"],
            "px": [0.0, float("nan"), 0.0],
            "py": [0.0, 0.0, 0.0],
            "var_x": [1.0, 1.0, float("inf")],
            "var_px": [1.0, 1.0, 1.0],
            "var_y": [1.0, 1.0, float("inf")],
            "var_py": [1.0, 1.0, 1.0],
        }
    )
    bad_bpms: list[str] = []

    detect_bad_bpms(pzs, {"BPM1", "BPM2", "BPM3"}, bad_bpms, log_individual=False)

    assert set(bad_bpms) == {"BPM1", "BPM2", "BPM3"}


def test_build_madng_twiss_table_uses_existing_twiss_file(tmp_path: Path) -> None:
    expected = tfs.TfsDataFrame(
        {"NAME": ["BPM1"], "S": [1.23], "BETX": [10.0]},
    )
    twiss_path = tmp_path / "twiss_ac.dat"
    tfs.write(twiss_path, expected, save_index=False)

    result = build_madng_twiss_table(
        model_dir=tmp_path,
        accelerator=SimpleNamespace(),  # ty:ignore[invalid-argument-type]
        output_dir=tmp_path,
        nattunes=[0.31, 0.32, 0.0],
        tunes=[0.28, 0.29, 0.0],
    )

    assert list(result["NAME"]) == ["BPM1"]
    assert result.loc[0, "S"] == 1.23
    assert result.loc[0, "BETX"] == 10.0
