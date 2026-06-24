from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import tfs

from aba_optimiser.measurements.loading import build_dataframe_file_indices
from aba_optimiser.measurements.preprocessing import preprocess_measurement_dataframe
from aba_optimiser.measurements.reconstruction import process_single_dataframe
from aba_optimiser.measurements.variances import (
    assign_known_noise_variances,
    assign_uniform_variances,
)
from tests.measurements.helpers import generate_fake_analysis_dir_from_twiss


def test_build_dataframe_file_indices_tracks_source_file() -> None:
    measurements = [
        SimpleNamespace(matrices=[object(), object()]),
        SimpleNamespace(matrices=[object()]),
        SimpleNamespace(matrices=[object(), object(), object()]),
    ]

    assert build_dataframe_file_indices(measurements) == [0, 0, 1, 2, 2, 2]  # ty:ignore[invalid-argument-type]


def test_process_single_dataframe_reconstructs_with_generated_analysis(tmp_path: Path) -> None:
    pytest.importorskip("tmom_recon")
    pytest.importorskip("omc3")

    base_dir = Path("tests/data/model_creator")
    analysis_dir = generate_fake_analysis_dir_from_twiss(
        tmp_path / "analysis",
        twiss_path=base_dir / "psb3_twiss.dat",
    )
    twiss = tfs.read(base_dir / "psb3_twiss_ac.dat", index="NAME")
    twiss.columns = [column.lower() for column in twiss.columns]
    twiss = twiss.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    twiss.index = twiss.index.astype(str)
    twiss.index.name = "name"
    # The fixture twiss predates vertical dispersion; PSB dy is ~0 and the
    # MAD-NG twiss table used in production includes it, so supply it here.
    twiss["dy"] = 0.0

    df = pd.DataFrame(
        {
            "name": [
                "BR3.BPM1L3",
                "BR3.BPM2L3",
                "BR3.BPM3L3",
                "BR3.BPM1L3",
                "BR3.BPM2L3",
                "BR3.BPM3L3",
            ],
            "turn": [1, 1, 1, 2, 2, 2],
            "x": [1e-6, 2e-6, 3e-6, 1.5e-6, 2.5e-6, 3.5e-6],
            "y": [2e-6, 3e-6, 4e-6, 2.5e-6, 3.5e-6, 4.5e-6],
        }
    )

    idx, result = process_single_dataframe(
        df_with_index=(7, df),
        twiss=twiss,
        bad_bpms=[],
        analysis_dir=analysis_dir,
        use_uniform_vars=True,
        beam=1,
    )

    assert idx == 7
    assert {"px", "py", "var_x", "var_y", "var_px", "var_py"} <= set(result.columns)
    assert not result[["px", "py"]].isna().any().any()
    assert set(result["name"]) == {"BR3.BPM1L3", "BR3.BPM2L3", "BR3.BPM3L3"}


def test_assign_uniform_variances_zero_weights_bad_bpms() -> None:
    df = pd.DataFrame({"name": ["BPM1", "BPM2"]})

    result = assign_uniform_variances(df, ["BPM2"], var_value=2.5)

    assert result.loc[0, "var_x"] == pytest.approx(2.5)
    assert result.loc[0, "var_py"] == pytest.approx(2.5)
    assert result.loc[1, "var_x"] == float("inf")
    assert result.loc[1, "var_y"] == float("inf")
    assert result.loc[1, "var_px"] == float("inf")
    assert result.loc[1, "var_py"] == float("inf")


def test_assign_known_noise_variances_allows_nan_variance_patterns_on_real_data() -> None:
    indexed = pd.DataFrame(
        {
            "x": [0.0, 1e-6, 2e-6],
            "y": [0.0, 3e-6, 4e-6],
        },
        index=pd.Index(["BI3.KSW1L4", "BR3.BPM2L3", "BR3.BPMT3L1"], name="name"),
    )

    result = assign_known_noise_variances(
        indexed,
        bad_bpms=[],
        nan_variance_patterns=[r"^BI3\.KSW1L4$", r"^BR3\.BPMT3L1$"],
        accelerator_type="psb",
    )

    assert pd.isna(result.loc["BI3.KSW1L4", "var_x"])
    assert pd.isna(result.loc["BI3.KSW1L4", "var_y"])
    assert pd.isna(result.loc["BR3.BPMT3L1", "var_x"])
    assert pd.isna(result.loc["BR3.BPMT3L1", "var_y"])
    assert result.loc["BR3.BPM2L3", "var_x"] > 0.0
    assert result.loc["BR3.BPM2L3", "var_y"] > 0.0


def test_preprocess_measurement_dataframe_requires_x_and_y() -> None:
    tws = pd.DataFrame({"x": [0.1], "y": [0.2]}, index=pd.Index(["BPM1"], name="name"))
    df = pd.DataFrame({"name": ["BPM1"], "turn": [1], "x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError, match="both x and y"):
        preprocess_measurement_dataframe(
            df,
            tws,
            remove_closed_orbit={"BPM1": {"x": 0.5}},
        )


def test_preprocess_measurement_dataframe_requires_px_and_py_together() -> None:
    tws = pd.DataFrame({"x": [0.1], "y": [0.2]}, index=pd.Index(["BPM1"], name="name"))
    df = pd.DataFrame({"name": ["BPM1"], "turn": [1], "x": [1.0], "y": [2.0]})

    with pytest.raises(ValueError, match="both px and py"):
        preprocess_measurement_dataframe(
            df,
            tws,
            remove_closed_orbit={"BPM1": {"x": 0.5, "y": 0.25, "px": 1e-6}},
        )


def test_preprocess_measurement_dataframe_accepts_name_column_and_warns_without_momenta() -> None:
    tws = pd.DataFrame({"x": [0.0], "y": [0.0]}, index=pd.Index(["BPM1"], name="name"))
    df = pd.DataFrame({"name": ["BPM1"], "turn": [1], "x": [1.0], "y": [2.0]})
    orbit = pd.DataFrame({"NAME": ["BPM1"], "x": [0.25], "y": [0.5]})

    with pytest.warns(UserWarning, match="px/py"):
        result = preprocess_measurement_dataframe(
            df,
            tws,
            remove_closed_orbit=orbit,
        )

    assert result.loc[0, "x"] == pytest.approx(0.75)
    assert result.loc[0, "y"] == pytest.approx(1.5)


def test_preprocess_measurement_dataframe_average_trims_from_kick() -> None:
    tws = pd.DataFrame(
        {"s": [1.0, 2.0, 3.0]},
        index=pd.Index(["BPM1", "BPM2", "BPM3"], name="name"),
    )
    df = pd.DataFrame(
        {
            "name": ["BPM1", "BPM2", "BPM3"] * 5,
            "turn": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "x": [0.0] * 9 + [0.0, 1e-3, 8e-4, -2e-4, 9e-4, 7e-4],
            "y": [0.0] * 15,
        }
    )

    result = preprocess_measurement_dataframe(
        df,
        tws,
        remove_closed_orbit="average",
        n_turns_free=3,
    )

    assert list(result["name"]) == ["BPM2", "BPM3", "BPM1", "BPM2", "BPM3"]
    assert list(result["turn"]) == [1, 1, 2, 2, 2]
    assert result.iloc[0]["x"] == pytest.approx(1e-3)

def test_preprocess_measurement_dataframe_average_skips_already_aligned() -> None:
    tws = pd.DataFrame({"s": [0.0]}, index=pd.Index(["KICKER"], name="name"))
    df = pd.DataFrame({"name": ["KICKER"], "turn": [1], "x": [0.0], "y": [0.0]})

    result = preprocess_measurement_dataframe(
        df,
        tws,
        remove_closed_orbit="average",
        kicker_name="KICKER",
    )

    pd.testing.assert_frame_equal(result, df)
