"""Integration tests for transverse momentum reconstruction using xtrack data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tfs

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")
pytest.importorskip("matplotlib")

# import matplotlib.pyplot as plt
from turn_by_turn import read_tbt

from aba_optimiser.config import POSITION_STD_DEV
from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.measurements.create_datafile import convert_measurements
from aba_optimiser.momentum_recon.transverse import calculate_pz


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))

def _add_var_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["var_x"] = POSITION_STD_DEV**2
    df["var_y"] = POSITION_STD_DEV**2
    return df


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(data_dir, model_dir_b1):
    tracking_path = data_dir / "tracking"
    tws = tfs.read(model_dir_b1 / "twiss_ac.dat")
    # rename all colmnms to lower case for consistency
    tws.columns = [col.lower() for col in tws.columns]
    tws = tws.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws.headers = {k.lower(): v for k, v in tws.headers.items()}
    tws = tws.set_index("name")

    truth = pd.read_parquet(tracking_path / "true_data.parquet")

    noisy_files = [read_tbt(tracking_path / f"acd_errs_noisy_{i}.sdds") for i in range(3)]
    noisy_files = convert_measurements(noisy_files)

    noisy_result = calculate_pz(
        orig_data=_add_var_columns(noisy_files[0]), inject_noise=False, tws=tws, info=False
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    names_left = noisy_result["name"].unique().tolist()
    truth = truth[truth["name"].isin(names_left)]

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(meas_df=noisy_files[0])
    cleaned_df = _add_var_columns(cleaned_df)
    cleaned_noise_result = calculate_pz(
        orig_data=cleaned_df, inject_noise=False, tws=tws, info=False
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    # Reduce truth to the only avalilable BPMs

    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_noisy = _rmse(
        merged_noisy["px"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = _rmse(
        merged_noisy["py"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = _rmse(
        merged_cleaned["px"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = _rmse(
        merged_cleaned["py"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    # # Check noisy within expected bounds
    assert px_rmse_noisy < 4e-6
    assert py_rmse_noisy < 3e-6
    # Check cleaned is better than noisy
    assert px_rmse_cleaned < 3.2e-6
    assert py_rmse_cleaned < 3e-7
