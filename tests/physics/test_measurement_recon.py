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

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.measurements.create_datafile import convert_measurements
from aba_optimiser.momentum_recon.transverse import (
    calculate_pz,
)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(data_dir, sequence_file):
    # optics_path = data_dir / "optics"
    tracking_path = data_dir / "tracking"
    model_dir = data_dir / "model"
    tws = tfs.read(model_dir / "twiss_ac.dat")
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

    noisy_files = [
        read_tbt(tracking_path / f"acd_errs_noisy_{i}.sdds") for i in range(3)
    ]
    noisy_files = convert_measurements(noisy_files)
    for df in noisy_files:
        # Add the px_true and py_true columns for comparison later
        df["x_weight"] = 1.0
        df["y_weight"] = 1.0

    noisy_result = calculate_pz(
        orig_data=noisy_files[0], inject_noise=False, tws=tws, info=False
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    names_left = noisy_result["name"].unique().tolist()
    truth = truth[truth["name"].isin(names_left)]

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(meas_df=noisy_files[0])
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

    # Plot phase space for the 10th BPM
    # bpm_names = sorted(names_left)
    # if len(bpm_names) > 9:
    #     bpm_name = bpm_names[9]
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #     datasets = [
    #         ("Truth", truth, "px", "py"),
    #         ("Noisy", merged_noisy, "px_calc", "py_calc"),
    #         ("Cleaned", merged_cleaned, "px_calc", "py_calc"),
    #     ]
    #     colors = ["blue", "red", "green", "orange"]
    #     for (label, df, px_col, py_col), color in zip(datasets, colors):
    #         df_bpm = df[df["name"] == bpm_name]
    #         axes[0].scatter(
    #             df_bpm["x"], df_bpm[px_col], s=1, color=color, label=label, alpha=0.7
    #         )
    #         axes[1].scatter(
    #             df_bpm["y"], df_bpm[py_col], s=1, color=color, label=label, alpha=0.7
    #         )
    #     axes[0].set_title(f"X vs Px for {bpm_name}")
    #     axes[0].set_xlabel("x")
    #     axes[0].set_ylabel("px")
    #     axes[0].legend()
    #     axes[1].set_title(f"Y vs Py for {bpm_name}")
    #     axes[1].set_xlabel("y")
    #     axes[1].set_ylabel("py")
    #     axes[1].legend()
    #     plt.tight_layout()
    #     plt.show()
