"""Shared utilities for momentum reconstruction integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.momentum_recon import inject_noise_xy

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import xtrack as xt


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.nanmean((predicted - actual) ** 2)))


def get_truth_and_twiss(
    baseline_line: xt.Line,
    tracking_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract truth momenta and prepare twiss from baseline line.

    Parameters
    ----------
    baseline_line : xtrack.Line
        The baseline accelerator line.
    tracking_df : pd.DataFrame
        The tracking DataFrame containing actual (true) momenta.

    Returns
    -------
    truth : pd.DataFrame
        DataFrame with true momenta (px_true, py_true).
    tws : pd.DataFrame
        Twiss parameters indexed by BPM names.
    """
    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    # Convert twiss to expected format
    ng = baseline_line.to_madng()
    ng["tws", "flw"] = ng.twiss(sequence=ng.seq, coupling=True)
    tws: pd.DataFrame = (
        ng.tws.to_df()
        .set_index("name")
        .rename(index=str.upper)
        .loc[lambda df: df.index.str.contains("BPM")]
    )

    return truth, tws


def verify_pz_reconstruction(
    tracking_df,
    truth: pd.DataFrame,
    tws: pd.DataFrame,
    calculate_pz_func: Callable[..., pd.DataFrame],  # Assuming return type is Any; adjust if needed
    px_clean_max: float,
    py_clean_max: float,
    px_noisy_min: float,
    px_noisy_max: float,
    py_noisy_min: float,
    py_noisy_max: float,
    px_divisor: float,
    py_divisor: float,
    rng_seed: int = 42,
    subtract_mean: bool = True,
):
    """Verify momentum reconstruction with noise and SVD cleaning.

    Tests three scenarios: clean data, noisy data, and SVD-cleaned data.
    Verifies that: (1) clean reconstruction meets accuracy thresholds,
    (2) noisy reconstruction degrades in expected range, and
    (3) SVD cleaning significantly improves reconstruction.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        The tracking data containing measurements.
    truth : pd.DataFrame
        The true momentum values (px_true, py_true).
    tws : tfs.TfsDataFrame
        Twiss parameters.
    calculate_pz_func : callable
        Function to calculate momentum (e.g., calculate_pz or calculate_transverse_pz).
    px_clean_max : float
        Maximum acceptable RMSE for clean px reconstruction.
    py_clean_max : float
        Maximum acceptable RMSE for clean py reconstruction.
    px_noisy_min : float or str
        Minimum expected RMSE for noisy px (or "px_rmse_clean" to use clean value).
    px_noisy_max : float
        Maximum acceptable RMSE for noisy px.
    py_noisy_min : float or str
        Minimum expected RMSE for noisy py (or "py_rmse_clean" to use clean value).
    py_noisy_max : float
        Maximum acceptable RMSE for noisy py.
    px_divisor : float
        Divisor to verify SVD improvement for px.
    py_divisor : float
        Divisor to verify SVD improvement for py.
    rng_seed : int
        Random seed for noise generation.
    subtract_mean : bool
        Whether to subtract mean in the calculation function.
    """
    no_noise_result = calculate_pz_func(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(rng_seed)
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy(
        noisy_df,
        tracking_df,
        rng,
        low_noise_bpms=[],
        noise_std=1e-4,
    )
    noisy_result = calculate_pz_func(
        noisy_df,
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_noise_result = calculate_pz_func(
        cleaned_df,
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_clean = rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_clean = rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    assert px_rmse_clean < px_clean_max
    assert py_rmse_clean < py_clean_max
    if isinstance(px_noisy_min, str) and px_noisy_min == "px_rmse_clean":
        px_noisy_min_val = px_rmse_clean
    else:
        px_noisy_min_val = px_noisy_min
    if isinstance(py_noisy_min, str) and py_noisy_min == "py_rmse_clean":
        py_noisy_min_val = py_rmse_clean
    else:
        py_noisy_min_val = py_noisy_min
    assert px_noisy_min_val < px_rmse_noisy < px_noisy_max
    assert py_noisy_min_val < py_rmse_noisy < py_noisy_max
    # Check cleaned is better than noisy
    assert px_rmse_cleaned < px_rmse_noisy / px_divisor
    assert py_rmse_cleaned < py_rmse_noisy / py_divisor
