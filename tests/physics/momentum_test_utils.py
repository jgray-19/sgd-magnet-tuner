"""Shared utilities for momentum reconstruction integration tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tfs
from omc3.optics_measurements.constants import (
    ERR,
    EXT,
    ORBIT,
    ORBIT_NAME,
)

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.momentum_recon import inject_noise_xy_inplace

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    import xtrack as xt

LOGGER = logging.getLogger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))

def xsuite_to_ngtws(tbl: xt.Table) -> pd.DataFrame:
    """Convert xsuite twiss table to ngtws format DataFrame.

    Args:
        line: xsuite Line object containing the twiss table.

    Returns:
        DataFrame in ngtws format.
    """
    df = tbl.to_pandas()
    df["beta11"] = df["betx"]
    df["beta22"] = df["bety"]
    df["alfa11"] = df["alfx"]
    df["alfa22"] = df["alfy"]
    df["mu1"] = df["mux"]
    df["mu2"] = df["muy"]
    df = tfs.TfsDataFrame(
        df,
        headers={"q1": tbl.qx, "q2": tbl.qy},
    )
    # remove
    df["name"] = df["name"].str.upper()  # ty:ignore[unresolved-attribute]
    df = df.set_index("name")
    bpm_names = df[df.index.str.match(r"^BPM.*\.B1$")].index.tolist()
    return df[df.index.isin(bpm_names)]


def get_truth(tracking_df: pd.DataFrame, tws: pd.DataFrame) -> pd.DataFrame:
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
    """
    df = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    # Ensure only BPMs present in twiss are included
    return df[df["name"].isin(tws.index)]


def verify_pz_reconstruction(
    tracking_df,
    truth: pd.DataFrame,
    tws: pd.DataFrame,
    calculate_pz_func: Callable[..., pd.DataFrame],  # Assuming return type is Any; adjust if needed
    px_nonoise_max: float,
    py_nonoise_max: float,
    px_noisy_min: float,
    px_noisy_max: float,
    py_noisy_min: float,
    py_noisy_max: float,
    px_cleaned_max: float,
    py_cleaned_max: float,
    rng_seed: int = 42,
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
    px_nonoise_max : float
        Maximum acceptable RMSE for nonoise px reconstruction.
    py_nonoise_max : float
        Maximum acceptable RMSE for nonoise py reconstruction.
    px_noisy_min : float
        Minimum expected RMSE for noisy px.
    px_noisy_max : float
        Maximum acceptable RMSE for noisy px.
    py_noisy_min : float
        Minimum expected RMSE for noisy py.
    py_noisy_max : float
        Maximum acceptable RMSE for noisy py.
    px_cleaned_max : float
        Maximum acceptable RMSE for SVD-cleaned px.
    py_cleaned_max : float
        Maximum acceptable RMSE for SVD-cleaned py.
    py_divisor : float
        Divisor to verify SVD improvement for py.
    rng_seed : int
        Random seed for noise generation.
    """
    no_noise_result = calculate_pz_func(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(rng_seed)
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(
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
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_noise_result = calculate_pz_func(
        cleaned_df,
        tws=tws,
        inject_noise=False,
        info=True,
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

    px_rmse_nonoise = rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_nonoise = rmse(
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

    LOGGER.info(
        f"PX RMSE no noise: {px_rmse_nonoise:.2e}, noisy: {px_rmse_noisy:.2e}, cleaned: {px_rmse_cleaned:.2e}"
    )
    LOGGER.info(
        f"PY RMSE no noise: {py_rmse_nonoise:.2e}, noisy: {py_rmse_noisy:.2e}, cleaned: {py_rmse_cleaned:.2e}"
    )

    assert px_rmse_nonoise < px_nonoise_max, f"PX no-noise RMSE {px_rmse_nonoise:.2e} should be < {px_nonoise_max:.2e}"
    assert py_rmse_nonoise < py_nonoise_max, f"PY no-noise RMSE {py_rmse_nonoise:.2e} should be < {py_nonoise_max:.2e}"
    assert px_noisy_min < px_rmse_noisy < px_noisy_max, f"PX noisy RMSE {px_rmse_noisy:.2e} should be in ({px_noisy_min:.2e}, {px_noisy_max:.2e})"
    assert py_noisy_min < py_rmse_noisy < py_noisy_max, f"PY noisy RMSE {py_rmse_noisy:.2e} should be in ({py_noisy_min:.2e}, {py_noisy_max:.2e})"
    # Check cleaned is better than noisy and meets absolute threshold
    assert px_rmse_cleaned < px_rmse_noisy, f"PX cleaned {px_rmse_cleaned:.2e} should be < noisy {px_rmse_noisy:.2e}"
    assert py_rmse_cleaned < py_rmse_noisy, f"PY cleaned {py_rmse_cleaned:.2e} should be < noisy {py_rmse_noisy:.2e}"
    assert px_rmse_cleaned < px_cleaned_max, f"PX cleaned RMSE {px_rmse_cleaned:.2e} should be < {px_cleaned_max:.2e}"
    assert py_rmse_cleaned < py_cleaned_max, f"PY cleaned RMSE {py_rmse_cleaned:.2e} should be < {py_cleaned_max:.2e}"

def add_error_to_orbit_measurement(fldr):
    for plane in ["x", "y"]:
        meas_file = fldr / f"{ORBIT_NAME}{plane}{EXT}"
        df = tfs.read(meas_file)
        df[f"{ERR}{ORBIT}{plane.upper()}"] = 1e-6
        tfs.write(meas_file, df)
