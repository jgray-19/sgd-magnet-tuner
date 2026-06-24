"""Utilities for computing averaged closed-orbit measurement data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weighted_mean_and_variance(
    sub: pd.DataFrame, value_col: str, var_col: str
) -> tuple[float, float]:
    """Compute inverse-variance weighted mean and its variance.

    Args:
        sub: DataFrame subset for a single BPM
        value_col: Column name for values to average
        var_col: Column name for variances

    Returns:
        Tuple of (weighted_mean, variance_of_mean)
    """
    vals = sub[value_col].to_numpy()
    vars_ = sub[var_col].to_numpy()
    mask = np.isfinite(vals) & np.isfinite(vars_) & (vars_ > 0)
    vals = vals[mask]
    vars_ = vars_[mask]

    if vals.size == 0:
        mu = float(sub[value_col].mean())
        n = sub[value_col].count()
        if n >= 2:
            v_unw = float(np.var(sub[value_col].to_numpy(), ddof=1))
            var_mean = v_unw / n
        else:
            var_mean = np.nan
        return mu, var_mean

    w = 1.0 / vars_
    sum_w = float(np.sum(w))
    mu = float(np.sum(w * vals) / sum_w)
    return mu, 1.0 / sum_w


def compute_three_turn_averages(pzs: pd.DataFrame) -> pd.DataFrame:
    """Average each BPM's measurements and replicate them across three turns.

    Per-BPM weighted means (and the variance of the mean, stored for downstream
    weighting) are computed for each observable, then duplicated over turns 1-3
    to match the tracking-data layout.
    """
    rows = []
    for name, sub in pzs.groupby("name"):
        mu_x,  vm_x  = compute_weighted_mean_and_variance(sub, "x",  "var_x")
        mu_y,  vm_y  = compute_weighted_mean_and_variance(sub, "y",  "var_y")
        mu_px, vm_px = compute_weighted_mean_and_variance(sub, "px", "var_px")
        mu_py, vm_py = compute_weighted_mean_and_variance(sub, "py", "var_py")
        rows.append({
            "name": name,
            "x": mu_x, "y": mu_y, "px": mu_px, "py": mu_py,
            "var_x": vm_x, "var_y": vm_y, "var_px": vm_px, "var_py": vm_py,
        })

    averaged = pd.DataFrame(rows)
    new_rows = [
        {**row.to_dict(), "turn": t}
        for t in [1, 2, 3]
        for _, row in averaged.iterrows()
    ]
    new_df = pd.DataFrame(new_rows)
    new_df["name"] = new_df["name"].astype("category")
    new_df["turn"] = new_df["turn"].astype("int32")
    return new_df
