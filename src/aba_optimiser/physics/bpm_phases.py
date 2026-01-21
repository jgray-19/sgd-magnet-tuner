import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _phase_pair_var_forward(
    var_mu: np.ndarray, i: np.ndarray, j: np.ndarray, total_var: float
) -> np.ndarray:
    """
    Variance of forward phase advance from i -> j (turns^2), with wrap-around.
    i, j are integer arrays of equal length.
    """
    out = np.empty_like(var_mu[i], dtype=float)
    mask = j >= i
    out[mask] = var_mu[j[mask]] - var_mu[i[mask]]
    out[~mask] = (total_var - var_mu[i[~mask]]) + var_mu[j[~mask]]
    # Numerical safety (should not be negative, but floating noise can happen)
    return np.maximum(out, 0.0)


def _find_bpm_phase(
    mu: pd.Series,
    tune: float,
    target: float,
    forward: bool,
    name: str,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Shared implementation for finding BPM with phase advance closest to target.
    """
    v = mu.to_numpy(float)
    n = len(v)

    if forward:
        diff = (v.reshape(1, n) - v.reshape(n, 1) + tune) % tune
    else:
        diff = (v.reshape(n, 1) - v.reshape(1, n) + tune) % tune

    np.fill_diagonal(diff, np.nan)

    abs_diff = np.abs(diff - target)
    idx = np.full(n, -1, dtype=int)
    for i in range(n):
        row = abs_diff[i, :]
        min_val = np.nanmin(row)
        candidates = np.where(row == min_val)[0]
        distances = np.minimum(np.abs(candidates - i), n - np.abs(candidates - i))
        idx[i] = candidates[np.argmin(distances)]

    delta = diff[np.arange(n), idx] - target
    names = mu.index[idx]

    out = pd.DataFrame({name: names, "delta": delta}, index=mu.index)

    # Add delta_err if variance info is provided
    if mu_var is not None and total_var is not None:
        var_arr = mu_var.to_numpy(float)
        i = np.arange(n, dtype=int)
        j = idx.astype(int)

        if forward:
            pair_var = _phase_pair_var_forward(var_arr, i, j, float(total_var))
        else:
            # backward from i to j corresponds to forward from j to i
            pair_var = _phase_pair_var_forward(var_arr, j, i, float(total_var))

        out["delta_err"] = np.sqrt(pair_var)

    return out


def prev_bpm_to_pi_2(
    mu: pd.Series,
    tune: float,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi/2 phase advance.
    Returns a DataFrame indexed by BPM_i with columns:
      - prev_bpm : name of BPM_j
      - delta    : (mu_i - mu_j - 0.25) signed error in turns
      - delta_err: uncertainty in delta (turns), if variance provided
    """
    return _find_bpm_phase(
        mu,
        tune,
        0.25,
        forward=False,
        name="prev_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def next_bpm_to_pi_2(
    mu: pd.Series,
    tune: float,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi/2 phase advance.
    """
    return _find_bpm_phase(
        mu,
        tune,
        0.25,
        forward=True,
        name="next_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def prev_bpm_to_pi(
    mu: pd.Series,
    tune: float,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi phase advance.
    Returns a DataFrame indexed by BPM_i with columns:
      - prev_bpm : name of BPM_j
      - delta    : (mu_i - mu_j - 0.5) signed error in turns
    """
    return _find_bpm_phase(
        mu,
        tune,
        0.5,
        forward=False,
        name="prev_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def next_bpm_to_pi(
    mu: pd.Series,
    tune: float,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi phase advance.
    """
    return _find_bpm_phase(
        mu,
        tune,
        0.5,
        forward=True,
        name="next_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )
