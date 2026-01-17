import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _find_bpm_phase(
    mu: pd.Series, tune: float, target: float, forward: bool, name: str
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

    return pd.DataFrame({name: names, "delta": delta}, index=mu.index)


def prev_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi/2 phase advance.

    Returns:
        DataFrame indexed by BPM_i with columns:

        - prev_bpm : name of BPM_j
        - delta    : (mu_i - mu_j - 0.25) signed error in turns
    """
    return _find_bpm_phase(mu, tune, 0.25, forward=False, name="prev_bpm")


def next_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi/2 phase advance.
    """
    return _find_bpm_phase(mu, tune, 0.25, forward=True, name="next_bpm")


def prev_bpm_to_pi(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi phase advance.

    Returns:
        DataFrame indexed by BPM_i with columns:

        - prev_bpm : name of BPM_j
        - delta    : (mu_i - mu_j - 0.5) signed error in turns
    """
    return _find_bpm_phase(mu, tune, 0.5, forward=False, name="prev_bpm")


def next_bpm_to_pi(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi phase advance.
    """
    return _find_bpm_phase(mu, tune, 0.5, forward=True, name="next_bpm")
