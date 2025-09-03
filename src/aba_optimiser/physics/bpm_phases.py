import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def prev_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi/2 phase advance.
    Returns a DataFrame indexed by BPM_i with columns:
      - prev_bpm : name of BPM_j
      - delta    : (mu_i - mu_j - 0.25) signed error in turns
    """
    # keep for public API; implementation is delegated to `_bpm_to_pi_2`
    return _bpm_to_pi_2(mu, tune, forward=False, name="prev_bpm")


# 1) define your new forward-phase helper
def next_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi/2 phase advance.
    """
    # keep for public API; implementation is delegated to `_bpm_to_pi_2`
    return _bpm_to_pi_2(mu, tune, forward=True, name="next_bpm")


# internal helper to avoid duplicating forward/backward matrix logic
def _bpm_to_pi_2(mu: pd.Series, tune: float, forward: bool, name: str) -> pd.DataFrame:
    """
    Shared implementation. If `forward` is True computes (mu_j - mu_i) mod Q,
    otherwise computes (mu_i - mu_j) mod Q and finds, for each i, the j whose
    phase difference is closest to 0.25 turns (pi/2).
    Returns a DataFrame indexed by `mu.index` with columns [name, 'delta'].
    """
    v = mu.to_numpy(float)
    n = len(v)

    if forward:
        # forward differences mod Q: (mu_j - mu_i) ∈ [0, 1)
        diff = (v.reshape(1, n) - v.reshape(n, 1) + tune) % tune
    else:
        # backward differences mod Q: (mu_i - mu_j) ∈ [0, 1)
        diff = (v.reshape(n, 1) - v.reshape(1, n) + tune) % tune

    np.fill_diagonal(diff, np.nan)

    # pick j minimising |Δ_ij - target|. target = 0.25 (*2pi) -> pi/2
    idx = np.nanargmin(np.abs(diff - 0.25), axis=1)
    delta = diff[np.arange(n), idx] - 0.25
    names = mu.index[idx]

    return pd.DataFrame({name: names, "delta": delta}, index=mu.index)
