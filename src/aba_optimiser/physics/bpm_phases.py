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
    Find BPM pairs with phase advance closest to a target value.

    For each BPM_i, finds the nearest BPM_j in the specified direction
    (forward or backward) whose phase advance is close to the target.

    Selection criteria (in order of priority):
    1. Candidate must be within stable_width (0.125 rotations) of target phase
    2. If any candidate has excellent phase match (< 0.01 rotations), prefer those
    3. Among remaining candidates, select closest by directional distance
    4. If tied on distance, select best phase match

    Args:
        mu: BPM phase advances (rotations)
        tune: Machine tune (rotations per turn)
        target: Target phase advance (turns), e.g., 0.25 for π/2, 0.5 for π
        forward: If True, search forward; if False, search backward
        name: Column name for the matched BPM in output DataFrame
        mu_var: Optional phase variance for each BPM (rotations²)
        total_var: Optional total phase variance around ring (rotations²)

    Returns:
        DataFrame with columns:
            - {name}: Name of matched BPM
            - delta: Signed phase error (actual - target) in rotations
            - delta_err: Phase error uncertainty (rotations), if variance provided
    """
    v = mu.to_numpy(float)
    n = len(v)

    # Compute phase advance matrix: diff[i, j] = phase from BPM_i to BPM_j
    if forward:
        diff = (v.reshape(1, n) - v.reshape(n, 1) + tune) % tune
    else:
        diff = (v.reshape(n, 1) - v.reshape(1, n) + tune) % tune

    np.fill_diagonal(diff, np.nan)

    # Find best match for each BPM
    stable_width = 0.125  # Only consider candidates within ±0.125 rotations of target

    # Set max BPM distance based on target phase advance
    # With ~0.125 turns per BPM average, π/2 needs ~2 BPMs, π needs ~4 BPMs
    # Allow for local variations: π/2 uses 11 BPMs, π has no limit (filtered by stable_width)
    max_bpm_distance = 11 if abs(target - 0.25) < 0.01 else 20

    idx = np.full(n, -1, dtype=int)

    for i in range(n):
        row = diff[i, :]

        # Filter to candidates within stable region
        mask = np.abs((row - target + 0.5) % 1 - 0.5) <= stable_width
        mask[i] = False  # Exclude self
        candidates = np.where(mask)[0]

        if len(candidates) == 0:
            idx[i] = -1
            continue

        # Filter to candidates within max circular distance (if limit is set)
        circular_distance = np.minimum(np.abs(candidates - i), n - np.abs(candidates - i))
        within_distance = circular_distance <= max_bpm_distance
        candidates = candidates[within_distance]

        if len(candidates) == 0:
            idx[i] = -1
            continue

        # Calculate phase error and directional distance for each candidate
        phase_error = np.abs(row[candidates] - target)
        distances = (candidates - i) % n if forward else (i - candidates) % n

        # Prefer excellent phase matches (< 0.01 rotations) if any exist
        excellent_match = phase_error < 0.02
        if np.any(excellent_match):
            candidates = candidates[excellent_match]
            distances = distances[excellent_match]
            phase_error = phase_error[excellent_match]

        # Select closest by directional distance
        min_distance = np.min(distances)
        closest_indices = distances == min_distance

        if np.sum(closest_indices) == 1:
            # Single closest candidate
            idx[i] = candidates[closest_indices][0]
        else:
            # Multiple at same distance: pick best phase match
            best_idx = np.argmin(phase_error[closest_indices])
            idx[i] = candidates[closest_indices][best_idx]

    # Build output DataFrame
    delta = np.full(n, np.nan)
    names = np.full(n, None, dtype=object)
    for i in range(n):
        if idx[i] != -1:
            delta[i] = diff[i, idx[i]] - target
            names[i] = mu.index[idx[i]]

    out = pd.DataFrame({name: names, "delta": delta}, index=mu.index)

    # Add uncertainty if variance information provided
    if mu_var is not None and total_var is not None:
        var_arr = mu_var.to_numpy(float)
        i_arr = np.arange(n, dtype=int)
        j_arr = idx.astype(int)

        if forward:
            pair_var = _phase_pair_var_forward(var_arr, i_arr, j_arr, float(total_var))
        else:
            # Backward from i to j is forward from j to i
            pair_var = _phase_pair_var_forward(var_arr, j_arr, i_arr, float(total_var))

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
    Find previous BPM at π/2 phase advance.

    For each BPM_i, finds the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to π/2 (0.25 turns).

    Returns:
        DataFrame with columns:
            - prev_bpm: Name of matched previous BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
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
    Find next BPM at π/2 phase advance.

    For each BPM_i, finds the next BPM_j whose forward phase advance
    (mu_j - mu_i) is closest to π/2 (0.25 turns).

    Returns:
        DataFrame with columns:
            - next_bpm: Name of matched next BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
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
    Find previous BPM at π phase advance.

    For each BPM_i, finds the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to π (0.5 turns).

    Returns:
        DataFrame with columns:
            - prev_bpm: Name of matched previous BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
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
    Find next BPM at π phase advance.

    For each BPM_i, finds the next BPM_j whose forward phase advance
    (mu_j - mu_i) is closest to π (0.5 turns).

    Returns:
        DataFrame with columns:
            - next_bpm: Name of matched next BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
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
