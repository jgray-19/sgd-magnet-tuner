"""
Transverse Momentum Calculation Module

This module provides functionality for estimating transverse momenta (px, py) of particles
in a particle accelerator using BPM (Beam Position Monitor) data. It uses phase-space
relationships between BPM pairs separated by approximately π/2 in betatron phase to
compute momentum components.

The main approach involves:
1. Identifying BPM pairs at π/2 phase difference
2. Computing momenta from position differences using lattice optics
3. Applying weighted averaging for improved accuracy
4. Optional noise injection and mean subtraction for robustness

Key functions:
- calculate_pz: Main entry point for momentum estimation
- Various helper functions for lattice handling, neighbor finding, and diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from aba_optimiser.config import (
    FILE_COLUMNS,
    POSITION_STD_DEV,
    SEQUENCE_FILE,
)
from aba_optimiser.mad.mad_interface import OptimisationMadInterface
from aba_optimiser.physics.bpm_phases import next_bpm_to_pi_2, prev_bpm_to_pi_2
from aba_optimiser.physics.dpp_calculation import get_mean_dpp

LOGGER = logging.getLogger(__name__)
OUT_COLS = list(FILE_COLUMNS)

if TYPE_CHECKING:
    import pandas as pd
    import tfs


# ---------- Small data containers ----------
@dataclass(frozen=True)
class LatticeMaps:
    """
    Container for lattice optics parameters mapped by BPM name.

    This dataclass stores various beta functions and alpha functions
    for both horizontal (x) and vertical (y) planes, which are essential
    for transverse momentum calculations.

    Attributes:
        sqrt_betax: Square root of horizontal beta function by BPM name
        sqrt_betay: Square root of vertical beta function by BPM name
        betax: Horizontal beta function by BPM name
        betay: Vertical beta function by BPM name
        alfax: Horizontal alpha function by BPM name
        alfay: Vertical alpha function by BPM name
    """

    sqrt_betax: dict
    sqrt_betay: dict
    betax: dict
    betay: dict
    alfax: dict
    alfay: dict
    dx: dict
    # No dy as we assume vertical dispersion is zero
    dpx: dict
    # No dpy as we assume vertical dispersion is zero


# ---------- Helpers: generic ----------
def _validate_input(df: tfs.TfsDataFrame) -> tuple[bool, bool]:
    """
    Validate input DataFrame for required columns and momentum presence.

    Args:
        df: Input DataFrame containing BPM data

    Returns:
        Tuple of (has_px, has_py) indicating presence of momentum columns

    Raises:
        ValueError: If required columns are missing
    """
    required = {"name", "turn", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    return ("px" in df.columns), ("py" in df.columns)


def _get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """
    Get a numpy random number generator, creating default if None provided.

    Args:
        rng: Optional random number generator

    Returns:
        Numpy random number generator instance
    """
    return rng or np.random.default_rng()


def _inject_noise_xy(
    df: tfs.TfsDataFrame,
    orig_df: tfs.TfsDataFrame,
    rng: np.random.Generator,
    low_noise_bpms: Sequence[str],
) -> None:
    """
    Inject Gaussian noise into x/y position measurements.

    Modifies the DataFrame in-place by adding noise to position columns.
    Certain BPMs can be designated as low-noise for reduced uncertainty.

    Args:
        df: DataFrame to modify with noise
        orig_df: Original DataFrame with true positions
        rng: Random number generator
        low_noise_bpms: List of BPM names with reduced noise level
    """
    n = len(df)
    LOGGER.debug("Adding Gaussian noise: std=%g", POSITION_STD_DEV)
    noise_x = rng.normal(0.0, POSITION_STD_DEV, size=n)
    noise_y = rng.normal(0.0, POSITION_STD_DEV, size=n)
    if low_noise_bpms:
        mask = df["name"].isin(low_noise_bpms).to_numpy()
        if mask.any():
            low_n = int(mask.sum())
            LOGGER.debug(
                "Reduced-noise at %d rows (%d BPMs)", low_n, len(low_noise_bpms)
            )
            noise_x[mask] = rng.normal(0.0, POSITION_STD_DEV / 10.0, size=low_n)
            noise_y[mask] = rng.normal(0.0, POSITION_STD_DEV / 10.0, size=low_n)
    df["x"] = orig_df["x"] + noise_x
    df["y"] = orig_df["y"] + noise_y


# ---------- Helpers: twiss / lattice ----------
def _ensure_twiss(tws: tfs.TfsDataFrame | None, info: bool) -> tfs.TfsDataFrame:
    """
    Ensure Twiss parameters are available, computing them if necessary.

    Args:
        tws: Existing Twiss DataFrame or None
        info: Whether to print tune information

    Returns:
        Twiss DataFrame with lattice optics
    """
    if tws is not None:
        return tws
    mad = OptimisationMadInterface(
        SEQUENCE_FILE, bpm_pattern="BPM", use_real_strengths=False
    )
    tws = mad.run_twiss()
    if info:
        print("Found tunes:", tws.q1, tws.q2)
    return tws


def _lattice_maps(tws: tfs.TfsDataFrame) -> LatticeMaps:
    """
    Extract lattice optics into convenient dictionary mappings.

    Args:
        tws: Twiss DataFrame

    Returns:
        LatticeMaps object with optics by BPM name
    """
    sqrt_betax = np.sqrt(tws["beta11"])
    sqrt_betay = np.sqrt(tws["beta22"])
    return LatticeMaps(
        sqrt_betax=sqrt_betax.to_dict(),
        sqrt_betay=sqrt_betay.to_dict(),
        betax=tws["beta11"].to_dict(),
        betay=tws["beta22"].to_dict(),
        alfax=tws["alfa11"].to_dict(),
        alfay=tws["alfa22"].to_dict(),
        dx=tws["dx"].to_dict(),
        dpx=tws["dpx"].to_dict(),
    )


def _neighbour_tables(
    tws: tfs.TfsDataFrame,
) -> tuple[tfs.TfsDataFrame, tfs.TfsDataFrame, tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """
    Create tables of BPM neighbors separated by π/2 in phase.

    Args:
        tws: Twiss DataFrame

    Returns:
        Tuple of (prev_x, prev_y, next_x, next_y) DataFrames
    """
    prev_x = prev_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
        columns={"prev_bpm": "prev_bpm_x", "delta": "delta_x"}
    )
    prev_y = prev_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
        columns={"prev_bpm": "prev_bpm_y", "delta": "delta_y"}
    )
    next_x = next_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
        columns={"next_bpm": "next_bpm_x", "delta": "delta_x"}
    )
    next_y = next_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
        columns={"next_bpm": "next_bpm_y", "delta": "delta_y"}
    )
    return prev_x, prev_y, next_x, next_y


def _attach_lattice_columns(df: tfs.TfsDataFrame, maps: LatticeMaps) -> None:
    """
    Add lattice optics columns to DataFrame using BPM name mapping.

    Args:
        df: DataFrame to modify
        maps: LatticeMaps with optics by BPM name
    """
    df["sqrt_betax"] = df["name"].map(maps.sqrt_betax)
    df["sqrt_betay"] = df["name"].map(maps.sqrt_betay)
    df["betax"] = df["name"].map(maps.betax)
    df["betay"] = df["name"].map(maps.betay)
    df["alfax"] = df["name"].map(maps.alfax)
    df["alfay"] = df["name"].map(maps.alfay)
    df["dx"] = df["name"].map(maps.dx)
    df["dpx"] = df["name"].map(maps.dpx)


# ---------- Helpers: neighbor coordinates ----------
def _compute_turn_wraps(
    data_p: tfs.TfsDataFrame, data_n: tfs.TfsDataFrame, bpm_index: dict
) -> tuple:
    """
    Compute turn numbers for neighbor BPMs, handling ring wrap-around.

    Args:
        data_p: DataFrame for previous BPM calculations
        data_n: DataFrame for next BPM calculations
        bpm_index: Mapping of BPM names to indices

    Returns:
        Tuple of turn adjustments for x/y planes
    """
    cur_i_p = data_p["name"].map(bpm_index)
    prev_ix = data_p["prev_bpm_x"].map(bpm_index)
    prev_iy = data_p["prev_bpm_y"].map(bpm_index)

    cur_i_n = data_n["name"].map(bpm_index)
    next_ix = data_n["next_bpm_x"].map(bpm_index)
    next_iy = data_n["next_bpm_y"].map(bpm_index)

    turn_x_p = data_p["turn"] - (cur_i_p < prev_ix).astype(np.int16)
    turn_y_p = data_p["turn"] - (cur_i_p < prev_iy).astype(np.int16)
    turn_x_n = data_n["turn"] + (cur_i_n > next_ix).astype(np.int16)
    turn_y_n = data_n["turn"] + (cur_i_n > next_iy).astype(np.int16)
    return turn_x_p, turn_y_p, turn_x_n, turn_y_n


def _merge_neighbor_coords(
    data_p: tfs.TfsDataFrame,
    data_n: tfs.TfsDataFrame,
    turn_x_p: pd.Series,
    turn_y_p: pd.Series,
    turn_x_n: pd.Series,
    turn_y_n: pd.Series,
):
    """
    Merge coordinates from neighboring BPMs into main DataFrames.

    Args:
        data_p: Previous BPM DataFrame
        data_n: Next BPM DataFrame
        turn_x_p, turn_y_p, turn_x_n, turn_y_n: Turn adjustments

    Returns:
        Tuple of modified (data_p, data_n)
    """
    coords_p = data_p[["turn", "name", "x", "y"]]
    coords_n = data_n[["turn", "name", "x", "y"]]

    coords_x_p = coords_p.rename(
        columns={"turn": "turn_x_p", "name": "prev_bpm_x", "x": "prev_x"}
    )[["turn_x_p", "prev_bpm_x", "prev_x"]]
    coords_y_p = coords_p.rename(
        columns={"turn": "turn_y_p", "name": "prev_bpm_y", "y": "prev_y"}
    )[["turn_y_p", "prev_bpm_y", "prev_y"]]
    coords_x_n = coords_n.rename(
        columns={"turn": "turn_x_n", "name": "next_bpm_x", "x": "next_x"}
    )[["turn_x_n", "next_bpm_x", "next_x"]]
    coords_y_n = coords_n.rename(
        columns={"turn": "turn_y_n", "name": "next_bpm_y", "y": "next_y"}
    )[["turn_y_n", "next_bpm_y", "next_y"]]

    data_p["turn_x_p"] = turn_x_p
    data_p["turn_y_p"] = turn_y_p
    data_n["turn_x_n"] = turn_x_n
    data_n["turn_y_n"] = turn_y_n

    data_p = data_p.merge(
        coords_x_p, on=["turn_x_p", "prev_bpm_x"], how="left", copy=False
    )
    data_p = data_p.merge(
        coords_y_p, on=["turn_y_p", "prev_bpm_y"], how="left", copy=False
    )
    data_n = data_n.merge(
        coords_x_n, on=["turn_x_n", "next_bpm_x"], how="left", copy=False
    )
    data_n = data_n.merge(
        coords_y_n, on=["turn_y_n", "next_bpm_y"], how="left", copy=False
    )

    # Default missing neighbors to 0 (rare)
    for frm, col in (
        (data_p, "prev_x"),
        (data_p, "prev_y"),
        (data_n, "next_x"),
        (data_n, "next_y"),
    ):
        frm[col] = frm[col].fillna(0)

    data_p.drop(columns=["turn_x_p", "turn_y_p"], inplace=True)
    data_n.drop(columns=["turn_x_n", "turn_y_n"], inplace=True)
    return data_p, data_n


# ---------- Helpers: momentum from next BPM ----------
def _momenta_from_next(data_n: tfs.TfsDataFrame, dpp_est: float) -> tfs.TfsDataFrame:
    """
    Calculate transverse momenta using the NEXT BPM at approximately +π/2 phase difference.

    This function solves for the transverse momenta at the current BPM using measurements
    from the next BPM in the ring. The calculation accounts for dispersion effects and
    uses the phase-space transfer matrix relationship between the two BPMs.

    The formula derives from the transfer matrix elements:
    - Position transfer: x2 = cos(φ) x1 + sin(φ) (px1 β1 + α1 x1)
    - Momentum transfer: px2 = -sin(φ)/β2 x1 + cos(φ) px1 - α2 sin(φ)/β2 x1

    Rearranging for px1 gives the implemented formula.

    Args:
        data_n: DataFrame containing current and next BPM data with lattice optics
        dpp_est: Estimated relative momentum deviation (dp/p) for dispersion correction

    Returns:
        Modified DataFrame with added 'px' and 'py' momentum columns
    """
    # Validate required columns for next BPM optics
    required_cols = ["sqrt_betax_n", "sqrt_betay_n", "dx_next"]
    missing = [col for col in required_cols if col not in data_n.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns for next BPM calculation: {missing}"
        )

    # Extract arrays for vectorised calculations
    x_current = data_n["x"].to_numpy()
    y_current = data_n["y"].to_numpy()
    x_next = data_n["next_x"].to_numpy()
    y_next = data_n["next_y"].to_numpy()

    # Lattice optics at current BPM
    sqrt_beta_x = data_n["sqrt_betax"].to_numpy()
    sqrt_beta_y = data_n["sqrt_betay"].to_numpy()
    alpha_x = data_n["alfax"].to_numpy()
    alpha_y = data_n["alfay"].to_numpy()

    # Lattice optics at next BPM
    sqrt_beta_x_next = data_n["sqrt_betax_n"].to_numpy()
    sqrt_beta_y_next = data_n["sqrt_betay_n"].to_numpy()

    # Dispersion corrections
    dx_current = data_n["dx"].to_numpy()
    dx_next = data_n["dx_next"].to_numpy()
    dpx_current = data_n["dpx"].to_numpy()

    # Phase advances in radians (delta is in turns)
    phi_x = data_n["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_n["delta_y"].to_numpy() * 2 * np.pi

    # Normalised positions (remove dispersion contribution)
    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_next_norm = (x_next - dpp_est * dx_next) / sqrt_beta_x_next
    y_current_norm = y_current / sqrt_beta_y
    y_next_norm = y_next / sqrt_beta_y_next

    # Trigonometric functions for transfer matrix
    cos_phi_x = np.cos(phi_x)
    cos_phi_y = np.cos(phi_y)
    tan_phi_x = np.tan(phi_x)
    tan_phi_y = np.tan(phi_y)

    # Secant (1/cos) with numerical stability
    sec_phi_x = np.divide(1.0, cos_phi_x, where=cos_phi_x != 0.0)
    sec_phi_y = np.divide(1.0, cos_phi_y, where=cos_phi_y != 0.0)

    # Calculate momenta using transfer matrix inversion
    # px = [x_next_norm * sec_phi_x + x_current_norm * (tan_phi_x - alpha_x)] / sqrt_beta_x + dpx_current * dpp_est
    px = (
        x_next_norm * sec_phi_x + x_current_norm * (tan_phi_x - alpha_x)
    ) / sqrt_beta_x + dpx_current * dpp_est
    py = (
        y_next_norm * sec_phi_y + y_current_norm * (tan_phi_y - alpha_y)
    ) / sqrt_beta_y

    # Store results
    data_n["px"] = px
    data_n["py"] = py

    return data_n


def _momenta_from_prev(data_p: tfs.TfsDataFrame, dpp_est: float) -> tfs.TfsDataFrame:
    """
    Calculate transverse momenta using the PREVIOUS BPM at approximately -π/2 phase difference.

    This function solves for the transverse momenta at the current BPM using measurements
    from the previous BPM in the ring. The calculation accounts for dispersion effects and
    uses the phase-space transfer matrix relationship between the two BPMs.

    The formula derives from the transfer matrix elements for the reverse direction:
    - Position transfer: x1 = cos(φ) x2 - sin(φ) (px2 β2 + α2 x2)
    - Momentum transfer: px1 = sin(φ)/β1 x2 + cos(φ) px2 + α1 sin(φ)/β1 x2

    Rearranging for px2 gives the implemented formula.

    Args:
        data_p: DataFrame containing current and previous BPM data with lattice optics
        dpp_est: Estimated relative momentum deviation (dp/p) for dispersion correction

    Returns:
        Modified DataFrame with added 'px' and 'py' momentum columns
    """
    # Validate required columns for previous BPM optics
    required_cols = ["sqrt_betax_p", "sqrt_betay_p", "dx_prev"]
    missing = [col for col in required_cols if col not in data_p.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns for previous BPM calculation: {missing}"
        )

    # Extract arrays for vectorised calculations
    x_current = data_p["x"].to_numpy()
    y_current = data_p["y"].to_numpy()
    x_prev = data_p["prev_x"].to_numpy()
    y_prev = data_p["prev_y"].to_numpy()

    # Lattice optics at current BPM
    sqrt_beta_x = data_p["sqrt_betax"].to_numpy()
    sqrt_beta_y = data_p["sqrt_betay"].to_numpy()
    alpha_x = data_p["alfax"].to_numpy()
    alpha_y = data_p["alfay"].to_numpy()

    # Lattice optics at previous BPM
    sqrt_beta_x_prev = data_p["sqrt_betax_p"].to_numpy()
    sqrt_beta_y_prev = data_p["sqrt_betay_p"].to_numpy()

    # Dispersion corrections
    dx_current = data_p["dx"].to_numpy()
    dx_prev = data_p["dx_prev"].to_numpy()
    dpx_current = data_p["dpx"].to_numpy()

    # Phase advances in radians (delta is in turns)
    phi_x = data_p["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_p["delta_y"].to_numpy() * 2 * np.pi

    # Normalised positions (remove dispersion contribution)
    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_prev_norm = (x_prev - dpp_est * dx_prev) / sqrt_beta_x_prev
    y_current_norm = y_current / sqrt_beta_y
    y_prev_norm = y_prev / sqrt_beta_y_prev

    # Trigonometric functions for transfer matrix
    cos_phi_x = np.cos(phi_x)
    sin_phi_x = np.sin(phi_x)
    tan_phi_x = np.tan(phi_x)
    cos_phi_y = np.cos(phi_y)
    sin_phi_y = np.sin(phi_y)
    tan_phi_y = np.tan(phi_y)

    # Calculate momenta using transfer matrix inversion
    # px = -[x_prev_norm * (cos_phi_x + sin_phi_x * tan_phi_x) + x_current_norm * (tan_phi_x + alpha_x)] / sqrt_beta_x + dpx_current * dpp_est
    px = (
        -(
            x_prev_norm * (cos_phi_x + sin_phi_x * tan_phi_x)
            + x_current_norm * (tan_phi_x + alpha_x)
        )
        / sqrt_beta_x
        + dpx_current * dpp_est
    )
    py = (
        -(
            y_prev_norm * (cos_phi_y + sin_phi_y * tan_phi_y)
            + y_current_norm * (tan_phi_y + alpha_y)
        )
        / sqrt_beta_y
    )

    # Store results
    data_p["px"] = px
    data_p["py"] = py

    return data_p


def _sync_endpoints(data_p: tfs.TfsDataFrame, data_n: tfs.TfsDataFrame) -> None:
    """
    Synchronise momentum values at ring endpoints for consistency.

    Matches the last point of previous-based calculation with first point
    of next-based calculation, and vice versa.

    Args:
        data_p: Previous BPM DataFrame
        data_n: Next BPM DataFrame
    """
    # match original behavior (edge constraints)
    data_n.iloc[-1, data_n.columns.get_loc("px")] = data_p.iloc[
        -1, data_p.columns.get_loc("px")
    ]
    data_n.iloc[-1, data_n.columns.get_loc("py")] = data_p.iloc[
        -1, data_p.columns.get_loc("py")
    ]
    data_p.iloc[0, data_p.columns.get_loc("px")] = data_n.iloc[
        0, data_n.columns.get_loc("px")
    ]
    data_p.iloc[0, data_p.columns.get_loc("py")] = data_n.iloc[
        0, data_n.columns.get_loc("py")
    ]


# ---------- Helpers: weighting & diagnostics ----------
def _weights(
    psi: np.ndarray, inv_beta1: np.ndarray, inv_beta2: np.ndarray
) -> np.ndarray:
    """
    Calculate 1/f weighting factors for momentum averaging.

    Implements the weighting scheme from literature (doi:10.2172/813017 §2.2)
    where smaller f corresponds to larger weight. The factor f
    depends on beta functions and phase differences.

    Args:
        psi: Phase differences in radians
        inv_beta1: Inverse beta function at first BPM
        inv_beta2: Inverse beta function at second BPM

    Returns:
        Array of weights (higher values for more reliable measurements)
    """
    pref = 1.0 / (np.sqrt(2.0) * np.abs(np.sin(psi)))
    inside = (
        inv_beta1
        + inv_beta2
        + np.sqrt(
            inv_beta1**2
            + inv_beta2**2
            + 2.0 * inv_beta1 * inv_beta2 * np.cos(2.0 * psi)
        )
    )
    f = pref * np.sqrt(inside)
    return 1.0 / f


def _weighted_average(data_p, data_n, beta_x, beta_y) -> tfs.TfsDataFrame:
    """
    Compute weighted average of momenta from prev/next BPM calculations.

    Combines the two independent momentum estimates using reliability weights
    based on lattice optics and phase relationships.

    Args:
        data_p: DataFrame with previous BPM momentum estimates
        data_n: DataFrame with next BPM momentum estimates
        beta_x: Beta function mapping for horizontal plane
        beta_y: Beta function mapping for vertical plane

    Returns:
        DataFrame with weighted average momenta
    """
    data_avg = data_p.copy(deep=True)

    psi_x_prev = (data_p["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_prev = (data_p["delta_y"].to_numpy() + 0.25) * 2 * np.pi
    psi_x_next = (data_n["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_next = (data_n["delta_y"].to_numpy() + 0.25) * 2 * np.pi

    inv_beta_x = 1.0 / data_p["betax"].to_numpy()
    inv_beta_y = 1.0 / data_p["betay"].to_numpy()
    inv_beta_p_x = 1.0 / data_p["prev_bpm_x"].map(beta_x).to_numpy()
    inv_beta_p_y = 1.0 / data_p["prev_bpm_y"].map(beta_y).to_numpy()
    inv_beta_n_x = 1.0 / data_n["next_bpm_x"].map(beta_x).to_numpy()
    inv_beta_n_y = 1.0 / data_n["next_bpm_y"].map(beta_y).to_numpy()

    wpx_prev = _weights(psi_x_prev, inv_beta_p_x, inv_beta_x)
    wpy_prev = _weights(psi_y_prev, inv_beta_p_y, inv_beta_y)
    wpx_next = _weights(psi_x_next, inv_beta_n_x, inv_beta_x)
    wpy_next = _weights(psi_y_next, inv_beta_n_y, inv_beta_y)

    eps = 0.0
    data_avg["px"] = (
        wpx_prev * data_p["px"].to_numpy() + wpx_next * data_n["px"].to_numpy()
    ) / (wpx_prev + wpx_next + eps)
    data_avg["py"] = (
        wpy_prev * data_p["py"].to_numpy() + wpy_next * data_n["py"].to_numpy()
    ) / (wpy_prev + wpy_next + eps)
    return data_avg


def _diagnostics(
    orig_data, data_p, data_n, data_avg, info: bool, has_px: bool, has_py: bool
) -> None:
    """
    Print diagnostic information comparing calculated vs original momenta.

    Provides statistics on differences between estimated and true momenta,
    including absolute and relative errors for validation.

    Args:
        orig_data: Original DataFrame with true momenta
        data_p: Previous BPM momentum estimates
        data_n: Next BPM momentum estimates
        data_avg: Weighted average momentum estimates
        info: Whether to print diagnostics
        has_px: Whether original data has px column
        has_py: Whether original data has py column
    """
    if not info:
        return
    if "x" in orig_data.columns:
        x_diff_p = data_p["x"] - orig_data["x"]
        y_diff_p = data_p["y"] - orig_data["y"]
        print("x_diff mean", x_diff_p.abs().mean(), "±", x_diff_p.std())
        print("y_diff mean", y_diff_p.abs().mean(), "±", y_diff_p.std())

    print("MOMENTUM DIFFERENCES ------")
    if has_px:
        px_diff_p = data_p["px"] - orig_data["px"]
        px_diff_n = data_n["px"] - orig_data["px"]
        px_diff_avg = data_avg["px"] - orig_data["px"]
        print("px_diff mean (prev w/ k)", px_diff_p.abs().mean(), "±", px_diff_p.std())
        print("px_diff mean (next w/ k)", px_diff_n.abs().mean(), "±", px_diff_n.std())
        print("px_diff mean (avg)", px_diff_avg.abs().mean(), "±", px_diff_avg.std())

    if has_py:
        py_diff_p = data_p["py"] - orig_data["py"]
        py_diff_n = data_n["py"] - orig_data["py"]
        py_diff_avg = data_avg["py"] - orig_data["py"]
        print("py_diff mean (prev w/ k)", py_diff_p.abs().mean(), "±", py_diff_p.std())
        print("py_diff mean (next w/ k)", py_diff_n.abs().mean(), "±", py_diff_n.std())
        print("py_diff mean (avg)", py_diff_avg.abs().mean(), "±", py_diff_avg.std())

    epsilon = 1e-10
    if has_px:
        mask_px = orig_data["px"].abs() > epsilon
        if mask_px.any():
            px_rel = (data_avg["px"] - orig_data["px"])[mask_px] / orig_data["px"][
                mask_px
            ]
            print("px_diff mean (avg rel)", px_rel.abs().mean(), "±", px_rel.std())
        else:
            print("px_diff mean (avg rel): No significant px values")
    if has_py:
        mask_py = orig_data["py"].abs() > epsilon
        if mask_py.any():
            py_rel = (data_avg["py"] - orig_data["py"])[mask_py] / orig_data["py"][
                mask_py
            ]
            print("py_diff mean (avg rel)", py_rel.abs().mean(), "±", py_rel.std())
        else:
            print("py_diff mean (avg rel): No significant py values")


# ---------- Public API ----------
def calculate_pz(
    orig_data: tfs.TfsDataFrame,
    inject_noise: bool = True,
    tws: tfs.TfsDataFrame | None = None,
    info: bool = True,
    rng: np.random.Generator | None = None,
    low_noise_bpms: Sequence[str] | None = None,
) -> tuple[tfs.TfsDataFrame, tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """
    Estimate transverse momenta px/py using BPM pairs ~π/2 apart (prev/next).

    This function implements a phase-space reconstruction method that uses BPM
    measurements at positions separated by approximately π/2 in betatron phase.
    The method provides three momentum estimates:
    1. Based on previous BPM in the ring
    2. Based on next BPM in the ring
    3. Weighted average of the two for improved accuracy

    The algorithm:
    1. Validates input data and optional noise injection
    2. Computes or retrieves Twiss parameters (lattice optics)
    3. Identifies BPM pairs at π/2 phase difference
    4. Calculates momenta using transfer matrix relationships
    5. Applies weighted averaging based on measurement reliability
    6. Provides diagnostic information if requested

    Args:
        orig_data: Input DataFrame with BPM measurements. Must contain columns:
                  'name', 'turn', 'x', 'y'. Optional: 'px', 'py' for validation.
        inject_noise: Whether to add Gaussian noise to position measurements
        tws: Pre-computed Twiss DataFrame. If None, computed from MAD-X model.
        info: Whether to print diagnostic information and statistics
        rng: Random number generator for noise injection. Auto-created if None.
        low_noise_bpms: List of BPM names with reduced noise (10x lower std dev)

    Returns:
        Tuple of three TfsDataFrames:
        - First: Momentum estimates using previous BPM
        - Second: Momentum estimates using next BPM
        - Third: Weighted average of the two estimates

    Raises:
        ValueError: If required columns are missing from input data

    Note:
        The weighted average typically provides the most accurate momentum estimates
        as it combines information from both neighboring BPMs with appropriate
        reliability weighting based on lattice optics.
    """
    low_noise_bpms = list(low_noise_bpms or [])
    LOGGER.info(
        "Calculating transverse momentum - inject_noise=%s, low_noise_bpms=%d BPMs",
        inject_noise,
        len(low_noise_bpms),
    )

    has_px, has_py = _validate_input(orig_data)
    data = orig_data.copy(deep=True)
    rng = _get_rng(rng)

    if inject_noise:
        _inject_noise_xy(data, orig_data, rng, low_noise_bpms)

    # Twiss & maps
    tws = _ensure_twiss(tws, info)
    dpp_est = get_mean_dpp(data, tws, info)
    maps = _lattice_maps(tws)
    prev_x_df, prev_y_df, next_x_df, next_y_df = _neighbour_tables(tws)

    bpm_list = tws.index.to_list()
    bpm_index = {b: i for i, b in enumerate(bpm_list)}

    # Join minimal info once
    data_p = data.join(prev_x_df, on="name", rsuffix="_px")
    data_p = data_p.join(prev_y_df, on="name", rsuffix="_py")
    data_n = data.join(next_x_df, on="name", rsuffix="_nx")
    data_n = data_n.join(next_y_df, on="name", rsuffix="_ny")

    # Lattice at current BPM
    _attach_lattice_columns(data_p, maps)
    _attach_lattice_columns(data_n, maps)

    # Neighbor √β for normalization
    data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(maps.sqrt_betax)
    data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(maps.sqrt_betay)
    data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(maps.sqrt_betax)
    data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(maps.sqrt_betay)

    # Beta and dispersions for prev/next BPMs
    data_p["dx_prev"] = data_p["prev_bpm_x"].map(maps.dx)
    data_n["dx_next"] = data_n["next_bpm_x"].map(maps.dx)

    # Turn wrap handling + merge neighbor coordinates
    turn_x_p, turn_y_p, turn_x_n, turn_y_n = _compute_turn_wraps(
        data_p, data_n, bpm_index
    )
    data_p, data_n = _merge_neighbor_coords(
        data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n
    )

    # Momenta from prev/next BPMs
    data_p = _momenta_from_prev(data_p, dpp_est)
    data_n = _momenta_from_next(data_n, dpp_est)

    # Synchronise endpoints (as in original)
    _sync_endpoints(data_p, data_n)

    # Weighted average
    data_avg = _weighted_average(data_p, data_n, maps.betax, maps.betay)

    # Diagnostics
    _diagnostics(
        orig_data=orig_data,
        data_p=data_p,
        data_n=data_n,
        data_avg=data_avg,
        info=info,
        has_px=has_px,
        has_py=has_py,
    )

    # Return filtered columns
    # return data_p[OUT_COLS], data_n[OUT_COLS], data_avg[OUT_COLS]
    return data_avg[OUT_COLS]
