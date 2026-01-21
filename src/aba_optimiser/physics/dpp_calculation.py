from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

# NEW: we need pandas for the aligned return
import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.physics.bpm_phases import (
    next_bpm_to_pi,
    next_bpm_to_pi_2,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)

LOGGER = logging.getLogger(__name__)
OUT_COLS = list(FILE_COLUMNS)

Direction = Literal["next", "prev"]

DX_TOL = 1e-2
DEN_TOL = 1e-10


@dataclass(frozen=True)
class TwissMaps:
    sqrt_betax: dict[str, float]
    dx: dict[str, float]
    x_co: dict[str, float]


def _twiss_maps(tws: pd.DataFrame) -> TwissMaps:
    sqrt_betax = np.sqrt(tws["beta11"]).to_dict()
    dx = tws["dx"].to_dict()
    x_co = tws.get("x", pd.Series(0.0, index=tws.index)).to_dict()
    return TwissMaps(sqrt_betax=sqrt_betax, dx=dx, x_co=x_co)


def _partner_tables(direction: Direction, mu1, q1):
    if direction == "next":
        tbl_pi = next_bpm_to_pi(mu1, q1).rename(columns={"next_bpm": "bpm_bar", "delta": "d_pi"})
        tbl_pi2 = next_bpm_to_pi_2(mu1, q1).rename(
            columns={"next_bpm": "bpm_tilde", "delta": "d_pi2"}
        )
    else:
        tbl_pi = prev_bpm_to_pi(mu1, q1).rename(columns={"prev_bpm": "bpm_bar", "delta": "d_pi"})
        tbl_pi2 = prev_bpm_to_pi_2(mu1, q1).rename(
            columns={"prev_bpm": "bpm_tilde", "delta": "d_pi2"}
        )
    return tbl_pi, tbl_pi2


def _wrap_turns(
    direction: Direction, df, bpm_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    cur_i = df["name"].map(bpm_index)
    i_bar = df["bpm_bar"].map(bpm_index)
    i_tilde = df["bpm_tilde"].map(bpm_index)
    if direction == "next":
        turn_bar = df["turn"] + (cur_i > i_bar).astype(np.int16)
        turn_tilde = df["turn"] + (cur_i > i_tilde).astype(np.int16)
    else:
        turn_bar = df["turn"] - (cur_i < i_bar).astype(np.int16)
        turn_tilde = df["turn"] - (cur_i < i_tilde).astype(np.int16)
    return turn_bar, turn_tilde


def _merge_partner_coords(df):
    coords = df[["turn", "name", "x"]]
    coords_bar = coords.rename(columns={"turn": "turn_bar", "name": "bpm_bar", "x": "x_bar"})[
        ["turn_bar", "bpm_bar", "x_bar"]
    ]
    coords_tilde = coords.rename(
        columns={"turn": "turn_tilde", "name": "bpm_tilde", "x": "x_tilde"}
    )[["turn_tilde", "bpm_tilde", "x_tilde"]]
    df = df.merge(coords_bar, on=["turn_bar", "bpm_bar"], how="left", copy=False)
    df = df.merge(coords_tilde, on=["turn_tilde", "bpm_tilde"], how="left", copy=False)

    if df[["x_bar", "x_tilde"]].isnull().any().any():
        # Check that the turn of the null values is either 0 (prev) or max turn (next), otherwise log
        max_turn = df["turn"].max()
        null_rows = df[df["x_bar"].isnull() | df["x_tilde"].isnull()]
        for _, row in null_rows.iterrows():
            if row["turn"] != 0 and row["turn"] != max_turn:
                LOGGER.warning(
                    "Missing partner coordinates for BPM %s at turn %d",
                    row["name"],
                    row["turn"],
                )
    # Clean up
    df.drop(columns=["turn_bar", "turn_tilde"], inplace=True)
    return df


def _phases_radians(d_pi: np.ndarray, d_pi2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi_bar = (d_pi + 0.5) * 2.0 * np.pi
    psi_tilde = (d_pi2 + 0.25) * 2.0 * np.pi
    return psi_bar, psi_tilde


def _compute_delta(
    x1: np.ndarray,
    x_bar: np.ndarray,
    x_tilde: np.ndarray,
    sqrt_beta_1: np.ndarray,
    sqrt_beta_bar: np.ndarray,
    sqrt_beta_tilde: np.ndarray,
    eta_1: np.ndarray,
    eta_bar: np.ndarray,
    eta_tilde: np.ndarray,
    psi_bar: np.ndarray,
    psi_tilde: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_bar = np.sin(psi_bar)
    sin_tilde = np.sin(psi_tilde)
    sin_bar_minus_tilde = np.sin(psi_bar - psi_tilde)
    num = (
        (x_tilde / sqrt_beta_tilde) * sin_bar
        - (x_bar / sqrt_beta_bar) * sin_tilde
        - (x1 / sqrt_beta_1) * sin_bar_minus_tilde
    )
    den = (
        (eta_tilde / sqrt_beta_tilde) * sin_bar
        - (eta_bar / sqrt_beta_bar) * sin_tilde
        - (eta_1 / sqrt_beta_1) * sin_bar_minus_tilde
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_all = num / den
    return num, den, delta_all


def _valid_mask(eta_1, eta_bar, eta_tilde, den) -> np.ndarray:
    valid_dx = (
        ~np.isclose(eta_1, 0.0, atol=DX_TOL)
        & ~np.isclose(eta_bar, 0.0, atol=DX_TOL)
        & ~np.isclose(eta_tilde, 0.0, atol=DX_TOL)
        & (eta_1 > 0.0)
        & (eta_bar > 0.0)
        & (eta_tilde > 0.0)
    )
    valid_den = np.abs(den) > DEN_TOL
    return valid_dx & valid_den


def _maybe_log_stats(delta: np.ndarray, info: bool, label: str = "") -> None:
    if not info:
        return
    prefix = f"[{label}] " if label else ""
    if delta.size:
        LOGGER.info(prefix + "Count: %d", delta.size)
        LOGGER.info(prefix + "δ mean: %s", np.nanmean(delta))
        LOGGER.info(prefix + "δ std:  %s", np.nanstd(delta))
        LOGGER.info(prefix + "δ min:  %s", np.nanmin(delta))
        LOGGER.info(prefix + "δ max:  %s", np.nanmax(delta))
    else:
        LOGGER.warning(prefix + "All samples were filtered out (check dx/den tolerances).")


def _calculate_dpp_direction_aligned(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    maps: TwissMaps,
    direction: Direction,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: compute per-row δ for a given direction.
    Returns:
        delta_all : np.ndarray (same length as data), may contain inf/NaN where invalid
        valid     : boolean mask of rows considered valid (dispersion & denominator)
    """
    df = data.copy(deep=True)
    mu1 = tws["mu1"]
    tbl_pi, tbl_pi2 = _partner_tables(direction, mu1, tws.q1)
    df = df.join(tbl_pi, on="name")
    df = df.join(tbl_pi2, on="name", rsuffix="_pi2")

    bpm_index = {b: i for i, b in enumerate(tws.index.to_list())}
    turn_bar, turn_tilde = _wrap_turns(direction, df, bpm_index)
    df["turn_bar"] = turn_bar
    df["turn_tilde"] = turn_tilde
    df = _merge_partner_coords(df)

    sqrt_beta_1 = df["name"].map(maps.sqrt_betax).to_numpy()
    sqrt_beta_bar = df["bpm_bar"].map(maps.sqrt_betax).to_numpy()
    sqrt_beta_tilde = df["bpm_tilde"].map(maps.sqrt_betax).to_numpy()

    eta_1 = df["name"].map(maps.dx).to_numpy()
    eta_bar = df["bpm_bar"].map(maps.dx).to_numpy()
    eta_tilde = df["bpm_tilde"].map(maps.dx).to_numpy()

    psi_bar, psi_tilde = _phases_radians(df["d_pi"].to_numpy(), df["d_pi2"].to_numpy())

    _, den, delta_all = _compute_delta(
        df["x"].to_numpy(),
        df["x_bar"].to_numpy(),
        df["x_tilde"].to_numpy(),
        sqrt_beta_1,
        sqrt_beta_bar,
        sqrt_beta_tilde,
        eta_1,
        eta_bar,
        eta_tilde,
        psi_bar,
        psi_tilde,
    )

    valid = _valid_mask(eta_1, eta_bar, eta_tilde, den)
    return delta_all, valid


def calculate_dpp_both(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    info: bool = True,
) -> pd.DataFrame:
    """
    Compute δ using BOTH 'prev' and 'next' partner BPM triplets.
    Returns a DataFrame aligned to the input rows with columns:
        - 'delta_prev'
        - 'delta_next'
        - 'delta_avg'  (simple unweighted average where both are valid; NaN otherwise)

    This leaves the original `calculate_dpp` API untouched.
    """
    LOGGER.info("Calculating δ using BOTH directions (prev + next)")
    if tws.index.name != "name":
        tws = tws.set_index("name")

    # Reduce data and tws rows based on BPMs present in data
    bpms_in_data = data["name"].unique().tolist()
    tws = tws.loc[tws.index.isin(bpms_in_data)]

    maps = _twiss_maps(tws)

    # remove closed orbit
    data["x"] = data["x"] - data["name"].map(maps.x_co)

    # Per-direction, aligned arrays and masks
    next_all, next_valid = _calculate_dpp_direction_aligned(data, tws, maps, "next")
    prev_all, prev_valid = _calculate_dpp_direction_aligned(data, tws, maps, "prev")

    # Build aligned Series with NaN where invalid
    delta_next = np.where(next_valid, next_all, np.nan)
    delta_prev = np.where(prev_valid, prev_all, np.nan)

    # Unweighted average only where both are valid
    both_valid = next_valid & prev_valid
    delta_avg = np.full_like(delta_next, np.nan, dtype=float)
    delta_avg[both_valid] = 0.5 * (delta_prev[both_valid] + delta_next[both_valid])

    # Diagnostics
    _maybe_log_stats(delta_prev[prev_valid], info, label="prev")
    _maybe_log_stats(delta_next[next_valid], info, label="next")
    _maybe_log_stats(delta_avg[np.isfinite(delta_avg)], info, label="avg")

    # Return as DataFrame aligned to input index
    return pd.DataFrame(
        {
            "delta_prev": delta_prev,
            "delta_next": delta_next,
            "delta_avg": delta_avg,
        },
        index=data.index,
    )


def get_mean_dpp(data: pd.DataFrame, tws: pd.DataFrame, info: bool = True) -> float:
    """
    Compute mean δ using BOTH 'prev' and 'next' partner BPM triplets.
    Returns a single float value representing the mean δ across all valid measurements.
    """
    dpp_df = calculate_dpp_both(data, tws=tws, info=info)
    mean_dpp = np.nanmean(dpp_df["delta_avg"])
    if info:
        LOGGER.info("Mean δ across all valid measurements: %s", mean_dpp)
    return mean_dpp if np.isfinite(mean_dpp) else 0.0
