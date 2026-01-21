from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import FILE_COLUMNS, POSITION_STD_DEV
from aba_optimiser.momentum_recon.schema import (
    CORE_ID_COLS,
    CORE_MOM_COLS,
    CORE_POS_COLS,
    POSITION_COLS,
)

LOGGER = logging.getLogger(__name__)
OUT_COLS = list(FILE_COLUMNS)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping, Sequence

    import pandas as pd


@dataclass(frozen=True)
class LatticeMaps:
    """Optics parameters mapped by BPM name."""

    sqrt_betax: Mapping[str, float]
    sqrt_betay: Mapping[str, float]
    betax: Mapping[str, float]
    betay: Mapping[str, float]
    alfax: Mapping[str, float]
    alfay: Mapping[str, float]
    dx: Mapping[str, float] | None = None
    dpx: Mapping[str, float] | None = None
    dy: Mapping[str, float] | None = None
    dpy: Mapping[str, float] | None = None


@dataclass(frozen=True)
class InputFeatures:
    has_px: bool
    has_py: bool


def validate_input(df: pd.DataFrame) -> InputFeatures:
    required = set(CORE_ID_COLS + CORE_POS_COLS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    return InputFeatures(
        has_px=("px" in df.columns),
        has_py=("py" in df.columns),
    )


def get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng or np.random.default_rng()


def neighbour_plane_factors(
    phi: np.ndarray, *, is_prev: bool
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Compute trigonometric factors for neighbor plane calculations.

    Args:
        phi: Phase differences array.
        is_prev: Whether this is previous neighbor calculation.

    Returns:
        Tuple of (sign, alpha_sign, cos_phi, tan_phi, sec_phi).
    """
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)
    sec_phi = 1.0 / cos_phi
    sign = -1 if is_prev else 1
    alpha_sign = 1 if is_prev else -1
    return sign, alpha_sign, cos_phi, tan_phi, sec_phi


def combine_two_estimates(
    value_a: np.ndarray,
    var_a: np.ndarray,
    value_b: np.ndarray,
    var_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two estimates using inverse-variance weighting.

    Handles NaN values, non-positive variances, and infinite variances.

    Args:
        value_a: Values from estimate A.
        var_a: Variances from estimate A.
        value_b: Values from estimate B.
        var_b: Variances from estimate B.

    Returns:
        Tuple of (combined_value, combined_var).
    """
    # Mask for valid estimates
    valid_a = np.isfinite(var_a) & (var_a > 0.0) & np.isfinite(value_a)
    valid_b = np.isfinite(var_b) & (var_b > 0.0) & np.isfinite(value_b)

    # Inverse variances (0 for invalid)
    inv_var_a = np.where(valid_a, 1.0 / var_a, 0.0)
    inv_var_b = np.where(valid_b, 1.0 / var_b, 0.0)

    # Combined inverse variance
    inv_var_combined = inv_var_a + inv_var_b

    # Combined value
    combined_value = np.where(
        inv_var_combined > 0.0,
        (inv_var_a * value_a + inv_var_b * value_b) / inv_var_combined,
        np.nan,  # or some fallback
    )

    # Fallbacks when only one is valid
    combined_value = np.where(valid_a & ~valid_b, value_a, combined_value)
    combined_value = np.where(valid_b & ~valid_a, value_b, combined_value)

    # Combined variance
    combined_var = np.where(inv_var_combined > 0.0, 1.0 / inv_var_combined, np.inf)
    combined_var = np.where(valid_a & ~valid_b, var_a, combined_var)
    combined_var = np.where(valid_b & ~valid_a, var_b, combined_var)

    return combined_value, combined_var


def inject_noise_xy_inplace(
    df: pd.DataFrame,
    orig_df: pd.DataFrame,
    rng: np.random.Generator,
    low_noise_bpms: Sequence[str],
    noise_std: float = POSITION_STD_DEV,
) -> None:
    n_rows = len(df)
    LOGGER.debug("Adding Gaussian noise: std=%g", noise_std)
    noise_x = rng.normal(0.0, noise_std, size=n_rows)
    noise_y = rng.normal(0.0, noise_std, size=n_rows)
    if low_noise_bpms:
        mask = df["name"].isin(low_noise_bpms).to_numpy()
        if mask.any():
            low_count = int(mask.sum())
            LOGGER.debug(
                "Reduced-noise applied to %d samples across %d BPMs",
                low_count,
                len(low_noise_bpms),
            )
            noise_x[mask] = rng.normal(0.0, noise_std / 10.0, size=low_count)
            noise_y[mask] = rng.normal(0.0, noise_std / 10.0, size=low_count)
    df["x"] = orig_df["x"] + noise_x
    df["y"] = orig_df["y"] + noise_y


def build_lattice_maps(
    tws: pd.DataFrame,
    *,
    include_dispersion: bool = False,
) -> LatticeMaps:
    sqrt_betax = np.sqrt(tws["beta11"])
    sqrt_betay = np.sqrt(tws["beta22"])
    params: dict[str, Mapping[str, float]] = {
        "sqrt_betax": sqrt_betax.to_dict(),
        "sqrt_betay": sqrt_betay.to_dict(),
        "betax": tws["beta11"].to_dict(),
        "betay": tws["beta22"].to_dict(),
        "alfax": tws["alfa11"].to_dict(),
        "alfay": tws["alfa22"].to_dict(),
    }
    if include_dispersion:
        params["dx"] = tws["dx"].to_dict()
        params["dpx"] = tws["dpx"].to_dict()
        params["dy"] = tws["dy"].to_dict()
        params["dpy"] = tws["dpy"].to_dict()
    return LatticeMaps(**params)


def attach_lattice_columns(df: pd.DataFrame, maps: LatticeMaps) -> None:
    df["sqrt_betax"] = df["name"].map(maps.sqrt_betax)
    df["sqrt_betay"] = df["name"].map(maps.sqrt_betay)
    df["betax"] = df["name"].map(maps.betax)
    df["betay"] = df["name"].map(maps.betay)
    df["alfax"] = df["name"].map(maps.alfax)
    df["alfay"] = df["name"].map(maps.alfay)

    if maps.dx is not None:
        df["dx"] = df["name"].map(maps.dx)
    if maps.dpx is not None:
        df["dpx"] = df["name"].map(maps.dpx)
    if maps.dy is not None:
        df["dy"] = df["name"].map(maps.dy)
    if maps.dpy is not None:
        df["dpy"] = df["name"].map(maps.dpy)


def weights(psi: np.ndarray, inv_beta1: np.ndarray, inv_beta2: np.ndarray) -> np.ndarray:
    pref = 1.0 / (np.sqrt(2.0) * np.abs(np.sin(psi)))
    inside = (
        inv_beta1
        + inv_beta2
        + np.sqrt(inv_beta1**2 + inv_beta2**2 + 2.0 * inv_beta1 * inv_beta2 * np.cos(2.0 * psi))
    )
    f = pref * np.sqrt(inside)
    return 1.0 / f


def align_by_name_turn(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames by sorting on name and turn."""
    a_aligned = a.sort_values(list(CORE_ID_COLS)).reset_index(drop=True)
    b_aligned = b.sort_values(list(CORE_ID_COLS)).reset_index(drop=True)
    return a_aligned, b_aligned


def weighted_average_from_weights(data_p: pd.DataFrame, data_n: pd.DataFrame) -> pd.DataFrame:
    # Align dataframes by sorting on name and turn
    data_p_aligned, data_n_aligned = align_by_name_turn(data_p, data_n)

    data_avg = data_p_aligned.copy(deep=True)

    # Combine px estimates
    px_combined, var_px_combined = combine_two_estimates(
        data_p_aligned["px"].to_numpy(),
        data_p_aligned["var_px"].to_numpy(),
        data_n_aligned["px"].to_numpy(),
        data_n_aligned["var_px"].to_numpy(),
    )
    data_avg["px"] = px_combined
    data_avg["var_px"] = var_px_combined

    # Combine py estimates
    py_combined, var_py_combined = combine_two_estimates(
        data_p_aligned["py"].to_numpy(),
        data_p_aligned["var_py"].to_numpy(),
        data_n_aligned["py"].to_numpy(),
        data_n_aligned["var_py"].to_numpy(),
    )
    data_avg["py"] = py_combined
    data_avg["var_py"] = var_py_combined

    return data_avg


def weighted_average_from_angles(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    beta_x_map: Mapping[str, float],
    beta_y_map: Mapping[str, float],
) -> pd.DataFrame:
    # Align dataframes by sorting on name and turn
    data_p_aligned, data_n_aligned = align_by_name_turn(data_p, data_n)

    data_avg = data_p_aligned.copy(deep=True)

    psi_x_prev = (data_p_aligned["delta_x_p"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_prev = (data_p_aligned["delta_y_p"].to_numpy() + 0.25) * 2 * np.pi
    psi_x_next = (data_n_aligned["delta_x_n"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_next = (data_n_aligned["delta_y_n"].to_numpy() + 0.25) * 2 * np.pi

    inv_beta_x = 1.0 / data_p_aligned["betax"].to_numpy()
    inv_beta_y = 1.0 / data_p_aligned["betay"].to_numpy()
    inv_beta_p_x = 1.0 / data_p_aligned["bpm_x_p"].map(beta_x_map).to_numpy()
    inv_beta_p_y = 1.0 / data_p_aligned["bpm_y_p"].map(beta_y_map).to_numpy()
    inv_beta_n_x = 1.0 / data_n_aligned["bpm_x_n"].map(beta_x_map).to_numpy()
    inv_beta_n_y = 1.0 / data_n_aligned["bpm_y_n"].map(beta_y_map).to_numpy()

    wpx_prev = weights(psi_x_prev, inv_beta_p_x, inv_beta_x)
    wpy_prev = weights(psi_y_prev, inv_beta_p_y, inv_beta_y)
    wpx_next = weights(psi_x_next, inv_beta_n_x, inv_beta_x)
    wpy_next = weights(psi_y_next, inv_beta_n_y, inv_beta_y)

    eps = 0.0
    data_avg["px"] = (
        wpx_prev * data_p_aligned["px"].to_numpy() + wpx_next * data_n_aligned["px"].to_numpy()
    ) / (wpx_prev + wpx_next + eps)
    data_avg["py"] = (
        wpy_prev * data_p_aligned["py"].to_numpy() + wpy_next * data_n_aligned["py"].to_numpy()
    ) / (wpy_prev + wpy_next + eps)

    # Handle NaNs: if one df has NaN, use the other df's value
    mask_px_p_nan = np.isnan(data_p_aligned["px"])
    mask_px_n_nan = np.isnan(data_n_aligned["px"])
    mask_py_p_nan = np.isnan(data_p_aligned["py"])
    mask_py_n_nan = np.isnan(data_n_aligned["py"])

    # fmt: off
    data_avg["px"] = np.where(mask_px_p_nan & ~mask_px_n_nan, data_n_aligned["px"], data_avg["px"])
    data_avg["px"] = np.where(mask_px_n_nan & ~mask_px_p_nan, data_p_aligned["px"], data_avg["px"])

    data_avg["py"] = np.where(mask_py_p_nan & ~mask_py_n_nan, data_n_aligned["py"], data_avg["py"])
    data_avg["py"] = np.where(mask_py_n_nan & ~mask_py_p_nan, data_p_aligned["py"], data_avg["py"])
    # fmt: on

    # Restore original order
    return data_avg


def sync_endpoints_inplace(data_p: pd.DataFrame, data_n: pd.DataFrame) -> None:
    for col in CORE_MOM_COLS:
        data_n.iloc[-1, data_n.columns.get_loc(col)] = data_p.iloc[-1, data_p.columns.get_loc(col)]
        data_p.iloc[0, data_p.columns.get_loc(col)] = data_n.iloc[0, data_n.columns.get_loc(col)]


def diagnostics(
    orig_data,
    data_p,
    data_n,
    data_avg,
    info: bool,
    features: InputFeatures,
) -> None:
    if not info:
        return

    # Merge dataframes to ensure proper alignment by name and turn
    # This prevents misleading diagnostics from index misalignment
    merge_cols = list(CORE_ID_COLS)

    # Merge prev estimates
    merged_p = orig_data.merge(
        data_p[merge_cols + list(POSITION_COLS + CORE_MOM_COLS) if features.has_px else merge_cols + list(POSITION_COLS)],
        on=merge_cols,
        suffixes=("_true", "_prev"),
    )

    # Merge next estimates
    merged_n = orig_data.merge(
        data_n[merge_cols + list(CORE_MOM_COLS) if features.has_px else merge_cols],
        on=merge_cols,
        suffixes=("_true", "_next"),
    )

    # Merge averaged estimates
    merged_avg = orig_data.merge(
        data_avg[merge_cols + list(CORE_MOM_COLS) if features.has_px else merge_cols],
        on=merge_cols,
        suffixes=("_true", "_avg"),
    )

    if "x_true" in merged_p.columns:
        x_diff = merged_p["x_prev"] - merged_p["x_true"]
        y_diff = merged_p["y_prev"] - merged_p["y_true"]
        LOGGER.info("x_diff mean %s ± %s", x_diff.abs().mean(), x_diff.std())
        LOGGER.info("y_diff mean %s ± %s", y_diff.abs().mean(), y_diff.std())

    LOGGER.info("MOMENTUM DIFFERENCES ------")
    if features.has_px:
        px_diff_p = merged_p["px_prev"] - merged_p["px_true"]
        px_diff_n = merged_n["px_next"] - merged_n["px_true"]
        px_diff_avg = merged_avg["px_avg"] - merged_avg["px_true"]
        LOGGER.info("px_diff mean (prev w/ k) %s ± %s", px_diff_p.abs().mean(), px_diff_p.std())
        LOGGER.info("px_diff mean (next w/ k) %s ± %s", px_diff_n.abs().mean(), px_diff_n.std())
        LOGGER.info("px_diff mean (avg) %s ± %s", px_diff_avg.abs().mean(), px_diff_avg.std())

    if features.has_py:
        py_diff_p = merged_p["py_prev"] - merged_p["py_true"]
        py_diff_n = merged_n["py_next"] - merged_n["py_true"]
        py_diff_avg = merged_avg["py_avg"] - merged_avg["py_true"]
        LOGGER.info("py_diff mean (prev w/ k) %s ± %s", py_diff_p.abs().mean(), py_diff_p.std())
        LOGGER.info("py_diff mean (next w/ k) %s ± %s", py_diff_n.abs().mean(), py_diff_n.std())
        LOGGER.info("py_diff mean (avg) %s ± %s", py_diff_avg.abs().mean(), py_diff_avg.std())

    epsilon = 1e-10
    if features.has_px and "px_true" in merged_avg.columns:
        mask_px = merged_avg["px_true"].abs() > epsilon
        if mask_px.any():
            px_rel = (merged_avg["px_avg"] - merged_avg["px_true"])[mask_px] / merged_avg[
                "px_true"
            ][mask_px]
            LOGGER.info("px_diff mean (avg rel) %s ± %s", px_rel.abs().mean(), px_rel.std())
        else:
            LOGGER.info("px_diff mean (avg rel): No significant px values")
    if features.has_py and "py_true" in merged_avg.columns:
        mask_py = merged_avg["py_true"].abs() > epsilon
        if mask_py.any():
            py_rel = (merged_avg["py_avg"] - merged_avg["py_true"])[mask_py] / merged_avg[
                "py_true"
            ][mask_py]
            LOGGER.info("py_diff mean (avg rel) %s ± %s", py_rel.abs().mean(), py_rel.std())
        else:
            LOGGER.info("py_diff mean (avg rel): No significant py values")


def remove_closed_orbit_inplace(data: pd.DataFrame, co: pd.DataFrame) -> None:
    """Remove closed orbit from tracking data in-place."""
    x_dict = co["x"].to_dict()
    y_dict = co["y"].to_dict()
    # Ensure arithmetic is performed on float dtype, not categorical
    data["x"] = data["x"].astype(float) - data["name"].map(x_dict).astype(float)
    data["y"] = data["y"].astype(float) - data["name"].map(y_dict).astype(float)


def restore_closed_orbit_and_reference_momenta_inplace(
    data: pd.DataFrame, co: pd.DataFrame
) -> None:
    """Restore closed orbit and add reference momenta to data in-place."""
    co_dict = co.to_dict()
    data["x"] = data["x"] + data["name"].map(co_dict["x"])
    data["y"] = data["y"] + data["name"].map(co_dict["y"])
    data["px"] = data["px"] + data["name"].map(co_dict["px"])
    data["py"] = data["py"] + data["name"].map(co_dict["py"])

    if "var_px" in co.columns and "var_py" in co.columns:
        data["var_px"] = data["var_px"] + data["name"].map(co_dict["var_px"])
        data["var_py"] = data["var_py"] + data["name"].map(co_dict["var_py"])
