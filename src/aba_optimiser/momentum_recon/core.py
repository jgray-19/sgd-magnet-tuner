from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import FILE_COLUMNS, POSITION_STD_DEV

LOGGER = logging.getLogger(__name__)
OUT_COLS = list(FILE_COLUMNS)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping, Sequence

    import tfs


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
    cox: Mapping[str, float] | None = None
    coy: Mapping[str, float] | None = None


def validate_input(df: tfs.TfsDataFrame) -> tuple[bool, bool]:
    required = {"name", "turn", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    return ("px" in df.columns), ("py" in df.columns)


def get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng or np.random.default_rng()


def inject_noise_xy(
    df: tfs.TfsDataFrame,
    orig_df: tfs.TfsDataFrame,
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
    tws: tfs.TfsDataFrame,
    *,
    include_dispersion: bool = False,
    include_orbit: bool = False,
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
    if include_orbit:
        params["cox"] = tws["x"].to_dict()
        params["coy"] = tws["y"].to_dict()
    return LatticeMaps(**params)


def attach_lattice_columns(df: tfs.TfsDataFrame, maps: LatticeMaps) -> None:
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
    if maps.cox is not None:
        df["cox"] = df["name"].map(maps.cox)
    if maps.coy is not None:
        df["coy"] = df["name"].map(maps.coy)


def weights(
    psi: np.ndarray, inv_beta1: np.ndarray, inv_beta2: np.ndarray
) -> np.ndarray:
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


def weighted_average_from_weights(
    data_p: tfs.TfsDataFrame, data_n: tfs.TfsDataFrame
) -> tfs.TfsDataFrame:
    # Align dataframes by sorting on name and turn, preserving original order
    data_p_aligned = data_p.sort_values(["name", "turn"]).reset_index(drop=True)
    data_n_aligned = data_n.sort_values(["name", "turn"]).reset_index(drop=True)

    data_avg = data_p_aligned.copy(deep=True)

    var_px_p = data_p_aligned["var_px"].to_numpy(dtype=float, copy=False)
    var_px_n = data_n_aligned["var_px"].to_numpy(dtype=float, copy=False)
    var_py_p = data_p_aligned["var_py"].to_numpy(dtype=float, copy=False)
    var_py_n = data_n_aligned["var_py"].to_numpy(dtype=float, copy=False)

    mask_px_p_nan = np.isnan(data_p_aligned["px"])
    mask_px_n_nan = np.isnan(data_n_aligned["px"])
    mask_py_p_nan = np.isnan(data_p_aligned["py"])
    mask_py_n_nan = np.isnan(data_n_aligned["py"])

    inv_var_px_p = np.zeros_like(var_px_p)
    inv_var_px_n = np.zeros_like(var_px_n)
    inv_var_py_p = np.zeros_like(var_py_p)
    inv_var_py_n = np.zeros_like(var_py_n)

    valid_px_p = np.isfinite(var_px_p) & (var_px_p > 0.0) & ~mask_px_p_nan
    valid_px_n = np.isfinite(var_px_n) & (var_px_n > 0.0) & ~mask_px_n_nan
    valid_py_p = np.isfinite(var_py_p) & (var_py_p > 0.0) & ~mask_py_p_nan
    valid_py_n = np.isfinite(var_py_n) & (var_py_n > 0.0) & ~mask_py_n_nan

    np.divide(1.0, var_px_p, out=inv_var_px_p, where=valid_px_p)
    np.divide(1.0, var_px_n, out=inv_var_px_n, where=valid_px_n)
    np.divide(1.0, var_py_p, out=inv_var_py_p, where=valid_py_p)
    np.divide(1.0, var_py_n, out=inv_var_py_n, where=valid_py_n)

    px_prev = data_p_aligned["px"].to_numpy(dtype=float, copy=False)
    px_next = data_n_aligned["px"].to_numpy(dtype=float, copy=False)
    py_prev = data_p_aligned["py"].to_numpy(dtype=float, copy=False)
    py_next = data_n_aligned["py"].to_numpy(dtype=float, copy=False)

    denom_px = inv_var_px_p + inv_var_px_n
    denom_py = inv_var_py_p + inv_var_py_n

    weighted_px = np.zeros_like(px_prev)
    weighted_py = np.zeros_like(py_prev)

    np.divide(
        inv_var_px_p * px_prev + inv_var_px_n * px_next,
        denom_px,
        out=weighted_px,
        where=denom_px > 0.0,
    )
    np.divide(
        inv_var_py_p * py_prev + inv_var_py_n * py_next,
        denom_py,
        out=weighted_py,
        where=denom_py > 0.0,
    )

    only_prev_px = valid_px_p & ~valid_px_n
    only_next_px = valid_px_n & ~valid_px_p
    only_prev_py = valid_py_p & ~valid_py_n
    only_next_py = valid_py_n & ~valid_py_p

    data_avg["px"] = np.where(denom_px > 0.0, weighted_px, px_prev)
    data_avg["px"] = np.where(only_prev_px, px_prev, data_avg["px"])
    data_avg["px"] = np.where(only_next_px, px_next, data_avg["px"])

    data_avg["py"] = np.where(denom_py > 0.0, weighted_py, py_prev)
    data_avg["py"] = np.where(only_prev_py, py_prev, data_avg["py"])
    data_avg["py"] = np.where(only_next_py, py_next, data_avg["py"])

    combined_inv_px = inv_var_px_p + inv_var_px_n
    combined_inv_py = inv_var_py_p + inv_var_py_n

    combined_var_px = np.full_like(var_px_p, np.inf)
    combined_var_py = np.full_like(var_py_p, np.inf)

    positive_inv_px = combined_inv_px > 0.0
    positive_inv_py = combined_inv_py > 0.0

    np.divide(1.0, combined_inv_px, out=combined_var_px, where=positive_inv_px)
    np.divide(1.0, combined_inv_py, out=combined_var_py, where=positive_inv_py)

    combined_var_px = np.where(valid_px_p & ~valid_px_n, var_px_p, combined_var_px)
    combined_var_px = np.where(valid_px_n & ~valid_px_p, var_px_n, combined_var_px)
    combined_var_py = np.where(valid_py_p & ~valid_py_n, var_py_p, combined_var_py)
    combined_var_py = np.where(valid_py_n & ~valid_py_p, var_py_n, combined_var_py)

    data_avg["var_px"] = combined_var_px
    data_avg["var_py"] = combined_var_py
    # Restore original order
    return data_avg


def weighted_average_from_angles(
    data_p: tfs.TfsDataFrame,
    data_n: tfs.TfsDataFrame,
    beta_x_map: Mapping[str, float],
    beta_y_map: Mapping[str, float],
) -> tfs.TfsDataFrame:
    # Align dataframes by sorting on name and turn, preserving original order
    data_p_aligned = data_p.sort_values(["name", "turn"]).reset_index(drop=True)
    data_n_aligned = data_n.sort_values(["name", "turn"]).reset_index(drop=True)

    data_avg = data_p_aligned.copy(deep=True)

    psi_x_prev = (data_p_aligned["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_prev = (data_p_aligned["delta_y"].to_numpy() + 0.25) * 2 * np.pi
    psi_x_next = (data_n_aligned["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_next = (data_n_aligned["delta_y"].to_numpy() + 0.25) * 2 * np.pi

    inv_beta_x = 1.0 / data_p_aligned["betax"].to_numpy()
    inv_beta_y = 1.0 / data_p_aligned["betay"].to_numpy()
    inv_beta_p_x = 1.0 / data_p_aligned["prev_bpm_x"].map(beta_x_map).to_numpy()
    inv_beta_p_y = 1.0 / data_p_aligned["prev_bpm_y"].map(beta_y_map).to_numpy()
    inv_beta_n_x = 1.0 / data_n_aligned["next_bpm_x"].map(beta_x_map).to_numpy()
    inv_beta_n_y = 1.0 / data_n_aligned["next_bpm_y"].map(beta_y_map).to_numpy()

    wpx_prev = weights(psi_x_prev, inv_beta_p_x, inv_beta_x)
    wpy_prev = weights(psi_y_prev, inv_beta_p_y, inv_beta_y)
    wpx_next = weights(psi_x_next, inv_beta_n_x, inv_beta_x)
    wpy_next = weights(psi_y_next, inv_beta_n_y, inv_beta_y)

    eps = 0.0
    data_avg["px"] = (
        wpx_prev * data_p_aligned["px"].to_numpy()
        + wpx_next * data_n_aligned["px"].to_numpy()
    ) / (wpx_prev + wpx_next + eps)
    data_avg["py"] = (
        wpy_prev * data_p_aligned["py"].to_numpy()
        + wpy_next * data_n_aligned["py"].to_numpy()
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


def sync_endpoints(data_p: tfs.TfsDataFrame, data_n: tfs.TfsDataFrame) -> None:
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


def diagnostics(
    orig_data,
    data_p,
    data_n,
    data_avg,
    info: bool,
    has_px: bool,
    has_py: bool,
) -> None:
    if not info:
        return

    # Merge dataframes to ensure proper alignment by name and turn
    # This prevents misleading diagnostics from index misalignment
    merge_cols = ["name", "turn"]

    # Merge prev estimates
    merged_p = orig_data.merge(
        data_p[merge_cols + ["x", "y", "px", "py"] if has_px else merge_cols + ["x", "y"]],
        on=merge_cols,
        suffixes=("_true", "_prev"),
    )

    # Merge next estimates
    merged_n = orig_data.merge(
        data_n[merge_cols + ["px", "py"] if has_px else merge_cols],
        on=merge_cols,
        suffixes=("_true", "_next"),
    )

    # Merge averaged estimates
    merged_avg = orig_data.merge(
        data_avg[merge_cols + ["px", "py"] if has_px else merge_cols],
        on=merge_cols,
        suffixes=("_true", "_avg"),
    )

    if "x_true" in merged_p.columns:
        x_diff = merged_p["x_prev"] - merged_p["x_true"]
        y_diff = merged_p["y_prev"] - merged_p["y_true"]
        print("x_diff mean", x_diff.abs().mean(), "±", x_diff.std())
        print("y_diff mean", y_diff.abs().mean(), "±", y_diff.std())

    print("MOMENTUM DIFFERENCES ------")
    if has_px:
        px_diff_p = merged_p["px_prev"] - merged_p["px_true"]
        px_diff_n = merged_n["px_next"] - merged_n["px_true"]
        px_diff_avg = merged_avg["px_avg"] - merged_avg["px_true"]
        print("px_diff mean (prev w/ k)", px_diff_p.abs().mean(), "±", px_diff_p.std())
        print("px_diff mean (next w/ k)", px_diff_n.abs().mean(), "±", px_diff_n.std())
        print("px_diff mean (avg)", px_diff_avg.abs().mean(), "±", px_diff_avg.std())

    if has_py:
        py_diff_p = merged_p["py_prev"] - merged_p["py_true"]
        py_diff_n = merged_n["py_next"] - merged_n["py_true"]
        py_diff_avg = merged_avg["py_avg"] - merged_avg["py_true"]
        print("py_diff mean (prev w/ k)", py_diff_p.abs().mean(), "±", py_diff_p.std())
        print("py_diff mean (next w/ k)", py_diff_n.abs().mean(), "±", py_diff_n.std())
        print("py_diff mean (avg)", py_diff_avg.abs().mean(), "±", py_diff_avg.std())

    epsilon = 1e-10
    if has_px and "px_true" in merged_avg.columns:
        mask_px = merged_avg["px_true"].abs() > epsilon
        if mask_px.any():
            px_rel = (merged_avg["px_avg"] - merged_avg["px_true"])[mask_px] / merged_avg[
                "px_true"
            ][mask_px]
            print("px_diff mean (avg rel)", px_rel.abs().mean(), "±", px_rel.std())
        else:
            print("px_diff mean (avg rel): No significant px values")
    if has_py and "py_true" in merged_avg.columns:
        mask_py = merged_avg["py_true"].abs() > epsilon
        if mask_py.any():
            py_rel = (merged_avg["py_avg"] - merged_avg["py_true"])[mask_py] / merged_avg[
                "py_true"
            ][mask_py]
            print("py_diff mean (avg rel)", py_rel.abs().mean(), "±", py_rel.std())
        else:
            print("py_diff mean (avg rel): No significant py values")
