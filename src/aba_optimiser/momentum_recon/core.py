from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import FILE_COLUMNS, POSITION_STD_DEV
from aba_optimiser.io.utils import get_lhc_file_path
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface

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
) -> None:
    n_rows = len(df)
    LOGGER.debug("Adding Gaussian noise: std=%g", POSITION_STD_DEV)
    noise_x = rng.normal(0.0, POSITION_STD_DEV, size=n_rows)
    noise_y = rng.normal(0.0, POSITION_STD_DEV, size=n_rows)
    if low_noise_bpms:
        mask = df["name"].isin(low_noise_bpms).to_numpy()
        if mask.any():
            low_count = int(mask.sum())
            LOGGER.debug(
                "Reduced-noise applied to %d samples across %d BPMs",
                low_count,
                len(low_noise_bpms),
            )
            noise_x[mask] = rng.normal(0.0, POSITION_STD_DEV / 10.0, size=low_count)
            noise_y[mask] = rng.normal(0.0, POSITION_STD_DEV / 10.0, size=low_count)
    df["x"] = orig_df["x"] + noise_x
    df["y"] = orig_df["y"] + noise_y


def ensure_twiss(tws: tfs.TfsDataFrame | None, info: bool) -> tfs.TfsDataFrame:
    if tws is not None:
        return tws
    # If no Twiss provided, we provide LHC beam 1 twiss
    mad = OptimisationMadInterface(
        get_lhc_file_path(beam=1), bpm_pattern="BPM", use_real_strengths=False
    )
    tws = mad.run_twiss()
    if info:
        LOGGER.info(
            "Found tunes: q1=%s q2=%s",
            getattr(tws, "q1", None),
            getattr(tws, "q2", None),
        )
    return tws


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


def weighted_average(
    data_p: tfs.TfsDataFrame,
    data_n: tfs.TfsDataFrame,
    beta_x_map: Mapping[str, float],
    beta_y_map: Mapping[str, float],
) -> tfs.TfsDataFrame:
    data_avg = data_p.copy(deep=True)

    psi_x_prev = (data_p["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_prev = (data_p["delta_y"].to_numpy() + 0.25) * 2 * np.pi
    psi_x_next = (data_n["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_next = (data_n["delta_y"].to_numpy() + 0.25) * 2 * np.pi

    inv_beta_x = 1.0 / data_p["betax"].to_numpy()
    inv_beta_y = 1.0 / data_p["betay"].to_numpy()
    inv_beta_p_x = 1.0 / data_p["prev_bpm_x"].map(beta_x_map).to_numpy()
    inv_beta_p_y = 1.0 / data_p["prev_bpm_y"].map(beta_y_map).to_numpy()
    inv_beta_n_x = 1.0 / data_n["next_bpm_x"].map(beta_x_map).to_numpy()
    inv_beta_n_y = 1.0 / data_n["next_bpm_y"].map(beta_y_map).to_numpy()

    wpx_prev = weights(psi_x_prev, inv_beta_p_x, inv_beta_x)
    wpy_prev = weights(psi_y_prev, inv_beta_p_y, inv_beta_y)
    wpx_next = weights(psi_x_next, inv_beta_n_x, inv_beta_x)
    wpy_next = weights(psi_y_next, inv_beta_n_y, inv_beta_y)

    eps = 0.0
    data_avg["px"] = (
        wpx_prev * data_p["px"].to_numpy() + wpx_next * data_n["px"].to_numpy()
    ) / (wpx_prev + wpx_next + eps)
    data_avg["py"] = (
        wpy_prev * data_p["py"].to_numpy() + wpy_next * data_n["py"].to_numpy()
    ) / (wpy_prev + wpy_next + eps)
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
    if "x" in orig_data.columns:
        x_diff = data_p["x"] - orig_data["x"]
        y_diff = data_p["y"] - orig_data["y"]
        print("x_diff mean", x_diff.abs().mean(), "±", x_diff.std())
        print("y_diff mean", y_diff.abs().mean(), "±", y_diff.std())

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
