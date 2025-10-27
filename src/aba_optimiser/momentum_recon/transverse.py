from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    BETA_NAME,
    NAME,
    NAME2,
    ORBIT,
    ORBIT_NAME,
    PHASE,
    PHASE_NAME,
)

from aba_optimiser.momentum_recon.core import (
    OUT_COLS,
    LatticeMaps,
    attach_lattice_columns,
    build_lattice_maps,
    diagnostics,
    ensure_twiss,
    get_rng,
    inject_noise_xy,
    sync_endpoints,
    validate_input,
    weighted_average,
)
from aba_optimiser.momentum_recon.momenta import (
    momenta_from_next,
    momenta_from_prev,
)
from aba_optimiser.momentum_recon.neighbors import (
    build_lattice_neighbor_tables,
    compute_turn_wraps,
    merge_neighbor_coords,
)

LOGGER = logging.getLogger(__name__)


def _subtract_bpm_means(df: tfs.TfsDataFrame, info: bool) -> None:
    bpm_means = df.groupby("name", observed=False)[["x", "y"]].mean()
    if info:
        print("BPM means (x, y):")
        print(bpm_means)
    df_ = df.merge(bpm_means, on="name", suffixes=("", "_mean"))
    df_["x"] = df_["x"] - df_["x_mean"]
    df_["y"] = df_["y"] - df_["y_mean"]
    df.drop(df.index, inplace=True)
    for col in df_.columns:
        df[col] = df_[col]


def _add_bpm_means(*dfs: tfs.TfsDataFrame) -> None:
    for df in dfs:
        if "x_mean" in df.columns and "y_mean" in df.columns:
            df["x"] += df["x_mean"]
            df["y"] += df["y_mean"]
        if "px_mean" in df.columns and "py_mean" in df.columns:
            df["px"] += df["px_mean"]
            df["py"] += df["py_mean"]


def _cleanup_mean_cols(*dfs: tfs.TfsDataFrame) -> None:
    for df in dfs:
        for column in ("x_mean", "y_mean", "px_mean", "py_mean"):
            if column in df.columns:
                df.drop(columns=[column], inplace=True)


def _find_neighbors_from_phase_diffs(
    phase_df: tfs.TfsDataFrame,
    phase_col: str,
    bpm_list: list[str],
    target_phase: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_prev: dict[str, tuple[str, float]] = {}
    best_next: dict[str, tuple[str, float]] = {}
    phase_df = phase_df.set_index([NAME, NAME2])
    bpm_count = len(bpm_list)
    for index, bpm in enumerate(bpm_list):
        phase = 0.0
        phase_diffs: dict[str, float] = {}
        current = bpm
        for offset in range(1, 12):
            prev_index = (index - offset) % bpm_count
            prev_bpm = bpm_list[prev_index]
            phase += phase_df.loc[(prev_bpm, current), phase_col]
            phase_diffs[prev_bpm] = phase
            if phase >= target_phase:
                break
            current = prev_bpm
        if not phase_diffs:
            raise ValueError(f"No previous BPM found for {bpm}")
        best_prev[bpm] = min(
            phase_diffs.items(), key=lambda entry: abs(entry[1] - target_phase)
        )

        phase = 0.0
        phase_diffs = {}
        current = bpm
        for offset in range(1, 12):
            next_index = (index + offset) % bpm_count
            next_bpm = bpm_list[next_index]
            phase += phase_df.loc[(current, next_bpm), phase_col]
            phase_diffs[next_bpm] = phase
            if phase >= target_phase:
                break
            current = next_bpm
        if not phase_diffs:
            raise ValueError(f"No next BPM found for {bpm}")
        best_next[bpm] = min(
            phase_diffs.items(), key=lambda entry: abs(entry[1] - target_phase)
        )

    prev_rows = [
        {"prev_bpm": best_prev[bpm][0], "delta": best_prev[bpm][1]} for bpm in bpm_list
    ]
    next_rows = [
        {"next_bpm": best_next[bpm][0], "delta": best_next[bpm][1]} for bpm in bpm_list
    ]
    prev_df = pd.DataFrame(prev_rows, index=bpm_list)
    next_df = pd.DataFrame(next_rows, index=bpm_list)
    return prev_df, next_df


def _add_final_row(phase_df: tfs.TfsDataFrame, plane: str) -> None:
    tune = "Q1" if plane == "X" else "Q2"
    first_bpm = phase_df.iloc[0][NAME]
    last_bpm = phase_df.iloc[-1][NAME2]
    total_phase = phase_df[PHASE + plane].sum()
    phase_df.loc[len(phase_df)] = {
        NAME: last_bpm,
        NAME2: first_bpm,
        PHASE + plane: phase_df[tune] - total_phase,
    }


def _measurement_neighbor_tables(
    phase_x: tfs.TfsDataFrame,
    phase_y: tfs.TfsDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _add_final_row(phase_x, "X")
    _add_final_row(phase_y, "Y")
    prev_x, next_x = _find_neighbors_from_phase_diffs(
        phase_x, PHASE + "X", phase_x[NAME].to_list()
    )
    prev_y, next_y = _find_neighbors_from_phase_diffs(
        phase_y, PHASE + "Y", phase_y[NAME].to_list()
    )
    prev_x = prev_x.rename(columns={"prev_bpm": "prev_bpm_x", "delta": "delta_x"})
    prev_y = prev_y.rename(columns={"prev_bpm": "prev_bpm_y", "delta": "delta_y"})
    next_x = next_x.rename(columns={"next_bpm": "next_bpm_x", "delta": "delta_x"})
    next_y = next_y.rename(columns={"next_bpm": "next_bpm_y", "delta": "delta_y"})
    return prev_x, prev_y, next_x, next_y


def calculate_pz(
    orig_data: tfs.TfsDataFrame,
    inject_noise: bool = True,
    tws: tfs.TfsDataFrame | None = None,
    info: bool = True,
    rng: np.random.Generator | None = None,
    low_noise_bpms: list[str] | None = None,
    subtract_mean: bool = False,
) -> tfs.TfsDataFrame:
    low_noise_bpms = list(low_noise_bpms or [])
    LOGGER.info(
        "Calculating transverse momentum - inject_noise=%s, low_noise_bpms=%d BPMs",
        inject_noise,
        len(low_noise_bpms),
    )

    has_px, has_py = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")
    for column in ("x", "y"):
        if column in data.columns:
            with contextlib.suppress(AttributeError, TypeError, ValueError):
                data[column] = data[column].astype(np.float32)
    rng = get_rng(rng)

    if inject_noise:
        inject_noise_xy(data, orig_data, rng, low_noise_bpms)

    if subtract_mean:
        _subtract_bpm_means(data, info)

    tws = ensure_twiss(tws, info)
    maps = build_lattice_maps(tws)
    prev_x_df, prev_y_df, next_x_df, next_y_df = build_lattice_neighbor_tables(tws)

    bpm_list = tws.index.to_list()
    bpm_index = {bpm: idx for idx, bpm in enumerate(bpm_list)}

    data_p = data.join(prev_x_df, on="name", rsuffix="_px")
    data_p = data_p.join(prev_y_df, on="name", rsuffix="_py")
    data_n = data.join(next_x_df, on="name", rsuffix="_nx")
    data_n = data_n.join(next_y_df, on="name", rsuffix="_ny")

    attach_lattice_columns(data_p, maps)
    attach_lattice_columns(data_n, maps)

    data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(maps.sqrt_betax)
    data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(maps.sqrt_betay)
    data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(maps.sqrt_betax)
    data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(maps.sqrt_betay)

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(
        data_p, data_n, bpm_index
    )
    data_p, data_n = merge_neighbor_coords(
        data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n
    )

    data_p = momenta_from_prev(data_p)
    data_n = momenta_from_next(data_n)

    sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n, maps.betax, maps.betay)

    if subtract_mean:
        _add_bpm_means(data_p, data_n, data_avg)
        _cleanup_mean_cols(data_p, data_n, data_avg)

    diagnostics(orig_data, data_p, data_n, data_avg, info, has_px, has_py)
    return data_avg[OUT_COLS]


def calculate_pz_from_measurements(
    data: pd.DataFrame,
    analysis_dir: str | Path,
    info: bool = True,
    subtract_mean: bool = False,
) -> pd.DataFrame:
    analysis_path = Path(analysis_dir)
    phase_x = tfs.read(analysis_path / (PHASE_NAME + "x.tfs"))
    phase_y = tfs.read(analysis_path / (PHASE_NAME + "y.tfs"))
    beta_x = tfs.read(analysis_path / (BETA_NAME + "x.tfs"), index=NAME)
    beta_y = tfs.read(analysis_path / (BETA_NAME + "y.tfs"), index=NAME)
    orbit_x = tfs.read(analysis_path / (ORBIT_NAME + "x.tfs"), index=NAME)
    orbit_y = tfs.read(analysis_path / (ORBIT_NAME + "y.tfs"), index=NAME)

    bpm_list = beta_x.index.to_list()
    prev_x_df, prev_y_df, next_x_df, next_y_df = _measurement_neighbor_tables(
        phase_x, phase_y
    )

    data = data[data["name"].isin(bpm_list)].copy()

    betay = BETA + "Y"
    betax = BETA + "X"
    alphax = ALPHA + "X"
    alphay = ALPHA + "Y"
    orbit_x_col = ORBIT + "X"
    orbit_y_col = ORBIT + "Y"

    sqrt_betax = np.sqrt(beta_x[betax])
    sqrt_betay = np.sqrt(beta_y[betay])
    maps = LatticeMaps(
        betax=beta_x[betax].to_dict(),
        betay=beta_y[betay].to_dict(),
        sqrt_betax=sqrt_betax.to_dict(),
        sqrt_betay=sqrt_betay.to_dict(),
        alfax=beta_x[alphax].to_dict(),
        alfay=beta_y[alphay].to_dict(),
    )

    if subtract_mean:
        data.loc[:, "x_mean"] = data["name"].map(orbit_x[orbit_x_col].to_dict())
        data.loc[:, "y_mean"] = data["name"].map(orbit_y[orbit_y_col].to_dict())
        data.loc[:, "x"] -= data["x_mean"]
        data.loc[:, "y"] -= data["y_mean"]

    bpm_index = {bpm: idx for idx, bpm in enumerate(bpm_list)}

    data_p = data.join(prev_x_df, on="name", rsuffix="_px")
    data_p = data_p.join(prev_y_df, on="name", rsuffix="_py")
    data_n = data.join(next_x_df, on="name", rsuffix="_nx")
    data_n = data_n.join(next_y_df, on="name", rsuffix="_ny")

    attach_lattice_columns(data_p, maps)
    attach_lattice_columns(data_n, maps)

    data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(maps.sqrt_betax)
    data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(maps.sqrt_betay)
    data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(maps.sqrt_betax)
    data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(maps.sqrt_betay)

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(
        data_p, data_n, bpm_index
    )
    data_p, data_n = merge_neighbor_coords(
        data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n
    )

    data_p = momenta_from_prev(data_p)
    data_n = momenta_from_next(data_n)

    sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n, maps.betax, maps.betay)

    if subtract_mean:
        _add_bpm_means(data_p, data_n, data_avg)
        _cleanup_mean_cols(data_p, data_n, data_avg)

    return data_avg[OUT_COLS]
