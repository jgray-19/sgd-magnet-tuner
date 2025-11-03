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
    DRIVEN_PHASE_NAME,
    # TOTAL_PHASE_NAME,
    NAME,
    NAME2,
    ORBIT,
    ORBIT_NAME,
    PHASE,
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
    output_bpm_list: list[str],
    phase_dict: dict[tuple[str, str], float],
    full_bpm_list: list[str],
    target_phase: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def get_forward_phase(from_bpm: str, to_bpm: str) -> float:
        i1 = full_bpm_list.index(from_bpm)
        i2 = full_bpm_list.index(to_bpm)
        n = len(full_bpm_list)
        dist = (i2 - i1) % n
        if dist == 0:
            return 0.0
        return sum(
            phase_dict[(full_bpm_list[(i1 + j) % n], full_bpm_list[(i1 + j + 1) % n])]
            for j in range(dist)
        )

    def find_best_neighbor(bpm: str, direction: str) -> tuple[str, float]:
        bpm_count = len(output_bpm_list)
        index = output_bpm_list.index(bpm)
        phase = 0.0
        delta = np.inf
        best_bpm = None
        current = bpm
        for offset in range(1, 12):
            if direction == "prev":
                neighbor_index = (index - offset) % bpm_count
                neighbor = output_bpm_list[neighbor_index]
                phase += get_forward_phase(neighbor, current)
                new_delta = target_phase - phase
            else:
                neighbor_index = (index + offset) % bpm_count
                neighbor = output_bpm_list[neighbor_index]
                phase += get_forward_phase(current, neighbor)
                new_delta = phase - target_phase

            # Check if this neighbor is better
            if abs(new_delta) < abs(delta):
                delta = new_delta
                best_bpm = neighbor

            # Check if we need to stop searching, if we passed the target phase
            # if phase > target_phase:
            #     break
            current = neighbor
        if not best_bpm:
            raise ValueError(f"No {direction} BPM found for {bpm}")
        return best_bpm, delta

    best_prev = {bpm: find_best_neighbor(bpm, "prev") for bpm in output_bpm_list}
    best_next = {bpm: find_best_neighbor(bpm, "next") for bpm in output_bpm_list}

    prev_df = pd.DataFrame(
        [(bpm, best[0], best[1]) for bpm, best in best_prev.items()],
        columns=["bpm", "prev_bpm", "delta"],
    ).set_index("bpm")
    next_df = pd.DataFrame(
        [(bpm, best[0], best[1]) for bpm, best in best_next.items()],
        columns=["bpm", "next_bpm", "delta"],
    ).set_index("bpm")
    return prev_df, next_df


def _measurement_neighbor_tables(
    phase_x: pd.DataFrame,
    phase_y: pd.DataFrame,
    bpm_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build phase dicts and full bpm list
    full_bpm_list = phase_x[NAME].tolist()
    phase_dict_x = {
        (row[NAME], row[NAME2]): row[PHASE + "X"] for _, row in phase_x.iterrows()
    }
    phase_dict_y = {
        (row[NAME], row[NAME2]): row[PHASE + "Y"] for _, row in phase_y.iterrows()
    }

    prev_x, next_x = _find_neighbors_from_phase_diffs(
        bpm_list, phase_dict_x, full_bpm_list
    )
    prev_y, next_y = _find_neighbors_from_phase_diffs(
        bpm_list, phase_dict_y, full_bpm_list
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

    bpm_list = data["name"].unique().tolist()
    tws = ensure_twiss(tws, info)
    tws = tws[tws.index.isin(bpm_list)]

    maps = build_lattice_maps(tws)
    prev_x_df, prev_y_df, next_x_df, next_y_df = build_lattice_neighbor_tables(tws)

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
    subtract_mean: bool = False,
    tws: tfs.TfsDataFrame | None = None,
) -> pd.DataFrame:
    raise DeprecationWarning(
        "This function is deprecated and will be removed in a future version."
    )
    analysis_path = Path(analysis_dir)
    phase_x = tfs.read(analysis_path / (DRIVEN_PHASE_NAME + "x.tfs"))
    phase_y = tfs.read(analysis_path / (DRIVEN_PHASE_NAME + "y.tfs"))
    beta_x = tfs.read(analysis_path / (BETA_NAME + "x.tfs"), index=NAME)
    beta_y = tfs.read(analysis_path / (BETA_NAME + "y.tfs"), index=NAME)
    orbit_x = tfs.read(analysis_path / (ORBIT_NAME + "x.tfs"), index=NAME)
    orbit_y = tfs.read(analysis_path / (ORBIT_NAME + "y.tfs"), index=NAME)

    # total_phase_x = tfs.read(analysis_path / (TOTAL_PHASE_NAME + "x.tfs"))
    # total_phase_y = tfs.read(analysis_path / (TOTAL_PHASE_NAME + "y.tfs"))

    # Add the final row from the total phase files to the phase files
    # final_row_x = total_phase_x.iloc[[-1]]
    # final_row_y = total_phase_y.iloc[[-1]]
    # phase_x = pd.concat([phase_x, final_row_x], ignore_index=True)
    # phase_y = pd.concat([phase_y, final_row_y], ignore_index=True)
    # del final_row_x, final_row_y, total_phase_x, total_phase_y

    # Select only necessary columns
    total_to_final_x = sum(phase_x["PHASEXMDL"]) + sum(phase_x["DELTAPHASEX"])
    total_to_final_y = sum(phase_y["PHASEYMDL"]) + sum(phase_y["DELTAPHASEY"])
    phase_x = phase_x[[NAME, NAME2, PHASE + "X"]]
    phase_y = phase_y[[NAME, NAME2, PHASE + "Y"]]

    # Add final row
    final_row_x = {
        NAME: phase_x.iloc[-1][NAME2],
        NAME2: phase_x.iloc[0][NAME],
        PHASE + "X": 0.27 - total_to_final_x % 1,
    }
    final_row_y = {
        NAME: phase_y.iloc[-1][NAME2],
        NAME2: phase_y.iloc[0][NAME],
        PHASE + "Y": 0.322 - total_to_final_y % 1,
    }
    phase_x = pd.concat([phase_x, pd.DataFrame([final_row_x])], ignore_index=True)
    phase_y = pd.concat([phase_y, pd.DataFrame([final_row_y])], ignore_index=True)

    bpm_list = list(
        set(phase_x[NAME].unique().tolist())
        & set(phase_y[NAME].unique().tolist())
        & set(phase_x[NAME2].unique().tolist())
        & set(phase_y[NAME2].unique().tolist())
        & set(beta_x.index.to_list())
        & set(beta_y.index.to_list())
        & set(orbit_x.index.to_list())
        & set(orbit_y.index.to_list())
    )

    # Reorder the bpm_list to match the order in beta_x
    bpm_list = [bpm for bpm in phase_x[NAME].to_list() if bpm in bpm_list]

    prev_x_df, prev_y_df, next_x_df, next_y_df = _measurement_neighbor_tables(
        phase_x, phase_y, bpm_list
    )
    if tws is not None:
        prev_x_df_tws, prev_y_df_tws, next_x_df_tws, next_y_df_tws = (
            build_lattice_neighbor_tables(tws)
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

    return data_n[OUT_COLS]  # data_p seems wrong -> avg will be wrong.
