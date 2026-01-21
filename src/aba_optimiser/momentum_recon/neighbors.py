from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.momentum_recon.core import attach_lattice_columns, build_lattice_maps
from aba_optimiser.momentum_recon.schema import (
    CORE_ID_COLS,
    NEXT,
    PLANE_X,
    PLANE_Y,
    POSITION_COLS,
    PREV,
    SUFFIX_NEXT,
    SUFFIX_PREV,
    VARIANCE_COLS,
)
from aba_optimiser.physics.bpm_phases import next_bpm_to_pi_2, prev_bpm_to_pi_2

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.momentum_recon.core import LatticeMaps


def prepare_neighbor_views(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    include_dispersion: bool = False,
    include_errors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], LatticeMaps]:
    """Build prev/next views with lattice columns and optional dispersion."""
    bpm_list = data["name"].unique().tolist()
    tws = tws[tws.index.isin(bpm_list)]
    maps = build_lattice_maps(tws, include_dispersion=include_dispersion)

    prev_x_df, prev_y_df, next_x_df, next_y_df = build_lattice_neighbor_tables(tws, include_errors)
    bpm_index = {bpm: idx for idx, bpm in enumerate(bpm_list)}

    data_p = data.join(prev_x_df, on="name", rsuffix="_px")
    data_p = data_p.join(prev_y_df, on="name", rsuffix="_py")
    data_n = data.join(next_x_df, on="name", rsuffix="_nx")
    data_n = data_n.join(next_y_df, on="name", rsuffix="_ny")

    attach_lattice_columns(data_p, maps)
    attach_lattice_columns(data_n, maps)

    data_p["sqrt_betax_p"] = data_p[PREV.bpm_x].map(maps.sqrt_betax)
    data_p["sqrt_betay_p"] = data_p[PREV.bpm_y].map(maps.sqrt_betay)
    data_n["sqrt_betax_n"] = data_n[NEXT.bpm_x].map(maps.sqrt_betax)
    data_n["sqrt_betay_n"] = data_n[NEXT.bpm_y].map(maps.sqrt_betay)

    if include_dispersion:
        if maps.dx is None or maps.dpx is None or maps.dy is None or maps.dpy is None:
            raise RuntimeError("Dispersion maps were not initialised correctly")
        data_p[PREV.dx] = data_p[PREV.bpm_x].map(maps.dx)
        data_p[PREV.dy] = data_p[PREV.bpm_y].map(maps.dy)
        data_n[NEXT.dx] = data_n[NEXT.bpm_x].map(maps.dx)
        data_n[NEXT.dy] = data_n[NEXT.bpm_y].map(maps.dy)

    return data_p, data_n, bpm_index, maps


def build_lattice_neighbor_tables(
    tws: pd.DataFrame,
    include_errors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build neighbor BPM tables from twiss data."""
    total_var_x, total_var_y, mu1_var, mu2_var = None, None, None, None
    if include_errors:
        total_var_x = tws.headers["mu1_total_var"]
        total_var_y = tws.headers["mu2_total_var"]
        mu1_var = tws["mu1_var"]
        mu2_var = tws["mu2_var"]

    prev_x = prev_bpm_to_pi_2(tws["mu1"], tws.q1, mu_var=mu1_var, total_var=total_var_x).rename(
        columns={"prev_bpm": PREV.bpm_x, "delta": PREV.delta_x, "delta_err": f"{PREV.delta_x}_err"}
    )
    prev_y = prev_bpm_to_pi_2(tws["mu2"], tws.q2, mu_var=mu2_var, total_var=total_var_y).rename(
        columns={"prev_bpm": PREV.bpm_y, "delta": PREV.delta_y, "delta_err": f"{PREV.delta_y}_err"}
    )
    next_x = next_bpm_to_pi_2(tws["mu1"], tws.q1, mu_var=mu1_var, total_var=total_var_x).rename(
        columns={"next_bpm": NEXT.bpm_x, "delta": NEXT.delta_x, "delta_err": f"{NEXT.delta_x}_err"}
    )
    next_y = next_bpm_to_pi_2(tws["mu2"], tws.q2, mu_var=mu2_var, total_var=total_var_y).rename(
        columns={"next_bpm": NEXT.bpm_y, "delta": NEXT.delta_y, "delta_err": f"{NEXT.delta_y}_err"}
    )
    return prev_x, prev_y, next_x, next_y


def compute_turn_wraps(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    bpm_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur_i_p = data_p["name"].map(bpm_index)
    prev_ix = data_p[PREV.bpm_x].map(bpm_index)
    prev_iy = data_p[PREV.bpm_y].map(bpm_index)

    cur_i_n = data_n["name"].map(bpm_index)
    next_ix = data_n[NEXT.bpm_x].map(bpm_index)
    next_iy = data_n[NEXT.bpm_y].map(bpm_index)

    turn_x_p = data_p["turn"] - (cur_i_p < prev_ix).astype(np.int16)
    turn_y_p = data_p["turn"] - (cur_i_p < prev_iy).astype(np.int16)
    turn_x_n = data_n["turn"] + (cur_i_n > next_ix).astype(np.int16)
    turn_y_n = data_n["turn"] + (cur_i_n > next_iy).astype(np.int16)
    return turn_x_p, turn_y_p, turn_x_n, turn_y_n


def merge_neighbor_coords(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    turn_x_p: np.ndarray,
    turn_y_p: np.ndarray,
    turn_x_n: np.ndarray,
    turn_y_n: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(VARIANCE_COLS)
    missing_p = required.difference(data_p.columns)
    missing_n = required.difference(data_n.columns)
    if missing_p or missing_n:
        raise KeyError(
            "Variance columns missing for neighbour merge: "
            f"data_p missing {sorted(missing_p)}; data_n missing {sorted(missing_n)}"
        )

    coords_p = data_p[list(CORE_ID_COLS + POSITION_COLS + VARIANCE_COLS)]
    coords_n = data_n[list(CORE_ID_COLS + POSITION_COLS + VARIANCE_COLS)]

    def get_neighbor_config(suffix: str, plane: str) -> tuple[str, str, dict[str, str], list[str]]:
        neighbor = PREV if suffix == "p" else NEXT
        rename_dict = {
            "turn": f"turn_{plane}_{suffix}",
            "name": getattr(neighbor, f"bpm_{plane}"),
            plane: getattr(neighbor, plane),
            f"var_{plane}": getattr(neighbor, f"var_{plane}"),
        }
        select_cols = [
            f"turn_{plane}_{suffix}",
            getattr(neighbor, f"bpm_{plane}"),
            getattr(neighbor, plane),
            getattr(neighbor, f"var_{plane}"),
        ]
        return suffix, plane, rename_dict, select_cols

    # Configurations for renaming and selecting columns
    configs = [get_neighbor_config(s, p) for s in [SUFFIX_PREV, SUFFIX_NEXT] for p in [PLANE_X, PLANE_Y]]

    # Create renamed coordinate dataframes
    coord_frames = {}
    for suffix, plane, rename_dict, select_cols in configs:
        source = coords_p if suffix == "p" else coords_n
        coord_frames[f"{suffix}_{plane}"] = source.rename(columns=rename_dict)[select_cols]

    # Add turn columns
    data_p["turn_x_p"] = turn_x_p
    data_p["turn_y_p"] = turn_y_p
    data_n["turn_x_n"] = turn_x_n
    data_n["turn_y_n"] = turn_y_n

    # Perform merges
    data_p = data_p.merge(coord_frames["p_x"], on=["turn_x_p", PREV.bpm_x], how="left", copy=False)
    data_p = data_p.merge(coord_frames["p_y"], on=["turn_y_p", PREV.bpm_y], how="left", copy=False)
    data_n = data_n.merge(coord_frames["n_x"], on=["turn_x_n", NEXT.bpm_x], how="left", copy=False)
    data_n = data_n.merge(coord_frames["n_y"], on=["turn_y_n", NEXT.bpm_y], how="left", copy=False)

    # Fill NaNs
    for col in (PREV.x, PREV.y):
        data_p[col] = data_p[col].fillna(0.0)
    for col in (NEXT.x, NEXT.y):
        data_n[col] = data_n[col].fillna(0.0)

    for col in (PREV.var_x, PREV.var_y):
        data_p[col] = data_p[col].fillna(np.inf)
    for col in (NEXT.var_x, NEXT.var_y):
        data_n[col] = data_n[col].fillna(np.inf)

    # Drop temporary turn columns
    data_p.drop(columns=["turn_x_p", "turn_y_p"], inplace=True)
    data_n.drop(columns=["turn_x_n", "turn_y_n"], inplace=True)
    return data_p, data_n
