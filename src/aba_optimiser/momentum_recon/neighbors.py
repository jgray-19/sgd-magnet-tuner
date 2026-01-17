from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.physics.bpm_phases import next_bpm_to_pi_2, prev_bpm_to_pi_2

if TYPE_CHECKING:
    import pandas as pd


def build_lattice_neighbor_tables(
    tws: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def compute_turn_wraps(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    bpm_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def merge_neighbor_coords(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    turn_x_p: np.ndarray,
    turn_y_p: np.ndarray,
    turn_x_n: np.ndarray,
    turn_y_n: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"var_x", "var_y"}
    missing_p = required.difference(data_p.columns)
    missing_n = required.difference(data_n.columns)
    if missing_p or missing_n:
        raise KeyError(
            "Variance columns missing for neighbour merge: "
            f"data_p missing {sorted(missing_p)}; data_n missing {sorted(missing_n)}"
        )

    coords_p = data_p[["turn", "name", "x", "y", "var_x", "var_y"]]
    coords_n = data_n[["turn", "name", "x", "y", "var_x", "var_y"]]

    coords_x_p = coords_p.rename(
        columns={
            "turn": "turn_x_p",
            "name": "prev_bpm_x",
            "x": "prev_x",
            "var_x": "prev_x_var",
        }
    )[["turn_x_p", "prev_bpm_x", "prev_x", "prev_x_var"]]

    coords_y_p = coords_p.rename(
        columns={
            "turn": "turn_y_p",
            "name": "prev_bpm_y",
            "y": "prev_y",
            "var_y": "prev_y_var",
        }
    )[["turn_y_p", "prev_bpm_y", "prev_y", "prev_y_var"]]

    coords_x_n = coords_n.rename(
        columns={
            "turn": "turn_x_n",
            "name": "next_bpm_x",
            "x": "next_x",
            "var_x": "next_x_var",
        }
    )[["turn_x_n", "next_bpm_x", "next_x", "next_x_var"]]

    coords_y_n = coords_n.rename(
        columns={
            "turn": "turn_y_n",
            "name": "next_bpm_y",
            "y": "next_y",
            "var_y": "next_y_var",
        }
    )[["turn_y_n", "next_bpm_y", "next_y", "next_y_var"]]

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

    for frame, column in (
        (data_p, "prev_x"),
        (data_p, "prev_y"),
        (data_n, "next_x"),
        (data_n, "next_y"),
    ):
        frame[column] = frame[column].fillna(0.0)

    for frame, column in (
        (data_p, "prev_x_var"),
        (data_p, "prev_y_var"),
        (data_n, "next_x_var"),
        (data_n, "next_y_var"),
    ):
        frame[column] = frame[column].fillna(np.inf)

    data_p.drop(columns=["turn_x_p", "turn_y_p"], inplace=True)
    data_n.drop(columns=["turn_x_n", "turn_y_n"], inplace=True)
    return data_p, data_n
