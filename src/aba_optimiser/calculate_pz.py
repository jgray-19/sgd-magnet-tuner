from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    MAGNET_RANGE,
    POSITION_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import next_bpm_to_pi_2, prev_bpm_to_pi_2

if TYPE_CHECKING:
    import tfs


def calculate_pz(
    orig_data: tfs.TfsDataFrame,
    inject_noise: bool = True,
    tws: None | tfs.TfsDataFrame = None,
    info: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """
    Generates two noisy DataFrames (data_p, data_n) based on config values.
    Prioritizes computation speed and avoids unnecessary RAM usage.
    """
    out_cols = ["name", "turn", "x", "px", "y", "py"]
    # np.random.seed(seed)

    # 1. READ TRACK DATA + ADD NOISE (only x/y, in-place)
    for col in out_cols:
        if col not in orig_data.columns and (info or col not in ["px", "py"]):
            raise ValueError(f"Column '{col}' not found in the original data.")
    data = orig_data.copy(deep=True)  # share underlying data, just new view

    if rng is None:
        rng = np.random.default_rng()  # Use new Generator API

    if inject_noise:
        data["x"] = orig_data["x"] + rng.normal(
            0, POSITION_STD_DEV, size=len(orig_data)
        )
        data["y"] = orig_data["y"] + rng.normal(
            0, POSITION_STD_DEV, size=len(orig_data)
        )

    # 2. INITIALISE MAD-X & GET TWISS (small, only once)
    if tws is None:
        mad = MadInterface(SEQUENCE_FILE, MAGNET_RANGE)
        mad.mad.send(f"tws = twiss{{sequence=MADX.{SEQ_NAME}, observe=1}}")
        tws = mad.mad.tws.to_df().set_index("name")

    # 3. PRECOMPUTE LATTICE FUNCTIONS (dicts for speed)
    sqrt_betax = np.sqrt(tws["beta11"])
    sqrt_betay = np.sqrt(tws["beta22"])
    alfax = tws["alfa11"]
    alfay = tws["alfa22"]
    map_sqrt_betax = sqrt_betax.to_dict()
    map_sqrt_betay = sqrt_betay.to_dict()
    map_alfax = alfax.to_dict()
    map_alfay = alfay.to_dict()

    # prev/next BPMs
    prev_x_df = prev_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
        columns={"prev_bpm": "prev_bpm_x", "delta": "delta_x"}
    )
    prev_y_df = prev_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
        columns={"prev_bpm": "prev_bpm_y", "delta": "delta_y"}
    )
    next_x_df = next_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
        columns={"next_bpm": "next_bpm_x", "delta": "delta_x"}
    )
    next_y_df = next_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
        columns={"next_bpm": "next_bpm_y", "delta": "delta_y"}
    )

    bpm_list = tws.index.to_list()
    bpm_index = {b: i for i, b in enumerate(bpm_list)}

    # 4. JOIN DATA (minimally: no repeated .copy())
    data_p = data.join(prev_x_df, on="name", rsuffix="_px")
    data_p = data_p.join(prev_y_df, on="name", rsuffix="_py")
    data_n = data.join(next_x_df, on="name", rsuffix="_nx")
    data_n = data_n.join(next_y_df, on="name", rsuffix="_ny")

    # Map lattice functions in-place
    for col, mapper in (
        ("sqrt_betax", map_sqrt_betax),
        ("sqrt_betay", map_sqrt_betay),
        ("alfax", map_alfax),
        ("alfay", map_alfay),
    ):
        data_p[col] = data_p["name"].map(mapper)
        data_n[col] = data_n["name"].map(mapper)

    # Compute indices (vectorised, single pass)
    cur_i_p = data_p["name"].map(bpm_index)
    prev_ix = data_p["prev_bpm_x"].map(bpm_index)
    prev_iy = data_p["prev_bpm_y"].map(bpm_index)
    cur_i_n = data_n["name"].map(bpm_index)
    next_ix = data_n["next_bpm_x"].map(bpm_index)
    next_iy = data_n["next_bpm_y"].map(bpm_index)

    turn_x_p = data_p["turn"] - (cur_i_p < prev_ix).astype(np.int8)
    turn_y_p = data_p["turn"] - (cur_i_p < prev_iy).astype(np.int8)
    turn_x_n = data_n["turn"] + (cur_i_n > next_ix).astype(np.int8)
    turn_y_n = data_n["turn"] + (cur_i_n > next_iy).astype(np.int8)

    # --- Construct minimal coord DataFrames for merging ---
    coords_p = data_p[["turn", "name", "x", "y"]]
    coords_n = data_n[["turn", "name", "x", "y"]]

    # These merges are tiny compared to the full DataFrame,
    # so it's ok (but we drop intermediates right after).
    coords_x_p = coords_p.rename(
        columns={"turn": "turn_x_p", "name": "prev_bpm_x", "x": "prev_x"}
    ).loc[:, ["turn_x_p", "prev_bpm_x", "prev_x"]]
    coords_y_p = coords_p.rename(
        columns={"turn": "turn_y_p", "name": "prev_bpm_y", "y": "prev_y"}
    ).loc[:, ["turn_y_p", "prev_bpm_y", "prev_y"]]
    coords_x_n = coords_n.rename(
        columns={"turn": "turn_x_n", "name": "next_bpm_x", "x": "next_x"}
    ).loc[:, ["turn_x_n", "next_bpm_x", "next_x"]]
    coords_y_n = coords_n.rename(
        columns={"turn": "turn_y_n", "name": "next_bpm_y", "y": "next_y"}
    ).loc[:, ["turn_y_n", "next_bpm_y", "next_y"]]
    del coords_p, coords_n

    # Attach turn_x_p/turn_y_p, merge in prev_x/prev_y
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

    data_p["prev_x"] = data_p["prev_x"].fillna(0)
    data_p["prev_y"] = data_p["prev_y"].fillna(0)
    data_n["next_x"] = data_n["next_x"].fillna(0)
    data_n["next_y"] = data_n["next_y"].fillna(0)

    data_p.drop(columns=["turn_x_p", "turn_y_p"], inplace=True)
    data_n.drop(columns=["turn_x_n", "turn_y_n"], inplace=True)
    del coords_x_p, coords_y_p, coords_x_n, coords_y_n  # release

    # Upstream lattice functions (fast, no copy)
    data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(map_sqrt_betax)
    data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(map_sqrt_betay)
    data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(map_sqrt_betax)
    data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(map_sqrt_betay)

    # 5. px/py COMPUTATION (use local variables, not DataFrame slicing)
    # --- prev (data_p) ---
    x1 = data_p["prev_x"].to_numpy() / data_p["sqrt_betax_p"].to_numpy()
    x2 = data_p["x"].to_numpy() / data_p["sqrt_betax"].to_numpy()
    y1 = data_p["prev_y"].to_numpy() / data_p["sqrt_betay_p"].to_numpy()
    y2 = data_p["y"].to_numpy() / data_p["sqrt_betay"].to_numpy()

    phi_x = data_p["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_p["delta_y"].to_numpy() * 2 * np.pi
    c_x, s_x, t_x = np.cos(phi_x), np.sin(phi_x), np.tan(phi_x)
    c_y, s_y, t_y = np.cos(phi_y), np.sin(phi_y), np.tan(phi_y)
    alfax = data_p["alfax"].to_numpy()
    alfay = data_p["alfay"].to_numpy()
    sqrt_betax = data_p["sqrt_betax"].to_numpy()
    sqrt_betay = data_p["sqrt_betay"].to_numpy()

    data_p["px"] = -(x1 * (c_x + s_x * t_x) + x2 * (t_x + alfax)) / sqrt_betax
    data_p["py"] = -(y1 * (c_y + s_y * t_y) + y2 * (t_y + alfay)) / sqrt_betay

    # --- next (data_n) ---
    x1 = data_n["x"].to_numpy() / data_n["sqrt_betax"].to_numpy()
    x2 = data_n["next_x"].to_numpy() / data_n["sqrt_betax_n"].to_numpy()
    y1 = data_n["y"].to_numpy() / data_n["sqrt_betay"].to_numpy()
    y2 = data_n["next_y"].to_numpy() / data_n["sqrt_betay_n"].to_numpy()

    phi_x = (data_n["delta_x"].to_numpy() + 0.25) * 2 * np.pi
    phi_y = (data_n["delta_y"].to_numpy() + 0.25) * 2 * np.pi
    c_x, s_x = np.cos(phi_x), np.sin(phi_x)
    c_y, s_y = np.cos(phi_y), np.sin(phi_y)
    alfax = data_n["alfax"].to_numpy()
    alfay = data_n["alfay"].to_numpy()
    sqrt_betax = data_n["sqrt_betax"].to_numpy()
    sqrt_betay = data_n["sqrt_betay"].to_numpy()

    data_n["px"] = ((x2 - x1 * c_x) / s_x - alfax * x1) / sqrt_betax
    data_n["py"] = ((y2 - y1 * c_y) / s_y - alfay * y1) / sqrt_betay

    # Final sync, as in original
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

    # 6. Print the differences and standard deviations
    if info:
        x_diff_p = data_p["x"] - orig_data["x"]
        px_diff_p = data_p["px"] - orig_data["px"]
        y_diff_p = data_p["y"] - orig_data["y"]
        py_diff_p = data_p["py"] - orig_data["py"]
        print("x_diff mean (prev w/ k)", x_diff_p.abs().mean(), "±", x_diff_p.std())
        print("y_diff mean (prev w/ k)", y_diff_p.abs().mean(), "±", y_diff_p.std())
        print("px_diff mean (prev w/ k)", px_diff_p.abs().mean(), "±", px_diff_p.std())
        print("py_diff mean (prev w/ k)", py_diff_p.abs().mean(), "±", py_diff_p.std())

        x_diff_n = data_n["x"] - orig_data["x"]
        px_diff_n = data_n["px"] - orig_data["px"]
        y_diff_n = data_n["y"] - orig_data["y"]
        py_diff_n = data_n["py"] - orig_data["py"]
        print("x_diff mean (next w/ k)", x_diff_n.abs().mean(), "±", x_diff_n.std())
        print("y_diff mean (next w/ k)", y_diff_n.abs().mean(), "±", y_diff_n.std())
        print("px_diff mean (next w/ k)", px_diff_n.abs().mean(), "±", px_diff_n.std())
        print("py_diff mean (next w/ k)", py_diff_n.abs().mean(), "±", py_diff_n.std())

    return data_p[out_cols], data_n[out_cols]
