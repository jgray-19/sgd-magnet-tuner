from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    MAGNET_RANGE,
    POSITION_STD_DEV,
    SEQUENCE_FILE,
)
from aba_optimiser.mad.mad_interface import MadInterface
from aba_optimiser.physics.bpm_phases import next_bpm_to_pi_2, prev_bpm_to_pi_2

if TYPE_CHECKING:
    import tfs

LOGGER = logging.getLogger(__name__)


def calculate_pz(
    orig_data: tfs.TfsDataFrame,
    inject_noise: bool = True,
    tws: None | tfs.TfsDataFrame = None,
    info: bool = True,
    rng: np.random.Generator | None = None,
    low_noise_bpms: list[str] = [],
) -> tuple[tfs.TfsDataFrame, tfs.TfsDataFrame]:
    """
    Generates two noisy DataFrames (data_p, data_n) based on config values.
    Prioritizes computation speed and avoids unnecessary RAM usage.
    """
    LOGGER.info(
        f"Calculating transverse momentum - inject_noise={inject_noise}, low_noise_bpms={len(low_noise_bpms) if low_noise_bpms else 0} BPMs"
    )

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
        LOGGER.debug(f"Adding Gaussian noise with std={POSITION_STD_DEV}")
        # Generate per-row Gaussian noise and reduce it for low-error BPMs
        n = len(orig_data)
        noise_x = rng.normal(0, POSITION_STD_DEV, size=n)
        noise_y = rng.normal(0, POSITION_STD_DEV, size=n)

        if low_noise_bpms:
            # mask of rows whose BPM name is in the low_error_bpms list
            mask = orig_data["name"].isin(low_noise_bpms)
            if mask.any():
                low_n = int(mask.sum())
                LOGGER.debug(
                    f"Reducing noise for {low_n} measurements at {len(low_noise_bpms)} low-noise BPMs"
                )
                # Replace noise for these BPMs with 10x lower std dev
                noise_x[mask.to_numpy()] = rng.normal(
                    0, POSITION_STD_DEV / 10.0, size=low_n
                )
                noise_y[mask.to_numpy()] = rng.normal(
                    0, POSITION_STD_DEV / 10.0, size=low_n
                )

        data["x"] = orig_data["x"] + noise_x
        data["y"] = orig_data["y"] + noise_y

    # Subtract mean coordinates at each BPM to center around closed orbit
    LOGGER.warning(
        "Not centering the BPMs - plus and minus deltap have different COs - Need to be discussed. "
    )
    # bpm_means = data.groupby("name", observed=False)[["x", "y"]].mean()
    # if info:
    #     print("BPM means (x, y):")
    #     print(bpm_means)
    # data = data.merge(bpm_means, on="name", suffixes=("", "_mean"))
    # data["x"] = data["x"] - data["x_mean"]
    # data["y"] = data["y"] - data["y_mean"]
    # data.drop(columns=["x_mean", "y_mean"], inplace=True)

    # Add weight_x and weight_y = 1 for all rows
    data["weight_x"] = 1.0
    data["weight_y"] = 1.0

    # Now the data needs to be discretised to simulate a real BPM measurement.
    # The unit of discretisation is 2.333e-5. But discretise to these values: ..., -3.333e-5, -1e-5, 1.333e-5, 3.666e-5, ...
    # data["x"] = np.round((data["x"] + 1e-5) / 2.333e-5) * 2.333e-5 - 1e-5
    # data["y"] = np.round((data["y"] + 1e-5) / 2.333e-5) * 2.333e-5 - 1e-5

    # 2. INITIALISE MAD-X & GET TWISS (small, only once)
    if tws is None:
        mad = MadInterface(SEQUENCE_FILE, MAGNET_RANGE, bpm_pattern="BPM")
        tws = mad.run_twiss()
        if info:
            print("Found tunes:", tws.q1, tws.q2)

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

    # Use Δμ directly (no +π/2 shift).
    # Derived formula:
    #   sinφ₁ * √(2J) = ( - z₂/√β₂  - (z₁/√β₁) * sinΔμ ) / cosΔμ
    #   p₁ = - (1/√β₁) * [ √(2J) sinφ₁ ] - α₁ * z₁/β₁
    #
    # where:
    #   z₁ = position at BPM1 (this BPM)
    #   z₂ = position at BPM2 (the next BPM, ~π/2 phase advance downstream)
    #   β₁, β₂ = beta functions at BPM1 and BPM2
    #   α₁ = alpha function at BPM1
    #   Δμ = phase advance BPM1 → BPM2

    # Normalised coordinates at BPM1 and BPM2
    k1x = data_n["x"].to_numpy() / data_n["sqrt_betax"].to_numpy()
    k2x = data_n["next_x"].to_numpy() / data_n["sqrt_betax_n"].to_numpy()
    k1y = data_n["y"].to_numpy() / data_n["sqrt_betay"].to_numpy()
    k2y = data_n["next_y"].to_numpy() / data_n["sqrt_betay_n"].to_numpy()

    # Phase advance to the *next* BPM (no +0.25 offset)
    phi_x = data_n["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_n["delta_y"].to_numpy() * 2 * np.pi
    cos_x, sin_x = np.cos(phi_x), np.sin(phi_x)
    cos_y, sin_y = np.cos(phi_y), np.sin(phi_y)

    # Compute √(2J) sinφ₁ from BPM1 & BPM2 positions
    sin_term_x = (-k2x - k1x * sin_x) / cos_x
    sin_term_y = (-k2y - k1y * sin_y) / cos_y

    # Lattice functions at BPM1
    alfax = data_n["alfax"].to_numpy()
    alfay = data_n["alfay"].to_numpy()
    sqrt_betax = data_n["sqrt_betax"].to_numpy()
    sqrt_betay = data_n["sqrt_betay"].to_numpy()
    beta_x = sqrt_betax**2
    beta_y = sqrt_betay**2

    # Final momentum estimates at BPM1
    data_n["px"] = (-sin_term_x / sqrt_betax) - alfax * (
        data_n["x"].to_numpy()
    ) / beta_x
    data_n["py"] = (-sin_term_y / sqrt_betay) - alfay * (
        data_n["y"].to_numpy()
    ) / beta_y

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

    # --- Average the two momentum estimates in a statistically sensible way ---
    # Both branches (prev/next) become ill-conditioned near Δμ ≈ π/2+nπ because
    # tanΔμ or 1/cosΔμ terms amplify noise. A simple mean can therefore bias the
    # result when one side sits close to a singular geometry.
    #
    # Practical fix: weight each estimate by cos²(Δμ) of the corresponding link,
    # which tracks the inverse variance of the estimator around those singular
    # points (cosΔμ→0 ⇒ weight→0). This uses the same Δμ (no +π/2) you already
    # used to compute p from each side.
    data_avg = data_p.copy(deep=True)  # keep same index/columns as data_p

    # Phase advances for the two links used to build px,py
    phi_x_prev = (data_p["delta_x"].to_numpy()) * 2 * np.pi  # BPM(prev) → BPM(this)
    phi_y_prev = (data_p["delta_y"].to_numpy()) * 2 * np.pi
    phi_x_next = (data_n["delta_x"].to_numpy()) * 2 * np.pi  # BPM(this) → BPM(next)
    phi_y_next = (data_n["delta_y"].to_numpy()) * 2 * np.pi

    # Inverse-variance style weights ∝ cos²Δμ
    wpx_prev = np.cos(phi_x_prev) ** 2
    wpy_prev = np.cos(phi_y_prev) ** 2
    wpx_next = np.cos(phi_x_next) ** 2
    wpy_next = np.cos(phi_y_next) ** 2

    # Small epsilon guards against (rare) cases where both weights ≈ 0
    _eps = 0  # 1e-12

    # Weighted averages
    data_avg["px"] = (
        wpx_prev * data_p["px"].to_numpy() + wpx_next * data_n["px"].to_numpy()
    ) / (wpx_prev + wpx_next + _eps)
    data_avg["py"] = (
        wpy_prev * data_p["py"].to_numpy() + wpy_next * data_n["py"].to_numpy()
    ) / (wpy_prev + wpy_next + _eps)

    # 6. Print the differences and standard deviations
    if info:
        x_diff_p = data_p["x"] - orig_data["x"]
        px_diff_p = data_p["px"] - orig_data["px"]
        y_diff_p = data_p["y"] - orig_data["y"]
        py_diff_p = data_p["py"] - orig_data["py"]
        print("x_diff mean", x_diff_p.abs().mean(), "±", x_diff_p.std())
        print("y_diff mean", y_diff_p.abs().mean(), "±", y_diff_p.std())

        print("MOMENTUM DIFFERENCES ------")
        print("px_diff mean (prev w/ k)", px_diff_p.abs().mean(), "±", px_diff_p.std())
        print("py_diff mean (prev w/ k)", py_diff_p.abs().mean(), "±", py_diff_p.std())

        px_diff_n = data_n["px"] - orig_data["px"]
        py_diff_n = data_n["py"] - orig_data["py"]
        print("px_diff mean (next w/ k)", px_diff_n.abs().mean(), "±", px_diff_n.std())
        print("py_diff mean (next w/ k)", py_diff_n.abs().mean(), "±", py_diff_n.std())

        px_diff_avg = data_avg["px"] - orig_data["px"]
        py_diff_avg = data_avg["py"] - orig_data["py"]
        print("px_diff mean (avg)", px_diff_avg.abs().mean(), "±", px_diff_avg.std())
        print("py_diff mean (avg)", py_diff_avg.abs().mean(), "±", py_diff_avg.std())

        # fmt: off
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        # Or filter out small original values
        mask_px = orig_data["px"].abs() > epsilon
        mask_py = orig_data["py"].abs() > epsilon

        if mask_px.any():
            px_diff_rel_filtered = px_diff_avg[mask_px] / orig_data["px"][mask_px]
            print("px_diff mean (avg rel)", px_diff_rel_filtered.abs().mean(), "±", px_diff_rel_filtered.std())
        else:
            print("px_diff mean (avg rel): No significant px values")

        if mask_py.any():
            py_diff_rel_filtered = py_diff_avg[mask_py] / orig_data["py"][mask_py]
            print("py_diff mean (avg rel)", py_diff_rel_filtered.abs().mean(), "±", py_diff_rel_filtered.std())
        else:
            print("py_diff mean (avg rel): No significant py values")
        # fmt: on

    return data_p[out_cols], data_n[out_cols], data_avg[out_cols]
