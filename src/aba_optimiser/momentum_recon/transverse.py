from __future__ import annotations

import contextlib
import logging

import numpy as np
import pandas as pd
from omc3.optics_measurements.constants import (
    NAME,
    NAME2,
    PHASE,
)

from aba_optimiser.config import POSITION_STD_DEV
from aba_optimiser.momentum_recon.core import (
    OUT_COLS,
    diagnostics,
    get_rng,
    inject_noise_xy_inplace,
    remove_closed_orbit_inplace,
    restore_closed_orbit_and_reference_momenta_inplace,
    sync_endpoints,
    validate_input,
)
from aba_optimiser.momentum_recon.core import (
    weighted_average_from_weights as weighted_average,
)
from aba_optimiser.momentum_recon.momenta import (
    momenta_from_next,
    momenta_from_prev,
)
from aba_optimiser.momentum_recon.neighbors import (
    compute_turn_wraps,
    merge_neighbor_coords,
    prepare_neighbor_views,
)

LOGGER = logging.getLogger(__name__)


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
        columns=["bpm", "prev_bpm", "delta"],  # ty:ignore[invalid-argument-type]
    ).set_index("bpm")
    next_df = pd.DataFrame(
        [(bpm, best[0], best[1]) for bpm, best in best_next.items()],
        columns=["bpm", "next_bpm", "delta"],  # ty:ignore[invalid-argument-type]
    ).set_index("bpm")
    return prev_df, next_df


def _measurement_neighbor_tables(
    phase_x: pd.DataFrame,
    phase_y: pd.DataFrame,
    bpm_list: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build phase dicts and full bpm list
    full_bpm_list = phase_x[NAME].tolist()
    phase_dict_x = {(row[NAME], row[NAME2]): row[PHASE + "X"] for _, row in phase_x.iterrows()}
    phase_dict_y = {(row[NAME], row[NAME2]): row[PHASE + "Y"] for _, row in phase_y.iterrows()}

    prev_x, next_x = _find_neighbors_from_phase_diffs(bpm_list, phase_dict_x, full_bpm_list)
    prev_y, next_y = _find_neighbors_from_phase_diffs(bpm_list, phase_dict_y, full_bpm_list)
    prev_x = prev_x.rename(columns={"prev_bpm": "bpm_x_p", "delta": "delta_x_p"})
    prev_y = prev_y.rename(columns={"prev_bpm": "bpm_y_p", "delta": "delta_y_p"})
    next_x = next_x.rename(columns={"next_bpm": "bpm_x_n", "delta": "delta_x_n"})
    next_y = next_y.rename(columns={"next_bpm": "bpm_y_n", "delta": "delta_y_n"})
    return prev_x, prev_y, next_x, next_y


def calculate_pz(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    inject_noise: bool | float = True,
    info: bool = True,
    rng: np.random.Generator | None = None,
    low_noise_bpms: list[str] | None = None,
) -> pd.DataFrame:
    low_noise_bpms = list(low_noise_bpms or [])
    LOGGER.info(
        "Calculating transverse momentum - inject_noise=%s, low_noise_bpms=%d BPMs",
        inject_noise,
        len(low_noise_bpms),
    )

    features = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")
    rng = get_rng(rng)

    if inject_noise is not False:
        noise_std = POSITION_STD_DEV if inject_noise is True else float(inject_noise)
        inject_noise_xy_inplace(data, orig_data, rng, low_noise_bpms, noise_std=noise_std)

    # Get the shared list of data and twiss BPMs
    tws_bpm_names = set(tws.index).intersection(data["name"].unique())
    data = data[data["name"].isin(tws_bpm_names)]
    tws = tws.loc[tws.index.isin(tws_bpm_names)]

    remove_closed_orbit_inplace(data, tws)

    data_p, data_n, bpm_index, _maps = prepare_neighbor_views(data, tws)

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(data_p, data_n, bpm_index)
    data_p, data_n = merge_neighbor_coords(data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n)

    data_p = momenta_from_prev(data_p)
    data_n = momenta_from_next(data_n)

    sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n)

    restore_closed_orbit_and_reference_momenta_inplace(data_avg, tws)

    # Restore original order of orig_data
    orig_order = orig_data.set_index(["name", "turn"]).index
    data_avg = data_avg.set_index(["name", "turn"]).reindex(orig_order).reset_index()

    diagnostics(orig_data, data_p, data_n, data_avg, info, features)
    return data_avg[OUT_COLS]
