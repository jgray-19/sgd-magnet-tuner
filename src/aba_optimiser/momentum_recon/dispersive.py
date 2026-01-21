from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
from aba_optimiser.momentum_recon.momenta import momenta_from_next, momenta_from_prev
from aba_optimiser.momentum_recon.neighbors import (
    compute_turn_wraps,
    merge_neighbor_coords,
    prepare_neighbor_views,
)
from aba_optimiser.physics.dpp_calculation import get_mean_dpp

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd
    from numpy.random import Generator

LOGGER = logging.getLogger(__name__)


def calculate_pz(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    inject_noise: bool = True,
    info: bool = True,
    rng: Generator | None = None,
    low_noise_bpms: list[str] | None = None,
) -> pd.DataFrame:
    low_noise_bpms = list(low_noise_bpms or [])
    LOGGER.info(
        "Calculating dispersive transverse momentum - inject_noise=%s, low_noise_bpms=%d BPMs",
        inject_noise,
        len(low_noise_bpms),
    )

    features = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    rng = get_rng(rng)

    if inject_noise:
        inject_noise_xy_inplace(data, orig_data, rng, low_noise_bpms)

    # Get the shared list of data and twiss BPMs
    tws_bpm_names = set(tws.index).intersection(data["name"].unique())
    data = data[data["name"].isin(tws_bpm_names)]
    tws = tws.loc[tws.index.isin(tws_bpm_names)]

    remove_closed_orbit_inplace(data, tws)

    dpp_est = get_mean_dpp(data, tws, info)
    data_p, data_n, bpm_index, _maps = prepare_neighbor_views(data, tws, include_dispersion=True)

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(
        data_p, data_n, bpm_index
    )
    data_p, data_n = merge_neighbor_coords(
        data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n
    )

    data_p = momenta_from_prev(data_p, dpp_est)
    data_n = momenta_from_next(data_n, dpp_est)

    sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n)

    restore_closed_orbit_and_reference_momenta_inplace(data_avg, tws)

    # Add to the header the dpp used
    data_avg.attrs["DPP_EST"] = dpp_est

    diagnostics(orig_data, data_p, data_n, data_avg, info, features)
    return data_avg[OUT_COLS]
