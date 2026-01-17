from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aba_optimiser.measurements.twiss_from_measurement import build_twiss_from_measurements
from aba_optimiser.momentum_recon.core import (
    OUT_COLS,
    attach_lattice_columns,
    build_lattice_maps,
    diagnostics,
    sync_endpoints,
    validate_input,
)
from aba_optimiser.momentum_recon.core import (
    weighted_average_from_weights as weighted_average,
)
from aba_optimiser.momentum_recon.momenta import momenta_from_next, momenta_from_prev
from aba_optimiser.momentum_recon.neighbors import (
    build_lattice_neighbor_tables,
    compute_turn_wraps,
    merge_neighbor_coords,
)
from aba_optimiser.physics.dpp_calculation import get_mean_dpp

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def calculate_pz_measurement(
    orig_data: pd.DataFrame,
    measurement_folder: str | Path,
    info: bool = True,
) -> pd.DataFrame:
    LOGGER.info(
        "Calculating dispersive transverse momentum from measurements - measurement_folder=%s",
        measurement_folder,
    )

    has_px, has_py = validate_input(orig_data)
    data = orig_data.copy(deep=True)

    bpm_list = data["name"].unique().tolist()
    tws = build_twiss_from_measurements(Path(measurement_folder), include_errors=False)
    tws = tws[tws.index.isin(bpm_list)]

    # Rename columns to match expected names in downstream functions
    tws = tws.rename(columns={
        "BETX": "beta11",
        "BETY": "beta22",
        "ALFX": "alfa11",
        "ALFY": "alfa22",
        "MUX": "mu1",
        "MUY": "mu2",
    })
    tws.columns = [col.lower() for col in tws.columns]
    tws.headers = {key.lower(): value for key, value in tws.headers.items()}

    dpp_est = get_mean_dpp(data, tws, info)
    maps = build_lattice_maps(tws, include_dispersion=True)
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

    if maps.dx is None or maps.dpx is None:
        raise RuntimeError("Dispersion maps were not initialised correctly")

    data_p["dx_prev"] = data_p["prev_bpm_x"].map(maps.dx)
    data_n["dx_next"] = data_n["next_bpm_x"].map(maps.dx)

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

    # Add to the header the dpp used
    data_avg.attrs["DPP_EST"] = dpp_est

    diagnostics(orig_data, data_p, data_n, data_avg, info, has_px, has_py)
    return data_avg[OUT_COLS]
