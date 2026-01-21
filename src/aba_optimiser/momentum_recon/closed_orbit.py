import logging

import pandas as pd

from aba_optimiser.momentum_recon.core import (
    sync_endpoints_inplace,
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

logger = logging.getLogger(__name__)

def estimate_closed_orbit(
    data: pd.DataFrame, tws: pd.DataFrame, dpp_est: float = 0.0
) -> pd.DataFrame:
    """Estimate closed orbit from tracking data.

    Args:
        data: Tracking data with BPM readings. Must contain columns: ["name", "x", "y"].
        tws: Twiss parameters DataFrame. Must have columns ["dx", "dy"] and be indexed by BPM name.
        dpp_est: Estimated relative momentum deviation.

    Returns:
        DataFrame indexed like tws.index with columns: x, y, var_x, var_y.
    """
    if "name" not in data.columns or "x" not in data.columns or "y" not in data.columns:
        raise ValueError('`data` must contain columns ["name", "x", "y"].')

    # Map dispersion to each row (per BPM), then correct positions turn-by-turn
    x_corr = data["x"] - dpp_est * data["name"].map(tws["dx"].to_dict())
    y_corr = data["y"] - dpp_est * data["name"].map(tws["dy"].to_dict())

    g = pd.DataFrame({"name": data["name"], "x_corr": x_corr, "y_corr": y_corr}).groupby(
        "name", sort=False
    )

    co_avg = pd.DataFrame(
        {
            "x": g["x_corr"].mean(),
            "y": g["y_corr"].mean(),
            "var_x": g["x_corr"].var(),
            "var_y": g["y_corr"].var(),
        }
    )

    logger.info("Estimated closed orbit at %d BPMs.", len(co_avg))
    logger.info("Mean closed orbit x: %.3e m, y: %.3e m", co_avg["x"].mean(), co_avg["y"].mean())

    # Align to Twiss order / include missing BPMs as NaN rows
    return co_avg.reindex(tws.index)


def extract_closed_orbit_from_measurements(tws: pd.DataFrame):
    """Extract closed orbit from BPM measurements.

    Args:
        tws: Twiss parameters DataFrame with closed orbit positions.
    Returns:
        DataFrame with closed orbit estimates (x, y).
    """
    co_data = tws[["x", "y"]].copy()
    co_data["var_x"] = tws["x_err"] ** 2
    co_data["var_y"] = tws["y_err"] ** 2
    return co_data


def compute_closed_orbit_momenta(co_data: pd.DataFrame, tws: pd.DataFrame) -> pd.DataFrame:
    """Compute closed orbit momenta from closed orbit data.

    Args:
        co_data: DataFrame with closed orbit positions (x, y).
        tws: Twiss DataFrame with optics parameters.
    Returns:
        DataFrame with closed orbit momenta (px, py, var_px, var_py).
    """

    # Create a 3 turn dataset for momentum calculation
    co_data = co_data.reset_index().copy()
    co_replicated = pd.concat(
        [
            co_data.assign(turn=0),
            co_data.assign(turn=1),
            co_data.assign(turn=2),
        ],
        ignore_index=True,
    )
    data_p, data_n, bpm_index, _maps = prepare_neighbor_views(
        co_replicated, tws, include_dispersion=False
    )

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(data_p, data_n, bpm_index)
    data_p, data_n = merge_neighbor_coords(data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n)

    data_p = momenta_from_prev(data_p)
    data_n = momenta_from_next(data_n)

    sync_endpoints_inplace(data_p, data_n)
    data_avg = weighted_average(data_p, data_n)

    # Extract just the second turn which corresponds to the closed orbit solution
    co_momenta = data_avg[data_avg["turn"] == 1].copy()
    co_momenta = co_momenta.set_index("name")[["x", "y", "px", "py", "var_px", "var_py"]]
    return co_momenta.reindex(tws.index)
