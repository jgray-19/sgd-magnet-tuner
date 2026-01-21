"""
Pipeline stages for momentum reconstruction from measurements.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.measurements.twiss_from_measurement import build_twiss_from_measurements
from aba_optimiser.momentum_recon.columns import (
    CURRENT_BPM_ERRORS,
    ERROR_RENAME_MAPPING,
    MEASUREMENT_RENAME_MAPPING,
)
from aba_optimiser.momentum_recon.core import (
    OUT_COLS,
    InputFeatures,
    diagnostics,
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
from aba_optimiser.momentum_recon.schema import PLANE_X, PLANE_Y, SUFFIX_NEXT, SUFFIX_PREV
from aba_optimiser.physics.dpp_calculation import get_mean_dpp

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from pathlib import Path

    import pandas as pd

LOGGER = logging.getLogger(__name__)

# Systematic plane definitions
PLANES = [(PLANE_X, "X", "11"), (PLANE_Y, "Y", "22")]


def prepare_data(orig_data: pd.DataFrame) -> tuple[pd.DataFrame, InputFeatures]:
    """Prepare and validate input data.

    Args:
        orig_data: Original tracking data.

    Returns:
        Tuple of (copied_data, input_features).
    """
    features = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    return data, features


def process_twiss(
    measurement_folder: Path,
    bpm_list: list[str],
    include_errors: bool,
) -> tuple[pd.DataFrame, bool]:
    """Process twiss data from measurements.

    Args:
        measurement_folder: Path to measurement files.
        bpm_list: List of BPM names to filter.
        include_errors: Whether to include error columns.

    Returns:
        Tuple of (processed_twiss, has_errors).
    """
    tws = build_twiss_from_measurements(measurement_folder, include_errors=include_errors)
    tws = tws[tws.index.isin(bpm_list)]

    # Check for errors upfront
    expected_error_cols = set(ERROR_RENAME_MAPPING.keys())
    has_errors = expected_error_cols & set(tws.columns)

    if has_errors and has_errors != expected_error_cols:
        missing = expected_error_cols - has_errors
        raise ValueError(
            f"Partial error columns found. If errors exist, all must exist. Missing: {missing}"
        )

    if has_errors:
        LOGGER.info("Error columns detected, will use uncertainty propagation")

    # Apply renaming (also index to lowercase)
    tws = tws.rename(columns=MEASUREMENT_RENAME_MAPPING)
    tws.index.name = tws.index.name.lower()
    if has_errors:
        tws = tws.rename(columns=ERROR_RENAME_MAPPING)
    tws.columns = [col.lower() for col in tws.columns]
    tws.headers = {key.lower(): value for key, value in tws.headers.items()}

    # Apply error propagation for sqrt(beta) if errors exist
    if has_errors:
        _apply_sqrt_beta_error_propagation(tws)

    return tws, bool(has_errors)


def _apply_sqrt_beta_error_propagation(tws: pd.DataFrame) -> None:
    """Apply error propagation for sqrt(beta) transformation (in-place).

    Formula: err_sqrt_beta = err_beta / (2 * sqrt(beta))

    Args:
        tws: Twiss DataFrame to modify in-place.
    """
    for plane_lower, plane_upper, plane_idx in PLANES:
        err_col = f"sqrt_beta{plane_lower}_err"
        beta_col = f"beta{plane_idx}"
        tws[err_col] = tws[err_col] / (2.0 * np.sqrt(tws[beta_col]))


def attach_errors_inplace(data_p: pd.DataFrame, data_n: pd.DataFrame, tws: pd.DataFrame) -> None:
    """Attach error columns to neighbor data (in-place).

    Args:
        data_p: Previous BPM data.
        data_n: Next BPM data.
        tws: Twiss DataFrame with error columns.
    """
    tws_dict = tws.to_dict()
    for err_col in CURRENT_BPM_ERRORS:
        data_p[err_col] = data_p["name"].map(tws_dict[err_col])
        data_n[err_col] = data_n["name"].map(tws_dict[err_col])

    # Attach neighbor errors
    for data, suffix in [(data_p, SUFFIX_PREV), (data_n, SUFFIX_NEXT)]:
        # Current BPM errors (same for both)
        for err_col in ["alfax_err", "alfay_err", "dx_err", "dy_err", "dpx_err", "dpy_err"]:
            data[err_col] = data["name"].map(tws_dict[err_col])


def attach_errors(data_p: pd.DataFrame, data_n: pd.DataFrame, tws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach error columns to neighbor data, returning copies."""
    data_p_copy = data_p.copy()
    data_n_copy = data_n.copy()
    attach_errors_inplace(data_p_copy, data_n_copy, tws)
    return data_p_copy, data_n_copy


def setup_momentum_calculation(
    data: pd.DataFrame, tws_meas: pd.DataFrame, model_tws: pd.DataFrame, info: bool
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Set up data for momentum calculation.

    Args:
        data: Tracking data.
        tws_meas: Processed measurement twiss data.
        model_tws: Model twiss data for closed orbit.
        info: Whether to log info.

    Returns:
        Tuple of (data_p, data_n, dpp_est).
    """
    # Remove closed orbit from tracking data before calculation
    remove_closed_orbit_inplace(data, model_tws)

    dpp_est = get_mean_dpp(data, tws_meas, info)
    data_p, data_n, bpm_index, maps = prepare_neighbor_views(
        data,
        tws_meas,
        include_dispersion=True,
        include_errors=True,
    )

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(data_p, data_n, bpm_index)
    data_p, data_n = merge_neighbor_coords(data_p, data_n, turn_x_p, turn_y_p, turn_x_n, turn_y_n)

    return data_p, data_n, dpp_est


def calculate_momenta(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    dpp_est: float,
    include_optics_errors: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate momenta for previous and next neighbors.

    Args:
        data_p: Previous neighbor data.
        data_n: Next neighbor data.
        dpp_est: Estimated DPP.
        include_optics_errors: Whether to include optics uncertainties.

    Returns:
        Tuple of (processed_data_p, processed_data_n).
    """
    data_p = momenta_from_prev(data_p, dpp_est, include_optics_errors=include_optics_errors)
    data_n = momenta_from_next(data_n, dpp_est, include_optics_errors=include_optics_errors)

    sync_endpoints(data_p, data_n)

    return data_p, data_n


def aggregate_results(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    model_tws: pd.DataFrame,
    dpp_est: float,
) -> pd.DataFrame:
    """Aggregate results and restore closed orbit.

    Args:
        data_p: Previous neighbor data with momenta.
        data_n: Next neighbor data with momenta.
        model_tws: Model twiss data.
        dpp_est: Estimated DPP.

    Returns:
        Final result DataFrame.
    """
    data_avg = weighted_average(data_p, data_n)

    # Restore closed orbit to x and y
    # Add reference momenta from twiss
    restore_closed_orbit_and_reference_momenta_inplace(data_avg, model_tws)

    # Add to the header the dpp used
    data_avg.attrs["DPP_EST"] = dpp_est

    return data_avg[OUT_COLS]


def run_diagnostics(
    orig_data: pd.DataFrame,
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    data_avg: pd.DataFrame,
    info: bool,
    features: InputFeatures,
) -> None:
    """Run diagnostic logging.

    Args:
        orig_data: Original data.
        data_p: Previous data.
        data_n: Next data.
        data_avg: Averaged data.
        info: Whether to log info.
        has_px: Whether original data has px.
        has_py: Whether original data has py.
    """
    diagnostics(orig_data, data_p, data_n, data_avg, info, features)
