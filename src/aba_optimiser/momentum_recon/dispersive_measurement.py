from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    DISPERSION,
    ERR,
    MOMENTUM_DISPERSION,
    PHASE_ADV,
)

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

# Define which error columns we expect
MEASUREMENT_RENAME_MAPPING: dict[str, str] = {
    f"{BETA}X": "beta11",
    f"{BETA}Y": "beta22",
    f"{ALPHA}X": "alfa11",
    f"{ALPHA}Y": "alfa22",
    f"{PHASE_ADV}X": "mu1",
    f"{PHASE_ADV}Y": "mu2",
    f"{DISPERSION}X": "dx",
    f"{DISPERSION}Y": "dy",
    f"{MOMENTUM_DISPERSION}X": "dpx",
    f"{MOMENTUM_DISPERSION}Y": "dpy",
}

ERROR_RENAME_MAPPING: dict[str, str] = {
    f"{ERR}{BETA}X": "sqrt_betax_err",
    f"{ERR}{BETA}Y": "sqrt_betay_err",
    f"{ERR}{ALPHA}X": "alfax_err",
    f"{ERR}{ALPHA}Y": "alfay_err",
    f"{ERR}{DISPERSION}X": "dx_err",
    f"{ERR}{MOMENTUM_DISPERSION}X": "dpx_err",
}


def _check_and_log_errors(tws: pd.DataFrame) -> bool:
    """Check if error columns exist and validate consistency.

    Args:
        tws: Twiss DataFrame to check.

    Returns:
        True if error columns exist, False otherwise.

    Raises:
        ValueError: If some error columns exist but not all.
    """
    expected_error_cols = set(ERROR_RENAME_MAPPING.keys())
    has_errors = expected_error_cols & set(tws.columns)

    if has_errors and has_errors != expected_error_cols:
        missing = expected_error_cols - has_errors
        raise ValueError(
            f"Partial error columns found. If errors exist, all must exist. Missing: {missing}"
        )

    if has_errors:
        LOGGER.info("Error columns detected, will use uncertainty propagation")

    return bool(has_errors)


def _apply_sqrt_beta_error_propagation(tws: pd.DataFrame) -> None:
    """Apply error propagation for sqrt(beta) transformation (in-place).

    Formula: err_sqrt_beta = err_beta / (2 * sqrt(beta))

    Args:
        tws: Twiss DataFrame to modify in-place.
    """
    tws["sqrt_betax_err"] = tws["sqrt_betax_err"] / (2.0 * np.sqrt(tws["beta11"]))
    tws["sqrt_betay_err"] = tws["sqrt_betay_err"] / (2.0 * np.sqrt(tws["beta22"]))


def _attach_current_bpm_errors(df: pd.DataFrame, tws: pd.DataFrame) -> None:
    """Attach error columns for current BPM to tracking data (in-place).

    Args:
        df: Tracking data to attach columns to.
        tws: Twiss DataFrame with error columns (indexed by BPM name).
    """
    error_cols = ["sqrt_betax_err", "sqrt_betay_err", "alfax_err", "alfay_err", "dx_err", "dpx_err"]
    tws_dict = tws.to_dict()

    for err_col in error_cols:
        df[err_col] = df["name"].map(tws_dict[err_col])


def _attach_neighbor_bpm_errors(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    tws: pd.DataFrame,
) -> None:
    """Attach error columns for neighboring BPMs to tracking data (in-place).

    Args:
        data_p: Previous BPM data to attach columns to.
        data_n: Next BPM data to attach columns to.
        tws: Twiss DataFrame with error columns (indexed by BPM name).
    """
    tws_dict = tws.to_dict()

    # Previous BPM errors
    data_p["sqrt_betax_p_err"] = data_p["prev_bpm_x"].map(tws_dict["sqrt_betax_err"])
    data_p["sqrt_betay_p_err"] = data_p["prev_bpm_y"].map(tws_dict["sqrt_betay_err"])
    data_p["alfax_err"] = data_p["name"].map(tws_dict["alfax_err"])
    data_p["alfay_err"] = data_p["name"].map(tws_dict["alfay_err"])
    data_p["dx_err"] = data_p["name"].map(tws_dict["dx_err"])
    data_p["dx_prev_err"] = data_p["prev_bpm_x"].map(tws_dict["dx_err"])
    data_p["dpx_err"] = data_p["name"].map(tws_dict["dpx_err"])

    # Next BPM errors
    data_n["sqrt_betax_n_err"] = data_n["next_bpm_x"].map(tws_dict["sqrt_betax_err"])
    data_n["sqrt_betay_n_err"] = data_n["next_bpm_y"].map(tws_dict["sqrt_betay_err"])
    data_n["alfax_err"] = data_n["name"].map(tws_dict["alfax_err"])
    data_n["alfay_err"] = data_n["name"].map(tws_dict["alfay_err"])
    data_n["dx_err"] = data_n["name"].map(tws_dict["dx_err"])
    data_n["dx_next_err"] = data_n["next_bpm_x"].map(tws_dict["dx_err"])
    data_n["dpx_err"] = data_n["name"].map(tws_dict["dpx_err"])


def calculate_pz_measurement(
    orig_data: pd.DataFrame,
    measurement_folder: str | Path,
    info: bool = True,
    include_errors: bool = False,
) -> pd.DataFrame:
    """Calculate transverse momenta from dispersive measurements.

    Args:
        orig_data: Original tracking data.
        measurement_folder: Path to measurement files.
        info: Whether to print diagnostic info.
        include_errors: Whether to include error columns from measurements.

    Returns:
        DataFrame with calculated px and py columns.

    Raises:
        ValueError: If error columns are inconsistent.
        RuntimeError: If dispersion maps not initialized correctly.
    """
    LOGGER.info(
        "Calculating dispersive transverse momentum from measurements - measurement_folder=%s",
        measurement_folder,
    )

    has_px, has_py = validate_input(orig_data)
    data = orig_data.copy(deep=True)

    bpm_list = data["name"].unique().tolist()
    tws = build_twiss_from_measurements(Path(measurement_folder), include_errors=include_errors)
    tws = tws[tws.index.isin(bpm_list)]

    # Check for errors upfront - if any error column exists, all must exist
    has_errors = _check_and_log_errors(tws)
    if include_errors and not has_errors:
        raise ValueError("include_errors=True but no error columns found in measurements")

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

    # Attach error columns if they exist
    if has_errors:
        _attach_current_bpm_errors(data_p, tws)
        _attach_current_bpm_errors(data_n, tws)

    data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(maps.sqrt_betax)
    data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(maps.sqrt_betay)
    data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(maps.sqrt_betax)
    data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(maps.sqrt_betay)

    # Attach error columns for neighboring BPMs if they exist
    if has_errors:
        _attach_neighbor_bpm_errors(data_p, data_n, tws)

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

    # Functions auto-detect and use uncertainty propagation if error columns exist
    data_p = momenta_from_prev(data_p, dpp_est)
    data_n = momenta_from_next(data_n, dpp_est)

    sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n)

    # Add to the header the dpp used
    data_avg.attrs["DPP_EST"] = dpp_est

    diagnostics(orig_data, data_p, data_n, data_avg, info, has_px, has_py)
    return data_avg[OUT_COLS]
