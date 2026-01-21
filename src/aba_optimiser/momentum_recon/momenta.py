from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.momentum_recon.core import neighbour_plane_factors
from aba_optimiser.momentum_recon.errors import (
    compute_measurement_errors,
    compute_optics_errors,
)
from aba_optimiser.momentum_recon.schema import (
    NEXT,
    POSITION_COLS,
    PREV,
    SUFFIX_NEXT,
    SUFFIX_PREV,
    VARIANCE_COLS,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def _column_or_zeros(frame, column: str, template: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.zeros_like(template, dtype=float)


def _require_columns(frame, cols: set[str], context: str) -> None:
    missing = cols.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for {context}: {sorted(missing)}")


def _require_momentum_columns(frame: pd.DataFrame, names, suffix: str, context: str) -> None:
    required = set(POSITION_COLS) | set(VARIANCE_COLS) | {
        names.x,
        names.y,
        names.var_x,
        names.var_y,
        f"sqrt_betax_{suffix}",
        f"sqrt_betay_{suffix}",
    }
    _require_columns(frame, required, context)


def _has_uncertainty_columns(data: pd.DataFrame, neighbor_suffix: str, names) -> bool:
    """Check if DataFrame has optical uncertainty columns.

    Args:
        data: DataFrame to check.
        neighbor_suffix: Suffix for neighbor beta columns ('p' for prev, 'n' for next).

    Returns:
        True if uncertainty columns exist.
    """
    required_err_cols = {
        "sqrt_betax_err",
        "sqrt_betay_err",
        f"sqrt_betax_{neighbor_suffix}_err",
        f"sqrt_betay_{neighbor_suffix}_err",
        "alfax_err",
        "alfay_err",
        names.delta_x_err,
        names.delta_y_err,
    }
    # Dispersion error columns are optional - check if dx column exists first
    has_dispersion = "dx" in data.columns
    if has_dispersion:
        # If dispersion exists, require all dispersion error columns
        dispersion_err_cols = {
            "dx_err",
            f"dx_{neighbor_suffix}_err",
            "dpx_err",
            "dy_err",
            f"dy_{neighbor_suffix}_err",
            "dpy_err",
        }
        required_err_cols |= dispersion_err_cols

    return required_err_cols.issubset(data.columns)


def _compute_nominal_momenta(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    *,
    is_prev: bool,
    dpp_est: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nominal (error-free) momentum values.

    Args:
        data: DataFrame with position and optics columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        dpp_est: Estimated DPP.

    Returns:
        Tuple of (px, py) arrays.
    """
    x_current = data["x"].to_numpy()
    y_current = data["y"].to_numpy()
    x_neighbor = data[names.x].to_numpy()
    y_neighbor = data[names.y].to_numpy()

    sqrt_beta_x = data["sqrt_betax"].to_numpy()
    sqrt_beta_y = data["sqrt_betay"].to_numpy()
    sqrt_beta_x_neigh = data[f"sqrt_betax_{neighbor_suffix}"].to_numpy()
    sqrt_beta_y_neigh = data[f"sqrt_betay_{neighbor_suffix}"].to_numpy()

    alpha_x = data["alfax"].to_numpy()
    alpha_y = data["alfay"].to_numpy()

    dx_current = _column_or_zeros(data, "dx", x_current)
    dx_neighbor = _column_or_zeros(data, names.dx, x_neighbor)
    dpx_current = _column_or_zeros(data, "dpx", x_current)
    dy_current = _column_or_zeros(data, "dy", y_current)
    dy_neighbor = _column_or_zeros(data, names.dy, y_neighbor)
    dpy_current = _column_or_zeros(data, "dpy", y_current)

    phi_x = data[names.delta_x].to_numpy() * 2 * np.pi
    phi_y = data[names.delta_y].to_numpy() * 2 * np.pi

    sign_x, alpha_sign_x, cos_phi_x, tan_phi_x, sec_phi_x = neighbour_plane_factors(phi_x, is_prev=is_prev)
    sign_y, alpha_sign_y, cos_phi_y, tan_phi_y, sec_phi_y = neighbour_plane_factors(phi_y, is_prev=is_prev)

    # Vertical dispersion should be very small, or typically 0, but included for completeness
    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_neighbor_norm = (x_neighbor - dpp_est * dx_neighbor) / sqrt_beta_x_neigh
    y_current_norm = (y_current - dpp_est * dy_current) / sqrt_beta_y
    y_neighbor_norm = (y_neighbor - dpp_est * dy_neighbor) / sqrt_beta_y_neigh

    # Nominal momenta
    px = (
        sign_x
        * (x_neighbor_norm * sec_phi_x + x_current_norm * (tan_phi_x + alpha_sign_x * alpha_x))
        / sqrt_beta_x
        + dpx_current * dpp_est
    )
    py = (
        sign_y
        * (y_neighbor_norm * sec_phi_y + y_current_norm * (tan_phi_y + alpha_sign_y * alpha_y))
        / sqrt_beta_y
        + dpy_current * dpp_est
    )

    return px, py


def _compute_momenta(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    *,
    is_prev: bool,
    dpp_est: float = 0.0,
    include_optics_errors: bool = False,
) -> pd.DataFrame:
    """Compute momenta with error propagation.

    Args:
        data: DataFrame with position and optics columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        dpp_est: Estimated DPP.
        include_optics_errors: Whether to include optics uncertainties.

    Returns:
        DataFrame with px, py, var_px, var_py columns added.
    """
    _require_momentum_columns(data, names, neighbor_suffix, "momenta")

    has_optics_uncertainties = _has_uncertainty_columns(data, neighbor_suffix, names)
    use_optics_errors = include_optics_errors and has_optics_uncertainties
    if use_optics_errors:
        LOGGER.debug("Including optical function uncertainties for %s momenta", neighbor_suffix)

    # Compute nominal momenta
    px, py = _compute_nominal_momenta(
        data, names, neighbor_suffix, is_prev=is_prev, dpp_est=dpp_est
    )

    # Compute measurement errors (always included)
    var_px, var_py = compute_measurement_errors(data, names, neighbor_suffix, is_prev)

    # Add optics errors if requested and available
    if use_optics_errors:
        var_px_opt_errors, var_py_opt_errors = compute_optics_errors(
            data, names, neighbor_suffix, is_prev, dpp_est
        )
        var_px = var_px + np.sum(var_px_opt_errors, axis=0)
        var_py = var_py + np.sum(var_py_opt_errors, axis=0)

    data["px"] = px
    data["py"] = py
    data["var_px"] = var_px
    data["var_py"] = var_py

    return data


def momenta_from_prev(
    data_p: pd.DataFrame, dpp_est: float = 0.0, *, include_optics_errors: bool = False
) -> pd.DataFrame:
    return _compute_momenta(
        data_p,
        PREV,
        SUFFIX_PREV,
        is_prev=True,
        dpp_est=dpp_est,
        include_optics_errors=include_optics_errors,
    )


def momenta_from_next(
    data_n: pd.DataFrame, dpp_est: float = 0.0, *, include_optics_errors: bool = False
) -> pd.DataFrame:
    return _compute_momenta(
        data_n,
        NEXT,
        SUFFIX_NEXT,
        is_prev=False,
        dpp_est=dpp_est,
        include_optics_errors=include_optics_errors,
    )
