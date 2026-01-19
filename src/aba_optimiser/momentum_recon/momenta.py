from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from uncertainties import unumpy

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


def _has_uncertainty_columns(data: pd.DataFrame, neighbor_suffix: str) -> bool:
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
    }
    # Dispersion error columns are optional - check if dx column exists first
    has_dispersion = "dx" in data.columns
    if has_dispersion:
        # If dispersion exists, check if error columns also exist
        dispersion_err_cols = {
            "dx_err",
            f"dx_{neighbor_suffix if neighbor_suffix == 'p' else neighbor_suffix}_err",
            "dpx_err",
        }
        # Only require dispersion errors if at least one is present
        if dispersion_err_cols & set(data.columns):
            required_err_cols |= dispersion_err_cols

    return required_err_cols.issubset(data.columns)


def _extract_values_and_errors(ufloat_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract nominal values and standard deviations from ufloat array.

    Args:
        ufloat_array: Array of ufloat objects.

    Returns:
        Tuple of (values, errors).
    """
    values = unumpy.nominal_values(ufloat_array)
    errors = unumpy.std_devs(ufloat_array)
    return values, errors


def momenta_from_prev(data_p: pd.DataFrame, dpp_est: float = 0.0) -> pd.DataFrame:
    _require_columns(
        data_p,
        {"x", "y", "prev_x", "prev_y", "var_x", "var_y", "prev_x_var", "prev_y_var"},
        "momenta_from_prev",
    )

    # Check if we should include optical function uncertainties
    has_optics_uncertainties = _has_uncertainty_columns(data_p, "p")
    if has_optics_uncertainties:
        LOGGER.debug("Including optical function uncertainties for prev momenta")

    x_current = data_p["x"].to_numpy()
    y_current = data_p["y"].to_numpy()
    x_prev = data_p["prev_x"].to_numpy()
    y_prev = data_p["prev_y"].to_numpy()

    sqrt_beta_x = data_p["sqrt_betax"].to_numpy()
    sqrt_beta_y = data_p["sqrt_betay"].to_numpy()
    sqrt_beta_x_prev = data_p["sqrt_betax_p"].to_numpy()
    sqrt_beta_y_prev = data_p["sqrt_betay_p"].to_numpy()

    alpha_x = data_p["alfax"].to_numpy()
    alpha_y = data_p["alfay"].to_numpy()

    dx_current = _column_or_zeros(data_p, "dx", x_current)
    dx_prev = _column_or_zeros(data_p, "dx_prev", x_prev)
    dpx_current = _column_or_zeros(data_p, "dpx", x_current)

    phi_x = data_p["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_p["delta_y"].to_numpy() * 2 * np.pi

    # Always use unumpy for position measurements
    x_current = unumpy.uarray(x_current, data_p["var_x"].to_numpy() ** 0.5)
    y_current = unumpy.uarray(y_current, data_p["var_y"].to_numpy() ** 0.5)
    x_prev = unumpy.uarray(x_prev, data_p["prev_x_var"].to_numpy() ** 0.5)
    y_prev = unumpy.uarray(y_prev, data_p["prev_y_var"].to_numpy() ** 0.5)

    # Conditionally add optical function uncertainties
    if has_optics_uncertainties:
        sqrt_betax_err = data_p["sqrt_betax_err"].to_numpy()
        sqrt_betay_err = data_p["sqrt_betay_err"].to_numpy()
        sqrt_betax_p_err = data_p["sqrt_betax_p_err"].to_numpy()
        sqrt_betay_p_err = data_p["sqrt_betay_p_err"].to_numpy()
        alfax_err = data_p["alfax_err"].to_numpy()
        alfay_err = data_p["alfay_err"].to_numpy()

        sqrt_beta_x = unumpy.uarray(sqrt_beta_x, sqrt_betax_err)
        sqrt_beta_y = unumpy.uarray(sqrt_beta_y, sqrt_betay_err)
        sqrt_beta_x_prev = unumpy.uarray(sqrt_beta_x_prev, sqrt_betax_p_err)
        sqrt_beta_y_prev = unumpy.uarray(sqrt_beta_y_prev, sqrt_betay_p_err)
        alpha_x = unumpy.uarray(alpha_x, alfax_err)
        alpha_y = unumpy.uarray(alpha_y, alfay_err)

        dx_current_err = data_p["dx_err"].to_numpy()
        dx_prev_err = data_p["dx_prev_err"].to_numpy()
        dpx_current_err = data_p["dpx_err"].to_numpy()
        dx_current = unumpy.uarray(dx_current, dx_current_err)
        dx_prev = unumpy.uarray(dx_prev, dx_prev_err)
        dpx_current = unumpy.uarray(dpx_current, dpx_current_err)

    # Compute momenta directly
    cos_phi_x = np.cos(phi_x)
    sin_phi_x = np.sin(phi_x)
    tan_phi_x = np.tan(phi_x)
    cos_phi_y = np.cos(phi_y)
    sin_phi_y = np.sin(phi_y)
    tan_phi_y = np.tan(phi_y)

    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_prev_norm = (x_prev - dpp_est * dx_prev) / sqrt_beta_x_prev
    y_current_norm = y_current / sqrt_beta_y
    y_prev_norm = y_prev / sqrt_beta_y_prev

    px = (
        -(
            x_prev_norm * (cos_phi_x + sin_phi_x * tan_phi_x)
            + x_current_norm * (tan_phi_x + alpha_x)
        )
        / sqrt_beta_x
        + dpx_current * dpp_est
    )
    py = (
        -(
            y_prev_norm * (cos_phi_y + sin_phi_y * tan_phi_y)
            + y_current_norm * (tan_phi_y + alpha_y)
        )
        / sqrt_beta_y
    )

    # Extract values and uncertainties
    px_vals, px_errs = _extract_values_and_errors(px)
    py_vals, py_errs = _extract_values_and_errors(py)
    data_p["px"] = px_vals
    data_p["py"] = py_vals
    data_p["var_px"] = px_errs**2
    data_p["var_py"] = py_errs**2

    return data_p


def momenta_from_next(data_n: pd.DataFrame, dpp_est: float = 0.0) -> pd.DataFrame:
    _require_columns(
        data_n,
        {"x", "y", "next_x", "next_y", "var_x", "var_y", "next_x_var", "next_y_var"},
        "momenta_from_next",
    )

    # Check if we should include optical function uncertainties
    has_optics_uncertainties = _has_uncertainty_columns(data_n, "n")
    if has_optics_uncertainties:
        LOGGER.debug("Including optical function uncertainties for next momenta")

    x_current = data_n["x"].to_numpy()
    y_current = data_n["y"].to_numpy()
    x_next = data_n["next_x"].to_numpy()
    y_next = data_n["next_y"].to_numpy()

    sqrt_beta_x = data_n["sqrt_betax"].to_numpy()
    sqrt_beta_y = data_n["sqrt_betay"].to_numpy()
    sqrt_beta_x_next = data_n["sqrt_betax_n"].to_numpy()
    sqrt_beta_y_next = data_n["sqrt_betay_n"].to_numpy()

    alpha_x = data_n["alfax"].to_numpy()
    alpha_y = data_n["alfay"].to_numpy()

    dx_current = _column_or_zeros(data_n, "dx", x_current)
    dx_next = _column_or_zeros(data_n, "dx_next", x_next)
    dpx_current = _column_or_zeros(data_n, "dpx", x_current)

    phi_x = data_n["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_n["delta_y"].to_numpy() * 2 * np.pi

    # Always use unumpy for position measurements
    x_current = unumpy.uarray(x_current, data_n["var_x"].to_numpy() ** 0.5)
    y_current = unumpy.uarray(y_current, data_n["var_y"].to_numpy() ** 0.5)
    x_next = unumpy.uarray(x_next, data_n["next_x_var"].to_numpy() ** 0.5)
    y_next = unumpy.uarray(y_next, data_n["next_y_var"].to_numpy() ** 0.5)

    # Conditionally add optical function uncertainties
    if has_optics_uncertainties:
        sqrt_betax_err = data_n["sqrt_betax_err"].to_numpy()
        sqrt_betay_err = data_n["sqrt_betay_err"].to_numpy()
        sqrt_betax_n_err = data_n["sqrt_betax_n_err"].to_numpy()
        sqrt_betay_n_err = data_n["sqrt_betay_n_err"].to_numpy()
        alfax_err = data_n["alfax_err"].to_numpy()
        alfay_err = data_n["alfay_err"].to_numpy()

        sqrt_beta_x = unumpy.uarray(sqrt_beta_x, sqrt_betax_err)
        sqrt_beta_y = unumpy.uarray(sqrt_beta_y, sqrt_betay_err)
        sqrt_beta_x_next = unumpy.uarray(sqrt_beta_x_next, sqrt_betax_n_err)
        sqrt_beta_y_next = unumpy.uarray(sqrt_beta_y_next, sqrt_betay_n_err)
        alpha_x = unumpy.uarray(alpha_x, alfax_err)
        alpha_y = unumpy.uarray(alpha_y, alfay_err)

        # Add dispersion uncertainties if available
        if "dx_err" in data_n.columns:
            dx_current_err = data_n["dx_err"].to_numpy()
            dx_next_err = data_n["dx_next_err"].to_numpy()
            dpx_current_err = data_n["dpx_err"].to_numpy()
            dx_current = unumpy.uarray(dx_current, dx_current_err)
            dx_next = unumpy.uarray(dx_next, dx_next_err)
            dpx_current = unumpy.uarray(dpx_current, dpx_current_err)

    # Compute momenta directly
    cos_phi_x = np.cos(phi_x)
    cos_phi_y = np.cos(phi_y)
    tan_phi_x = np.tan(phi_x)
    tan_phi_y = np.tan(phi_y)

    sec_phi_x = 1.0 / cos_phi_x
    sec_phi_y = 1.0 / cos_phi_y

    x_next_norm = (x_next - dpp_est * dx_next) / sqrt_beta_x_next
    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    y_next_norm = y_next / sqrt_beta_y_next
    y_current_norm = y_current / sqrt_beta_y

    px = (
        x_next_norm * sec_phi_x + x_current_norm * (tan_phi_x - alpha_x)
    ) / sqrt_beta_x + dpx_current * dpp_est
    py = (y_next_norm * sec_phi_y + y_current_norm * (tan_phi_y - alpha_y)) / sqrt_beta_y

    # Extract values and uncertainties
    px_vals, px_errs = _extract_values_and_errors(px)
    py_vals, py_errs = _extract_values_and_errors(py)
    data_n["px"] = px_vals
    data_n["py"] = py_vals
    data_n["var_px"] = px_errs**2
    data_n["var_py"] = py_errs**2

    return data_n
