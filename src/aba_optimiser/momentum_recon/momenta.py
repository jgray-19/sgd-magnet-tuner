from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd


def _column_or_zeros(frame, column: str, template: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.zeros_like(template, dtype=float)


def _require_columns(frame, cols: set[str], context: str) -> None:
    missing = cols.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for {context}: {sorted(missing)}")


def momenta_from_prev(data_p: pd.DataFrame, dpp_est: float = 0.0) -> pd.DataFrame:
    _require_columns(
        data_p,
        {"x", "y", "prev_x", "prev_y", "var_x", "var_y", "prev_x_var", "prev_y_var"},
        "momenta_from_prev",
    )

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

    betax = data_p["betax"].to_numpy()
    betay = data_p["betay"].to_numpy()

    dx_current = _column_or_zeros(data_p, "dx", x_current)
    dx_prev = _column_or_zeros(data_p, "dx_prev", x_prev)
    dpx_current = _column_or_zeros(data_p, "dpx", x_current)

    phi_x = data_p["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_p["delta_y"].to_numpy() * 2 * np.pi

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

    data_p["px"] = (
        -(
            x_prev_norm * (cos_phi_x + sin_phi_x * tan_phi_x)
            + x_current_norm * (tan_phi_x + alpha_x)
        )
        / sqrt_beta_x
        + dpx_current * dpp_est
    )
    data_p["py"] = (
        -(
            y_prev_norm * (cos_phi_y + sin_phi_y * tan_phi_y)
            + y_current_norm * (tan_phi_y + alpha_y)
        )
        / sqrt_beta_y
    )

    curr_x_var = data_p["var_x"].to_numpy()
    prev_x_var = data_p["prev_x_var"].to_numpy()
    curr_y_var = data_p["var_y"].to_numpy()
    prev_y_var = data_p["prev_y_var"].to_numpy()

    # Analytical derivatives for variance propagation
    dpx_dx_current = -(tan_phi_x + alpha_x) / betax
    dpx_dx_prev = -(cos_phi_x + sin_phi_x * tan_phi_x) / (sqrt_beta_x * sqrt_beta_x_prev)
    dpy_dy_current = -(tan_phi_y + alpha_y) / betay
    dpy_dy_prev = -(cos_phi_y + sin_phi_y * tan_phi_y) / (sqrt_beta_y * sqrt_beta_y_prev)

    data_p["var_px"] = dpx_dx_current**2 * curr_x_var + dpx_dx_prev**2 * prev_x_var
    data_p["var_py"] = dpy_dy_current**2 * curr_y_var + dpy_dy_prev**2 * prev_y_var
    return data_p


def momenta_from_next(data_n: pd.DataFrame, dpp_est: float = 0.0) -> pd.DataFrame:
    _require_columns(
        data_n,
        {"x", "y", "next_x", "next_y", "var_x", "var_y", "next_x_var", "next_y_var"},
        "momenta_from_next",
    )

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

    betax = data_n["betax"].to_numpy()
    betay = data_n["betay"].to_numpy()

    dx_current = _column_or_zeros(data_n, "dx", x_current)
    dx_next = _column_or_zeros(data_n, "dx_next", x_next)
    dpx_current = _column_or_zeros(data_n, "dpx", x_current)

    phi_x = data_n["delta_x"].to_numpy() * 2 * np.pi
    phi_y = data_n["delta_y"].to_numpy() * 2 * np.pi

    cos_phi_x = np.cos(phi_x)
    cos_phi_y = np.cos(phi_y)
    tan_phi_x = np.tan(phi_x)
    tan_phi_y = np.tan(phi_y)

    sec_phi_x = 1.0 / cos_phi_x
    sec_phi_y = 1.0 / cos_phi_y

    data_n["px"] = (
        ((x_next - dpp_est * dx_next) / sqrt_beta_x_next) * sec_phi_x
        + ((x_current - dpp_est * dx_current) / sqrt_beta_x) * (tan_phi_x - alpha_x)
    ) / sqrt_beta_x + dpx_current * dpp_est
    data_n["py"] = (
        ((y_next) / sqrt_beta_y_next) * sec_phi_y
        + (y_current / sqrt_beta_y) * (tan_phi_y - alpha_y)
    ) / sqrt_beta_y

    # Analytical derivatives for variance propagation
    dpx_dx_current = (tan_phi_x - alpha_x) / betax
    dpx_dx_next = sec_phi_x / (sqrt_beta_x * sqrt_beta_x_next)
    dpy_dy_current = (tan_phi_y - alpha_y) / betay
    dpy_dy_next = sec_phi_y / (sqrt_beta_y * sqrt_beta_y_next)

    curr_x_var = data_n["var_x"].to_numpy()
    next_x_var = data_n["next_x_var"].to_numpy()
    curr_y_var = data_n["var_y"].to_numpy()
    next_y_var = data_n["next_y_var"].to_numpy()

    data_n["var_px"] = dpx_dx_current**2 * curr_x_var + dpx_dx_next**2 * next_x_var
    data_n["var_py"] = dpy_dy_current**2 * curr_y_var + dpy_dy_next**2 * next_y_var
    return data_n
