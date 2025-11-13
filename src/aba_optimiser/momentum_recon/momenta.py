from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import tfs


def _column_or_zeros(frame, column: str, template: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.zeros_like(template, dtype=float)


def _propagated_variance(
    coeff_first: np.ndarray,
    coeff_second: np.ndarray,
    var_first: np.ndarray,
    var_second: np.ndarray,
) -> np.ndarray:
    return np.square(coeff_first) * var_first + np.square(coeff_second) * var_second


def _require_columns(frame, cols: set[str], context: str) -> None:
    missing = cols.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for {context}: {sorted(missing)}")


def momenta_from_prev(
    data_p: tfs.TfsDataFrame, dpp_est: float = 0.0
) -> tfs.TfsDataFrame:
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

    denom_prev_x = sqrt_beta_x * sqrt_beta_x_prev
    coeff_prev_x = np.divide(
        cos_phi_x + sin_phi_x * tan_phi_x,
        denom_prev_x,
        out=np.zeros_like(denom_prev_x, dtype=float),
        where=denom_prev_x != 0.0,
    )
    coeff_curr_x = np.divide(
        tan_phi_x + alpha_x,
        betax,
        out=np.zeros_like(betax, dtype=float),
        where=betax != 0.0,
    )

    prev_x_var = data_p["prev_x_var"].to_numpy()
    curr_x_var = data_p["var_x"].to_numpy()
    var_px = _propagated_variance(coeff_prev_x, coeff_curr_x, prev_x_var, curr_x_var)
    data_p["var_px"] = var_px

    denom_prev_y = sqrt_beta_y * sqrt_beta_y_prev
    coeff_prev_y = np.divide(
        cos_phi_y + sin_phi_y * tan_phi_y,
        denom_prev_y,
        out=np.zeros_like(denom_prev_y, dtype=float),
        where=denom_prev_y != 0.0,
    )
    coeff_curr_y = np.divide(
        tan_phi_y + alpha_y,
        betay,
        out=np.zeros_like(betay, dtype=float),
        where=betay != 0.0,
    )

    prev_y_var = data_p["prev_y_var"].to_numpy()
    curr_y_var = data_p["var_y"].to_numpy()
    var_py = _propagated_variance(coeff_prev_y, coeff_curr_y, prev_y_var, curr_y_var)
    data_p["var_py"] = var_py
    return data_p


def momenta_from_next(
    data_n: tfs.TfsDataFrame, dpp_est: float = 0.0
) -> tfs.TfsDataFrame:
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

    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_next_norm = (x_next - dpp_est * dx_next) / sqrt_beta_x_next
    y_current_norm = y_current / sqrt_beta_y
    y_next_norm = y_next / sqrt_beta_y_next

    cos_phi_x = np.cos(phi_x)
    cos_phi_y = np.cos(phi_y)
    tan_phi_x = np.tan(phi_x)
    tan_phi_y = np.tan(phi_y)

    sec_phi_x = np.divide(1.0, cos_phi_x, where=cos_phi_x != 0.0)
    sec_phi_y = np.divide(1.0, cos_phi_y, where=cos_phi_y != 0.0)

    data_n["px"] = (
        x_next_norm * sec_phi_x + x_current_norm * (tan_phi_x - alpha_x)
    ) / sqrt_beta_x + dpx_current * dpp_est
    data_n["py"] = (
        y_next_norm * sec_phi_y + y_current_norm * (tan_phi_y - alpha_y)
    ) / sqrt_beta_y

    denom_next_x = sqrt_beta_x * sqrt_beta_x_next
    coeff_next_x = np.divide(
        sec_phi_x,
        denom_next_x,
        out=np.zeros_like(denom_next_x, dtype=float),
        where=denom_next_x != 0.0,
    )
    coeff_curr_x = np.divide(
        tan_phi_x - alpha_x,
        betax,
        out=np.zeros_like(betax, dtype=float),
        where=betax != 0.0,
    )

    next_x_var = data_n["next_x_var"].to_numpy()
    curr_x_var = data_n["var_x"].to_numpy()
    var_px = _propagated_variance(coeff_next_x, coeff_curr_x, next_x_var, curr_x_var)
    data_n["var_px"] = var_px

    denom_next_y = sqrt_beta_y * sqrt_beta_y_next
    coeff_next_y = np.divide(
        sec_phi_y,
        denom_next_y,
        out=np.zeros_like(denom_next_y, dtype=float),
        where=denom_next_y != 0.0,
    )
    coeff_curr_y = np.divide(
        tan_phi_y - alpha_y,
        betay,
        out=np.zeros_like(betay, dtype=float),
        where=betay != 0.0,
    )

    next_y_var = data_n["next_y_var"].to_numpy()
    curr_y_var = data_n["var_y"].to_numpy()
    var_py = _propagated_variance(coeff_next_y, coeff_curr_y, next_y_var, curr_y_var)
    data_n["var_py"] = var_py
    return data_n
