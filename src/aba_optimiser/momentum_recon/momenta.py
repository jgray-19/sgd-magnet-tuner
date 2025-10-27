from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import tfs


def _column_or_zeros(frame, column: str, template: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.zeros_like(template, dtype=float)


def momenta_from_prev(
    data_p: tfs.TfsDataFrame, dpp_est: float = 0.0
) -> tfs.TfsDataFrame:
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
    return data_p


def momenta_from_next(
    data_n: tfs.TfsDataFrame, dpp_est: float = 0.0
) -> tfs.TfsDataFrame:
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
    return data_n
