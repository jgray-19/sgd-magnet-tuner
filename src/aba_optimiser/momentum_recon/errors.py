"""
Error propagation utilities for momentum reconstruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.momentum_recon.core import neighbour_plane_factors

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd


def compute_measurement_errors(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    is_prev: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute measurement-only error contributions to momentum variances.

    Args:
        data: DataFrame with position and variance columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.

    Returns:
        Tuple of (var_px_measurement, var_py_measurement).
    """
    sigma_x_current = data["var_x"].to_numpy() ** 0.5
    sigma_y_current = data["var_y"].to_numpy() ** 0.5
    sigma_x_neighbor = data[names.var_x].to_numpy() ** 0.5
    sigma_y_neighbor = data[names.var_y].to_numpy() ** 0.5

    sqrt_beta_x = data["sqrt_betax"].to_numpy()
    sqrt_beta_y = data["sqrt_betay"].to_numpy()
    sqrt_beta_x_neigh = data[f"sqrt_betax_{neighbor_suffix}"].to_numpy()
    sqrt_beta_y_neigh = data[f"sqrt_betay_{neighbor_suffix}"].to_numpy()

    alpha_x = data["alfax"].to_numpy()
    alpha_y = data["alfay"].to_numpy()


    phi_x = data[names.delta_x].to_numpy() * 2 * np.pi
    phi_y = data[names.delta_y].to_numpy() * 2 * np.pi

    sign_x, alpha_sign_x, cos_phi_x, tan_phi_x, sec_phi_x = neighbour_plane_factors(phi_x, is_prev=is_prev)
    sign_y, alpha_sign_y, cos_phi_y, tan_phi_y, sec_phi_y = neighbour_plane_factors(phi_y, is_prev=is_prev)

    # Compute variances analytically
    var_px = (
        sigma_x_neighbor**2 * (sign_x * sec_phi_x / (sqrt_beta_x * sqrt_beta_x_neigh)) ** 2
        + sigma_x_current**2 * (sign_x * (tan_phi_x + alpha_sign_x * alpha_x) / sqrt_beta_x**2) ** 2
    )

    var_py = (
        sigma_y_neighbor**2 * (sign_y * sec_phi_y / (sqrt_beta_y * sqrt_beta_y_neigh)) ** 2
        + sigma_y_current**2 * (sign_y * (tan_phi_y + alpha_sign_y * alpha_y) / sqrt_beta_y**2) ** 2
    )

    return var_px, var_py


def compute_optics_errors(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    is_prev: bool,
    dpp_est: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optics error contributions to momentum variances.

    Args:
        data: DataFrame with optics and error columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        dpp_est: Estimated DPP.

    Returns:
        Tuple of (var_px_errors, var_py_errors), where each is a 2D numpy array of shape (7, N).
        Each row is one error contribution. Sum along axis 0 and add to measurement variance for total variance.
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

    dx_current = data.get("dx", np.zeros_like(x_current))
    dx_neighbor = data.get(names.dx, np.zeros_like(x_neighbor))
    dpx_current = data.get("dpx", np.zeros_like(x_current))
    dy_current = data.get("dy", np.zeros_like(y_current))
    dy_neighbor = data.get(names.dy, np.zeros_like(y_neighbor))
    dpy_current = data.get("dpy", np.zeros_like(y_current))
    if np.all(dx_current == 0) and dpp_est != 0.0:
        raise ValueError("Dispersion columns missing but dpp_est is non-zero.")

    phi_x = data[names.delta_x].to_numpy() * 2 * np.pi
    phi_y = data[names.delta_y].to_numpy() * 2 * np.pi

    # Optics errors
    sigma_sqrt_betax = data["sqrt_betax_err"].to_numpy()
    sigma_sqrt_betay = data["sqrt_betay_err"].to_numpy()
    sigma_sqrt_betax_neigh = data[f"sqrt_betax_{neighbor_suffix}_err"].to_numpy()
    sigma_sqrt_betay_neigh = data[f"sqrt_betay_{neighbor_suffix}_err"].to_numpy()
    sigma_alpha_x = data["alfax_err"].to_numpy()
    sigma_alpha_y = data["alfay_err"].to_numpy()
    sigma_dx_current = data.get("dx_err", np.zeros_like(dx_current))
    sigma_dpx_current = data.get("dpx_err", np.zeros_like(dpx_current))
    sigma_dy_current = data.get("dy_err", np.zeros_like(dy_current))
    sigma_dpy_current = data.get("dpy_err", np.zeros_like(dpy_current))
    sigma_dx_neighbor = data.get(f"{names.dx}_err", np.zeros_like(dx_neighbor))
    sigma_dy_neighbor = data.get(f"{names.dy}_err", np.zeros_like(dy_neighbor))
    # Phase uncertainty is stored in TURNS in the dataframe
    sigma_delta_x = data[names.delta_x_err].to_numpy()
    sigma_delta_y = data[names.delta_y_err].to_numpy()

    sign_x, alpha_sign_x, cos_phi_x, tan_phi_x, sec_phi_x = neighbour_plane_factors(phi_x, is_prev=is_prev)
    sign_y, alpha_sign_y, cos_phi_y, tan_phi_y, sec_phi_y = neighbour_plane_factors(phi_y, is_prev=is_prev)

    # Normalized coordinates
    x_current_norm = (x_current - dpp_est * dx_current) / sqrt_beta_x
    x_neighbor_norm = (x_neighbor - dpp_est * dx_neighbor) / sqrt_beta_x_neigh
    y_current_norm = (y_current - dpp_est * dy_current) / sqrt_beta_y
    y_neighbor_norm = (y_neighbor - dpp_est * dy_neighbor) / sqrt_beta_y_neigh

    # Add optics contributions
    # Horizontal
    a_x = tan_phi_x + alpha_sign_x * alpha_x
    sec_tan_x = sec_phi_x * tan_phi_x
    sec2_x = sec_phi_x**2

    dpx_ddx_neigh = sign_x * (-dpp_est) * sec_phi_x / (sqrt_beta_x * sqrt_beta_x_neigh)
    dpx_ddx_curr = sign_x * (-dpp_est) * a_x / sqrt_beta_x**2
    dpx_ddpx = dpp_est
    dpx_dalpha = sign_x * alpha_sign_x * x_current_norm / sqrt_beta_x
    dpx_ds = -(sign_x / sqrt_beta_x**2) * (x_neighbor_norm * sec_phi_x + 2.0 * x_current_norm * a_x)
    dpx_ds_neigh = -sign_x * x_neighbor_norm * sec_phi_x / (sqrt_beta_x * sqrt_beta_x_neigh)
    dpx_dphi = (sign_x / sqrt_beta_x) * (
        x_neighbor_norm * sec_tan_x + x_current_norm * sec2_x
    )

    # Phase error is in turns; dpx_dphi is w.r.t. phi in radians
    # Chain rule: dpx/d(delta_turns) = dpx/d(phi_radians) * d(phi_radians)/d(delta_turns) = dpx_dphi * 2π
    dpx_ddelta = dpx_dphi * 2.0 * np.pi

    var_px_errors = np.vstack((
        sigma_dx_neighbor**2 * dpx_ddx_neigh**2,
        sigma_dx_current**2 * dpx_ddx_curr**2,
        sigma_dpx_current**2 * dpx_ddpx**2,
        sigma_alpha_x**2 * dpx_dalpha**2,
        sigma_sqrt_betax**2 * dpx_ds**2,
        sigma_sqrt_betax_neigh**2 * dpx_ds_neigh**2,
        sigma_delta_x**2 * dpx_ddelta**2,
    ))

    # Vertical
    a_y = tan_phi_y + alpha_sign_y * alpha_y
    sec_tan_y = sec_phi_y * tan_phi_y
    sec2_y = sec_phi_y**2

    dpy_ddy_neigh = sign_y * (-dpp_est) * sec_phi_y / (sqrt_beta_y * sqrt_beta_y_neigh)
    dpy_ddy_curr = sign_y * (-dpp_est) * a_y / sqrt_beta_y**2
    dpy_ddpy = dpp_est
    dpy_dalpha = sign_y * alpha_sign_y * y_current_norm / sqrt_beta_y
    dpy_ds = -(sign_y / sqrt_beta_y**2) * (y_neighbor_norm * sec_phi_y + 2.0 * y_current_norm * a_y)
    dpy_ds_neigh = -sign_y * y_neighbor_norm * sec_phi_y / (sqrt_beta_y * sqrt_beta_y_neigh)
    dpy_dphi = (sign_y / sqrt_beta_y) * (
        y_neighbor_norm * sec_tan_y + y_current_norm * sec2_y
    )

    dpy_ddelta = dpy_dphi * 2.0 * np.pi

    var_py_errors = np.vstack((
        sigma_dy_neighbor**2 * dpy_ddy_neigh**2,
        sigma_dy_current**2 * dpy_ddy_curr**2,
        sigma_dpy_current**2 * dpy_ddpy**2,
        sigma_alpha_y**2 * dpy_dalpha**2,
        sigma_sqrt_betay**2 * dpy_ds**2,
        sigma_sqrt_betay_neigh**2 * dpy_ds_neigh**2,
        sigma_delta_y**2 * dpy_ddelta**2,
    ))

    return var_px_errors, var_py_errors
