import numpy as np
import pandas as pd

from aba_optimiser.momentum_recon.errors import compute_measurement_errors, compute_optics_errors
from aba_optimiser.momentum_recon.momenta import _compute_nominal_momenta
from aba_optimiser.momentum_recon.schema import PREV, SUFFIX_PREV


def make_row_prev(*, with_optics_errs: bool) -> pd.DataFrame:
    row = {
        "name": "BPM.1",
        "turn": 0,
        "x": 2e-3,
        "y": -1e-3,
        "var_x": (1e-4) ** 2,
        "var_y": (1e-4) ** 2,
        "x_p": -1.5e-3,
        "y_p": 1.2e-3,
        "var_x_p": (1e-4) ** 2,
        "var_y_p": (1e-4) ** 2,
        "sqrt_betax": 10.0,
        "sqrt_betay": 12.0,
        "sqrt_betax_p": 11.0,
        "sqrt_betay_p": 13.0,
        "alfax": 0.7,
        "alfay": -0.4,
        # phase in TURNS
        "delta_x_p": 0.13,
        "delta_y_p": 0.21,
    }

    if with_optics_errs:
        row.update(
            {
                "sqrt_betax_err": 5e-2,
                "sqrt_betay_err": 5e-2,
                "sqrt_betax_p_err": 5e-2,
                "sqrt_betay_p_err": 5e-2,
                "alfax_err": 2e-2,
                "alfay_err": 2e-2,
                # phase error in TURNS
                "delta_x_p_err": 1e-4,
                "delta_y_p_err": 1e-4,
            }
        )

    return pd.DataFrame([row])


def _px_of(df: pd.DataFrame) -> float:
    px, _ = _compute_nominal_momenta(df, PREV, SUFFIX_PREV, is_prev=True, dpp_est=0.0)
    return float(px[0])


def _py_of(df: pd.DataFrame) -> float:
    _, py = _compute_nominal_momenta(df, PREV, SUFFIX_PREV, is_prev=True, dpp_est=0.0)
    return float(py[0])


def central_diff(df: pd.DataFrame, col: str, func, eps: float) -> float:
    q0 = float(df.loc[0, col])
    df_p = df.copy(deep=True)
    df_m = df.copy(deep=True)
    df_p.loc[0, col] = q0 + eps
    df_m.loc[0, col] = q0 - eps
    return (func(df_p) - func(df_m)) / (2.0 * eps)


def eps_for(col: str, q0: float) -> float:
    """
    Finite-difference step size.

    Needs to be large enough to avoid catastrophic cancellation, but small enough
    to stay in the linear regime. These values are conservative for double precision.
    """
    if col.startswith("delta_"):
        return 1e-7  # turns

    if col.startswith("sqrt_beta"):
        return 1e-5 * max(1.0, abs(q0))

    if col.startswith("alfa"):
        return 1e-7 * max(1.0, abs(q0))

    return 1e-6 * max(1.0, abs(q0))


def numerical_variance(df: pd.DataFrame, cols: list[str], sig2: dict[str, float], func, eps_scale: float) -> float:
    v = 0.0
    for c in cols:
        q0 = float(df.loc[0, c])
        eps = eps_scale * eps_for(c, q0)
        g = central_diff(df, c, func, eps)
        v += g * g * sig2[c]
    return float(v)


def variance_contributions(df: pd.DataFrame, cols: list[str], sig2: dict[str, float], func) -> dict[str, float]:
    contrib = {}
    for c in cols:
        q0 = float(df.loc[0, c])
        g = central_diff(df, c, func, eps_for(c, q0))
        contrib[c] = float(g * g * sig2[c])
    return contrib


def test_jacobian_measurement_only_prev():
    df = make_row_prev(with_optics_errs=False)

    var_px_a, var_py_a = compute_measurement_errors(df, PREV, SUFFIX_PREV, is_prev=True)

    # px depends on x, x_p for measurement part
    g_x = central_diff(df, "x", _px_of, eps_for("x", float(df.loc[0, "x"])))
    g_xp = central_diff(df, "x_p", _px_of, eps_for("x_p", float(df.loc[0, "x_p"])))
    var_px_num = g_x**2 * float(df.loc[0, "var_x"]) + g_xp**2 * float(df.loc[0, "var_x_p"])

    assert np.isfinite(var_px_num)
    assert np.isfinite(float(var_px_a[0]))
    assert var_px_num >= 0.0
    assert float(var_px_a[0]) >= 0.0

    np.testing.assert_allclose(var_px_num, float(var_px_a[0]), rtol=1e-4, atol=0.0)

    # py depends on y, y_p for measurement part
    g_y = central_diff(df, "y", _py_of, eps_for("y", float(df.loc[0, "y"])))
    g_yp = central_diff(df, "y_p", _py_of, eps_for("y_p", float(df.loc[0, "y_p"])))
    var_py_num = g_y**2 * float(df.loc[0, "var_y"]) + g_yp**2 * float(df.loc[0, "var_y_p"])

    assert np.isfinite(var_py_num)
    assert np.isfinite(float(var_py_a[0]))
    assert var_py_num >= 0.0
    assert float(var_py_a[0]) >= 0.0

    np.testing.assert_allclose(var_py_num, float(var_py_a[0]), rtol=1e-4, atol=0.0)


def test_jacobian_full_optics_prev():
    df = make_row_prev(with_optics_errs=True)

    var_px_meas, var_py_meas = compute_measurement_errors(df, PREV, SUFFIX_PREV, is_prev=True)

    var_px_opt_errors, var_py_opt_errors = compute_optics_errors(
        df, PREV, SUFFIX_PREV, is_prev=True, dpp_est=0.0
    )

    var_px_full = var_px_meas + np.sum(var_px_opt_errors, axis=0)
    var_py_full = var_py_meas + np.sum(var_py_opt_errors, axis=0)

    var_px_opt_analytic = float(var_px_full[0]) - float(var_px_meas[0])
    var_py_opt_analytic = float(var_py_full[0]) - float(var_py_meas[0])

    assert var_px_opt_analytic >= 0.0
    assert var_py_opt_analytic >= 0.0

    # X-plane: individual term comparison
    # Order: dx_neighbor, dx_current, dpx_current, alpha, sqrt_beta_current, sqrt_beta_neighbor, phase
    optics_vars_x = ["dx_p", "dx", "dpx", "alfax", "sqrt_betax", "sqrt_betax_p", "delta_x_p"]
    sig2_x_optics = {
        "dx_p": 0.0,  # not in test data
        "dx": 0.0,     # not in test data
        "dpx": 0.0,    # not in test data
        "alfax": float(df.loc[0, "alfax_err"]) ** 2,
        "sqrt_betax": float(df.loc[0, "sqrt_betax_err"]) ** 2,
        "sqrt_betax_p": float(df.loc[0, "sqrt_betax_p_err"]) ** 2,
        "delta_x_p": float(df.loc[0, "delta_x_p_err"]) ** 2,
    }

    failures_x = []
    for i, var_name in enumerate(optics_vars_x):
        if sig2_x_optics[var_name] == 0.0:
            # Skip dispersion terms that are zero
            continue

        q0 = float(df.loc[0, var_name])
        g = central_diff(df, var_name, _px_of, eps_for(var_name, q0))
        var_num = g * g * sig2_x_optics[var_name]
        var_analytic = float(var_px_opt_errors[i][0])

        try:
            np.testing.assert_allclose(var_num, var_analytic, rtol=2e-9, atol=0.0)
        except AssertionError:
            rel_diff = abs(var_num - var_analytic) / max(abs(var_analytic), 1e-20)
            failures_x.append(f"{var_name}: num={var_num:.6e}, analytic={var_analytic:.6e}, rel_diff={rel_diff:.2e}")

    if failures_x:
        raise AssertionError("X-plane optics derivative failures (rtol=0.5%):\n" + "\n".join(failures_x))

    # Y-plane: individual term comparison
    optics_vars_y = ["dy_p", "dy", "dpy", "alfay", "sqrt_betay", "sqrt_betay_p", "delta_y_p"]
    sig2_y_optics = {
        "dy_p": 0.0,  # not in test data
        "dy": 0.0,     # not in test data
        "dpy": 0.0,    # not in test data
        "alfay": float(df.loc[0, "alfay_err"]) ** 2,
        "sqrt_betay": float(df.loc[0, "sqrt_betay_err"]) ** 2,
        "sqrt_betay_p": float(df.loc[0, "sqrt_betay_p_err"]) ** 2,
        "delta_y_p": float(df.loc[0, "delta_y_p_err"]) ** 2,
    }

    failures_y = []
    for i, var_name in enumerate(optics_vars_y):
        if sig2_y_optics[var_name] == 0.0:
            # Skip dispersion terms that are zero
            continue

        q0 = float(df.loc[0, var_name])
        g = central_diff(df, var_name, _py_of, eps_for(var_name, q0))
        var_num = g * g * sig2_y_optics[var_name]
        var_analytic = float(var_py_opt_errors[i][0])

        try:
            np.testing.assert_allclose(var_num, var_analytic, rtol=5e-9, atol=0.0)
        except AssertionError:
            rel_diff = abs(var_num - var_analytic) / max(abs(var_analytic), 1e-20)
            failures_y.append(f"{var_name}: num={var_num:.6e}, analytic={var_analytic:.6e}, rel_diff={rel_diff:.2e}")

    if failures_y:
        raise AssertionError("Y-plane optics derivative failures (rtol=0.5%):\n" + "\n".join(failures_y))
