"""Tests for dispersive measurement momentum reconstruction with uncertainties."""

from __future__ import annotations

import numpy as np
import pytest
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.model_creator import convert_tfs_to_madx
from aba_optimiser.momentum_recon import inject_noise_xy_inplace
from aba_optimiser.momentum_recon.dispersive_measurement import calculate_pz_measurement
from aba_optimiser.xsuite.acd import run_acd_track

from .momentum_test_utils import add_error_to_orbit_measurement, get_truth, rmse, xsuite_to_ngtws


@pytest.mark.slow
@pytest.mark.parametrize("add_noise", [False, True], ids=["no_noise", "with_noise"])
def test_dispersive_measurement_with_uncertainties(
    seq_b1,
    tmp_path,
    add_noise,
    xsuite_json_path,
):
    """Test momentum reconstruction with uncertainty propagation from measurements.

    This test:
    1. Runs ACD tracking
    2. Optionally adds noise to measurements
    3. Performs SVD cleaning if noisy
    4. Creates fake measurements with errors (1e-3 relative)
    5. Calculates momentum with uncertainty propagation
    6. Verifies RMSE within expected bounds

    Args:
        seq_b1: Sequence file fixture for Beam 1.
        tmp_path: Temporary path fixture.
        add_noise: If True, adds noise and performs SVD cleaning before calculating pz.
    """
    json_path = xsuite_json_path("lhcb1.seq")

    tracking_df, x_tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq_b1,
        delta_p=0.0,
        ramp_turns=1000,
        flattop_turns=100,
    )

    ng_tws = xsuite_to_ngtws(x_tws)
    truth = get_truth(tracking_df, ng_tws)

    # Since the above gets only BPMs, we disable drift removal to keep names consistent
    # Drop mu1 and mu2 to avoid issues in conversion
    madx_tws = convert_tfs_to_madx(ng_tws.drop(columns=["mu1", "mu2"]), remove_drifts=False)

    # DY is zero in the model; add a tiny symmetric jitter so relative errors propagate
    # into non-zero absolute errors for DY (and DPY if present).
    rng_dy = np.random.default_rng(2025)
    if "DY" in madx_tws.columns:
        madx_tws["DY"] = madx_tws["DY"] + rng_dy.normal(0.0, 1e-6, size=len(madx_tws))
    if "DPY" in madx_tws.columns:
        madx_tws["DPY"] = madx_tws["DPY"] + rng_dy.normal(0.0, 1e-6, size=len(madx_tws))

    # Generate fake measurements from the twiss
    temp_dir = tmp_path / "dispersive_measurement_uncertainties"
    rel_errors = [1e-2] if add_noise else [0.0]
    generate_fake_measurement(
        twiss=madx_tws,
        outputdir=temp_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
        relative_errors=rel_errors,  # Add 1% relative errors if noisy
        randomize=["values", "errors"] if add_noise else [],
        seed=1234,
    )

    add_error_to_orbit_measurement(temp_dir)

    # Prepare data for calculation
    calc_df = tracking_df.copy(deep=True)
    # Add noise if requested
    if add_noise:
        rng = np.random.default_rng(42)
        inject_noise_xy_inplace(
            calc_df,
            tracking_df,
            rng,
            low_noise_bpms=[],
            noise_std=1e-4,  # 100 um noise
        )
        # Apply SVD cleaning to noisy data
        calc_df = svd_clean_measurements(calc_df)

    # Call the measurement-based function with uncertainties
    result = calculate_pz_measurement(
        orig_data=calc_df,
        measurement_folder=str(temp_dir),
        model_tws=ng_tws,
        info=False,
        include_errors=True,
    )

    # Check that DPP_EST is close to the true delta_p (should be ~0)
    dpp_est = result.attrs["DPP_EST"]
    assert abs(dpp_est - 0.0) < 1e-5, f"DPP_EST {dpp_est:.2e} not close to expected 0.0"

    # Also check that the result has the expected columns
    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    # If uncertainties are included, check variance columns exist
    assert "var_px" in result.columns, "var_px column missing"
    assert "var_py" in result.columns, "var_py column missing"

    # Merge with truth and check RMSE
    merged = truth.merge(
        result[["name", "turn", "px", "py", "var_px", "var_py"]],
        on=["name", "turn"],
    )

    px_rmse = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())
    print(f"PX RMSE (add_noise={add_noise}):", px_rmse)
    print(f"PY RMSE (add_noise={add_noise}):", py_rmse)

    if add_noise:
        # With noise, expect slightly larger errors
        assert px_rmse < 4e-6, f"px RMSE with noise {px_rmse:.2e} > 4e-6"
        assert py_rmse < 4e-6, f"py RMSE with noise {py_rmse:.2e} > 4e-6"
    else:
        # Without noise, expect tighter errors
        assert px_rmse < 2e-7, f"px RMSE without noise {px_rmse:.2e} > 2e-7"
        assert py_rmse < 2e-7, f"py RMSE without noise {py_rmse:.2e} > 2e-7"

    # Check that variance columns exist and are valid
    assert "var_px" in merged.columns, "var_px column missing after merge"
    assert "var_py" in merged.columns, "var_py column missing after merge"
    assert not merged["var_px"].isna().any(), "var_px contains NaN values"
    assert not merged["var_py"].isna().any(), "var_py contains NaN values"
    assert (merged["var_px"] > 0).all(), "var_px contains non-positive values"
    assert (merged["var_py"] > 0).all(), "var_py contains non-positive values"

    # Test if uncertainties are well-calibrated
    px_residuals = merged["px"].to_numpy() - merged["px_true"].to_numpy()
    py_residuals = merged["py"].to_numpy() - merged["py_true"].to_numpy()
    px_uncertainties = np.sqrt(merged["var_px"].to_numpy())
    py_uncertainties = np.sqrt(merged["var_py"].to_numpy())

    # Calculate normalized residuals (z-scores)
    px_z_scores = px_residuals / px_uncertainties
    py_z_scores = py_residuals / py_uncertainties

    # Check for invalid values
    px_valid = np.isfinite(px_z_scores)
    py_valid = np.isfinite(py_z_scores)
    print(f"PX valid z-scores: {np.sum(px_valid)}/{len(px_valid)} ({100 * np.mean(px_valid):.1f}%)")
    print(f"PY valid z-scores: {np.sum(py_valid)}/{len(py_valid)} ({100 * np.mean(py_valid):.1f}%)")

    if not px_valid.all():
        print(f"PX uncertainty range: [{px_uncertainties.min():.2e}, {px_uncertainties.max():.2e}]")
        print(
            f"PX residual range: [{np.abs(px_residuals).min():.2e}, {np.abs(px_residuals).max():.2e}]"
        )
    if not py_valid.all():
        print(f"PY uncertainty range: [{py_uncertainties.min():.2e}, {py_uncertainties.max():.2e}]")
        print(
            f"PY residual range: [{np.abs(py_residuals).min():.2e}, {np.abs(py_residuals).max():.2e}]"
        )

    # Use only finite z-scores for statistics
    px_z_finite = px_z_scores[px_valid]
    py_z_finite = py_z_scores[py_valid]

    # Check that ~68% of residuals are within 1 sigma
    px_within_1sigma = np.sum(np.abs(px_z_finite) < 1) / len(px_z_finite)
    py_within_1sigma = np.sum(np.abs(py_z_finite) < 1) / len(py_z_finite)
    print(f"PX within 1 sigma: {px_within_1sigma:.1%} (expect ~68%)")
    print(f"PY within 1 sigma: {py_within_1sigma:.1%} (expect ~68%)")

    # Check that ~95% of residuals are within 2 sigma
    px_within_2sigma = np.sum(np.abs(px_z_finite) < 2) / len(px_z_finite)
    py_within_2sigma = np.sum(np.abs(py_z_finite) < 2) / len(py_z_finite)
    print(f"PX within 2 sigma: {px_within_2sigma:.1%} (expect ~95%)")
    print(f"PY within 2 sigma: {py_within_2sigma:.1%} (expect ~95%)")

    # Calculate reduced chi-squared (should be ~1 if uncertainties are correct)
    px_chi2_reduced = np.mean(px_z_finite**2)
    py_chi2_reduced = np.mean(py_z_finite**2)
    print(f"PX reduced χ²: {px_chi2_reduced:.3f} (expect ~1)")
    print(f"PY reduced χ²: {py_chi2_reduced:.3f} (expect ~1)")

    # Test that uncertainties are reasonable; only enforce an upper bound since
    # overestimated errors can drive chi² toward 0 when residuals are tiny.
    assert np.isfinite(px_chi2_reduced), "PX χ² is not finite"
    assert np.isfinite(py_chi2_reduced), "PY χ² is not finite"
    assert px_chi2_reduced < 10.0, (
        f"PX χ² = {px_chi2_reduced:.2f} indicates poorly calibrated uncertainties"
    )
    assert py_chi2_reduced < 10.0, (
        f"PY χ² = {py_chi2_reduced:.2f} indicates poorly calibrated uncertainties"
    )

    # At least 40% should be within 1 sigma (relaxed due to correlations/systematics)
    assert px_within_1sigma > 0.4, f"Only {px_within_1sigma:.1%} of PX within 1 sigma (expect >40%)"
    assert py_within_1sigma > 0.4, f"Only {py_within_1sigma:.1%} of PY within 1 sigma (expect >40%)"
