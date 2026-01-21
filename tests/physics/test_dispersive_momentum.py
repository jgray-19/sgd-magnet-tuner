"""Integration tests for dispersive momentum reconstruction using xtrack data."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")

from aba_optimiser.filtering.svd import svd_clean_measurements  # noqa: E402
from aba_optimiser.momentum_recon.dispersive import (  # noqa: E402
    calculate_pz as dispersive_calc,
)
from aba_optimiser.momentum_recon.transverse import (  # noqa: E402
    calculate_pz as transverse_calc,
)
from aba_optimiser.momentum_recon.transverse import (
    inject_noise_xy_inplace,
)
from aba_optimiser.xsuite.acd import run_acd_track

from .momentum_test_utils import get_truth, rmse, xsuite_to_ngtws


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
def test_dispersive_momentum_on_momentum(seq_file, data_dir, xsuite_json_path):
    """Test dispersive momentum reconstruction for on-momentum beam.

    For on-momentum particles (δp=0), dispersive and transverse methods
    should produce nearly identical results.
    """
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)

    tracking_df, tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=0.0,
        ramp_turns=1000,
        flattop_turns=100,
    )

    # Convert twiss to ngtws format, NAME will be index
    tws = xsuite_to_ngtws(tws)
    truth = get_truth(tracking_df, tws)

    # Transverse reconstruction (baseline)
    trans_result = transverse_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_trans", "py": "py_trans"})

    # Dispersive reconstruction
    disp_result = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_disp", "py": "py_disp"})

    # Merge results
    merged = truth.merge(
        trans_result[["name", "turn", "px_trans", "py_trans"]],
        on=["name", "turn"],
    ).merge(
        disp_result[["name", "turn", "px_disp", "py_disp"]],
        on=["name", "turn"],
    )

    assert len(merged) == len(truth)

    # Compute RMSE for both methods
    px_rmse_trans = rmse(merged["px_true"].to_numpy(), merged["px_trans"].to_numpy())
    py_rmse_trans = rmse(merged["py_true"].to_numpy(), merged["py_trans"].to_numpy())
    px_rmse_disp = rmse(merged["px_true"].to_numpy(), merged["px_disp"].to_numpy())
    py_rmse_disp = rmse(merged["py_true"].to_numpy(), merged["py_disp"].to_numpy())

    # For on-momentum, both methods should give reasonable results
    # Note: Using driven motion with AC dipole introduces some systematic offset
    # from the natural optics model, so tolerances are relaxed
    assert px_rmse_trans < 3e-7, f"Transverse px RMSE {px_rmse_trans:.2e} > 3e-7"
    assert py_rmse_trans < 3e-7, f"Transverse py RMSE {py_rmse_trans:.2e} > 3e-7"
    assert px_rmse_disp < 3e-7, f"Dispersive px RMSE {px_rmse_disp:.2e} > 3e-7"
    assert py_rmse_disp < 3e-7, f"Dispersive py RMSE {py_rmse_disp:.2e} > 3e-7"

    # Both methods should be equivalent for on-momentum (dispersive <= transverse)
    assert px_rmse_disp <= px_rmse_trans, (
        f"Dispersive px RMSE {px_rmse_disp:.2e} > transverse {px_rmse_trans:.2e}"
    )
    assert py_rmse_disp <= py_rmse_trans, (
        f"Dispersive py RMSE {py_rmse_disp:.2e} > transverse {py_rmse_trans:.2e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize("delta_p", [-5e-4, 4e-4])
def test_dispersive_momentum_off_momentum(seq_file, delta_p, data_dir, xsuite_json_path):
    """Test dispersive momentum reconstruction for off-momentum beam.

    For off-momentum particles (δp≠0), the dispersive method should correct
    for the dispersive contribution to the x coordinate, resulting in better
    px reconstruction than the transverse method.

    The py reconstruction should be unaffected by dispersion and both methods
    should perform equally well.
    """
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)

    tracking_df, tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=delta_p,
        ramp_turns=1000,
        flattop_turns=100,
    )
    tws = xsuite_to_ngtws(tws)
    truth = get_truth(tracking_df, tws)

    # Transverse reconstruction (no dispersion correction)
    trans_result = transverse_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_trans", "py": "py_trans"})

    # Dispersive reconstruction (with dispersion correction)
    disp_result = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_disp", "py": "py_disp"})

    # Merge results
    merged = truth.merge(
        trans_result[["name", "turn", "px_trans", "py_trans"]],
        on=["name", "turn"],
    ).merge(
        disp_result[["name", "turn", "px_disp", "py_disp"]],
        on=["name", "turn"],
    )

    assert len(merged) == len(truth)

    # Compute RMSE for both methods
    px_rmse_trans = rmse(merged["px_true"].to_numpy(), merged["px_trans"].to_numpy())
    py_rmse_trans = rmse(merged["py_true"].to_numpy(), merged["py_trans"].to_numpy())
    px_rmse_disp = rmse(merged["px_true"].to_numpy(), merged["px_disp"].to_numpy())
    py_rmse_disp = rmse(merged["py_true"].to_numpy(), merged["py_disp"].to_numpy())

    # For off-momentum:
    # - Transverse px should be degraded due to uncorrected dispersion
    # - Dispersive px should be better (dispersion corrected)
    # - py should be similar for both (no dispersion in y)

    # py should still be reasonably accurate for both methods
    assert py_rmse_trans < 2e-7, f"Transverse py RMSE {py_rmse_trans:.2e} > 2e-7"
    assert py_rmse_disp < 2e-7, f"Dispersive py RMSE {py_rmse_disp:.2e} > 2e-7"

    # Dispersive px should be 20x better than transverse px
    assert px_rmse_disp <= px_rmse_trans / 20, (
        f"Dispersive px RMSE {px_rmse_disp:.2e} should be <= transverse {px_rmse_trans:.2e}"
    )

    # Both should give reasonable results
    tol = 6e-6 if "crossing" not in seq_file else 7.2e-6
    assert px_rmse_disp < 2e-7, f"Dispersive px RMSE {px_rmse_disp:.2e} > 2e-7"
    assert px_rmse_trans < tol, f"Transverse px RMSE {px_rmse_trans:.2e} > {tol:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize("delta_p", [-5e-4, 4e-4])
def test_dispersive_momentum_off_momentum_with_noise(seq_file, delta_p, data_dir, xsuite_json_path):
    """Test dispersive momentum reconstruction with noise for off-momentum beam.

    For off-momentum particles (δp≠0), verify that SVD cleaning improves
    reconstruction quality for noisy data compared to noisy reconstruction.
    """
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)

    tracking_df, tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=delta_p,
        ramp_turns=1000,
        flattop_turns=100,
    )

    tws = xsuite_to_ngtws(tws)
    truth = get_truth(tracking_df, tws)

    # Clean reconstruction (no noise)
    clean_result = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_clean", "py": "py_clean"})

    # Noisy reconstruction - inject noise manually then calculate
    rng = np.random.default_rng(42)
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(noisy_df, tracking_df, rng, low_noise_bpms=[])
    noisy_result = dispersive_calc(
        noisy_df,
        tws=tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_noisy", "py": "py_noisy"})

    # SVD cleaned reconstruction - apply SVD to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    svd_result = dispersive_calc(
        cleaned_df,
        tws=tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_svd", "py": "py_svd"})

    # Merge all results
    merged = truth.merge(
        clean_result[["name", "turn", "px_clean", "py_clean"]],
        on=["name", "turn"],
    ).merge(
        noisy_result[["name", "turn", "px_noisy", "py_noisy"]],
        on=["name", "turn"],
    ).merge(
        svd_result[["name", "turn", "px_svd", "py_svd"]],
        on=["name", "turn"],
    )

    assert len(merged) == len(truth)

    # Compute RMSE
    px_rmse_clean = rmse(merged["px_true"].to_numpy(), merged["px_clean"].to_numpy())
    py_rmse_clean = rmse(merged["py_true"].to_numpy(), merged["py_clean"].to_numpy())
    px_rmse_noisy = rmse(merged["px_true"].to_numpy(), merged["px_noisy"].to_numpy())
    py_rmse_noisy = rmse(merged["py_true"].to_numpy(), merged["py_noisy"].to_numpy())
    px_rmse_svd = rmse(merged["px_true"].to_numpy(), merged["px_svd"].to_numpy())
    py_rmse_svd = rmse(merged["py_true"].to_numpy(), merged["py_svd"].to_numpy())

    # Noisy should be significantly worse than clean (at least 70% worse)
    assert px_rmse_noisy >= px_rmse_clean * 1.70, (
        f"Noisy px RMSE {px_rmse_noisy:.2e} should be >= clean {px_rmse_clean:.2e} * 1.70"
    )
    assert py_rmse_noisy >= py_rmse_clean * 1.95, (
        f"Noisy py RMSE {py_rmse_noisy:.2e} should be >= clean {py_rmse_clean:.2e} * 1.95"
    )

    # SVD should be slightly worse than clean but still acceptable (e.g. within 30%)
    assert px_rmse_svd <= px_rmse_clean * 1.30, (
        f"SVD px RMSE {px_rmse_svd:.2e} should be <= clean {px_rmse_clean:.2e} * 1.30"
    )
    assert py_rmse_svd <= py_rmse_clean * 1.30, (
        f"SVD py RMSE {py_rmse_svd:.2e} should be <= clean {py_rmse_clean:.2e} * 1.30"
    )

    # SVD should significantly improve over noisy (at least 50% reduction)
    assert px_rmse_svd <= px_rmse_noisy * 0.65, (
        f"SVD px RMSE {px_rmse_svd:.2e} should be <= noisy {px_rmse_noisy:.2e} * 0.65"
    )
    assert py_rmse_svd <= py_rmse_noisy * 0.60, (
        f"SVD py RMSE {py_rmse_svd:.2e} should be <= noisy {py_rmse_noisy:.2e} * 0.60"
    )
