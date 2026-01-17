"""Tests for dispersive measurement momentum reconstruction."""

from __future__ import annotations

import pytest
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")

from aba_optimiser.model_creator import convert_tfs_to_madx
from aba_optimiser.momentum_recon.dispersive_measurement import calculate_pz_measurement
from aba_optimiser.xsuite.acd import run_acd_track

from .momentum_test_utils import get_truth_and_twiss, rmse


@pytest.mark.slow
@pytest.mark.parametrize("delta_p", [0.0, 4e-4])
def test_dispersive_measurement_recovers_dpp(json_b1, seq_b1, tmp_path_factory, delta_p):
    """Test that calculate_pz_measurement recovers the true DPP from measurements."""
    json_path = json_b1

    tracking_df, tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq_b1,
        delta_p=delta_p,
        ramp_turns=1000,
        flattop_turns=100,
    )

    truth, tws = get_truth_and_twiss(baseline_line, tracking_df)
    # Select only BPMs that match ^BPM.*\.B1$
    bpm_names = tws[tws.index.str.match(r"^BPM.*\.B1$")].index.tolist()
    tws = tws[tws.index.isin(bpm_names)]

    tws = convert_tfs_to_madx(tws)

    # Generate fake measurements from the twiss
    temp_dir = tmp_path_factory.mktemp("dispersive_measurement")
    generate_fake_measurement(
        twiss=tws,
        outputdir=temp_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
    )

    measurement_dir = temp_dir

    # Call the measurement-based function
    result = calculate_pz_measurement(
        orig_data=tracking_df.copy(deep=True),
        measurement_folder=str(measurement_dir),
        info=False,
    )

    # Check that DPP_EST is close to the true delta_p
    dpp_est = result.attrs["DPP_EST"]
    assert abs(dpp_est - delta_p) < 1e-5, f"DPP_EST {dpp_est:.2e} not close to true {delta_p:.2e}"

    # Also check that the result has the expected columns
    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    # Merge with truth and check RMSE
    merged = truth.merge(
        result[["name", "turn", "px", "py"]],
        on=["name", "turn"],
    )

    px_rmse = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())

    assert px_rmse < 3e-7, f"px RMSE {px_rmse:.2e} > 3e-7"
    assert py_rmse < 3e-7, f"py RMSE {py_rmse:.2e} > 3e-7"
