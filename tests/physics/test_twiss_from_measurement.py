from pathlib import Path

import pandas as pd
import pytest
from omc3.scripts.fake_measurement_from_model import generate

from aba_optimiser.model_creator import convert_tfs_to_madx
from src.aba_optimiser.measurements.twiss_from_measurement import build_twiss_from_measurements
from src.aba_optimiser.physics.bpm_phases import (
    next_bpm_to_pi,
    # next_bpm_to_pi_2,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)


def _setup_twiss_and_measurements(
    tmp_path: Path, seq_b1: Path, include_dispersion: bool = True
) -> tuple[pd.DataFrame, Path]:
    """Set up twiss data and fake measurements for testing.

    Args:
        tmp_path: Temporary directory for measurements.
        seq_b1: Path to B1 sequence file.
        include_dispersion: Whether to include dispersion in generated measurements.
    """
    from aba_optimiser.mad.base_mad_interface import BaseMadInterface

    interface = BaseMadInterface()
    interface.load_sequence(seq_b1, "lhcb1")
    interface.setup_beam(particle="proton", beam_energy=6800.0)

    # Run twiss using MAD interface
    interface.observe_elements("BPM.*%.B1$")
    twiss_df: pd.DataFrame = interface.run_twiss(coupling=True)
    twiss_df = convert_tfs_to_madx(twiss_df)

    # Generate fake measurements
    temp_dir = tmp_path / "twiss_measurement"
    temp_dir.mkdir()
    parameters = ["BETX", "BETY", "X", "Y", "PHASEX", "PHASEY"]
    if include_dispersion:
        parameters.extend(["DX", "DY"])

    generate(
        twiss=twiss_df,
        outputdir=temp_dir,
        parameters=parameters,
    )

    measurement_dir = temp_dir
    return twiss_df, measurement_dir


def test_twiss_from_measurement_mu(tmp_path, seq_b1):
    """Test that build_twiss_from_measurements produces correct MU columns."""
    twiss_df, measurement_dir = _setup_twiss_and_measurements(tmp_path, seq_b1)

    twiss_from_meas, _ = build_twiss_from_measurements(measurement_dir, include_errors=False)

    q1 = twiss_df.headers["Q1"]
    q2 = twiss_df.headers["Q2"]

    # Test for X plane MU
    mu_x_original = twiss_df["MUX"]
    mu_x_meas = twiss_from_meas["MUX"]
    result_original_x = prev_bpm_to_pi_2(mu_x_original, q1)
    result_meas_x = prev_bpm_to_pi_2(mu_x_meas, q1)
    pd.testing.assert_frame_equal(result_original_x, result_meas_x)

    # Test for Y plane MU
    mu_y_original = twiss_df["MUY"]
    mu_y_meas = twiss_from_meas["MUY"]
    result_original_y = prev_bpm_to_pi(mu_y_original, q2)
    result_meas_y = prev_bpm_to_pi(mu_y_meas, q2)
    pd.testing.assert_frame_equal(result_original_y, result_meas_y)

    # Test another function
    result_original_next = next_bpm_to_pi(mu_x_original, q1)
    result_meas_next = next_bpm_to_pi(mu_x_meas, q1)
    pd.testing.assert_frame_equal(result_original_next, result_meas_next)


def test_twiss_from_measurement_other_columns(tmp_path, seq_b1):
    """Test that build_twiss_from_measurements produces correct beta, alpha, dispersion columns."""
    twiss_df, measurement_dir = _setup_twiss_and_measurements(tmp_path, seq_b1)

    twiss_from_meas, _ = build_twiss_from_measurements(measurement_dir, include_errors=False)

    # Check beta columns
    pd.testing.assert_series_equal(twiss_df["BETX"], twiss_from_meas["BETX"], check_names=False)
    pd.testing.assert_series_equal(twiss_df["BETY"], twiss_from_meas["BETY"], check_names=False)

    # Check alpha columns
    pd.testing.assert_series_equal(twiss_df["ALFX"], twiss_from_meas["ALFX"], check_names=False)
    pd.testing.assert_series_equal(twiss_df["ALFY"], twiss_from_meas["ALFY"], check_names=False)

    # Check dispersion columns
    pd.testing.assert_series_equal(twiss_df["DX"], twiss_from_meas["DX"], check_names=False)
    pd.testing.assert_series_equal(twiss_df["DY"], twiss_from_meas["DY"], check_names=False)

    # Check S column
    pd.testing.assert_series_equal(twiss_df["S"], twiss_from_meas["S"], check_names=False)

@pytest.mark.parametrize(
    "include_dispersion, include_errors, expected_dispersion",
    [
        (False, False, False),
        (True, False, True),
        (False, True, False),
        (True, True, True),
    ],
    ids=[
        "no-disp-no-errors",
        "disp-no-errors",
        "no-disp-errors",
        "disp-errors",
    ],
)
def test_branches_dispersion_and_errors(
    tmp_path, seq_b1, include_dispersion, include_errors, expected_dispersion
):
    """Exercise all four branches of dispersion/error handling."""
    twiss_df_unused, measurement_dir = _setup_twiss_and_measurements(
        tmp_path, seq_b1, include_dispersion=include_dispersion
    )

    twiss_from_meas, dispersion_found = build_twiss_from_measurements(
        measurement_dir, include_errors=include_errors
    )

    # Dispersion presence matches expectation
    assert dispersion_found is expected_dispersion

    # Check dispersion columns when expected
    has_dx = "DX" in twiss_from_meas.columns
    has_dy = "DY" in twiss_from_meas.columns
    if expected_dispersion:
        assert has_dx and has_dy
    else:
        assert not has_dx and not has_dy

    # Error columns expectation
    error_cols = [col for col in twiss_from_meas.columns if "ERR" in col.upper()]
    if include_errors:
        assert error_cols, "Should expose error columns when include_errors=True"
    else:
        assert not error_cols, "Should not expose error columns when include_errors=False"
