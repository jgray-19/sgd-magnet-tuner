from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import tfs


@pytest.fixture(scope="module")
def twiss_and_measurements(seq_b1, tmp_path_factory):
    """Fixture to create twiss and fake measurements once per module."""
    from aba_optimiser.mad.base_mad_interface import BaseMadInterface

    interface = BaseMadInterface()
    interface.load_sequence(seq_b1, "lhcb1")
    interface.setup_beam(particle="proton", beam_energy=6800.0)

    # Run twiss using MAD interface
    interface.observe_elements("BPM.*%.B1$")
    twiss_df: tfs.TfsDataFrame = interface.run_twiss(coupling=True)
    twiss_df = convert_tfs_to_madx(twiss_df)

    # Generate fake measurements
    temp_dir = tmp_path_factory.mktemp("twiss_measurement")
    generate(
        twiss=twiss_df,
        outputdir=temp_dir,
        parameters=["BETX", "BETY", "DX", "DY", "X", "Y", "PHASEX", "PHASEY"],
    )

    measurement_dir = Path(temp_dir)
    return twiss_df, measurement_dir


def test_twiss_from_measurement_mu(twiss_and_measurements):
    """Test that build_twiss_from_measurements produces correct MU columns."""
    twiss_df, measurement_dir = twiss_and_measurements
    twiss_from_meas = build_twiss_from_measurements(measurement_dir, include_errors=False)

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


def test_twiss_from_measurement_other_columns(twiss_and_measurements):
    """Test that build_twiss_from_measurements produces correct beta, alpha, dispersion columns."""
    twiss_df, measurement_dir = twiss_and_measurements
    twiss_from_meas = build_twiss_from_measurements(measurement_dir, include_errors=False)

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
