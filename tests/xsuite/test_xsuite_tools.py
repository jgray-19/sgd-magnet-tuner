"""
Tests for xsuite_tools module.

This module contains pytest tests for the xsuite_tools functions.
"""

import os
import re
import time

import numpy as np
import pandas as pd
import pytest
import xpart as xp
import xtrack as xt

from aba_optimiser.config import (
    BEAM_ENERGY,
    LHCB1_SEQ_NAME,
)
from aba_optimiser.xsuite.acd import insert_ac_dipole, run_acd_twiss
from aba_optimiser.xsuite.env import create_xsuite_environment, initialise_env
from aba_optimiser.xsuite.monitors import insert_particle_monitors_at_pattern, line_to_dataframes
from aba_optimiser.xsuite.tracking import run_tracking


@pytest.fixture(scope="function")
def test_env():
    """Create a test environment with a simple line."""
    env = xt.Environment()

    # Create a more realistic test line with BPMs and proper optics
    test_line = xt.Line(
        elements=[
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),  # bpm.1
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),  # bpm.2
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=0.2),
            xt.Marker(),  # bpm.3
            xt.Drift(length=2.0),
            xt.Quadrupole(length=1.0, k1=-0.2),
            xt.Marker(),  # bpm.4
            xt.Marker(),  # mkqa.6l4.b1
        ],
        element_names=[
            "drift1",
            "quad.1",
            "bpm.1",
            "drift2",
            "quad.2",
            "bpm.2",
            "drift3",
            "quad.3",
            "bpm.3",
            "drift4",
            "quad.4",
            "mkqa.6l4.b1",
            "bpm.4",
        ],
    )

    # Set particle reference
    test_line.particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=450e9,  # 450 GeV
    )

    env.lines["test_line"] = test_line
    return env


@pytest.fixture(scope="function")
def test_line(test_env):
    """Get test line from environment."""
    return test_env.lines["test_line"]


@pytest.fixture
def twiss_table(test_line):
    """Compute twiss table for test line."""
    return test_line.twiss(method="4d")


@pytest.mark.slow
def test_create_xsuite_environment(tmp_path, seq_b1):
    # Test basic creation
    json_file = tmp_path / "lhcb1.json"
    json_file.unlink(missing_ok=True)
    env = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    assert LHCB1_SEQ_NAME in env.lines
    assert json_file.exists()
    line = env.lines[LHCB1_SEQ_NAME]
    assert np.isclose(line.particle_ref.energy0[0], BEAM_ENERGY * 1e9, rtol=1e-10)
    assert len(line.particle_ref.energy0) == 1

    # Test rerun_madx flag
    mod_time_before = json_file.stat().st_mtime
    env2 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    mod_time_after = json_file.stat().st_mtime
    assert mod_time_before == mod_time_after
    assert LHCB1_SEQ_NAME in env2.lines

    env3 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, rerun_madx=True, json_file=json_file
    )
    mod_time_after_rerun = json_file.stat().st_mtime
    assert mod_time_after_rerun > mod_time_after
    assert LHCB1_SEQ_NAME in env3.lines

    # Test custom json and energy
    temp_json = tmp_path / "temp_xsuite.json"
    env4 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=temp_json, beam_energy=450
    )
    assert LHCB1_SEQ_NAME in env4.lines
    line4 = env4.lines[LHCB1_SEQ_NAME]
    assert np.isclose(line4.particle_ref.energy0[0], 450e9, rtol=1e-10)
    assert len(line4.particle_ref.energy0) == 1

    # Test sequence file newer than json
    mod_time_before = json_file.stat().st_mtime
    # Make sequence file appear newer
    future_time = time.time() + 1
    os.utime(str(seq_b1), (future_time, future_time))
    env5 = create_xsuite_environment(
        sequence_file=seq_b1, seq_name=LHCB1_SEQ_NAME, json_file=json_file
    )
    mod_time_after = json_file.stat().st_mtime
    assert mod_time_after > mod_time_before
    assert LHCB1_SEQ_NAME in env5.lines


@pytest.mark.parametrize(
    "qx, qy, k1_mqy, k0_mb, k2_mcs",
    [
        (0.31, 0.32, 0.00323, 0.0003166, -1.3),
        (0.28, 0.31, 0.0025, 0.00025, -1.0),
    ],
    ids=[
        "Init test case 1",
        "Init test case 2",
    ],
)
def test_initialise_env(corrector_table, seq_b1, qx, qy, k1_mqy, k0_mb, k2_mcs, tmp_path):
    """Test initialise_env function."""
    json_file = tmp_path / "temp_xsuite.json"
    json_file.unlink(missing_ok=True)  # Ensure fresh environment
    matched_tunes = {"dqx_b1_op": qx, "dqy_b1_op": qy}
    magnet_strengths = {
        "mqy.b5l2.b1.k1": k1_mqy,
        "mb.b8r2.b1.k0": k0_mb,
        "mcs.b8r2.b1.k2": k2_mcs,
    }

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        sequence_file=seq_b1,
        beam_energy=BEAM_ENERGY,
        seq_name=LHCB1_SEQ_NAME,
        json_file=json_file,
    )
    assert env["dqx.b1_op"] == qx  # ty:ignore[not-subscriptable]
    assert env["dqy.b1_op"] == qy  # ty:ignore[not-subscriptable]
    assert np.isclose(env["mqy.b5l2.b1"].k1, k1_mqy)  # ty:ignore[not-subscriptable]
    assert np.isclose(env["mb.b8r2.b1"].k0, k0_mb)  # ty:ignore[not-subscriptable]
    assert np.isclose(env["mcs.b8r2.b1"].k2, k2_mcs)  # ty:ignore[not-subscriptable]

    for row in corrector_table.itertuples():
        assert len(env[row.ename.lower()].knl) == 1  # ty:ignore[not-subscriptable]
        assert np.isclose(env[row.ename.lower()].knl[0], -row.hkick)  # ty:ignore[not-subscriptable]
        assert len(env[row.ename.lower()].ksl) == 1  # ty:ignore[not-subscriptable]
        assert np.isclose(env[row.ename.lower()].ksl[0], row.vkick)  # ty:ignore[not-subscriptable]f


@pytest.mark.parametrize(
    "pattern, inplace, pre_add_existing, num_turns, num_part",
    [
        ("bpm.", False, False, 10, 5),
        ("quad.", False, False, 5, 2),
        ("bpm.", True, False, 10, 5),
        ("bpm.", False, True, 10, 1),
    ],
    ids=[
        "bpm pattern, not inplace, no existing",
        "quad pattern, not inplace, no existing",
        "bpm pattern, inplace, no existing",
        "bpm pattern, not inplace, with existing",
    ],
)
def test_insert_particle_monitors_at_pattern(
    test_line: xt.Line,
    pattern: str,
    inplace: bool,
    pre_add_existing: bool,
    num_turns: int,
    num_part: int,
):
    """Test inserting particle monitors at various patterns with different options."""
    if pre_add_existing:
        # Manually add a monitor with uppercase name
        bpm_name = "bpm.1"
        monitor_name = bpm_name.upper()
        test_line.env._element_dict[monitor_name] = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=10, num_particles=1
        )
        bpm_index = test_line.element_names.index(bpm_name)
        test_line.insert(monitor_name, at=bpm_index + 1)

    initial_num_elements = len(test_line.elements)
    new_line = insert_particle_monitors_at_pattern(
        test_line, pattern, num_turns=num_turns, num_particles=num_part, inplace=inplace
    )

    if inplace:
        assert new_line is test_line
    else:
        assert new_line is not test_line

    num_matches = sum(1 for name in test_line.element_names if re.match(pattern, name))
    expected_add = num_matches if not pre_add_existing else num_matches - 1  # since one replaced
    assert len(new_line.elements) == initial_num_elements + expected_add

    # Check that monitors exist with correct properties
    for name in test_line.element_names:
        if re.match(pattern, name):
            monitor_name = name.upper()
            assert monitor_name in new_line.element_names
            monitor = new_line[monitor_name]
            assert isinstance(monitor, xt.ParticlesMonitor)
            assert monitor.x.shape == (num_part, num_turns)
            assert monitor.y.shape == (num_part, num_turns)


def test_insert_ac_dipole(test_line: xt.Line, twiss_table: xt.TwissTable):
    """Test inserting AC dipole into the line."""
    beam = 1
    acd_ramp = 100
    total_turns = 1000
    driven_tunes = [0.16, 0.18]

    initial_num_elements = len(test_line.elements)
    new_line = insert_ac_dipole(test_line, twiss_table, beam, acd_ramp, total_turns, driven_tunes)

    # Check that a new line is returned (function copies)
    assert new_line is not test_line
    assert len(test_line.elements) == initial_num_elements

    # Check that AC dipoles are inserted
    acd_marker = f"mkqa.6l4.b{beam}"
    acdh_name = f"mkach.6l4.b{beam}"
    acdv_name = f"mkacv.6l4.b{beam}"

    assert acdh_name in new_line.element_names
    assert acdv_name in new_line.element_names

    # Check that they are ACDipole elements
    acdh = new_line[acdh_name]
    acdv = new_line[acdv_name]
    assert isinstance(acdh, xt.ACDipole)
    assert isinstance(acdv, xt.ACDipole)

    # Check properties
    assert acdh.plane == "h"  # horizontal
    assert acdv.plane == "v"  # vertical
    assert acdh.freq == driven_tunes[0]
    assert acdv.freq == driven_tunes[1]

    # Check that they are at the same position as the marker
    marker_pos = new_line.get_s_position(acd_marker)
    acdh_pos = new_line.get_s_position(acdh_name)
    acdv_pos = new_line.get_s_position(acdv_name)
    assert acdh_pos == marker_pos
    assert acdv_pos == marker_pos

    # Check that the parameters are set correctly
    assert all(acdh.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])
    assert all(acdv.ramp == [0, acd_ramp, total_turns, total_turns + acd_ramp])

    # Check that the twiss is unaffected by the insertion
    new_twiss = new_line.twiss(method="4d")
    # Remove the row corresponding to the AC dipole for comparison
    pd_twiss = new_twiss.to_pandas()
    pd_twiss = pd_twiss[pd_twiss["name"] != acdh_name]
    pd_twiss = pd_twiss[pd_twiss["name"] != acdv_name]
    pd_twiss.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(twiss_table.to_pandas(), pd_twiss, check_exact=False, rtol=1e-10)


def test_run_acd_twiss(test_line: xt.Line):
    """Test running AC dipole twiss."""
    driven_tunes = [0.16, 0.18]
    beam = 1
    dpp = 0.05

    twiss_orig = test_line.twiss(method="4d")
    acd_twiss_on_mom = run_acd_twiss(test_line, beam, 0, driven_tunes)
    acd_twiss_off_mom = run_acd_twiss(test_line, beam, dpp, driven_tunes)
    assert isinstance(acd_twiss_on_mom, xt.TwissTable)
    assert isinstance(acd_twiss_off_mom, xt.TwissTable)

    assert f"mkach.6l4.b{beam}" in acd_twiss_on_mom["name"]
    assert f"mkacv.6l4.b{beam}" in acd_twiss_on_mom["name"]
    assert f"mkach.6l4.b{beam}" in acd_twiss_off_mom["name"]
    assert f"mkacv.6l4.b{beam}" in acd_twiss_off_mom["name"]

    assert np.isclose(acd_twiss_on_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_on_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qx"], driven_tunes[0], atol=1e-10, rtol=1e-10)
    assert np.isclose(acd_twiss_off_mom["qy"], driven_tunes[1], atol=1e-10, rtol=1e-10)

    # Check that the off and on momentum twiss are different
    assert not np.isclose(
        acd_twiss_on_mom["betx"],
        acd_twiss_off_mom["betx"],
        atol=1e-2,
        rtol=1e-2,
    ).all()

    # Check original line unaffected
    twiss_after = test_line.twiss(method="4d")
    assert np.isclose(twiss_orig["qx"], twiss_after["qx"], atol=1e-10, rtol=1e-10)
    assert np.isclose(twiss_orig["qy"], twiss_after["qx"], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    "nturns, num_particles",
    [
        (1, 1),
        (20, 5),
    ],
    ids=[
        "1_turn_1_particle",
        "20_turns_5_particles",
    ],
)
def test_run_tracking(test_line: xt.Line, nturns: int, num_particles: int):
    """Test running tracking simulation with various turns and particles."""
    # Create particles at origin
    particles = test_line.build_particles(
        x=np.zeros(num_particles),
        px=np.zeros(num_particles),
        y=np.zeros(num_particles),
        py=np.zeros(num_particles),
        delta=np.zeros(num_particles),
    )

    # Insert monitors at BPM locations with more turns than we'll track
    monitor_turns = nturns + 10  # More than nturns
    monitored_line = insert_particle_monitors_at_pattern(
        test_line,
        pattern="bpm.",
        num_turns=monitor_turns,
        num_particles=num_particles,
        inplace=False,
    )

    # Run tracking for nturns
    run_tracking(monitored_line, particles, nturns)

    # Check that tracking completed successfully for all particles
    assert np.all(particles.state == 1)

    # Verify monitors recorded data for exactly nturns turns per particle
    bpm_names = ["bpm.1", "bpm.2", "bpm.3", "bpm.4"]
    for bpm in bpm_names:
        monitor_name = bpm.upper()
        assert monitor_name in monitored_line.element_names
        monitor = monitored_line[monitor_name]
        assert isinstance(monitor, xt.ParticlesMonitor)

        # Check that state shows exactly nturns * num_particles successful recordings
        expected_successful = nturns * num_particles
        successful_turns = np.sum(monitor.data.state[:] == 1)
        assert successful_turns == expected_successful, (
            f"BPM {bpm} recorded {successful_turns} turns, expected {expected_successful}"
        )

        # Check shapes: (num_particles, monitor_turns)
        assert monitor.x.shape == (num_particles, monitor_turns)

        # Check that the first nturns have state=1 for all particles, rest have state!=1
        state_data = monitor.data.state[:].reshape(num_particles, monitor_turns)
        assert np.all(state_data[:, :nturns] == 1)
        assert np.all(state_data[:, nturns:] != 1)


def test_line_to_dataframes(test_line: xt.Line):
    """Test converting tracked line to list of DataFrames."""
    # Create a simple line with 2 monitors
    env = xt.Environment()
    simple_line = xt.Line(
        elements=[
            xt.Drift(length=1.0),
            xt.ParticlesMonitor(num_particles=2, start_at_turn=0, stop_at_turn=2),
            xt.Drift(length=1.0),
            xt.ParticlesMonitor(num_particles=2, start_at_turn=0, stop_at_turn=2),
        ],
        element_names=["drift1", "MONITOR1", "drift2", "MONITOR2"],
    )
    env.lines["simple"] = simple_line

    # Set up simple tracking data for 2 particles, 2 turns
    # Data format: [p0_t0, p1_t0, p0_t1, p1_t1]
    simple_line["MONITOR1"].data.particle_id[:] = [0, 1, 0, 1]
    simple_line["MONITOR1"].data.x[:] = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]  # Monitor 1: p0: [1,3], p1: [2,4]
    simple_line["MONITOR1"].data.px[:] = [0.1, 0.2, 0.3, 0.4]
    simple_line["MONITOR1"].data.y[:] = [0.01, 0.02, 0.03, 0.04]
    simple_line["MONITOR1"].data.py[:] = [0.001, 0.002, 0.003, 0.004]

    simple_line["MONITOR2"].data.particle_id[:] = [0, 1, 0, 1]
    simple_line["MONITOR2"].data.x[:] = [
        10.0,
        20.0,
        30.0,
        40.0,
    ]  # Monitor 2: p0: [10,30], p1: [20,40]
    simple_line["MONITOR2"].data.px[:] = [0.01, 0.02, 0.03, 0.04]
    simple_line["MONITOR2"].data.y[:] = [0.001, 0.002, 0.003, 0.004]
    simple_line["MONITOR2"].data.py[:] = [0.0001, 0.0002, 0.0003, 0.0004]

    # Call the function
    dataframes = line_to_dataframes(simple_line)

    # Should return one DataFrame per particle
    assert len(dataframes) == 2
    assert all(isinstance(df, pd.DataFrame) for df in dataframes)

    # Each DataFrame should have correct structure
    expected_columns = ["name", "turn", "x", "px", "y", "py"]
    for df in dataframes:
        assert list(df.columns) == expected_columns
        assert len(df) == 4  # 2 monitors × 2 turns

    # Check particle 0 data
    df_p0 = dataframes[0]
    # Should have data from both monitors for both turns
    monitor1_data = df_p0[df_p0["name"] == "MONITOR1"]
    monitor2_data = df_p0[df_p0["name"] == "MONITOR2"]

    # Particle 0: MONITOR1 should have x=[1.0, 3.0] for turns [1, 2]
    assert list(monitor1_data["x"]) == [1.0, 3.0]
    assert list(monitor1_data["turn"]) == [1, 2]

    # Particle 0: MONITOR2 should have x=[10.0, 30.0] for turns [1, 2]
    assert list(monitor2_data["x"]) == [10.0, 30.0]
    assert list(monitor2_data["turn"]) == [1, 2]

    # Check particle 1 data
    df_p1 = dataframes[1]
    monitor1_data = df_p1[df_p1["name"] == "MONITOR1"]
    monitor2_data = df_p1[df_p1["name"] == "MONITOR2"]

    # Particle 1: MONITOR1 should have x=[2.0, 4.0] for turns [1, 2]
    assert list(monitor1_data["x"]) == [2.0, 4.0]
    assert list(monitor1_data["turn"]) == [1, 2]

    # Particle 1: MONITOR2 should have x=[20.0, 40.0] for turns [1, 2]
    assert list(monitor2_data["x"]) == [20.0, 40.0]
    assert list(monitor2_data["turn"]) == [1, 2]


def test_line_to_dataframes_no_monitors(test_line: xt.Line):
    """Test line_to_dataframes raises error when no monitors present."""
    with pytest.raises(ValueError, match="No ParticlesMonitor found"):
        line_to_dataframes(test_line)


def test_line_to_dataframes_particle_loss():
    """Test line_to_dataframes raises error when particles are lost."""
    # Create a simple line with one monitor
    env = xt.Environment()
    line = xt.Line(
        elements=[
            xt.Drift(length=1.0),
            xt.ParticlesMonitor(num_particles=2, start_at_turn=0, stop_at_turn=2),
        ],
        element_names=["drift", "MONITOR"],
    )
    env.lines["test"] = line

    # Simulate particle loss: last particle_id is not the max
    monitor = line["MONITOR"]
    # Particle 1 lost, last is 0 but max is 1
    monitor.data.particle_id[:] = [0, 1, 0, 0]

    with pytest.raises(AssertionError, match="Some particles were lost"):
        line_to_dataframes(line)


def test_line_to_dataframes_inconsistent_particles():
    """Test line_to_dataframes raises error when monitors have different particle counts."""
    # Create a line with two monitors having different particle counts
    env = xt.Environment()
    line = xt.Line(
        elements=[
            xt.Drift(length=1.0),
            xt.ParticlesMonitor(num_particles=2, start_at_turn=0, stop_at_turn=2),
            xt.Drift(length=1.0),
            xt.ParticlesMonitor(
                num_particles=3, start_at_turn=0, stop_at_turn=2
            ),  # Different count
        ],
        element_names=["drift1", "MONITOR1", "drift2", "MONITOR2"],
    )
    env.lines["test"] = line

    # Set different numbers of unique particle IDs
    line["MONITOR1"].data.particle_id[:] = [0, 0, 1, 1]  # 2 unique
    line["MONITOR2"].data.particle_id[:] = [0, 0, 0, 1, 1, 2]  # 3 unique

    with pytest.raises(ValueError, match="Monitors have different number of particles"):
        line_to_dataframes(line)
