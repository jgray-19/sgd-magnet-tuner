"""
Tests for AbaMadInterface.

This module contains pytest tests for the AbaMadInterface class.
"""

from __future__ import annotations

import numpy as np
import pytest

from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from src.aba_optimiser.physics.deltap import dp2pt as physics_dp2pt
from tests.mad.helpers import (
    check_corrector_strengths,
    check_corrector_strengths_zero,
    check_interface_basic_init,
    cleanup_interface,
    get_marker_and_element_positions,
)


@pytest.mark.parametrize(
    "py_name, expected_py_name, var_name, var_value",
    [
        (None, "py", "a", 2),
        ("test_py", "test_py", "b", 3),
    ],
    ids=["default_py_name", "custom_py_name"],
)
def test_init(py_name, expected_py_name, var_name, var_value) -> None:
    """Test initialization of AbaMadInterface."""
    interface = AbaMadInterface() if py_name is None else AbaMadInterface(py_name=py_name)
    check_interface_basic_init(interface, expected_py_name)
    interface.mad.send(f"{var_name} = {var_value}")
    assert getattr(interface.mad, var_name) == var_value
    cleanup_interface(interface)



def test_cycle_sequence(loaded_interface: AbaMadInterface) -> None:
    """Test cycling sequence to a marker."""
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    assert loaded_interface.mad.recv() is None

    loaded_interface.cycle_sequence("IP5")
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    cycle_marker = loaded_interface.mad.recv()
    assert cycle_marker == "IP5"

    loaded_interface.cycle_sequence()
    loaded_interface.mad.send("""py:send(loaded_sequence:raw_get("__cycle"))""")
    cycle_marker = loaded_interface.mad.recv()
    assert cycle_marker is None


@pytest.mark.parametrize(
    "element, marker_name, offset, expected_marker_name, index_check, pos_check",
    [
        (
            "S.DS.L1.B1",
            None,
            None,
            "S.DS.L1.B1_marker",
            lambda m, e: m + 1 == e,
            lambda m, e: m == e - 1e-10,
        ),
        (
            "S.DS.L1.B1",
            "MyMarker",
            5e-6,
            "MyMarker",
            lambda m, e: m - 1 == e,
            lambda m, e: m == e + 5e-6,
        ),
    ],
    ids=["default_marker", "custom_marker"],
)
def test_install_marker(
    loaded_interface: AbaMadInterface,
    element,
    marker_name,
    offset,
    expected_marker_name,
    index_check,
    pos_check,
) -> None:
    """Test installing a marker element."""
    interface = loaded_interface
    if marker_name and offset is not None:
        ret_name = interface.install_marker(
            element, marker_name=marker_name, offset=offset
        )
    else:
        ret_name = interface.install_marker(element)
    marker_position, marker_index, elem_position, elem_index = (
        get_marker_and_element_positions(interface, expected_marker_name, element)
    )
    assert index_check(marker_index, elem_index)
    assert pos_check(marker_position, elem_position)
    assert ret_name == expected_marker_name


def test_getset_variables(interface: AbaMadInterface) -> None:
    """Test setting MAD variables."""
    interface.set_variables(**{"KQTL_1L1_B1": 1.2, "KQTL_1L2_B1": 2.3})
    assert interface.mad.KQTL_1L1_B1 == 1.2
    assert interface.mad.KQTL_1L2_B1 == 2.3

    v1, v2 = interface.get_variables("KQTL_1L1_B1", "KQTL_1L2_B1")
    assert v1 == 1.2
    assert v2 == 2.3


def test_set_madx_variables(interface: AbaMadInterface) -> None:
    """Test setting MAD-X variables."""
    interface.set_madx_variables(**{"kqtl_1l1_b1": 1.5, "KQTL_1L2_B1": 2.5})
    assert interface.mad.MADX.KQTL_1L1_B1 == 1.5
    assert interface.mad.MADX.kqtl_1l2_b1 == 2.5


def test_set_magnet_strength(loaded_interface: AbaMadInterface) -> None:
    """Test setting magnet strengths."""
    magnet_strengths = {
        "MB.A8R2.B1.k0": 3.566169870533780e-04,
        "MB.B8R2.B1.k0": 3.566320017035819e-04,
        "MQ.11R2.B1.k1": -8.555311397913858e-03,
        "MS.11R2.B1.k2": -1.366585087094679e-01,
    }
    strengths_before = {}
    for mag_name, new_strength in magnet_strengths.items():
        mag_base, strength_num = mag_name.rsplit(".", 1)
        strengths_before[mag_base] = loaded_interface.mad[
            f"MADX['{mag_base}'].{strength_num}"
        ]

    loaded_interface.set_magnet_strengths(magnet_strengths)
    for mag_name, new_strength in magnet_strengths.items():
        mag_base, strength_num = mag_name.rsplit(".", 1)
        updated_strength = loaded_interface.mad[f"MADX['{mag_base}'].{strength_num}"]

        assert updated_strength != strengths_before[mag_base], (
            f"Magnet {mag_name} strength did not change from previous value."
        )
        assert updated_strength == new_strength, (
            f"Magnet {mag_name} strength not updated correctly: "
            f"{updated_strength} != {new_strength}"
        )


def test_set_magnet_strengths_error(loaded_interface: AbaMadInterface) -> None:
    """Test setting magnet strengths with incorrect naming raises error."""
    with pytest.raises(ValueError):
        loaded_interface.set_magnet_strengths(
            {
                "MOB.A8R2.B1.k4": 3.566169870533780e-04,  # Invalid magnet type
            }
        )

    magnet_strengths_invalid_suffix = {
        "MB.A8R2.B1.k": 3.566169870533780e-04,  # Invalid suffix
    }
    with pytest.raises(ValueError):
        loaded_interface.set_magnet_strengths(magnet_strengths_invalid_suffix)

    with pytest.raises(ValueError):
        loaded_interface.set_magnet_strengths(
            {
                "MB.A8R2.B1_k0": 3.566169870533780e-04,  # Invalid format
            }
        )


def test_apply_corrector_strengths(loaded_interface: AbaMadInterface, corrector_table):
    # Check initial strengths are zero
    check_corrector_strengths_zero(loaded_interface, corrector_table)

    loaded_interface.apply_corrector_strengths(corrector_table)

    # Check strengths updated correctly
    check_corrector_strengths(loaded_interface, corrector_table)


def test_twiss(loaded_interface_with_beam: AbaMadInterface):
    """Test twiss function."""
    interface = loaded_interface_with_beam
    twiss_df = interface.run_twiss()

    # Assert the columns we expect are present
    expected_columns = [
        "s",
        "beta11",
        "beta22",
        "alfa11",
        "alfa22",
        "mu1",
        "mu2",
        "dx",
        "dy",
    ]
    for col in expected_columns:
        assert col in twiss_df.columns, (
            f"Expected column {col} not found in twiss output"
        )
    assert "name" not in twiss_df.columns, "Column 'name' should be the index"
    assert twiss_df.index.name == "name", (
        f"Expected index name to be 'name', got {twiss_df.index.name}"
    )

    # There should only be one entry, since by default observe = 1 and nothing has been set to be observed
    assert len(twiss_df) == 1, f"Expected 1 twiss entry, got {len(twiss_df)}"
    # Check the marker is named $end
    assert twiss_df.index[0] == "$end", (
        f"Expected marker name '$end', got {twiss_df.index[0]}"
    )

    # Check the tunes
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.q1}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.q2}"

    # Now set observe to 0
    twiss_df = interface.run_twiss(observe=0)
    # There should be loads of entries now, including drifts
    assert len(twiss_df) > 1000, f"Expected >1000 twiss entries, got {len(twiss_df)}"
    assert "drift__3" in twiss_df.index, (
        "Expected to find drift elements in twiss output"
    )
    assert "MB.A33R2.B1" in twiss_df.index, (
        "Expected to find magnet elements in twiss output"
    )
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.headers['q1']}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.headers['q2']}"

    # Now observe BPMs
    interface.observe_elements("BPM")
    twiss_df = interface.run_twiss()
    # There should only be BPMs observed
    assert len(twiss_df) == 563, f"Expected 563 twiss entries, got {len(twiss_df)}"
    assert all(twiss_df.index.str.match(r"^BPM.*")), (
        "Expected only BPM elements in twiss output"
    )
    assert abs(twiss_df.headers["q1"] - 62.28) < 3e-7, f"Unexpected Qx: {twiss_df.headers['q1']}"
    assert abs(twiss_df.headers["q2"] - 60.31) < 3e-7, f"Unexpected Qy: {twiss_df.headers['q2']}"
    interface.unobserve_elements(["BPM"])


@pytest.mark.parametrize(
    "target_qx,target_qy,qx_knob,qy_knob",
    [
        (0.2801, 0.3101, None, None),
        (0.29, 0.32, "dqx_b1", "dqy_b1"),
        (0.27, 0.30, None, None),
    ],
)
def test_match_tunes(
    loaded_interface_with_beam: AbaMadInterface,
    target_qx: float,
    target_qy: float,
    qx_knob: str,
    qy_knob: str,
):
    """Test matching tunes."""
    interface = loaded_interface_with_beam
    default_qx = qx_knob or "dqx_b1_op"
    default_qy = qy_knob or "dqy_b1_op"
    knobs = {
        default_qx: interface.mad.MADX[default_qx],
        default_qy: interface.mad.MADX[default_qy],
    }
    print("Initial knobs:", knobs)

    kwargs = {}
    if qx_knob:
        kwargs["qx_knob"] = qx_knob
    if qy_knob:
        kwargs["qy_knob"] = qy_knob

    new_knobs = interface.match_tunes(
        target_qx=target_qx, target_qy=target_qy, **kwargs
    )

    twiss_df = interface.run_twiss()
    assert abs((twiss_df.headers["q1"] % 1) - target_qx) < 1e-5, (
        f"Qx not matched: {twiss_df.headers['q1'] % 1} != {target_qx}"
    )
    assert abs((twiss_df.headers["q2"] % 1) - target_qy) < 1e-5, (
        f"Qy not matched: {twiss_df.headers['q2'] % 1} != {target_qy}"
    )

    # Check that knobs have been changed
    for knob, old_value in knobs.items():
        new_value = interface.mad.MADX[knob]
        assert new_value != old_value, f"Knob {knob} value did not change"
        assert new_value == new_knobs[knob], (
            f"Returned knob value for {knob} does not match MAD value"
        )

class TestDp2pt:
    def test_zero_dp(self, loaded_interface_with_beam: AbaMadInterface):
        """Test dp2pt with dp=0 returns 0"""
        interface = loaded_interface_with_beam
        pt = interface.dp2pt(0.0)
        assert np.isclose(pt, 0.0, rtol=1e-12, atol=1e-15)

    def test_positive_dp(self, loaded_interface_with_beam: AbaMadInterface):
        """Test dp2pt with positive dp"""
        interface = loaded_interface_with_beam
        dp = 0.01
        pt = interface.dp2pt(dp)
        # Compare with physics calculation
        mass = 0.938  # proton mass
        energy = interface.mad.loaded_sequence.beam.energy
        expected_pt = physics_dp2pt(dp, mass, energy)
        assert np.isclose(pt, expected_pt, rtol=1e-12, atol=1e-13)

    def test_negative_dp(self, loaded_interface_with_beam: AbaMadInterface):
        """Test dp2pt with negative dp"""
        interface = loaded_interface_with_beam
        dp = -0.005
        pt = interface.dp2pt(dp)
        # Compare with physics calculation
        mass = 0.938
        energy = interface.mad.loaded_sequence.beam.energy
        expected_pt = physics_dp2pt(dp, mass, energy)
        assert np.isclose(pt, expected_pt, rtol=1e-12, atol=1e-13)

    def test_high_energy_approximation(
        self, loaded_interface_with_beam: AbaMadInterface
    ):
        """Test at high energy where pt ≈ dp"""
        interface = loaded_interface_with_beam
        dp = 0.01
        pt = interface.dp2pt(dp)
        # At high energy, pt should be close to dp
        assert np.isclose(pt, dp, rtol=1e-3, atol=1e-6)


class TestPt2dp:
    def test_zero_pt(self, loaded_interface_with_beam: AbaMadInterface):
        """Test pt2dp with pt=0 returns 0"""
        interface = loaded_interface_with_beam
        dp = interface.pt2dp(0.0)
        assert np.isclose(dp, 0.0, rtol=1e-12, atol=1e-15)

    def test_positive_pt(self, loaded_interface_with_beam: AbaMadInterface):
        """Test pt2dp with positive pt"""
        interface = loaded_interface_with_beam
        pt = 0.01
        dp = interface.pt2dp(pt)
        # Check inverse consistency
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15)

    def test_negative_pt(self, loaded_interface_with_beam: AbaMadInterface):
        """Test pt2dp with negative pt"""
        interface = loaded_interface_with_beam
        pt = -0.005
        dp = interface.pt2dp(pt)
        # Check inverse consistency
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15)


def test_pt2dp_dp2pt_consistency(loaded_interface_with_beam: AbaMadInterface):
    """Test that pt2dp and dp2pt are inverses of each other."""
    interface = loaded_interface_with_beam
    test_dps = [0.0, 0.001, -0.001, 0.01, -0.005]
    test_pts = [0.0, 0.001, -0.001, 0.01, -0.005]

    for dp in test_dps:
        pt = interface.dp2pt(dp)
        dp_back = interface.pt2dp(pt)
        assert np.isclose(dp_back, dp, rtol=1e-12, atol=1e-15), (
            f"Inconsistency for dp={dp}: pt={pt}, dp_back={dp_back}"
        )

    for pt in test_pts:
        dp = interface.pt2dp(pt)
        pt_back = interface.dp2pt(dp)
        assert np.isclose(pt_back, pt, rtol=1e-12, atol=1e-15), (
            f"Inconsistency for pt={pt}: dp={dp}, pt_back={pt_back}"
        )


@pytest.mark.parametrize("end_num", [10, 11, 12])
def test_get_bpm_list(loaded_interface: AbaMadInterface, end_num: int) -> None:
    """Test getting list of BPM names within a range."""
    interface = loaded_interface

    # Set up BPM observation
    interface.observe_elements("BPM")

    # Test getting BPM list for a range
    start_num = 9
    start_bpm = f"BPM.{start_num}R2.B1"
    end_bpm = f"BPM.{end_num}R2.B1"
    bpm_range = f"{start_bpm}/{end_bpm}"
    expected_bpms = [f"BPM.{i}R2.B1" for i in range(start_num, end_num + 1)]
    all_bpms, bpm_names = interface.get_bpm_list(bpm_range)

    # Verify all_bpms is a list of strings
    assert isinstance(all_bpms, list)
    assert all(isinstance(name, str) for name in all_bpms)

    # Verify bpm_names (in range) is a list of strings
    assert isinstance(bpm_names, list)
    assert all(isinstance(name, str) for name in bpm_names)

    # Check that the list has expected BPMs
    assert bpm_names == expected_bpms, f"Expected BPMs {expected_bpms}, got {bpm_names}"
    interface.unobserve_elements(["BPM"])
