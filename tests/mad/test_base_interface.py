"""
Tests for BaseMadInterface.

This module contains pytest tests for the BaseMadInterface class.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from tests.mad.helpers import (
    check_element_observations,
    get_marker_and_element_positions,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "py_name, expected_py_name, var_name, var_value",
    [
        (None, "py", "a", 2),
        ("test_py", "test_py", "b", 3),
    ],
    ids=["default_py_name", "custom_py_name"],
)
def test_init(py_name, expected_py_name, var_name, var_value) -> None:
    """Test initialization of BaseMadInterface."""
    if py_name is None:
        interface = BaseMadInterface()
    else:
        interface = BaseMadInterface(py_name=py_name)
    assert interface.py_name == expected_py_name
    interface.mad.send(f"{var_name} = {var_value}")
    assert getattr(interface.mad, var_name) == var_value
    with contextlib.suppress(Exception):
        del interface


def test_load_sequence(interface: BaseMadInterface, sequence_file: Path) -> None:
    """Test loading a sequence file."""
    # this test explicitly checks load_sequence behaviour
    interface.load_sequence(sequence_file, "lhcb1")
    assert interface.mad.SEQ_NAME == "lhcb1"
    assert (
        interface.mad.loaded_sequence is not None and interface.mad.loaded_sequence != 0
    )
    assert interface.mad.MADX.lhcb1 is not None and interface.mad.MADX.lhcb1 != 0


@pytest.mark.parametrize("energy", [6500.0, 7000.0])
def test_setup_beam(loaded_interface: BaseMadInterface, energy) -> None:
    """Test setting up beam parameters."""
    interface = loaded_interface
    interface.setup_beam(particle="proton", beam_energy=energy)
    assert interface.mad.loaded_sequence.beam.particle == "proton"
    assert interface.mad.loaded_sequence.beam.energy == energy
    assert interface.mad.loaded_sequence.beam.charge == 1
    assert interface.mad.loaded_sequence.beam.spin == 0.5


@pytest.mark.parametrize(
    "pattern",
    ["MB.A33R2.B1", "BPM"],
    ids=["SingleElement", "BPMElements"],
)
def test_observe_elements(
    loaded_interface: BaseMadInterface,
    pattern: str,
) -> None:
    """Test configuring element observation."""
    loaded_interface.observe_elements(pattern)
    check_element_observations(loaded_interface, f"elm.name:match('{pattern}')")
    loaded_interface.unobserve_elements(pattern)


def test_cycle_sequence(loaded_interface: BaseMadInterface) -> None:
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
    loaded_interface: BaseMadInterface,
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


def test_getset_variables(interface: BaseMadInterface) -> None:
    """Test setting MAD variables."""
    interface.set_variables(**{"KQTL_1L1_B1": 1.2, "KQTL_1L2_B1": 2.3})
    assert interface.mad.KQTL_1L1_B1 == 1.2
    assert interface.mad.KQTL_1L2_B1 == 2.3

    v1, v2 = interface.get_variables("KQTL_1L1_B1", "KQTL_1L2_B1")
    assert v1 == 1.2
    assert v2 == 2.3


def test_set_madx_variables(interface: BaseMadInterface) -> None:
    """Test setting MAD-X variables."""
    interface.set_madx_variables(**{"kqtl_1l1_b1": 1.5, "KQTL_1L2_B1": 2.5})
    assert interface.mad.MADX.KQTL_1L1_B1 == 1.5
    assert interface.mad.MADX.kqtl_1l2_b1 == 2.5


def test_set_magnet_strength(loaded_interface: BaseMadInterface) -> None:
    """Test setting magnet strengths."""
    magnet_strengths = {
        "MB.A8R2.B1_k": 3.566169870533780e-04,
        "MB.B8R2.B1_k": 3.566320017035819e-04,
        "MQ.11R2.B1_k": -8.555311397913858e-03,
        "MS.11R2.B1_k": -1.366585087094679e-01,
    }
    mag_type_to_num = {"MB": 0, "MQ": 1, "MS": 2}
    strengths_before = {}
    for mag_name, new_strength in magnet_strengths.items():
        mag_base = mag_name[:-2]
        strength_num = mag_type_to_num[mag_name[:2]]
        strengths_before[mag_base] = loaded_interface.mad[
            f"MADX['{mag_base}'].k{strength_num}"
        ]

    loaded_interface.set_magnet_strengths(magnet_strengths)
    for mag_name, new_strength in magnet_strengths.items():
        mag_base = mag_name[:-2]
        strength_num = mag_type_to_num[mag_name[:2]]
        updated_strength = loaded_interface.mad[f"MADX['{mag_base}'].k{strength_num}"]

        assert updated_strength != strengths_before[mag_base], (
            f"Magnet {mag_name} strength did not change from previous value."
        )
        assert updated_strength == new_strength, (
            f"Magnet {mag_name} strength not updated correctly: "
            f"{updated_strength} != {new_strength}"
        )


def test_apply_corrector_strengths(loaded_interface: BaseMadInterface, data_dir: Path):
    corrector_file = data_dir / "corrector_strengths.tfs"
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Check initial strengths are zero
    for row in corrector_table.itertuples():
        strength = loaded_interface.mad[f"MADX['{row.name}'].kick"]
        assert strength == 0.0, f"Initial strength for {row.name} not zero: {strength}"

    loaded_interface.apply_corrector_strengths(corrector_table)

    # Check strengths updated correctly
    for row in corrector_table.itertuples():
        h_or_v = row.kind[0].lower()
        expected = getattr(row, f"{h_or_v}kick")
        actual = loaded_interface.mad[f"MADX['{row.name}'].kick"]
        assert actual == expected, (
            f"{row.name} strength mismatch: {actual} != {expected}"
        )


def test_twiss(loaded_interface_with_beam: BaseMadInterface):
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
    assert abs(twiss_df.q1 - 62.28) < 1e-8, f"Unexpected Qx: {twiss_df.q1}"
    assert abs(twiss_df.q2 - 60.31) < 1e-8, f"Unexpected Qy: {twiss_df.q2}"

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
    assert twiss_df.q1 - 62.28 < 1e-8, f"Unexpected Qx: {twiss_df.q1}"
    assert twiss_df.q2 - 60.31 < 1e-8, f"Unexpected Qy: {twiss_df.q2}"

    # Now observe BPMs
    interface.observe_elements("BPM")
    twiss_df = interface.run_twiss()
    # There should only be BPMs observed
    assert len(twiss_df) == 563, f"Expected 563 twiss entries, got {len(twiss_df)}"
    assert all(twiss_df.index.str.match(r"^BPM.*")), (
        "Expected only BPM elements in twiss output"
    )
    assert twiss_df.q1 - 62.28 < 1e-8, f"Unexpected Qx: {twiss_df.q1}"
    assert twiss_df.q2 - 60.31 < 1e-8, f"Unexpected Qy: {twiss_df.q2}"
    interface.unobserve_elements("BPM")


@pytest.mark.parametrize(
    "target_qx,target_qy,qx_knob,qy_knob",
    [
        (0.28, 0.31, None, None),
        (0.29, 0.32, "dqx_b1", "dqy_b1"),
        (0.27, 0.30, None, None),
    ],
)
def test_match_tunes(
    loaded_interface_with_beam: BaseMadInterface,
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
    assert abs((twiss_df.q1 % 1) - target_qx) < 1e-5, (
        f"Qx not matched: {twiss_df.q1 % 1} != {target_qx}"
    )
    assert abs((twiss_df.q2 % 1) - target_qy) < 1e-5, (
        f"Qy not matched: {twiss_df.q2 % 1} != {target_qy}"
    )

    # Check that knobs have been changed
    for knob, old_value in knobs.items():
        new_value = interface.mad.MADX[knob]
        assert new_value != old_value, f"Knob {knob} value did not change"
        assert new_value == new_knobs[knob], (
            f"Returned knob value for {knob} does not match MAD value"
        )


def test_run_tracking_default(loaded_interface_with_beam: BaseMadInterface):
    """Test running tracking simulation with default parameters."""
    interface = loaded_interface_with_beam

    interface.run_tracking()
    results_df = interface.mad.trk.to_df()

    # Check that results dataframe has expected columns
    expected_columns = ["turn", "name", "x", "px", "y", "py", "t", "pt"]
    assert all(col in results_df.columns for col in expected_columns), (
        f"Expected columns {expected_columns}, got {results_df.columns}"
    )

    # Check that there is only one row (one turn, one element observed)
    assert len(results_df) == 1, f"Expected 1 tracking row, got {len(results_df)}"

    # Check that the observed element is $end
    assert results_df.iloc[0]["name"] == "$end", (
        f"Expected observed element '$end', got {results_df.iloc[0]['name']}"
    )

    # Check that x, y, px, py, t, pt are all zero as we started on-axis with no errors
    for coord in ["x", "y", "px", "py", "t", "pt"]:
        assert abs(results_df.iloc[0][coord]) < 1e-15, (
            f"Expected {coord} to be ~0, got {results_df.iloc[0][coord]}"
        )


def test_run_tracking_multiple_turns(loaded_interface_with_beam: BaseMadInterface):
    """Test running tracking simulation for multiple turns."""
    interface = loaded_interface_with_beam

    nturns = 10
    interface.run_tracking(nturns=nturns)
    results_df = interface.mad.trk.to_df()

    # Check that there are nturns rows
    assert len(results_df) == nturns, (
        f"Expected {nturns} tracking rows, got {len(results_df)}"
    )

    # Check that turn numbers are correct
    for turn in range(1, nturns + 1):
        turn_row = results_df[results_df["turn"] == turn]
        assert len(turn_row) == 1, (
            f"Expected 1 row for turn {turn}, got {len(turn_row)}"
        )
        assert turn_row.iloc[0]["turn"] == turn, (
            f"Expected turn number {turn}, got {turn_row.iloc[0]['turn']}"
        )

    # Check that x, y, px, py, t, pt are still all zero
    for coord in ["x", "y", "px", "py", "t", "pt"]:
        assert all(abs(results_df[coord]) < 1e-15), (
            f"Expected all {coord} to be ~0, got {results_df[coord].values}"
        )


def test_run_tracking_nonzero_initial(loaded_interface_with_beam: BaseMadInterface):
    """Test running tracking simulation with non-zero initial conditions."""
    interface = loaded_interface_with_beam

    # Add a non-zero initial condition and check tracking
    x0, px0, y0, py0, t0, pt0 = 1e-3, 1e-4, -1e-3, -1e-4, 1e-7, 1e-5
    interface.run_tracking(x0=x0, px0=px0, y0=y0, py0=py0, t0=t0, pt0=pt0, nturns=5)
    results_df = interface.mad.trk.to_df()

    # Check that there are 5 rows
    assert len(results_df) == 5, f"Expected 5 tracking rows, got {len(results_df)}"

    # Check that all rows have non-zero values
    for coord in ["x", "y", "px", "py", "t", "pt"]:
        assert all(abs(results_df[coord]) > 1e-8), (
            f"Expected all {coord} to be non-zero, got {results_df[coord].values}"
        )


def test_run_tracking_with_bpms(loaded_interface_with_beam: BaseMadInterface):
    """Test running tracking simulation with BPMs observed."""
    interface = loaded_interface_with_beam

    # Observe BPMs and check tracking output
    interface.observe_elements("BPM")
    interface.run_tracking(nturns=1)
    results_df = interface.mad.trk.to_df()

    # Check that there are multiple rows for the BPMs
    assert len(results_df) == 563, (
        f"Expected 563 tracking rows for BPMs, got {len(results_df)}"
    )
    interface.unobserve_elements("BPM")
