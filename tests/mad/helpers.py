"""
Common helper functions for MAD interface tests.

This module contains utility functions used across MAD interface test modules.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tfs

if TYPE_CHECKING:
    from aba_optimiser.mad.base_mad_interface import BaseMadInterface


def _recv_n(mad: Any, n: int) -> list[Any]:
    """Helper to receive n messages from a MAD instance."""
    results = []
    for _ in range(n):
        results.append(mad.recv())
    return results


def check_interface_basic_init(interface: Any, py_name: str = "py") -> None:
    """Check basic interface initialization properties."""
    assert interface.py_name == py_name
    assert hasattr(interface, "mad")


def cleanup_interface(interface: Any) -> None:
    """Clean up interface instance."""
    with contextlib.suppress(Exception):
        del interface


def check_sequence_loaded(interface: BaseMadInterface, seq_name: str = "lhcb1") -> None:
    """Check that sequence is loaded correctly."""
    assert seq_name == interface.mad.SEQ_NAME
    assert interface.mad.loaded_sequence is not None


def check_beam_setup(
    interface: BaseMadInterface,
    particle: str = "proton",
    energy: float | None = None,
    charge: int = 1,
    spin: float = 0.5,
) -> None:
    """Check beam setup properties."""
    assert particle == interface.mad.loaded_sequence.beam.particle
    if energy is not None:
        assert energy == interface.mad.loaded_sequence.beam.energy
    assert charge == interface.mad.loaded_sequence.beam.charge
    assert spin == interface.mad.loaded_sequence.beam.spin


def check_element_observations(
    interface: BaseMadInterface,
    condition: str,
) -> None:
    """Helper to check element observations in MAD."""
    interface.mad.send(f"""
for _, elm in loaded_sequence:iter() do
    if {condition} then
        assert(
            elm:is_observed(),
            "Found expected observation missing for element " .. elm.name
        )
    else
        assert(
            not elm:is_observed(),
            "Found unexpected observation for element " .. elm.name
        )
    end
end
py:send("PASS")""")
    assert interface.mad.recv() == "PASS"


def get_marker_and_element_positions(
    interface: BaseMadInterface, marker_name: str, element_name: str
) -> tuple[float, int, float, int]:
    """Helper to get positions and indices of marker and element."""
    interface.mad.send(f"""
local marker_pos, elm_pos, marker_idx, elm_idx
for i, elm, s, ds in loaded_sequence:siter() do
    if elm.name == "{marker_name}" then
        marker_idx = i
        marker_pos = s
    end
    if elm.name == "{element_name}" then
        elm_idx = i
        elm_pos = s
    end
end
py:send(marker_pos)
py:send(marker_idx)
py:send(elm_pos)
py:send(elm_idx)""")
    return _recv_n(interface.mad, 4)


def check_corrector_strengths_zero(
    interface: BaseMadInterface, corrector_table: tfs.TfsDataFrame
) -> None:
    """Check that all corrector strengths are initially zero."""
    for row in corrector_table.itertuples():
        strength = interface.mad[f"MADX['{row.name}'].kick"]
        assert strength == 0.0, f"Initial strength for {row.name} not zero: {strength}"


def check_corrector_strengths(
    interface: BaseMadInterface,
    corrector_table: tfs.TfsDataFrame,
) -> None:
    """Check that corrector strengths match expected values.

    Args:
        interface: The MAD interface to check
        corrector_table: The corrector table containing expected values
        use_real_strengths: If True, expect values from table; if False, expect zeros
    """
    for row in corrector_table.itertuples():
        h_or_v = row.kind[0].lower()
        expected = getattr(row, f"{h_or_v}kick")
        actual = interface.mad[f"MADX['{row.name}'].kick"]
        assert actual == expected, (
            f"{row.name} strength mismatch: {actual} != {expected}"
        )
