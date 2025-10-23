"""
Common helper functions for MAD interface tests.

This module contains utility functions used across MAD interface test modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aba_optimiser.mad.base_mad_interface import BaseMadInterface


def _recv_n(mad: Any, n: int) -> list[Any]:
    """Helper to receive n messages from a MAD instance."""
    results = []
    for _ in range(n):
        results.append(mad.recv())
    return results


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
