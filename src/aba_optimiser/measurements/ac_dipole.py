"""Shared AC-dipole optimisation window definition.

Single source of truth for the AC-dipole window used by both the closed-orbit
optimisation workflow and the squeeze quadrupole tuning pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ACDipoleOptimisationWindow:
    """Window definition for full-turn arc-by-arc tracking around the AC dipole."""

    bpm_upstream: str
    bpm_downstream: str


def window_from_attrs(attrs: dict) -> ACDipoleOptimisationWindow | None:
    """Build an :class:`ACDipoleOptimisationWindow` from a DataFrame's attrs.

    Returns ``None`` when the upstream/downstream BPM metadata is absent.
    """
    upstream = attrs.get("ac_dipole_bpm_upstream")
    downstream = attrs.get("ac_dipole_bpm_downstream")
    if not upstream or not downstream:
        return None
    return ACDipoleOptimisationWindow(
        bpm_upstream=str(upstream),
        bpm_downstream=str(downstream),
    )
