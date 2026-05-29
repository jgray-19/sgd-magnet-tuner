"""Shared arc/range configuration for LHC closed-orbit measurement workflows.

Several measurement workflows (closed-orbit optimisation, the batch data-file
loop, the beam-2 wrapper) build the same per-arc magnet/BPM ranges. The magnet
ranges and the L/R-suffix structure are identical per beam; only the set of BPM
indices selected within each arc differs between workflows. This module is the
single source of truth for those dataclasses and for building the ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Number of arcs in the LHC ring.
N_ARCS = 8


@dataclass
class RangeConfig:
    """Per-arc magnet and BPM ranges (one entry per arc) for one beam."""

    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


@dataclass
class MeasurementSetupConfig:
    """Input metadata describing one closed-orbit measurement campaign."""

    beam: int
    model_dir: str
    arc_config: RangeConfig
    folder: str
    name_prefix: str
    times: list[str]
    title: str


def arc_magnet_ranges(beam: int) -> list[str]:
    """Return the ``BPM/BPM`` magnet range string for each of the 8 arcs."""
    if beam == 1:
        return [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    if beam == 2:
        return [f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
    raise ValueError(f"Unsupported beam {beam!r}; expected 1 or 2")


def arc_ranges(
    beam: int,
    start_indices: Iterable[int],
    end_indices: Iterable[int],
) -> RangeConfig:
    """Build per-arc magnet/BPM ranges for ``beam``.

    ``start_indices`` and ``end_indices`` select which BPM numbers populate the
    start and end of each arc (e.g. ``range(9, 14)`` for the first five BPMs, or
    ``range(9, 35, 5)`` for a sparser selection). The per-arc iteration order and
    L/R suffix logic are fixed per beam and shared across all callers.
    """
    start_indices = list(start_indices)
    end_indices = list(end_indices)
    if beam == 1:
        arcs = range(1, 9)
        bpm_starts = [[f"BPM.{i}R{s}.B1" for i in start_indices] for s in arcs]
        bpm_end_points = [[f"BPM.{i}L{s % 8 + 1}.B1" for i in end_indices] for s in arcs]
    elif beam == 2:
        arcs = range(8, 0, -1)
        bpm_starts = [[f"BPM.{i}L{s}.B2" for i in start_indices] for s in arcs]
        bpm_end_points = [[f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in end_indices] for s in arcs]
    else:
        raise ValueError(f"Unsupported beam {beam!r}; expected 1 or 2")
    return RangeConfig(
        magnet_ranges=arc_magnet_ranges(beam),
        bpm_starts=bpm_starts,
        bpm_end_points=bpm_end_points,
    )
