"""Characterisation tests for shared arc/range configuration.

These pin the exact BPM/magnet range strings produced by the measurement
workflows, so the de-duplication into ``arc_config`` is behaviour-preserving.
The expected values are recomputed here directly from the original inline
expressions that lived in each workflow module.
"""

from __future__ import annotations

import pytest

from aba_optimiser.measurements.arc_config import (
    arc_magnet_ranges,
    arc_ranges,
)


def test_arc_magnet_ranges_beam1_matches_original_expression() -> None:
    expected = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    assert arc_magnet_ranges(1) == expected


def test_arc_magnet_ranges_beam2_matches_original_expression() -> None:
    expected = [f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
    assert arc_magnet_ranges(2) == expected


def test_arc_magnet_ranges_rejects_unknown_beam() -> None:
    with pytest.raises(ValueError, match="beam"):
        arc_magnet_ranges(3)


def test_beam1_dense_ranges_match_create_datafile_loop() -> None:
    # create_datafile_loop used the first five BPMs (indices 9..13).
    indices = range(9, 14)
    rc = arc_ranges(1, indices, indices)
    assert rc.magnet_ranges == [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
    assert rc.bpm_starts == [[f"BPM.{i}R{s}.B1" for i in indices] for s in range(1, 9)]
    assert rc.bpm_end_points == [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in indices] for s in range(1, 9)
    ]


def test_beam2_dense_ranges_match_create_datafile_loop() -> None:
    indices = range(9, 14)
    rc = arc_ranges(2, indices, indices)
    assert rc.magnet_ranges == [
        f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)
    ]
    assert rc.bpm_starts == [[f"BPM.{i}L{s}.B2" for i in indices] for s in range(8, 0, -1)]
    assert rc.bpm_end_points == [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in indices] for s in range(8, 0, -1)
    ]


@pytest.mark.parametrize("skip_step", [3, 5])
def test_beam1_sparse_ranges_match_optimise_closed_orbit(skip_step: int) -> None:
    # optimise_closed_orbit used asymmetric ranges: starts 9..34, ends 9..33.
    rc = arc_ranges(1, range(9, 35, skip_step), range(9, 34, skip_step))
    assert rc.bpm_starts == [
        [f"BPM.{i}R{s}.B1" for i in range(9, 35, skip_step)] for s in range(1, 9)
    ]
    assert rc.bpm_end_points == [
        [f"BPM.{i}L{s % 8 + 1}.B1" for i in range(9, 34, skip_step)] for s in range(1, 9)
    ]


@pytest.mark.parametrize("skip_step", [3, 5])
def test_beam2_sparse_ranges_match_optimise_closed_orbit(skip_step: int) -> None:
    # optimise_closed_orbit beam 2: starts 9..33, ends 9..34.
    rc = arc_ranges(2, range(9, 34, skip_step), range(9, 35, skip_step))
    assert rc.bpm_starts == [
        [f"BPM.{i}L{s}.B2" for i in range(9, 34, skip_step)] for s in range(8, 0, -1)
    ]
    assert rc.bpm_end_points == [
        [f"BPM.{i}R{(s - 2) % 8 + 1}.B2" for i in range(9, 35, skip_step)]
        for s in range(8, 0, -1)
    ]
