from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.measurements.analysis import _collect_bad_bpms

if TYPE_CHECKING:
    from pathlib import Path


def test_collect_bad_bpms_merges_x_and_y_summaries(tmp_path: Path) -> None:
    analysed_a = tmp_path / "meas_a_bunchID0"
    analysed_b = tmp_path / "meas_b_bunchID0"

    (tmp_path / "meas_a_bunchID0.bad_bpms_x").write_text("BPM.1 bad\nBPM.2 bad\n")
    (tmp_path / "meas_a_bunchID0.bad_bpms_y").write_text("BPM.2 bad\nBPM.3 bad\n")
    (tmp_path / "meas_b_bunchID0.bad_bpms_x").write_text("BPM.4 bad\n")

    result = _collect_bad_bpms([analysed_a, analysed_b])

    assert result == ["BPM.1", "BPM.2", "BPM.3", "BPM.4"]


def test_collect_bad_bpms_ignores_missing_summary_files(tmp_path: Path) -> None:
    analysed = tmp_path / "meas_bunchID0"

    result = _collect_bad_bpms([analysed])

    assert result == []
