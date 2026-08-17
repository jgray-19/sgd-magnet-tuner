from __future__ import annotations

from pathlib import Path

from aba_optimiser.measurements.bad_bpms import find_all_bad_bpms, find_all_bad_bpms_from_analysis


def test_find_all_bad_bpms_reads_all_summary_files(tmp_path: Path) -> None:
    (tmp_path / "file_a.bad_bpms_x").write_text("BPM1 reason\nBPM2 reason\n")
    (tmp_path / "file_b.bad_bpms_y").write_text("BPM2 reason\nBPM3 reason\n")

    assert find_all_bad_bpms(tmp_path) == {"BPM1", "BPM2", "BPM3"}


def test_find_all_bad_bpms_from_analysis_reads_measurement_folders(tmp_path: Path) -> None:
    measurement_a = tmp_path / "measurement_a"
    measurement_b = tmp_path / "measurement_b"
    measurement_a.mkdir()
    measurement_b.mkdir()
    (measurement_a / "a.bad_bpms_x").write_text("BPM1 x\n")
    (measurement_b / "b.bad_bpms_y").write_text("BPM2 y\n")
    (tmp_path / "analysis.ini").write_text(
        f"[DEFAULT]\nfiles = ['{measurement_a / 'a.sdds'}', '{measurement_b / 'b.sdds'}']\n"
    )

    assert find_all_bad_bpms_from_analysis(tmp_path) == {"BPM1", "BPM2"}


def test_find_all_bad_bpms_from_analysis_returns_empty_for_invalid_config(tmp_path: Path) -> None:
    (tmp_path / "analysis.ini").write_text("[DEFAULT]\nfiles = [\n")

    assert find_all_bad_bpms_from_analysis(tmp_path) == set()
