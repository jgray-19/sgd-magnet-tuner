from __future__ import annotations

import os
from typing import TYPE_CHECKING

from aba_optimiser.measurements.uncompensated_analysis import (
    NO_COMPENSATION_SUFFIX,
    get_uncompensated_analysis_dir,
    rerun_optics_analysis_without_compensation,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_uncompensated_analysis_dir_appends_suffix(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"

    result = get_uncompensated_analysis_dir(analysis_dir)

    assert result == tmp_path / f"analysis{NO_COMPENSATION_SUFFIX}"


def test_rerun_optics_analysis_without_compensation_uses_latest_ini(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_analysis_dir = tmp_path / "analysis"
    source_analysis_dir.mkdir()

    older_ini = source_analysis_dir / "analysis_older.ini"
    older_ini.write_text('[DEFAULT]\ncompensation = "equation"\nbeam = 1\n')

    latest_ini = source_analysis_dir / "analysis_latest.ini"
    latest_ini.write_text(
        "\n".join(
            [
                "[DEFAULT]",
                "harpy = True",
                "optics = True",
                'compensation = "equation"',
                "beam = 2",
                'model_dir = "/from_ini/model"',
                f'files = [PosixPath("{tmp_path / "meas.sdds"}")]',
                "three_bpm_method = True",
                'nat_tunes = ""',
                'drv_tunes = ""',
            ]
        )
    )

    # Filesystem mtime resolution can be too coarse to distinguish back-to-back
    # writes, so make the "latest" ini unambiguously newer than the older one.
    os.utime(older_ini, (1_000, 1_000))
    os.utime(latest_ini, (2_000, 2_000))

    analysed_stem = tmp_path / "target" / "lin_files" / "meas"
    calls: list[dict] = []

    def fake_hio(**kwargs) -> None:
        calls.append(kwargs)
        analysed_stem.parent.mkdir(parents=True, exist_ok=True)
        analysed_stem.with_suffix(".linx").write_text("x")
        analysed_stem.with_suffix(".liny").write_text("y")
        analysed_stem.parent.joinpath("meas.bad_bpms_x").write_text("BPM.1 bad\n")

    monkeypatch.setattr("aba_optimiser.measurements.uncompensated_analysis.hole_in_one_entrypoint", fake_hio)

    target_dir, bad_bpms = rerun_optics_analysis_without_compensation(
        source_analysis_dir,
        model_dir=tmp_path / "ignored_model",
        beam=7,
        target_analysis_dir=tmp_path / "target",
    )

    assert target_dir == tmp_path / "target"
    assert calls
    assert calls[0]["compensation"] == "none"
    assert calls[0]["outputdir"] == tmp_path / "target"
    assert calls[0]["beam"] == 2
    assert calls[0]["model_dir"] == "/from_ini/model"
    assert calls[0]["three_bpm_method"] is True
    assert calls[0]["files"] == [tmp_path / "meas.sdds"]
    assert "nat_tunes" not in calls[0]
    assert "drv_tunes" not in calls[0]
    assert bad_bpms == ["BPM.1"]


def test_rerun_optics_analysis_without_compensation_reuses_configured_analysed_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_analysis_dir = tmp_path / "analysis"
    source_analysis_dir.mkdir()

    analysed_stem = source_analysis_dir / "linfiles" / "meas_bunchID0"
    analysed_stem.parent.mkdir(parents=True)
    analysed_stem.with_suffix(".bad_bpms_x").write_text("BPM.2 bad\n")

    latest_ini = source_analysis_dir / "analysis_latest.ini"
    latest_ini.write_text(
        "\n".join(
            [
                "[DEFAULT]",
                "optics = True",
                'compensation = "equation"',
                f'files = ["{source_analysis_dir.name}/linfiles/{analysed_stem.name}"]',
                f'outputdir = "{source_analysis_dir.name}"',
            ]
        )
    )

    calls: list[dict] = []

    def fake_hio(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("aba_optimiser.measurements.uncompensated_analysis.hole_in_one_entrypoint", fake_hio)

    target_dir, bad_bpms = rerun_optics_analysis_without_compensation(
        source_analysis_dir,
        target_analysis_dir=tmp_path / "target",
    )

    assert target_dir == tmp_path / "target"
    assert calls
    assert calls[0]["compensation"] == "none"
    assert bad_bpms == ["BPM.2"]
