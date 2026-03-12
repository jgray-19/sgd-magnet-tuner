from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omc3.model.constants import JOB_MODEL_MADX_NOMINAL

from aba_optimiser.model_creator.madx_utils import (
    _detect_accelerator_from_model_dir,
    _detect_lhc_beam_from_model_dir,
    _detect_lhc_year_from_model_dir,
    _load_nominal_creator_from_model_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_nominal_job(model_dir: Path, lines: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / JOB_MODEL_MADX_NOMINAL).write_text(lines)


def test_detect_lhc_year_from_nominal_header(tmp_path: Path) -> None:
    model_dir = tmp_path / "model_b1"
    _write_nominal_job(
        model_dir,
        '\n'.join(
            [
                'title, "LHC Model created by omc3";',
                "! LHC year               2025",
                "use, sequence=lhcb1;",
            ]
        ),
    )

    assert _detect_lhc_year_from_model_dir(model_dir) == "2025"


def test_detect_lhc_year_requires_omc3_header(tmp_path: Path) -> None:
    model_dir = tmp_path / "model_b1"
    _write_nominal_job(model_dir, "use, sequence=lhcb1;")

    with pytest.raises(ValueError, match="Could not infer LHC year"):
        _detect_lhc_year_from_model_dir(model_dir)


def test_load_nominal_creator_passes_detected_lhc_year(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model_b2"
    _write_nominal_job(
        model_dir,
        '\n'.join(
            [
                f"! Acc-Models             {model_dir / 'acc-models-lhc'}",
                "! LHC year               2022",
                "use, sequence=lhcb2;",
            ]
        ),
    )

    captured: dict[str, object] = {}

    class DummyCreator:
        def __init__(self, accel: object) -> None:
            self.accel = accel

    class FakeLhc:
        LOCAL_REPO_NAME = "acc-models-lhc"

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("aba_optimiser.model_creator.madx_utils.Lhc", FakeLhc)
    monkeypatch.setattr(
        "aba_optimiser.model_creator.madx_utils.get_model_creator_class",
        lambda accel, creator_type: DummyCreator,
    )

    creator, accelerator, beam = _load_nominal_creator_from_model_dir(model_dir)

    assert isinstance(creator, DummyCreator)
    assert accelerator == "lhc"
    assert beam == 2
    assert captured == {"model_dir": model_dir, "beam": 2, "year": "2022"}


def test_detect_accelerator_from_sps_job_header(tmp_path: Path) -> None:
    model_dir = tmp_path / "sps_model"
    _write_nominal_job(model_dir, "call, file='acc-models-sps/sps.seq';\nuse, sequence=sps;")

    assert _detect_accelerator_from_model_dir(model_dir) == "sps"


def test_detect_lhc_beam_falls_back_to_directory_name(tmp_path: Path) -> None:
    model_dir = tmp_path / "some_model_b1"
    _write_nominal_job(model_dir, "! LHC year               2025")

    assert _detect_lhc_beam_from_model_dir(model_dir) == 1
