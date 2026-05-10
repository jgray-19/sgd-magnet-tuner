from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


def _read_tunes_from_twiss_table(table_path: Path) -> tuple[float, float]:
    q1 = None
    q2 = None
    pattern = re.compile(r"^@\s+(Q[12])\s+%\S+\s+([-+0-9.eE]+)\s*$")

    for line in table_path.read_text().splitlines():
        match = pattern.match(line)
        if match is None:
            continue

        name, value = match.groups()
        if name == "Q1":
            q1 = float(value)
        elif name == "Q2":
            q2 = float(value)

    if q1 is None or q2 is None:
        raise ValueError(f"Could not read Q1/Q2 headers from {table_path}")
    return q1, q2


def _write_psb3_sequence_without_acd(source_path: Path, target_path: Path) -> Path:
    seq_text = source_path.read_text()
    filtered_lines = [
        line
        for line in seq_text.splitlines()
        if "hacmap" not in line.lower() and "vacmap" not in line.lower()
    ]
    target_path.write_text("\n".join(filtered_lines) + "\n")
    return target_path


@pytest.fixture
def psb3_saved_sequence(data_dir: Path, tmp_path: Path) -> Path:
    source = data_dir / "model_creator" / "psb3_saved.seq"
    target = tmp_path / source.name
    shutil.copy2(source, target)
    return target


@pytest.fixture
def psb3_saved_sequence_without_acd(data_dir: Path, tmp_path: Path) -> Path:
    source = data_dir / "model_creator" / "psb3_saved.seq"
    return _write_psb3_sequence_without_acd(source, tmp_path / "psb3_saved_no_acd.seq")


@pytest.mark.parametrize(
    ("sequence_path_fixture", "twiss_table"),
    [
        ("psb3_saved_sequence_without_acd", "psb3_twiss.dat"),
        ("psb3_saved_sequence", "psb3_twiss_ac.dat"),
    ],
    ids=["acd_off", "acd_on"],
)
def test_psb3_saved_sequence_runs_twiss_and_matches_expected_tunes(
    request: pytest.FixtureRequest,
    data_dir: Path,
    sequence_path_fixture: str,
    twiss_table: str,
) -> None:
    twiss_table_path = data_dir / "model_creator" / twiss_table
    expected_q1, expected_q2 = _read_tunes_from_twiss_table(twiss_table_path)
    sequence_path = request.getfixturevalue(sequence_path_fixture)

    interface = AbaMadInterface(accelerator=PSB(ring=3, sequence_file=sequence_path))
    twiss = interface.run_twiss(observe=0)

    assert len(twiss) > 100
    assert twiss.headers["q1"] == pytest.approx(expected_q1, rel=1e-8)
    assert twiss.headers["q2"] == pytest.approx(expected_q2, rel=1e-8)
