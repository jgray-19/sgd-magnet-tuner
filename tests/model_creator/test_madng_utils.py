from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import tfs

from aba_optimiser.accelerators import LHC
from aba_optimiser.model_creator import ModelCreatorMadngInterface
from aba_optimiser.model_creator.config import DRV_TUNES, NAT_TUNES

SEQUENCE_DIR = Path("tests/data/sequences")


def _copy_lhc_sequence(tmp_path: Path, beam: int = 1) -> Path:
    seq_name = f"lhcb{beam}"
    source_seq = SEQUENCE_DIR / f"{seq_name}.seq"
    source_mad = SEQUENCE_DIR / f"{seq_name}.mad"
    target_seq = tmp_path / f"{seq_name}_saved.seq"
    target_mad = target_seq.with_suffix(".mad")

    shutil.copy2(source_seq, target_seq)
    shutil.copy2(source_mad, target_mad)
    return target_seq


def _name_column(df: tfs.TfsDataFrame) -> str:
    return "NAME" if "NAME" in df.columns else "name"


@pytest.fixture
def beam1_sequence(tmp_path: Path) -> Path:
    return _copy_lhc_sequence(tmp_path, beam=1)


def test_model_creator_madng_interface_loads_real_sequence_and_reports_tunes(
    beam1_sequence: Path,
) -> None:
    accelerator = LHC(beam=1, sequence_file=beam1_sequence)

    with ModelCreatorMadngInterface(accelerator, stdout="/dev/null", redirect_stderr=True) as interface:
        q1, q2 = interface.get_current_tunes()

        assert interface.accelerator is accelerator
        assert interface.beam == 1
        assert interface.accelerator.seq_name == "lhcb1"
        assert q1 == pytest.approx(62.28, abs=1e-3)
        assert q2 == pytest.approx(60.31, abs=1e-3)


def test_model_creator_madng_interface_matches_real_tunes(
    beam1_sequence: Path,
) -> None:
    accelerator = LHC(beam=1, sequence_file=beam1_sequence)

    with ModelCreatorMadngInterface(accelerator, stdout="/dev/null", redirect_stderr=True) as interface:
        interface.match_model_tunes(NAT_TUNES)
        q1, q2 = interface.get_current_tunes("Matched")
        interface.initialise_model(NAT_TUNES)
        q1_after_init, q2_after_init = interface.get_current_tunes("Initialised")

    assert q1 % 1 == pytest.approx(NAT_TUNES[0], abs=1e-5)
    assert q2 % 1 == pytest.approx(NAT_TUNES[1], abs=1e-5)
    assert q1_after_init % 1 == pytest.approx(NAT_TUNES[0], abs=1e-5)
    assert q2_after_init % 1 == pytest.approx(NAT_TUNES[1], abs=1e-5)


def test_model_creator_madng_interface_exports_real_raw_twiss_tables(
    beam1_sequence: Path,
    tmp_path: Path,
) -> None:
    accelerator = LHC(beam=1, sequence_file=beam1_sequence)
    output_dir = tmp_path / "raw_export"
    output_dir.mkdir()

    with ModelCreatorMadngInterface(accelerator, stdout="/dev/null", redirect_stderr=True) as interface:
        interface.initialise_model(NAT_TUNES)
        interface.compute_and_export_twiss_tables(
            output_dir,
            tunes=NAT_TUNES,
            drv_tunes=DRV_TUNES,
        )

    twiss = tfs.read(output_dir / "twiss.dat")
    twiss_ac = tfs.read(output_dir / "twiss_ac.dat")
    twiss_elements = tfs.read(output_dir / "twiss_elements.dat")

    assert (output_dir / "twiss.dat").exists()
    assert (output_dir / "twiss_ac.dat").exists()
    assert (output_dir / "twiss_elements.dat").exists()
    assert "mu1" in twiss.columns
    assert "MUX" not in twiss.columns
    assert _name_column(twiss) == "name"
    assert twiss["name"].astype(str).str.contains("BPM").all()
    assert len(twiss) > 500
    assert len(twiss_ac) == len(twiss)
    assert "mu1" in twiss_elements.columns
    assert twiss_elements["name"].astype(str).isin(["$start", "$end"]).sum() == 2
    assert len(twiss_elements) > len(twiss)


def test_model_creator_madng_interface_update_model_writes_converted_twiss_tables(
    beam1_sequence: Path,
    tmp_path: Path,
) -> None:
    accelerator = LHC(beam=1, sequence_file=beam1_sequence)
    output_dir = tmp_path / "converted_export"
    output_dir.mkdir()

    with ModelCreatorMadngInterface(accelerator, stdout="/dev/null", redirect_stderr=True) as interface:
        interface.update_model(output_dir, tunes=NAT_TUNES, drv_tunes=DRV_TUNES)

    twiss = tfs.read(output_dir / "twiss.dat")
    twiss_ac = tfs.read(output_dir / "twiss_ac.dat")
    twiss_elements = tfs.read(output_dir / "twiss_elements.dat")

    assert "NAME" in twiss.columns
    assert "MUX" in twiss.columns
    assert "mu1" not in twiss.columns
    assert twiss.headers["Q1"] == pytest.approx(62.28, abs=1e-3)
    assert twiss.headers["Q2"] == pytest.approx(60.31, abs=1e-3)
    assert twiss["NAME"].astype(str).str.contains("BPM").all()
    assert len(twiss_ac) == len(twiss)
    assert "MUX" in twiss_elements.columns
    assert twiss_elements["NAME"].astype(str).isin(["$start", "$end"]).sum() == 0
    assert len(twiss_elements) > len(twiss)
    assert twiss_ac.headers["Q1"] == pytest.approx(62.27, abs=1e-3)
    assert twiss_ac.headers["Q2"] == pytest.approx(60.322, abs=1e-3)
