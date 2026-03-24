"""MAD-X utility functions for model sequence creation."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from cpymad.madx import Madx
from omc3.model.accelerators.lhc import Lhc
from omc3.model.accelerators.sps import Sps
from omc3.model.constants import JOB_MODEL_MADX_NOMINAL
from omc3.model.model_creators.manager import CreatorType, get_model_creator_class

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omc3.model.model_creators.abstract_model_creator import ModelCreator

LOGGER = logging.getLogger(__name__)

_DEFINE_NOMINAL_BEAMS_RE = re.compile(r"exec,\s*define_nominal_beams\([^;]*\)\s*;", re.IGNORECASE)
_LHC_USE_SEQUENCE_RE = re.compile(r"use\s*,\s*sequence\s*=\s*lhcb([12])", re.IGNORECASE)
_LHC_SEQUENCE_TOKEN_RE = re.compile(r"\blhcb([12])\b", re.IGNORECASE)
_LHC_YEAR_RE = re.compile(r"^\s*!\s*LHC year\s+(?P<year>\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_POST_OPTICS_INSERT_MARKERS = (
    "\n! ----- Remove IR symmetry definitions -----\n",
    "\n! ----- Finalize Sequence -----\n",
)


def _adapt_script_to_beam4(base_script: str, beam: int, energy: float) -> str:
    """Adapt creator base script for beam4 tracking (beam 2 only)."""
    if beam != 2:
        raise ValueError("beam4 script adaptation is only valid for beam 2.")

    script = base_script.replace("lhc.seq", "lhcb4.seq")
    beam4_cmd = f"beam, sequence=LHCB2, particle=proton, energy={energy}, bv=1;"

    if not _DEFINE_NOMINAL_BEAMS_RE.search(script):
        raise ValueError("Could not find define_nominal_beams call to adapt for beam4.")

    return _DEFINE_NOMINAL_BEAMS_RE.sub(beam4_cmd, script, count=1)


def _detect_accelerator_from_model_dir(model_dir: Path) -> str:
    """Infer accelerator family from model folder contents."""
    if (model_dir / Lhc.LOCAL_REPO_NAME).exists():
        return "lhc"
    if (model_dir / Sps.LOCAL_REPO_NAME).exists():
        return "sps"

    job_file = model_dir / JOB_MODEL_MADX_NOMINAL
    if job_file.exists():
        job_text = job_file.read_text(errors="ignore").lower()
        if Lhc.LOCAL_REPO_NAME in job_text or "lhcb1" in job_text or "lhcb2" in job_text:
            return "lhc"
        if Sps.LOCAL_REPO_NAME in job_text or "sps.seq" in job_text or "sequence=sps" in job_text:
            return "sps"

    raise ValueError(
        f"Could not infer accelerator type from model directory: {model_dir}. "
        "Expected LHC or SPS model layout."
    )


def _detect_lhc_beam_from_model_dir(model_dir: Path) -> int:
    """Infer LHC beam number from model folder contents."""
    job_file = model_dir / JOB_MODEL_MADX_NOMINAL
    if job_file.exists():
        text = job_file.read_text(errors="ignore")
        use_matches = {int(match.group(1)) for match in _LHC_USE_SEQUENCE_RE.finditer(text)}
        if len(use_matches) == 1:
            beam = use_matches.pop()
            LOGGER.info(
                "Inferred LHC beam %d from use statement in MAD-X nominal job file: %s",
                beam,
                job_file,
            )
            return beam

        token_matches = {int(match.group(1)) for match in _LHC_SEQUENCE_TOKEN_RE.finditer(text)}
        if len(token_matches) == 1:
            beam = token_matches.pop()
            LOGGER.info(
                "Inferred LHC beam %d from sequence tokens in MAD-X nominal job file: %s",
                beam,
                job_file,
            )
            return beam

    name = model_dir.name.lower()
    has_b1 = "b1" in name
    has_b2 = "b2" in name
    if has_b1 and not has_b2:
        LOGGER.info("Inferred LHC beam 1 from model directory name: %s", model_dir)
        return 1
    if has_b2 and not has_b1:
        LOGGER.info("Inferred LHC beam 2 from model directory name: %s", model_dir)
        return 2

    raise ValueError(
        f"Could not infer LHC beam from model directory: {model_dir}. "
        "Provide a model directory with an unambiguous beam."
    )


def _detect_lhc_year_from_model_dir(model_dir: Path) -> str:
    """Infer LHC optics year from the nominal omc3 MAD-X header."""
    job_file = model_dir / JOB_MODEL_MADX_NOMINAL
    if job_file.exists():
        match = _LHC_YEAR_RE.search(job_file.read_text(errors="ignore"))
        if match is not None:
            return match.group("year")

    raise ValueError(
        f"Could not infer LHC year from model directory: {model_dir}. "
        "Expected the omc3 nominal header comment '! LHC year ...'."
    )


def _inject_post_optics_calls(madx_script: str, madx_files: Sequence[str]) -> str:
    """Insert additional MAD-X files after the optics modifiers section."""
    if not madx_files:
        return madx_script

    extra_calls = "\n! ----- Additional post-optics modifiers -----\n" + "".join(
        f"call, file = '{madx_file}';\n" for madx_file in madx_files
    )

    for marker in _POST_OPTICS_INSERT_MARKERS:
        marker_index = madx_script.find(marker)
        if marker_index != -1:
            return madx_script[:marker_index] + extra_calls + madx_script[marker_index:]

    return madx_script + extra_calls


def _load_nominal_creator_from_model_dir(model_dir: Path) -> tuple[ModelCreator, str, int | None]:
    """Load OMC3 nominal creator from an existing LHC or SPS model directory."""
    accelerator = _detect_accelerator_from_model_dir(model_dir)
    if accelerator == "lhc":
        beam = _detect_lhc_beam_from_model_dir(model_dir)
        year = _detect_lhc_year_from_model_dir(model_dir)
        accel = Lhc(model_dir=model_dir, beam=beam, year=year)
    else:
        beam = None
        accel = Sps(model_dir=model_dir)

    creator_class = get_model_creator_class(accel, CreatorType.NOMINAL)  # ty:ignore[invalid-argument-type]
    creator = creator_class(accel)
    # creator.prepare_run()
    return creator, accelerator, beam


def make_madx_sequence(
    model_dir: Path | str,
    *,
    seq_outdir: Path | None = None,
    beam4: bool = False,
    post_optics_madx_files: Sequence[Path | str] | None = None,
) -> Path:
    """Generate and save MAD-X sequence file from LHC or SPS model folder.

    - Detect accelerator family from ``model_dir`` (LHC or SPS)
    - Detect beam for LHC model directories
    - Load matching OMC3 nominal model creator
    - Build ``base_script + save_script``
    - Optionally adapt script for LHC beam4 (beam 2 only)
    - Execute using ``cpymad``
    """

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    creator, accelerator, beam = _load_nominal_creator_from_model_dir(model_dir)

    madx_script = creator.get_base_madx_script()
    if beam4:
        if accelerator != "lhc" or beam != 2:
            raise ValueError("beam4 sequence adaptation is only supported for LHC beam 2.")
        creator_energy = creator.accel.energy
        if creator_energy is None:
            raise ValueError("Could not determine beam energy from model creator accelerator.")

        madx_script = _adapt_script_to_beam4(madx_script, beam, float(creator_energy))

    if post_optics_madx_files:
        madx_script = _inject_post_optics_calls(
            madx_script,
            [
                f"{creator.resolve_path_for_madx(Path(madx_file).resolve())}"
                for madx_file in post_optics_madx_files
            ],
        )

    outdir = Path(seq_outdir) if seq_outdir is not None else model_dir
    outdir.mkdir(parents=True, exist_ok=True)
    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))
        madx.input(madx_script)

        # Always move to outdir, in case you don't have write permissions in the original model_dir
        madx.chdir(str(outdir))
        madx.input(creator.get_save_sequence_script())

    saved_seq = outdir / creator.save_sequence_filename
    if not saved_seq.exists():
        raise FileNotFoundError(
            f"Expected saved sequence file not produced by creator: {saved_seq}"
        )

    desired_seq_name = f"{creator.sequence_name.lower()}_saved.seq"
    seq_path = outdir / desired_seq_name

    if saved_seq.resolve() != seq_path.resolve():
        shutil.copy2(saved_seq, seq_path)
        saved_seq.unlink(missing_ok=True)

    LOGGER.info(
        "Saved MAD-X sequence for accelerator %s%s to %s",
        accelerator.upper(),
        f" beam {beam}" if accelerator == "lhc" else "",
        seq_path,
    )
    return seq_path
