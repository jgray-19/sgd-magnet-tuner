"""External optics-analysis helpers for measurement processing."""

from __future__ import annotations

import logging
from pathlib import Path

from omc3.hole_in_one import hole_in_one_entrypoint

LOGGER = logging.getLogger(__name__)


def run_measurement_analysis(
    analysis_dir: str | Path,
    model_dir: str | Path,
    files: list[Path],
    *,
    beam: int,
    nattunes: list[float],
    tunes: list[float],
) -> list[str]:
    """Run external optics analysis and return the discovered bad BPMs."""
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    hole_in_one_entrypoint(
        harpy=True,
        files=files,
        outputdir=analysis_dir / "lin_files",
        unit="mm",
        driven_excitation="acd",
        first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        is_free_kick=False,
        keep_exact_zeros=False,
        max_peak=0.02,
        nattunes=nattunes,
        num_svd_iterations=3,
        opposite_direction=beam == 2,
        output_bits=10,
        peak_to_peak=1e-08,
        resonances=4,
        sing_val=12,
        svd_dominance_limit=0.925,
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],
        tune_clean_limit=1e-05,
        tunes=tunes,
        turn_bits=14,
        model_dir=model_dir,
        turns=[0, 50000],
        clean=True,
    )

    analysed_files = [
        created_file.with_suffix("")
        for created_file in (analysis_dir / "lin_files").glob("*_bunchID*.linx")
    ]

    hole_in_one_entrypoint(
        optics=True,
        files=analysed_files,
        outputdir=analysis_dir,
        analyse_dpp=0,
        chromatic_beating=False,
        compensation="equation",
        coupling_method=2,
        coupling_pairing=0,
        isolation_forest=False,
        nonlinear=[],
        only_coupling=False,
        range_of_bpms=11,
        second_order_dispersion=False,
        three_bpm_method=False,
        three_d_excitation=False,
        union=False,
        accel="lhc",
        ats=False,
        beam=beam,
        dpp=0.0,
        model_dir=model_dir,
        xing=False,
        year="2025",
    )

    bad_bpms = _collect_bad_bpms(analysed_files)
    LOGGER.info("Identified %d bad BPMs from analysis.", len(bad_bpms))
    return bad_bpms


def _collect_bad_bpms(analysed_files: list[Path]) -> list[str]:
    bad_bpms: set[str] = set()
    for file in analysed_files:
        for suffix in (".bad_bpms_x", ".bad_bpms_y"):
            summary_file = file.parent / f"{file.name}{suffix}"
            if summary_file.exists():
                with summary_file.open("r") as handle:
                    bad_bpms.update(line.split(" ")[0] for line in handle.readlines())
    return sorted(bad_bpms)
