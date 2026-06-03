"""Orchestration pipeline for LHC squeeze quadrupole optimisation."""

from __future__ import annotations

import argparse
import logging
import shutil
from typing import TYPE_CHECKING

from pymadng_utils.accelerators.lhc import LHC as MadngLHCAccelerator  # noqa: N811
from pymadng_utils.model_creator.madng_utils import update_model_with_madng

if TYPE_CHECKING:
    from pathlib import Path

from aba_optimiser.config import MEASUREMENTS_ARTIFACTS_ROOT
from aba_optimiser.measurements.ac_dipole import ACDipoleOptimisationWindow
from aba_optimiser.measurements.b2_errors import resolve_b2_error_table
from aba_optimiser.measurements.squeeze.constants import MEAS_TIMES, ZEROHZ, get_beam_paths
from aba_optimiser.measurements.squeeze.io import (
    get_knob_files,
    get_sequence_creation_time,
    load_bad_bpms,
    load_frequency_results,
    load_metadata,
    prepare_frequency_metadata,
    process_frequency_results,
    save_bad_bpms,
    update_metadata,
)
from aba_optimiser.measurements.squeeze.optimisation import (
    optimise_arc,
    resolve_restore_resume,
)
from aba_optimiser.measurements.squeeze_config import MODEL_DIRS
from aba_optimiser.measurements.squeeze_helpers import (
    extract_tunes_from_job_file,
    get_or_make_sequence,
    get_results_dir,
    reconstruct_ac_dipole_measurements,
)

logger = logging.getLogger(__name__)


def process_measurements_fresh(
    beam: int,
    squeeze_step: str,
    meas_times_for_step: dict[str, list[str]],
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    temp_analysis_dir: Path,
    sequence_path: Path,
    use_weighted_svd: bool = True,
) -> tuple[list[dict], set[str], float, ACDipoleOptimisationWindow]:
    """Process raw SDDS files: reconstruct, validate, persist parquets, return descriptors."""
    freq_metadata: dict[str, tuple[list[Path], Path, Path]] = {}
    all_files: list[Path] = []
    acd_tune_knobs_files: list[Path | None] = []
    all_bad_bpms: set[str] = set()
    energy = 0.0

    for freq, times in meas_times_for_step.items():
        if not times:
            raise ValueError(f"No measurement times found for frequency {freq}")
        logger.info("  Frequency %s: %d measurements", freq, len(times))
        files, tune_knobs_file, corrector_knobs_file, bad_bpms, freq_energy = prepare_frequency_metadata(
            freq, times, beam, meas_base_dir, results_dir, squeeze_step
        )
        if freq == ZEROHZ:
            energy = freq_energy
        freq_metadata[freq] = (files, tune_knobs_file, corrector_knobs_file)
        all_files.extend(files)
        acd_tune_knobs_files.extend([tune_knobs_file] * len(files))
        all_bad_bpms.update(bad_bpms)

    b2_errors = resolve_b2_error_table(beam, energy)
    update_metadata(temp_analysis_dir, energy=energy, b2_errors=str(b2_errors))

    logger.info("Processing %d measurement files with AC-dipole reconstruction...", len(all_files))
    pzs_dict = reconstruct_ac_dipole_measurements(
        measurement_files=all_files,
        model_dir=model_dir,
        sequence_path=sequence_path,
        beam=beam,
        energy=energy,
        use_weighted_svd=use_weighted_svd,
        tune_knobs_files=acd_tune_knobs_files or None,
        num_workers=8,
    )

    missing = sorted(
        stem for stem, pzs in pzs_dict.items()
        if not all(pzs.attrs.get(k) for k in ("ac_dipole_marker", "ac_dipole_bpm_upstream", "ac_dipole_bpm_downstream"))
    )
    if missing:
        raise ValueError(f"AC-dipole window metadata missing in {len(missing)} reconstruction(s): {missing}")

    unique_markers = {pzs.attrs["ac_dipole_marker"] for pzs in pzs_dict.values()}
    unique_upstreams = {pzs.attrs["ac_dipole_bpm_upstream"] for pzs in pzs_dict.values()}
    unique_downstreams = {pzs.attrs["ac_dipole_bpm_downstream"] for pzs in pzs_dict.values()}
    if len(unique_markers) > 1 or len(unique_upstreams) > 1 or len(unique_downstreams) > 1:
        raise ValueError(
            f"Inconsistent AC-dipole window across reconstructions: "
            f"markers={unique_markers}, upstreams={unique_upstreams}, downstreams={unique_downstreams}"
        )

    ac_dipole_marker = next(iter(unique_markers))
    bpm_upstream = next(iter(unique_upstreams))
    bpm_downstream = next(iter(unique_downstreams))
    window = ACDipoleOptimisationWindow(bpm_upstream=bpm_upstream, bpm_downstream=bpm_downstream)
    logger.info("AC-dipole: marker=%s, upstream=%s, downstream=%s", ac_dipole_marker, bpm_upstream, bpm_downstream)

    update_metadata(
        temp_analysis_dir,
        ac_dipole_marker=ac_dipole_marker,
        ac_dipole_bpm_upstream=bpm_upstream,
        ac_dipole_bpm_downstream=bpm_downstream,
    )

    all_measurements: list[dict] = []
    for freq, (files_freq, tune_knobs_freq, corrector_knobs_freq) in freq_metadata.items():
        all_measurements.extend(
            process_frequency_results(
                freq,
                [str(f) for f in files_freq],
                pzs_dict,
                tune_knobs_freq,
                corrector_knobs_freq,
                temp_analysis_dir,
            )
        )

    return all_measurements, all_bad_bpms, energy, window


def load_measurements_from_reload(
    temp_analysis_dir: Path,
    results_dir: Path,
    squeeze_step: str,
    meas_times_for_step: dict[str, list[str]],
) -> tuple[list[dict], float, ACDipoleOptimisationWindow]:
    """Reload previously persisted parquet files and recover energy and AC-dipole window."""
    metadata = load_metadata(temp_analysis_dir)
    energy = metadata["energy"]

    upstream = metadata.get("ac_dipole_bpm_upstream", "")
    downstream = metadata.get("ac_dipole_bpm_downstream", "")
    if not upstream or not downstream:
        raise ValueError(
            "AC-dipole window missing in metadata for --skip-reload. "
            "Run once without --skip-reload to generate it."
        )
    window = ACDipoleOptimisationWindow(bpm_upstream=upstream, bpm_downstream=downstream)

    all_measurements: list[dict] = []
    for freq, times in meas_times_for_step.items():
        if not times:
            continue
        logger.info("  Frequency %s: %d measurements (loading)", freq, len(times))
        tune_knobs_file, corrector_knobs_file = get_knob_files(results_dir, squeeze_step, freq)
        all_measurements.extend(
            load_frequency_results(freq, len(times), tune_knobs_file, corrector_knobs_file, temp_analysis_dir)
        )

    return all_measurements, energy, window


def process_squeeze_step(
    beam: int,
    squeeze_step: str,
    meas_times: dict,
    meas_base_dir: Path,
    model_dir: Path,
    results_dir: Path,
    checkpoint_every_n_epochs: int,
    skip_reload: bool = False,
    cleanup_temp: bool = False,
    restore_bends_opt: bool = False,
    restore_quads_opt: bool = False,
    hessian_parallelism: int = 1,
    use_weighted_svd: bool = True,
) -> None:
    """Orchestrate measurement loading, reconstruction, and optimisation for one squeeze step."""
    logger.info("Processing squeeze step %s for beam %d", squeeze_step, beam)

    if ZEROHZ not in meas_times[squeeze_step]:
        raise NotImplementedError("Please include 0Hz measurements to build the closed-orbit reference.")

    results_dir.mkdir(exist_ok=True)
    temp_analysis_dir = (
        MEASUREMENTS_ARTIFACTS_ROOT
        / "temp"
        / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
    )
    bad_bpms_file = results_dir / f"bad_bpms_{squeeze_step}.txt"
    meas_times_for_step = meas_times[squeeze_step]

    if skip_reload:
        if not temp_analysis_dir.exists():
            raise FileNotFoundError(
                f"Temp analysis directory {temp_analysis_dir} not found. "
                "Run without --skip-reload first to generate processed data."
            )
        logger.info("Using existing temp directory: %s", temp_analysis_dir)
    else:
        temp_analysis_dir.mkdir(exist_ok=True)

    madng_model_dir = temp_analysis_dir / "madng_model"

    if skip_reload and madng_model_dir.exists():
        logger.info("--skip-reload: reusing existing MAD-NG model at %s", madng_model_dir)
        sequence_path = get_or_make_sequence(beam, madng_model_dir)
    else:
        sequence_time = get_sequence_creation_time(meas_times_for_step, squeeze_step)
        job_file = model_dir / "job.create_model_nominal.madx"
        nat_x, nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
        logger.info("Tunes for %s: nat=(%s,%s), drv=(%s,%s)", squeeze_step, nat_x, nat_y, drv_x, drv_y)
        if madng_model_dir.exists():
            shutil.rmtree(madng_model_dir)
        shutil.copytree(model_dir, madng_model_dir, symlinks=True)
        sequence_path = get_or_make_sequence(beam, madng_model_dir, time=sequence_time)
        madng_accel = MadngLHCAccelerator(beam=beam, sequence_file=sequence_path)
        update_model_with_madng(madng_accel, madng_model_dir, tunes=[nat_x, nat_y], convert_to_madx=False)

    if skip_reload:
        all_measurements, energy, window = load_measurements_from_reload(
            temp_analysis_dir, results_dir, squeeze_step, meas_times_for_step
        )
        all_bad_bpms = load_bad_bpms(bad_bpms_file)
    else:
        all_measurements, all_bad_bpms, energy, window = process_measurements_fresh(
            beam=beam,
            squeeze_step=squeeze_step,
            meas_times_for_step=meas_times_for_step,
            meas_base_dir=meas_base_dir,
            model_dir=madng_model_dir,
            results_dir=results_dir,
            temp_analysis_dir=temp_analysis_dir,
            sequence_path=sequence_path,
            use_weighted_svd=use_weighted_svd,
        )
        save_bad_bpms(bad_bpms_file, all_bad_bpms)

    b2_errors = resolve_b2_error_table(beam, energy)
    update_metadata(temp_analysis_dir, b2_errors=str(b2_errors))

    logger.info("Total bad BPMs: %d", len(all_bad_bpms))
    logger.info("Total measurements: %d", len(all_measurements))
    logger.info("Machine deltaps: %s", [m["machine_deltap"] for m in all_measurements])

    checkpoint_dir = temp_analysis_dir / "checkpoints"
    arc_numbers, restore_bends_opt, restore_quads_opt, restore_arc = resolve_restore_resume(
        arc_numbers=[1],
        checkpoint_dir=checkpoint_dir,
        beam=beam,
        squeeze_step=squeeze_step,
        restore_bends_opt=restore_bends_opt,
        restore_quads_opt=restore_quads_opt,
    )

    for i, arc_num in enumerate(arc_numbers):
        optimise_arc(
            arc_num=arc_num,
            beam=beam,
            sequence_path=sequence_path,
            measurements=all_measurements,
            temp_analysis_dir=temp_analysis_dir,
            results_dir=results_dir,
            squeeze_step=squeeze_step,
            all_bad_bpms=all_bad_bpms,
            energy=energy,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            rewrite_file=(i == 0),
            window=window,
            b2_errors=b2_errors,
            restore_bends_opt=restore_bends_opt and restore_arc == arc_num,
            restore_quads_opt=restore_quads_opt and restore_arc == arc_num,
            hessian_parallelism=hessian_parallelism,
        )

    if cleanup_temp:
        logger.info("Cleaning up temp directory: %s", temp_analysis_dir)
        shutil.rmtree(temp_analysis_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Optimise LHC squeeze quadrupoles using measurement data")
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True)
    parser.add_argument("--squeeze-step", type=str, required=True, help="e.g. '1.2m', '0.6m'")
    parser.add_argument("--skip-reload", action="store_true", help="Reuse existing processed data")
    parser.add_argument("--cleanup-temp", action="store_true", help="Delete temp directory after completion")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=3)
    parser.add_argument("--restore-bends-opt", action="store_true")
    parser.add_argument("--restore-quads-opt", action="store_true")
    parser.add_argument("--hessian-parallelism", type=int, default=3)
    parser.add_argument("--use-weighted-svd", action="store_true", default=True)
    parser.add_argument("--no-weighted-svd", action="store_false", dest="use_weighted_svd")
    args = parser.parse_args()

    if args.squeeze_step not in MEAS_TIMES[args.beam]:
        raise ValueError(f"Unknown squeeze step {args.squeeze_step!r} for beam {args.beam}")
    if args.checkpoint_every_epochs < 0:
        raise ValueError("--checkpoint-every-epochs must be >= 0")
    if args.hessian_parallelism < 1:
        raise ValueError("--hessian-parallelism must be >= 1")
    if args.restore_bends_opt and args.restore_quads_opt:
        raise ValueError("Choose only one restore option: --restore-bends-opt or --restore-quads-opt")

    meas_base_dir, model_base_dir = get_beam_paths(args.beam, args.squeeze_step)
    model_dir = model_base_dir / MODEL_DIRS[args.beam][args.squeeze_step]

    process_squeeze_step(
        beam=args.beam,
        squeeze_step=args.squeeze_step,
        meas_times=MEAS_TIMES[args.beam],
        meas_base_dir=meas_base_dir,
        model_dir=model_dir,
        results_dir=get_results_dir(args.beam),
        checkpoint_every_n_epochs=args.checkpoint_every_epochs,
        skip_reload=args.skip_reload,
        cleanup_temp=args.cleanup_temp,
        restore_bends_opt=args.restore_bends_opt,
        restore_quads_opt=args.restore_quads_opt,
        hessian_parallelism=args.hessian_parallelism,
        use_weighted_svd=args.use_weighted_svd,
    )


if __name__ == "__main__":
    main()
