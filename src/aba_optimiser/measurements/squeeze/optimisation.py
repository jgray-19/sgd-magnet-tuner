"""Optimisation helpers for LHC squeeze quadrupole tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.measurements.ac_dipole import ACDipoleOptimisationWindow

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.measurements.orbit_averaging import compute_three_turn_averages
from aba_optimiser.measurements.squeeze.io import save_arc_estimates
from aba_optimiser.training.config.models import (
    CheckpointConfig,
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.controller import Controller

logger = logging.getLogger(__name__)


def _first_bpm_for_beam(beam: int) -> str:
    """Return the BPM each measurement turn is recorded from for this beam."""
    return "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"


def _measurement_details(
    measurement: dict, b2_errors: Path | None, first_bpm: str | None
) -> MeasurementDetails:
    """Build per-measurement MAD interface options from a squeeze descriptor."""
    return MeasurementDetails(
        interface_options={
            "corrector_strengths": measurement["corrector_file"],
            "tune_knobs_file": measurement["tune_knobs_file"],
            "b2_errors": b2_errors,
        },
        first_bpm=first_bpm,
    )


def _create_averaged_measurement_config(
    measurements: list[dict],
    temp_analysis_dir: Path,
    arc_num: int,
    beam: int,
    b2_errors: Path | None = None,
) -> MeasurementConfig:
    """Load measurement parquets, average over all turns, and return a MeasurementConfig."""
    avg_files = []
    for i, m in enumerate(measurements):
        df = pd.read_parquet(m["file"])
        avg_df = compute_three_turn_averages(df)
        avg_path = temp_analysis_dir / f"avg_orbit_arc{arc_num}_file{i}.parquet"
        avg_df.to_parquet(avg_path)
        avg_files.append(avg_path)
    first_bpm = _first_bpm_for_beam(beam)
    return MeasurementConfig(
        {
            avg_path: _measurement_details(m, b2_errors, first_bpm)
            for avg_path, m in zip(avg_files, measurements)
        }
    )


def get_ac_dipole_bpm_points(
    beam: int, window: ACDipoleOptimisationWindow
) -> tuple[str, list[str], list[str]]:
    """Return (magnet_range, bpm_start_points, bpm_end_points) anchored around the AC dipole."""
    suffix = f".B{beam}"
    if not window.bpm_upstream.endswith(suffix):
        raise ValueError(f"Upstream BPM {window.bpm_upstream} does not match beam suffix {suffix}")
    if not window.bpm_downstream.endswith(suffix):
        raise ValueError(
            f"Downstream BPM {window.bpm_downstream} does not match beam suffix {suffix}"
        )
    return (
        f"{window.bpm_downstream}/{window.bpm_upstream}",
        [window.bpm_downstream],
        [window.bpm_upstream],
    )


def create_configs(
    beam: int,
    all_bad_bpms: set[str],
    measurements: list[dict],
    window: ACDipoleOptimisationWindow,
    b2_errors: Path | None = None,
) -> tuple[SequenceConfig, list[str], list[str], MeasurementConfig]:
    """Build SequenceConfig and MeasurementConfig from resolved measurement descriptors."""
    magnet_range, bpm_start_points, bpm_end_points = get_ac_dipole_bpm_points(beam, window)
    sequence_config = SequenceConfig(
        magnet_range=magnet_range,
        bad_bpms=list(all_bad_bpms),
    )
    first_bpm = _first_bpm_for_beam(beam)
    measurement_config = MeasurementConfig(
        {m["file"]: _measurement_details(m, b2_errors, first_bpm) for m in measurements}
    )
    return sequence_config, bpm_start_points, bpm_end_points, measurement_config


def get_default_simulation_config(
    tracks_per_worker: int = 300,
    num_batches: int = 20,
) -> SimulationConfig:
    """Return default simulation config for optimisation stages."""
    return SimulationConfig(
        tracks_per_worker=tracks_per_worker,
        num_batches=num_batches,
        num_workers=60,
        use_fixed_bpm=True,
        run_arc_by_arc=True,
        n_run_turns=1,
        optimise_momenta=False,
        bpm_loss_outlier_sigma=20,
        enable_preloop_outlier_screening = False,
    )


def resolve_restore_resume(
    arc_numbers: list[int],
    checkpoint_dir: Path,
    beam: int,
    squeeze_step: str,
    restore_bends_opt: bool,
    restore_quads_opt: bool,
) -> tuple[list[int], bool, bool, int | None]:
    """Find the most recent stage checkpoint and trim the arc list to resume from it."""
    if not (restore_bends_opt or restore_quads_opt):
        return arc_numbers, restore_bends_opt, restore_quads_opt, None

    restore_stage = "bends" if restore_bends_opt else "quads"
    squeeze_step_id = squeeze_step.replace(".", "_")
    prefix = f"checkpoint_b{beam}_{squeeze_step_id}_arc"
    suffix = f"_{restore_stage}.json"
    candidates = [
        p
        for p in checkpoint_dir.glob("*.json")
        if p.name.startswith(prefix) and p.name.endswith(suffix)
    ]
    if not candidates:
        logger.warning(
            "No %s checkpoint files found in %s for beam %d, squeeze step %s. Running without restore.",
            restore_stage,
            checkpoint_dir,
            beam,
            squeeze_step,
        )
        return arc_numbers, False, False, None

    checkpoint_file = max(candidates, key=lambda p: p.stat().st_mtime)
    restore_arc = int(checkpoint_file.stem.rsplit("_", 2)[1].replace("arc", ""))
    logger.info(
        "Resuming from most recent %s checkpoint: arc %d (%s)",
        restore_stage,
        restore_arc,
        checkpoint_file,
    )

    if restore_arc not in set(arc_numbers):
        logger.warning(
            "Most recent restore arc %d is incompatible with current run mode (allowed: %s). Running without restore.",
            restore_arc,
            sorted(arc_numbers),
        )
        return arc_numbers, False, False, None

    resumed_arcs = [arc for arc in arc_numbers if arc >= restore_arc]
    logger.info("Continuing arc loop from restored arc %d: %s", restore_arc, resumed_arcs)
    return resumed_arcs, restore_bends_opt, restore_quads_opt, restore_arc


def optimise_arc(
    arc_num: int,
    beam: int,
    sequence_path: Path,
    measurements: list[dict],
    temp_analysis_dir: Path,
    results_dir: Path,
    squeeze_step: str,
    all_bad_bpms: set[str],
    energy: float,
    checkpoint_dir: Path,
    checkpoint_every_n_epochs: int,
    rewrite_file: bool = False,
    window: ACDipoleOptimisationWindow | None = None,
    b2_errors: Path | None = None,
    restore_bends_opt: bool = False,
    restore_quads_opt: bool = False,
    hessian_parallelism: int = 1,
) -> None:
    """Run bend then quadrupole optimisation for one arc."""
    if window is None:
        raise ValueError("AC-dipole window is required for squeeze optimisation")

    logger.info("Optimising arc %d for %s", arc_num, squeeze_step)
    sequence_config, bpm_start_points, bpm_end_points, measurement_config = create_configs(
        beam, all_bad_bpms, measurements, window, b2_errors=b2_errors
    )
    output_cfg = OutputConfig(
        include_uncertainty=True,
        mad_logfile=temp_analysis_dir / "mad_log.txt",
        python_logfile=temp_analysis_dir / "python_worker_log.txt",
        parallel_hessian=hessian_parallelism,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    squeeze_step_id = squeeze_step.replace(".", "_")
    bend_checkpoint_cfg = CheckpointConfig(
        checkpoint_path=checkpoint_dir
        / f"checkpoint_b{beam}_{squeeze_step_id}_arc{arc_num}_bends.json",
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        restore_from_checkpoint=restore_bends_opt,
    )
    quad_checkpoint_cfg = CheckpointConfig(
        checkpoint_path=checkpoint_dir
        / f"checkpoint_b{beam}_{squeeze_step_id}_arc{arc_num}_quads.json",
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        restore_from_checkpoint=restore_quads_opt,
    )

    bend_estimates: dict[str, float] | None = None
    if restore_quads_opt:
        logger.info(
            "Skipping bend stage for arc %d because quadrupole checkpoint restore is enabled.",
            arc_num,
        )
    else:
        bend_estimates: dict[str, float] | None = None
        if not restore_bends_opt:
            logger.info("Pre-stage: averaged closed-orbit bend fit for arc %d", arc_num)
            avg_measurement_config = _create_averaged_measurement_config(
                measurements, temp_analysis_dir, arc_num, beam, b2_errors=b2_errors
            )
            avg_bend_ctrl = Controller(
                LHC(
                    beam=beam,
                    kinetic_energy=energy,
                    sequence_file=sequence_path,
                    optimise_bends=True,
                    optimise_quad_dy=True,
                ),
                OptimiserConfig(
                    max_epochs=1000,
                    warmup_epochs=30,
                    warmup_lr_start=5e-7,
                    max_lr=1e-7,
                    min_lr=1e-7,
                    gradient_converged_value=1e-3,
                    optimiser_type="adam",
                ),
                SimulationConfig(
                    tracks_per_worker=1,
                    num_batches=1,
                    num_workers=60,
                    use_fixed_bpm=True,
                    run_arc_by_arc=True,
                    optimise_momenta=False,
                    bpm_loss_outlier_sigma=5,
                    enable_preloop_outlier_screening=False,
                ),
                sequence_config,
                avg_measurement_config,
                bpm_start_points,
                bpm_end_points,
                initial_knob_strengths=None,
                true_strengths=None,
                output_config=OutputConfig(
                    include_uncertainty=False,
                    mad_logfile=output_cfg.mad_logfile,
                    python_logfile=output_cfg.python_logfile,
                    parallel_hessian=output_cfg.parallel_hessian,
                ),
                checkpoint_config=bend_checkpoint_cfg,
            )
            bend_estimates, _ = avg_bend_ctrl.run()

    quad_ctrl_without_b2 = Controller(
        LHC(
            beam=beam,
            kinetic_energy=energy,
            sequence_file=sequence_path,
            optimise_quadrupoles=True,
            optimise_bends=True,
            optimise_other_quadrupoles=True,
            optimise_quad_dx=True,
            optimise_quad_dy=True,
        ),
        OptimiserConfig(
            max_epochs=300,
            warmup_epochs=5,
            warmup_lr_start=1e-6,
            max_lr=2e-6 if "inj" in squeeze_step else 5e-7,
            min_lr=2e-6 if "inj" in squeeze_step else 5e-7,
            gradient_converged_value=1e-7,
            optimiser_type="adam",
        ),
        get_default_simulation_config(num_batches=30),
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        initial_knob_strengths=bend_estimates,
        true_strengths=None,
        output_config=output_cfg,
        checkpoint_config=quad_checkpoint_cfg,
        debug=False,
    )
    estimates, uncertainties = quad_ctrl_without_b2.run()

    quad_ctrl_with_b2 = Controller(
        LHC(
            beam=beam,
            kinetic_energy=energy,
            sequence_file=sequence_path,
            optimise_quadrupoles=True,
            optimise_bends=True,
            optimise_other_quadrupoles=True,
            optimise_quad_dx=True,
            optimise_quad_dy=True,
        ),
        OptimiserConfig(
            max_epochs=1000,
            warmup_epochs=100,
            warmup_lr_start=1e-14,
            max_lr=1e-8,
            min_lr=1e-6,
            gradient_converged_value=1e-7,
            optimiser_type="adam",
        ),
        get_default_simulation_config(num_batches=30),
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        # initial_knob_strengths=bend_estimates,
        initial_knob_strengths=estimates,
        true_strengths=None,
        output_config=output_cfg,
        checkpoint_config=quad_checkpoint_cfg,
        # checkpoint_config=None,  # Don't checkpoint the final run with b2 errors, to avoid accidentally restoring from it
        debug=False,
    )
    estimates, uncertainties = quad_ctrl_with_b2.run()

    save_arc_estimates(
        results_dir, squeeze_step, arc_num, estimates, uncertainties, rewrite_file=rewrite_file
    )
