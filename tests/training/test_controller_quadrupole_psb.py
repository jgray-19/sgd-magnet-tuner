"""PSB quadrupole-focused integration tests for controller logic."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_simulation_config_quad,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


PSB_TARGET_QX = 0.21
PSB_TARGET_QY = 0.24
PSB_TRACK_BPM_PATTERN = r"br1\.bpm.*"
PSB_BPM_START_POINTS = [
    "BR1.BPM1L3",
    "BR1.BPM5L3",
    "BR1.BPM9L3",
    "BR1.BPM13L3",
]


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="PSB quadrupole optimisation still under investigation")
def test_controller_quad_opt_psb_ring1(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
) -> None:
    """Run a PSB ring-1 quadrupole optimisation scenario."""
    flattop_turns = 256
    off_magnet_path = tmp_path / "track_off_magnet_psb.parquet"
    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        perturb_quads=True,
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
    )

    base_sim = _make_simulation_config_quad()
    simulation_config = dataclasses.replace(
        base_sim,
        tracks_per_worker=1,
        num_workers=4,
        num_batches=4,
        run_arc_by_arc=False,
        n_run_turns=1,
        bpm_loss_outlier_sigma=20,
        worker_loss_outlier_sigma=20,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=600,
        warmup_epochs=40,
        warmup_lr_start=1e-8,
        max_lr=3e-6,
        min_lr=3e-6,
        gradient_converged_value=5e-15,
        expected_rel_error=loaded_psb_interface.accelerator.get_perturbation_families()["q"]["default_rel_std"],  # ty:ignore[invalid-assignment, index]
        optimiser_type="adam",
    )

    sequence_config = SequenceConfig("$start/$end")
    measurement_config = MeasurementConfig(
        measurement_files=off_magnet_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )
    accelerator = PSB(
        ring=1,
        beam_energy=loaded_psb_interface.accelerator.beam_energy,
        sequence_file=seq_psb,
        optimise_quadrupoles=True,
    )

    ctrl = Controller(
        accelerator,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points=PSB_BPM_START_POINTS,
        bpm_end_points=[],
        show_plots=False,
        true_strengths=magnet_strengths,
        debug=False,
        mad_logfile=tmp_path / "controller_quad_opt_psb.log",
        write_tensorboard_logs=False,
        plots_dir=tmp_path / "plots",
    )
    estimate, unc = ctrl.run()

    assert set(estimate) == set(magnet_strengths)
    assert set(unc) == set(magnet_strengths)
    for magnet, true_value in magnet_strengths.items():
        est_value = estimate[magnet]
        rel_diff = (
            abs(est_value - true_value) / abs(true_value) if true_value != 0 else abs(est_value)
        )
        assert rel_diff < 5e-3, f"Relative difference for {magnet} is too high: {rel_diff:.2%}"
