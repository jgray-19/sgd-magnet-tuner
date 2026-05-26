"""Kicker-focused integration tests for controller logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import (
    KickerConfig,
    MeasurementConfig,
    OutputConfig,
    SequenceConfig,
)
from tests.training.controller_test_utils import (
    _generate_kicker_track,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

logger = logging.getLogger(__name__)


def _build_kicker_controller(
    *,
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
) -> tuple[Controller, dict[str, float]]:
    magnet_range = "$start/$end"
    bpm_start_points: list[str] = []
    bpm_end_points: list[str] = []

    flattop_turns = 10
    off_magnet_path = tmp_path / "track_kicker_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file, kicker_name = _generate_kicker_track(
        loaded_psb_interface,
        flattop_turns,
        off_magnet_path,
        dpp_value=0.0,
        bpm_pattern=r"(?i)br3\.bpm.*",
    )
    optimiser_config = OptimiserConfig(
        max_epochs=300,
        warmup_epochs=50,
        warmup_lr_start=1e-4,
        max_lr=1e-2,
        min_lr=1e-2,
        gradient_converged_value=5e-20,
    )

    ctrl = Controller(
        PSB(
            ring=3,
            kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
            sequence_file=seq_psb,
            optimise_quadrupoles=True,
        ),
        optimiser_config,
        _make_simulation_config_quad(),
        SequenceConfig(
            magnet_range=magnet_range,
            first_bpm=kicker_name,
        ),
        MeasurementConfig(
            measurement_files=off_magnet_path,
            corrector_files=corrector_file,
            tune_knobs_files=tune_knobs_file,
            flattop_turns=flattop_turns,
            bunches_per_file=1,
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "mad_logfile_kicker.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
        kicker_config=KickerConfig(
            kicker_name=kicker_name,
            turns_after_kicker=flattop_turns,
        ),
    )
    return ctrl, magnet_strengths.copy()

@pytest.mark.slow
def test_controller_quad_opt_with_kicker(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
    controller_test_mode: str,
) -> None:
    ctrl, magnet_strengths = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
    )
    logger.info(
        "Starting kicker controller with logfile at %s",
        tmp_path / "mad_logfile_kicker.log",
    )

    initial_loss = evaluate_controller_worker_loss(ctrl, ctrl.initial_knobs)

    if controller_test_mode == "loss_regression":
        true_loss = evaluate_controller_worker_loss(ctrl, magnet_strengths)
        assert true_loss < 1e-18, (
            f"True-strength kicker loss should be numerically tiny, got {true_loss:.3e}"
        )
        assert true_loss < initial_loss * 1e-3, (
            "Kicker loss should strongly prefer the true quadrupole strengths "
            f"(initial={initial_loss:.3e}, true={true_loss:.3e})"
        )
        return

    initial_sum_true_diff = sum(
        abs(ctrl.initial_knobs[magnet] - magnet_strengths[magnet])
        for magnet in magnet_strengths
    )
    estimate, _unc = ctrl.run()
    final_sum_true_diff = sum(
        abs(estimate[magnet] - magnet_strengths[magnet])
        for magnet in magnet_strengths
    )
    final_ctrl, _ = _build_kicker_controller(
        tmp_path=tmp_path,
        seq_psb=seq_psb,
        loaded_psb_interface=loaded_psb_interface,
    )
    final_loss = evaluate_controller_worker_loss(final_ctrl, estimate)

    assert final_loss < initial_loss * 0.05, (
        "Kicker optimisation should reduce the worker loss substantially "
        f"(initial={initial_loss:.3e}, final={final_loss:.3e})"
    )
    assert final_sum_true_diff < initial_sum_true_diff, (
        "Kicker optimisation should move the estimate closer to the true strengths "
        f"(initial diff={initial_sum_true_diff:.3e}, final diff={final_sum_true_diff:.3e})"
    )
