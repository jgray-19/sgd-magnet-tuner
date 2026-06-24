"""Controller-level tracking tests for PSB BPM displacement optimisation.

Because xtrack does not support BPM misalignment, the workflow mirrors
what would happen in a real machine measurement:
  1. Generate clean noiseless tracking data with xtrack (perfect BPMs).
  2. Manually shift the x (or y) readings for chosen BPMs in the parquet
     file – this simulates displaced BPMs producing an offset measurement.
  3. Run the Controller with optimise_bpm_dx=True.
  4. Verify that the true displacements give a substantially lower worker
     loss than the default all-zero displacements.

This validates the entire pipeline: knob creation → MAD misalign chain →
gradient computation → loss evaluation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.controller import Controller
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_simulation_config_quad,
    evaluate_controller_worker_loss,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

pytestmark = pytest.mark.serial

PSB_TARGET_QX = 0.17
PSB_TARGET_QY = 0.225
PSB_TRACK_BPM_PATTERN = r"br3\.bpm.*"
PSB_BPM_START_POINTS = [
    "BR3.BPM1L3",
    "BR3.BPM5L3",
    "BR3.BPM9L3",
    "BR3.BPM13L3",
]

# BPMs to displace and the applied offsets (metres).
_BPM_OFFSETS_X: dict[str, float] = {
    "BR3.BPM3L3": 1.5e-3,
    "BR3.BPM7L3": -1.0e-3,
}


def _apply_bpm_offsets_to_parquet(
    parquet_path: Path,
    bpm_offsets: dict[str, float],
    plane: str = "x",
) -> dict[str, float]:
    """Subtract ``bpm_offsets[bpm]`` from the ``plane`` column of each named BPM.

    Returns the offsets expressed as knob values (positive offset → positive
    displacement, because in MAD-NG a monitor with misalign.dx = d observes
    x_beam - d, so we need to shift the data by -d to match).

    The raw data x values for each BPM are shifted by *-offset*, and the
    matching knob value (misalignment) is *+offset*.
    """
    df = pd.read_parquet(parquet_path)
    for bpm_name, offset in bpm_offsets.items():
        mask = df["name"].str.upper() == bpm_name.upper()
        df.loc[mask, plane] -= offset
    df.to_parquet(parquet_path, index=False)
    # Knob values are the displacements (positive → BPM displaced in +x direction)
    knob_suffix = f"d{plane}"
    return {f"{bpm.upper()}.{knob_suffix}": offset for bpm, offset in bpm_offsets.items()}


@pytest.mark.slow
def test_controller_bpm_dx_loss_regression_psb_ring3(
    tmp_path: Path,
    seq_psb: Path,
    loaded_psb_interface: AbaMadInterface,
) -> None:
    """BPM dx optimisation: true displacements give much lower loss than zero.

    Procedure:
      1. Generate noiseless tracking data (no BPM displacement in xtrack).
      2. Manually subtract known offsets from BPM x readings.
      3. Build Controller with optimise_bpm_dx=True.
      4. Assert: loss(true displacements) << loss(zero displacements).
    """
    flattop_turns = 256
    track_path = tmp_path / "track_bpm_dx_psb.parquet"

    corrector_file, _magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_psb_interface,
        flattop_turns,
        track_path,
        0.0,
        perturb_quads=False,
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
    )

    # Inject known BPM displacements into the measurement data.
    true_bpm_knobs = _apply_bpm_offsets_to_parquet(track_path, _BPM_OFFSETS_X, plane="x")

    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        tracks_per_worker=1,
        num_workers=4,
        num_batches=4,
        run_arc_by_arc=False,
        n_run_turns=1,
        bpm_loss_outlier_sigma=20,
        worker_loss_outlier_sigma=20,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=1,
        warmup_epochs=0,
        warmup_lr_start=1e-6,
        max_lr=1e-4,
        min_lr=1e-4,
        gradient_converged_value=1e-15,
        optimiser_type="adam",
    )

    sequence_config = SequenceConfig("$start/$end")
    measurement_config = MeasurementConfig(
        measurement_files=track_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )
    accelerator = PSB(
        ring=3,
        kinetic_energy=loaded_psb_interface.accelerator.kinetic_energy,
        sequence_file=seq_psb,
        optimise_bpm_dx=True,
    )

    ctrl = Controller(
        accelerator,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points=PSB_BPM_START_POINTS,
        bpm_end_points=[],
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_bpm_dx_psb.log",
            write_tensorboard_logs=False,
            include_uncertainty=False,
        ),
        debug=False,
    )

    # Build the knob dict that the controller recognises (lowercase element names).
    controller_knobs_by_upper = {knob.upper(): knob for knob in ctrl.initial_knobs}
    true_knobs_for_ctrl = {
        controller_knobs_by_upper[knob.upper()]: val
        for knob, val in true_bpm_knobs.items()
        if knob.upper() in controller_knobs_by_upper
    }
    zero_knobs = dict.fromkeys(true_knobs_for_ctrl, 0.0)

    true_loss = evaluate_controller_worker_loss(
        ctrl,
        true_knobs_for_ctrl,
        enable_validation=False,
    )
    zero_loss = evaluate_controller_worker_loss(
        ctrl,
        zero_knobs,
        enable_validation=False,
    )

    assert true_loss < zero_loss, (
        f"Expected true BPM displacements to give lower loss than zero; "
        f"true_loss={true_loss:.4f}, zero_loss={zero_loss:.4f}"
    )
