"""PSB energy-focused integration tests for controller logic."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig
from tests.training.controller_test_utils import (
    _make_simulation_config_energy,
    _run_energy_optimisation_case,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


PSB_TARGET_QX = 0.21
PSB_TARGET_QY = 0.24
PSB_DPP_VALUE = 2.0e-4
PSB_TRACK_BPM_PATTERN = r"br1\.bpm.*"
PSB_BPM_START_POINTS = [
    "BR1.BPM1L3",
    "BR1.BPM5L3",
    "BR1.BPM9L3",
    "BR1.BPM13L3",
]


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="PSB energy optimisation still under investigation")
def test_controller_energy_opt_psb(
    tmp_path: Path,
    loaded_psb_interface: AbaMadInterface,
) -> None:
    """Run a PSB ring-1 energy optimisation scenario."""
    base_config = _make_simulation_config_energy()
    simulation_config = dataclasses.replace(
        base_config,
        run_arc_by_arc=False,
        n_run_turns=1,
        bpm_loss_outlier_sigma=100,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=400,
        warmup_epochs=20,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-6,
        gradient_converged_value=5e-11,
        expected_rel_error=0,
    )

    estimate, unc = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_psb_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=PSB_BPM_START_POINTS,
        bpm_end_points=[],
        magnet_range="$start/$end",
        mad_log_name="controller_energy_opt_psb.log",
        bpm_pattern=PSB_TRACK_BPM_PATTERN,
        apply_orbit_correction=False,
        target_qx=PSB_TARGET_QX,
        target_qy=PSB_TARGET_QY,
        dpp_value=PSB_DPP_VALUE,
    )

    assert np.allclose(estimate.pop("deltap"), PSB_DPP_VALUE, rtol=5e-2, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert 0 < uncertainty < 5e-4
    assert not estimate
    assert not unc
