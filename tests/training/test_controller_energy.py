"""
Energy-focused integration tests for controller logic.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig
from tests.training.controller_test_utils import (
    DPP_VALUE,
    _make_simulation_config_energy,
    _run_energy_optimisation_case,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


@pytest.mark.slow
@pytest.mark.parametrize("optimise_momenta", [False, True], ids=["position_only", "with_momenta"])
def test_controller_energy_opt(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    optimise_momenta: bool
) -> None:
    simulation_config = _make_simulation_config_energy(optimise_momenta)
    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=1,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-7,
        gradient_converged_value=5e-10,
        expected_rel_error=0,
    )

    estimate, unc = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=["BPM.9R2.B1", "BPM.10R2.B1", "BPM.11R2.B1"],
        bpm_end_points=["BPM.9L3.B1", "BPM.10L3.B1", "BPM.11L3.B1"],
        magnet_range="BPM.9R2.B1/BPM.9L3.B1",
        mad_log_name="controller_energy_opt.log",
    )

    assert np.allclose(estimate.pop("deltap"), DPP_VALUE, rtol=2e-3, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert 0 < uncertainty < 3e-6
    assert not estimate
    assert not unc


@pytest.mark.slow
def test_controller_energy_opt_sps(
    tmp_path: Path,
    seq_sps: Path,
    loaded_sps_interface: AbaMadInterface,
) -> None:
    sps_dpp_value = -1.5e-4
    simulation_config = _make_simulation_config_energy()
    optimiser_config = OptimiserConfig(
        max_epochs=600,
        warmup_epochs=1,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-6,
        gradient_converged_value=5e-10,
        expected_rel_error=0,
    )

    estimate, unc = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_sps_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=["BPH.13008", "BPH.13208", "BPH.13408"],
        bpm_end_points=["BPH.13608", "BPH.20208", "BPH.20408"],
        magnet_range="BPH.13008/BPH.20408",
        mad_log_name="controller_energy_opt_sps.log",
        bpm_pattern="bp[hv].*",
        apply_orbit_correction=True,
        dpp_value=sps_dpp_value,
        target_qx=0.13,
        target_qy=0.18,
    )

    assert np.allclose(estimate.pop("deltap"), sps_dpp_value, rtol=1e-2, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert 0 < uncertainty < 1e-4
    assert not estimate
    assert not unc


@pytest.mark.slow
@pytest.mark.parametrize("n_run_turns", [2, 3, 5], ids=["2_turns", "3_turns", "5_turns"])
def test_controller_energy_opt_multi_turn(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    n_run_turns: int,
) -> None:
    base_config = _make_simulation_config_energy()
    simulation_config = dataclasses.replace(
        base_config,
        run_arc_by_arc=False,
        n_run_turns=n_run_turns,
        bpm_loss_outlier_sigma=100,
    )

    optimiser_config = OptimiserConfig(
        max_epochs=200,
        warmup_epochs=30,
        warmup_lr_start=2e-5 * n_run_turns,
        max_lr=3e-6,
        min_lr=3e-6,
        gradient_converged_value=5e-10,
        expected_rel_error=0,
    )

    estimate, unc = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=[f"BPM.9R{ip}.B1" for ip in range(1, 8)],
        bpm_end_points=[],
        magnet_range="$start/$end",
        mad_log_name="controller_energy_opt_multi_turn.log",
        apply_orbit_correction=True,
    )

    assert np.allclose(estimate.pop("deltap"), DPP_VALUE, rtol=5e-3 * n_run_turns, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert 0 < uncertainty < 2e-6
    assert not estimate
    assert not unc


@pytest.mark.slow
@pytest.mark.parametrize("n_run_turns", [2, 3], ids=["2_turns_sps", "3_turns_sps"])
def test_controller_energy_opt_sps_multi_turn(
    tmp_path: Path,
    seq_sps: Path,
    loaded_sps_interface: AbaMadInterface,
    n_run_turns: int,
) -> None:
    sps_dpp_value = -3e-4
    base_config = _make_simulation_config_energy()
    simulation_config = dataclasses.replace(
        base_config,
        run_arc_by_arc=False,
        n_run_turns=n_run_turns,
        bpm_loss_outlier_sigma=100,
    )

    optimiser_config = OptimiserConfig(
        max_epochs=250,
        warmup_epochs=20,
        warmup_lr_start=2e-5 * n_run_turns,
        max_lr=2e-6,
        min_lr=2e-6,
        gradient_converged_value=5e-10,
        expected_rel_error=0,
    )

    estimate, unc = _run_energy_optimisation_case(
        tmp_path=tmp_path,
        loaded_interface=loaded_sps_interface,
        simulation_config=simulation_config,
        optimiser_config=optimiser_config,
        bpm_start_points=["BPH.13208", "BPV.13308", "BPH.13608", "BPV.20108"],
        bpm_end_points=[],
        magnet_range="$start/$end",
        mad_log_name="controller_energy_opt_sps_multi_turn.log",
        bpm_pattern="bp[hv][tc]?.*",
        apply_orbit_correction=True,
        dpp_value=sps_dpp_value,
        target_qx=0.13,
        target_qy=0.18,
    )

    assert np.allclose(estimate.pop("deltap"), sps_dpp_value, rtol=1e-1, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert 0 < uncertainty < 2e-4
    assert not estimate
    assert not unc
