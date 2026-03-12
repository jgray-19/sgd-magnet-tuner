"""
Quadrupole-focused integration tests for controller logic.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import LHC, SPS
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_optimiser_config_quad,
    _make_simulation_config_quad,
)

if TYPE_CHECKING:
    from pathlib import Path



logger = logging.getLogger(__name__)


@pytest.mark.slow
@pytest.mark.parametrize("start_marker", ["MSIA.EXIT.B1", "E.CELL.12.B1"])
def test_controller_quad_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    start_marker: str,
    loaded_interface: AbaMadInterface,
) -> None:
    magnet_range = "BPM.9R1.B1/BPM.9L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(9, 14)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(9, 14)]

    flattop_turns = 1000
    off_magnet_path = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    optimiser_config = _make_optimiser_config_quad()
    simulation_config = _make_simulation_config_quad()
    true_values = magnet_strengths.copy()

    sequence_config = SequenceConfig(
        magnet_range=magnet_range,
        first_bpm=start_marker,
    )
    measurement_config = MeasurementConfig(
        measurement_files=off_magnet_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )

    accelerator = LHC(
        beam=1,
        beam_energy=6800,
        sequence_file=seq_b1,
        optimise_quadrupoles=True,
    )

    ctrl = Controller(
        accelerator,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        show_plots=False,
        plots_dir=tmp_path / "plots",
        true_strengths=true_values,
        debug=False,
        mad_logfile=tmp_path / "mad_logfile.log",
        optimise_knobs=None,
        write_tensorboard_logs=False,
    )
    logger.info("Starting controller with logfile at %s", tmp_path / "mad_logfile.log")
    estimate, _unc = ctrl.run()

    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - true_values[magnet]) / abs(true_values[magnet])
            if true_values[magnet] != 0
            else abs(value)
        )
        assert rel_diff < 0.001, f"Relative difference for {magnet} is too high: {rel_diff:.2%}"


@pytest.mark.slow
@pytest.mark.parametrize("use_diagonal_kicks", [False], ids=["separate_hv_kicks"])
def test_controller_quad_opt_sps_multi_turn_all_quads(
    tmp_path: Path,
    seq_sps: Path,
    loaded_sps_interface: AbaMadInterface,
    use_diagonal_kicks: bool,
) -> None:
    """SPS quadrupole optimisation using separate horizontal and vertical files."""
    flattop_turns = 256
    off_magnet_path = tmp_path / "track_off_magnet_sps.parquet"
    measurement_files = (
        off_magnet_path
        if use_diagonal_kicks
        else [
            off_magnet_path.with_name("track_off_magnet_sps_particle_0.parquet"),
            off_magnet_path.with_name("track_off_magnet_sps_particle_1.parquet"),
        ]
    )

    loaded_sps_interface.observe_bpms(loaded_sps_interface.accelerator.bpm_pattern)
    no_error = loaded_sps_interface.run_twiss()
    loaded_sps_interface.unobserve_elements([loaded_sps_interface.accelerator.bpm_pattern])

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_sps_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        perturb_quads=True,
        bpm_pattern="bp[hv].*",
        apply_orbit_correction=False,
        target_qx=0.13,
        target_qy=0.18,
        num_particles=1,
        use_diagonal_kicks=use_diagonal_kicks,
    )

    base_sim = _make_simulation_config_quad()
    simulation_config = dataclasses.replace(
        base_sim,
        tracks_per_worker=2,
        num_workers=8,
        num_batches=1,
        optimise_momenta=True,
        run_arc_by_arc=False,
        n_run_turns=1,
        bpm_loss_outlier_sigma=10,
        worker_loss_outlier_sigma=10,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=30,
        warmup_lr_start=5e-9,
        max_lr=3e-7,
        min_lr=5e-7,
        gradient_converged_value=5e-16,
        expected_rel_error=loaded_sps_interface.accelerator.get_perturbation_families()["q"]["default_rel_std"],  # ty:ignore[invalid-argument-type]
        optimiser_type="adam",
    )

    sequence_config = SequenceConfig("$start/$end")
    # sequence_config = SequenceConfig("BPH.13008/BPH.13408")
    measurement_config = MeasurementConfig(
        measurement_files=measurement_files,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )

    accelerator = SPS(
        beam_energy=450.0,
        sequence_file=seq_sps,
        optimise_quadrupoles=True,
    )
    # Take 15 equally spaced v bpms and 15 equally spaced h bpms throughout the ring
    loaded_sps_interface.observe_bpms()
    all_bpms, _ = loaded_sps_interface.get_bpm_list("$start/$end")

    # Filter horizontal (BPH) and vertical (BPV) BPMs
    h_bpms = [bpm for bpm in all_bpms if bpm.startswith("BPH")]
    v_bpms = [bpm for bpm in all_bpms if bpm.startswith("BPV")]

    print(f"Total BPMs: {len(all_bpms)}, Horizontal BPMs: {len(h_bpms)}, Vertical BPMs: {len(v_bpms)}")

    # Take 15 equally spaced BPMs from each category
    n_bpms = 4
    h_spacing = max(1, len(h_bpms) // n_bpms)
    v_spacing = max(1, len(v_bpms) // n_bpms)
    h_bpms_selected = h_bpms[::h_spacing][:n_bpms]
    v_bpms_selected = v_bpms[::v_spacing][:n_bpms]

    # # Combine h and v BPMs for start points
    bpm_start_points = h_bpms_selected + v_bpms_selected
    bpm_end_points = []
    # bpm_start_points = ['BPH.13008', "BPV.13108", "BPH.13208"]
    # bpm_end_points = ["BPH.13208", "BPV.13308", "BPH.13408"]

    all_errors = loaded_sps_interface.run_twiss()

    ctrl = Controller(
        accelerator,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        show_plots=False,
        true_strengths=magnet_strengths,
        debug=False,
        mad_logfile=tmp_path / "controller_quad_opt_sps_multi_turn.log",
        write_tensorboard_logs = False,
        plots_dir=tmp_path / "plots",
    )
    estimate, unc = ctrl.run()

    iface = AbaMadInterface(accelerator=SPS(sequence_file=seq_sps, beam_energy=450.0))
    iface.set_magnet_strengths(estimate)
    iface.observe_bpms()
    est_errors = iface.run_twiss()

    # For debugging plot and show the betas and phase at all points.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(all_errors["s"], (all_errors["beta11"] - no_error["beta11"]) / no_error["beta11"] * 100, label="Perturbed dβx/βx")
    plt.plot(est_errors["s"], (est_errors["beta11"] - no_error["beta11"]) / no_error["beta11"] * 100, label="Estimated dβx/βx", linestyle="--")
    plt.xlabel("s (m)")
    plt.ylabel("dβx/βx (%)")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(all_errors["s"], (all_errors["beta22"] - no_error["beta22"]) / no_error["beta22"] * 100, label="Perturbed dβy/βy")
    plt.plot(est_errors["s"], (est_errors["beta22"] - no_error["beta22"]) / no_error["beta22"] * 100, label="Estimated dβy/βy", linestyle="--")
    plt.xlabel("s (m)")
    plt.ylabel("dβy/βy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(tmp_path / "beta_comparison.png")
    plt.show()

    # Check that magnet_strengths and estimate are close for all magnets
    for magnet, true_value in magnet_strengths.items():
        est_value = estimate[magnet]
        rel_diff = (
            abs(est_value - true_value) / abs(true_value) if true_value != 0 else abs(est_value)
        )
        assert rel_diff < 2e-4, f"Relative difference for {magnet} is too high: {rel_diff:.2%}"
