"""
Quadrupole-focused integration tests for controller logic.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import LHC, SPS
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import (
    MeasurementConfig,
    OutputConfig,
    SequenceConfig,
)
from tests.training.controller_test_utils import (
    _generate_nonoise_track,
    _make_optimiser_config_quad,
    _make_simulation_config_quad,
)

if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def _build_lhc_quad_controller(
    *,
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    start_marker: str,
) -> tuple[Controller, dict[str, float]]:
    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(13, 18)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(13, 18)]

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

    ctrl = Controller(
        LHC(
            beam=1,
            beam_energy=6800,
            sequence_file=seq_b1,
            optimise_quadrupoles=True,
        ),
        _make_optimiser_config_quad(),
        _make_simulation_config_quad(),
        SequenceConfig(
            magnet_range=magnet_range,
            first_bpm=start_marker,
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
            show_plots=False,
            plots_dir=tmp_path / "plots",
            mad_logfile=tmp_path / "mad_logfile.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
        debug=False,
        optimise_knobs=None,
    )
    return ctrl, magnet_strengths.copy()


def _assert_estimate_matches_true(
    estimate: dict[str, float],
    true_values: dict[str, float],
    *,
    max_rel_diff: float,
) -> None:
    worst_magnet = ""
    worst_rel_diff = -np.inf
    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - true_values[magnet]) / abs(true_values[magnet])
            if true_values[magnet] != 0
            else abs(value)
        )
        if rel_diff > worst_rel_diff:
            worst_magnet = magnet
            worst_rel_diff = rel_diff
        assert rel_diff < max_rel_diff, (
            f"Relative difference for {magnet} is too high: {rel_diff:.2%} "
            f"(worst so far: {worst_magnet} at {worst_rel_diff:.2%})"
        )


@pytest.mark.slow
@pytest.mark.parametrize("start_marker", ["MSIA.EXIT.B1", "E.CELL.12.B1"])
def test_controller_quad_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    start_marker: str,
    loaded_interface: AbaMadInterface,
) -> None:
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker=start_marker,
    )
    logger.info("Starting controller with logfile at %s", tmp_path / "mad_logfile.log")
    estimate, _unc = ctrl.run()
    _assert_estimate_matches_true(estimate, true_values, max_rel_diff=1e-4)


@pytest.mark.slow
def test_controller_quad_opt_simple_without_early_stopping_reaches_truth(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
) -> None:
    ctrl, true_values = _build_lhc_quad_controller(
        tmp_path=tmp_path,
        seq_b1=seq_b1,
        loaded_interface=loaded_interface,
        start_marker="MSIA.EXIT.B1",
    )

    ctrl.optimisation_loop._should_stop_for_loss_change = (  # type: ignore[method-assign]
        lambda epoch, epoch_loss, prev_loss: False
    )
    ctrl.optimisation_loop.gradient_converged_value = -1.0

    estimate, _unc = ctrl.run()
    _assert_estimate_matches_true(estimate, true_values, max_rel_diff=1e-4)


@pytest.mark.slow
def test_controller_quad_worker_gradients_match_finite_differences(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
) -> None:
    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = [f"BPM.{i}R1.B1" for i in range(13, 18)]
    bpm_end_points = [f"BPM.{i}L2.B1" for i in range(13, 18)]

    flattop_turns = 1000
    off_magnet_path = tmp_path / "track_off_magnet.parquet"

    corrector_file, _magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        off_magnet_path,
        0.0,
        start_marker="MSIA.EXIT.B1",
        perturb_quads=True,
    )

    optimiser_config = _make_optimiser_config_quad()
    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        enable_preloop_outlier_screening=False,
    )

    ctrl = Controller(
        LHC(
            beam=1,
            beam_energy=6800,
            sequence_file=seq_b1,
            optimise_quadrupoles=True,
        ),
        optimiser_config,
        simulation_config,
        SequenceConfig(
            magnet_range=magnet_range,
            first_bpm="MSIA.EXIT.B1",
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
            show_plots=False,
            plots_dir=tmp_path / "plots",
            mad_logfile=tmp_path / "worker_gradients.log",
            write_tensorboard_logs=False,
        ),
        true_strengths={},
        debug=False,
        optimise_knobs=None,
    )

    ctrl.worker_manager.start_workers(
        ctrl.data_manager.track_data,
        ctrl.data_manager.turn_batches,
        ctrl.data_manager.file_map,
        ctrl.config_manager.start_bpms,
        ctrl.config_manager.end_bpms,
        ctrl.simulation_config,
        ctrl.machine_deltaps,
        ctrl.initial_knobs,
    )

    def evaluate(knob_values: dict[str, float]) -> dict[int, tuple[np.ndarray, float]]:
        channels = ctrl.worker_manager.channels
        if channels is None:
            raise RuntimeError("Worker channels were not initialised")

        channels.send_all((knob_values, 0))
        return {
            int(worker_id): (np.asarray(grad, dtype=float).reshape(-1), float(loss))
            for worker_id, grad, loss in channels.recv_all()
        }

    try:
        baseline = evaluate(ctrl.initial_knobs)
        assert len(baseline) == len(ctrl.worker_manager.worker_metadata)
        assert len(baseline) > 0

        grad_norms: list[float] = []
        losses: list[float] = []
        representative_workers: dict[int, tuple[int, float]] = {}
        n_knobs = len(ctrl.initial_knobs)

        for meta in ctrl.worker_manager.worker_metadata:
            grad_arr, loss = baseline[meta.worker_id]
            grad_norm = float(np.linalg.norm(grad_arr))

            assert grad_arr.shape == (n_knobs,)
            assert np.isfinite(grad_arr).all()
            assert np.isfinite(loss)
            assert loss > 0.0
            assert grad_norm > 0.0
            assert np.count_nonzero(grad_arr) > 0

            grad_norms.append(grad_norm)
            losses.append(loss)
            current = representative_workers.get(meta.sdir)
            if current is None or grad_norm > current[1]:
                representative_workers[meta.sdir] = (meta.worker_id, grad_norm)

        assert max(grad_norms) > min(grad_norms)
        assert max(losses) > min(losses)
        assert set(representative_workers) == {1, -1}

        knob_names = ctrl.config_manager.knob_names
        fd_step = 1e-6
        for sdir, (worker_id, _grad_norm) in representative_workers.items():
            grad_arr, _loss = baseline[worker_id]
            knob_idx = int(np.argmax(np.abs(grad_arr)))
            knob_name = knob_names[knob_idx]
            analytic_grad = float(grad_arr[knob_idx])

            plus_knobs = dict(ctrl.initial_knobs)
            minus_knobs = dict(ctrl.initial_knobs)
            plus_knobs[knob_name] += fd_step
            minus_knobs[knob_name] -= fd_step

            plus_loss = evaluate(plus_knobs)[worker_id][1]
            minus_loss = evaluate(minus_knobs)[worker_id][1]
            fd_grad = (plus_loss - minus_loss) / (2.0 * fd_step)

            assert np.isclose(fd_grad, analytic_grad, rtol=1e-5, atol=1e-12), (
                f"Worker {worker_id} (sdir={sdir}) knob {knob_name}: "
                f"analytic={analytic_grad:.6e}, fd={fd_grad:.6e}"
            )
    finally:
        ctrl.worker_manager.termination_and_hessian(
            len(ctrl.initial_knobs),
            estimate_hessian=False,
        )

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
        output_config=OutputConfig(
            show_plots=False,
            plots_dir=tmp_path / "plots",
            mad_logfile=tmp_path / "controller_quad_opt_sps_multi_turn.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths,
        debug=False,
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
