from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.training.controller import Controller, _estimate_uncertainties_from_hessian
from aba_optimiser.training.config.models import OutputConfig
from aba_optimiser.workers.common import WeightProcessor


def test_estimate_uncertainties_from_hessian_handles_negative_mode() -> None:
    hessian = np.array(
        [
            [4.0, 0.0],
            [0.0, -1e-12],
        ],
        dtype=np.float64,
    )

    uncertainties = _estimate_uncertainties_from_hessian(hessian)

    assert np.isclose(uncertainties[0], 0.5)
    assert np.isclose(uncertainties[1], 1e4)


def test_finalise_results_uses_finite_non_negative_uncertainties_for_indefinite_hessian() -> None:
    ctrl = Controller.__new__(Controller)
    ctrl.output_config = OutputConfig(include_uncertainty=True)
    ctrl.final_knobs = {"kq1": 1.0, "kq2": 2.0}
    ctrl.filtered_true_strengths = {"kq1": 1.1, "kq2": 2.1}
    ctrl.accelerator = SimpleNamespace(optimise_energy=False)
    ctrl.config_manager = SimpleNamespace(
        knob_names=["kq1", "kq2"],
        mad_iface=SimpleNamespace(
            convert_uncertainties_to_absolute=lambda knob_names, uncertainties: np.asarray(
                uncertainties,
                dtype=np.float64,
            )
        ),
    )
    ctrl.output_knob_names = ["kq1", "kq2"]

    ctrl.final_knobs = {"kq1": 0.9, "kq2": 1.9}
    uncertainties = ctrl._finalise_results(
        np.array([[4.0, 0.0], [0.0, -1e-12]], dtype=np.float64),
        writer=None,
    )

    assert np.all(np.isfinite(uncertainties))
    assert np.all(uncertainties >= 0.0)
    assert np.isclose(uncertainties[0], 0.5)
    assert np.isclose(uncertainties[1], 1e4)


def _collect_epoch_gradient(ctrl: Controller, knob_updates: dict[str, float]) -> np.ndarray:
    """Return the raw aggregated training gradient summed over all workers and batches."""
    gradient = np.zeros(len(ctrl.config_manager.knob_names), dtype=np.float64)
    channels = ctrl.worker_manager._channels()
    for batch in range(ctrl.simulation_config.num_batches):
        channels.send_all((knob_updates, batch))
        for result in channels.recv_all():
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(f"Unexpected worker result payload: {result!r}")
            _worker_id, grad, _loss = result
            gradient += np.asarray(grad, dtype=np.float64).reshape(-1)
    return gradient


def _compute_training_weight_normaliser(ctrl: Controller) -> float:
    """Rebuild the worker payload weights and return the global gradient normaliser."""
    payloads = ctrl.worker_manager.create_worker_payloads(
        ctrl.data_manager.track_data,
        ctrl.data_manager.turn_batches,
        ctrl.data_manager.file_map,
        ctrl.config_manager.start_bpms,
        ctrl.config_manager.end_bpms,
        ctrl.simulation_config,
        ctrl.machine_deltaps,
    )
    if not payloads:
        raise AssertionError("Expected at least one worker payload")

    payload_data: list[list[np.ndarray]] = []
    for data, _config, _file_idx in payloads:
        n_init = len(data.init_coords)
        payload_data.append(
            [
                data.position_variances[:n_init, :, 0],
                data.position_variances[:n_init, :, 1],
                data.momentum_variances[:n_init, :, 0],
                data.momentum_variances[:n_init, :, 1],
            ]
        )

    all_variances = [[var_slices[i] for var_slices in payload_data] for i in range(4)]
    floors = [
        WeightProcessor.compute_variance_floor(
            np.concatenate([values.reshape(-1) for values in dim_vars])
        )
        for dim_vars in all_variances
    ]

    global_max = 0.0
    for var_slices in payload_data:
        raw_weights = [
            WeightProcessor.variance_to_weight(
                WeightProcessor.floor_variances(var_slice, floor_value=floor)
            )
            for var_slice, floor in zip(var_slices, floors, strict=True)
        ]
        global_max = max(
            global_max,
            max((float(np.max(weights)) if weights.size else 0.0) for weights in raw_weights),
        )

    return global_max if global_max > 0.0 else 1.0


@pytest.mark.slow
@pytest.mark.serial
def test_controller_worker_hessian_matches_finite_difference_on_reduced_knob_subset(
    tmp_path,
    seq_b1,
    loaded_interface,
) -> None:
    from aba_optimiser.accelerators import LHC
    from aba_optimiser.training.config.models import MeasurementConfig, SequenceConfig
    from tests.training.controller_test_utils import (
        _generate_nonoise_track,
        _make_optimiser_config_quad,
        _make_simulation_config_quad,
    )

    magnet_range = "BPM.13R1.B1/BPM.13L2.B1"
    bpm_start_points = ["BPM.13R1.B1"]
    bpm_end_points = ["BPM.13L2.B1"]
    flattop_turns = 64
    start_marker = "MSIA.EXIT.B1"
    measurement_file = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_interface,
        flattop_turns,
        measurement_file,
        0.0,
        start_marker=start_marker,
        perturb_quads=True,
    )

    simulation_config = dataclasses.replace(
        _make_simulation_config_quad(),
        tracks_per_worker=4,
        num_workers=4,
        num_batches=2,
    )
    ctrl = Controller(
        LHC(
            beam=1,
            kinetic_energy=6800,
            sequence_file=seq_b1,
            optimise_quadrupoles=True,
            optimise_other_quadrupoles=False,
        ),
        _make_optimiser_config_quad(),
        simulation_config,
        SequenceConfig(
            magnet_range=magnet_range,
            first_bpm=start_marker,
        ),
        MeasurementConfig(
            measurement_files=measurement_file,
            corrector_files=corrector_file,
            tune_knobs_files=tune_knobs_file,
            flattop_turns=flattop_turns,
            bunches_per_file=1,
        ),
        bpm_start_points,
        bpm_end_points,
        output_config=OutputConfig(
            mad_logfile=tmp_path / "controller_hessian.log",
            write_tensorboard_logs=False,
        ),
        true_strengths=magnet_strengths.copy(),
    )
    weight_normaliser = _compute_training_weight_normaliser(ctrl)

    terminated = False
    try:
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
        normalisation_points = {
            len(meta.bpm_names) * meta.n_run_turns for meta in ctrl.worker_manager.worker_metadata
        }
        assert len(normalisation_points) == 1
        gradient_normalisation = float(normalisation_points.pop())

        base_knobs = ctrl.filtered_true_strengths.copy()
        base_vec = np.array(
            [base_knobs[name] for name in ctrl.config_manager.knob_names],
            dtype=np.float64,
        )
        n_knobs = len(base_vec)
        centre = n_knobs // 2
        subset = np.array([centre - 1, centre, centre + 1], dtype=int)

        fd_matrix = np.zeros((subset.size, subset.size), dtype=np.float64)
        for col, knob_idx in enumerate(subset):
            knob_name = ctrl.config_manager.knob_names[knob_idx]
            step = max(1e-6, 1e-2 * max(abs(base_vec[knob_idx]), 1e-4))

            plus_knobs = base_knobs.copy()
            minus_knobs = base_knobs.copy()
            plus_knobs[knob_name] += step
            minus_knobs[knob_name] -= step

            grad_plus = _collect_epoch_gradient(ctrl, plus_knobs)
            grad_minus = _collect_epoch_gradient(ctrl, minus_knobs)
            fd_matrix[:, col] = (grad_plus[subset] - grad_minus[subset]) / (2.0 * step)

        total_hessian = ctrl.worker_manager.termination_and_hessian(n_knobs, estimate_hessian=True)
        terminated = True
    finally:
        if not terminated:
            ctrl.worker_manager.terminate_workers()

    assert total_hessian.shape == (n_knobs, n_knobs)
    assert np.all(np.isfinite(total_hessian))
    sym_hessian = 0.5 * (total_hessian + total_hessian.T)
    predicted = 2.0 * (
        sym_hessian[np.ix_(subset, subset)] / (weight_normaliser * gradient_normalisation)
    )
    difference = fd_matrix - predicted
    reference_scale = max(np.linalg.norm(predicted), 1.0)

    assert np.linalg.norm(predicted) > 0.0
    assert np.linalg.norm(difference) / reference_scale < 0.25
    assert np.allclose(fd_matrix, predicted, rtol=0.25, atol=1e-2)
