from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.accelerators import SPS
from aba_optimiser.training.worker_payloads import WorkerPayloadBuilder
from aba_optimiser.training.worker_setup import WorkerObservationPlan, WorkerRangeSpec
from aba_optimiser.workers.common import TrackingData, WorkerConfig


def _make_builder(tmp_path: Path, all_bpms: list[str] | None = None) -> WorkerPayloadBuilder:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    accelerator = SPS(sequence_file=seq_file, beam_energy=450.0)
    accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    return WorkerPayloadBuilder(
        accelerator=accelerator,
        all_bpms=all_bpms or ["BPH.13208", "BPV.13308"],
        beam_energy=450.0,
    )


def test_make_worker_payload_keeps_only_active_plane_measurements(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    df = pd.DataFrame(
        {
            "turn": [1, 1],
            "name": ["BPH.13208", "BPV.13308"],
            "x": [0.0, 0.0],
            "y": [0.0, 1.2e-3],
            "px": [2.5e-5, 0.0],
            "py": [0.0, 3.1e-5],
            "var_x": [1.0, np.inf],
            "var_y": [np.inf, 1.0],
            "var_px": [1.0, np.inf],
            "var_py": [np.inf, 1.0],
        }
    ).set_index(["turn", "name"])

    pos, mom, pos_var, mom_var, init_coords, pts = builder.make_worker_payload(
        turn_batch=[1],
        file_turn_map={1: 0},
        start_bpm="BPH.13208",
        end_bpm="BPV.13308",
        sdir=1,
        bpm_names=["BPH.13208"],
        kick_plane="x",
        machine_deltaps=[0.0],
        arrays_cache={0: builder.extract_arrays(df)},
        track_data={0: df},
        n_run_turns=1,
    )

    assert np.allclose(init_coords[0, :4], [0.0, 2.5e-5, 0.0, 0.0])
    assert np.allclose(pts, [0.0])
    assert pos.shape == (1, 1, 2)
    assert mom.shape == (1, 1, 2)
    assert pos_var[0, :, 0].tolist() == [1.0]
    assert mom_var[0, :, 0].tolist() == [1.0]
    assert np.allclose(pos[0, :, 1], 0.0)
    assert np.allclose(mom[0, :, 1], 0.0)
    assert not np.isfinite(pos_var[0, :, 1]).any()
    assert not np.isfinite(mom_var[0, :, 1]).any()


def test_make_worker_payload_rejects_single_plane_bpms_for_dual_plane_worker(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    df = pd.DataFrame(
        {
            "turn": [1, 1],
            "name": ["BPH.13208", "BPV.13308"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "px": [0.1, 0.2],
            "py": [0.3, 0.4],
            "var_x": [1.0, 1.0],
            "var_y": [1.0, 1.0],
            "var_px": [1.0, 1.0],
            "var_py": [1.0, 1.0],
        }
    ).set_index(["turn", "name"])

    with pytest.raises(ValueError, match="Dual-plane worker received single-plane BPMs"):
        builder.make_worker_payload(
            turn_batch=[1],
            file_turn_map={1: 0},
            start_bpm="BPH.13208",
            end_bpm="BPV.13308",
            sdir=1,
            bpm_names=["BPH.13208", "BPV.13308"],
            kick_plane="xy",
            machine_deltaps=[0.0],
            arrays_cache={0: builder.extract_arrays(df)},
            track_data={0: df},
            n_run_turns=1,
        )


def test_diagnostic_loss_per_bpm_sums_turns(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path, all_bpms=["BPH.13208", "BPH.13608"])

    per_bpm = builder.diagnostic_loss_per_bpm(
        loss_per_point=np.array([1.0, 2.0, 3.0, 4.0]),
        bpm_names=["BPH.13208", "BPH.13608"],
        n_run_turns=2,
        worker_id=0,
    )

    assert per_bpm.tolist() == [4.0, 6.0]


def test_attach_global_weights_normalises_all_observables(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path, all_bpms=["BPH.13208"])
    data = TrackingData(
        position_comparisons=np.zeros((1, 1, 2), dtype=np.float64),
        momentum_comparisons=np.zeros((1, 1, 2), dtype=np.float64),
        position_variances=np.array([[[2.0, 3.0]]], dtype=np.float64),
        momentum_variances=np.array([[[4.0, 5.0]]], dtype=np.float64),
        init_coords=np.array([[1.0, 0.1, 0.0, 0.0, 0.0, 0.01]], dtype=np.float64),
        init_pts=np.array([0.01], dtype=np.float64),
        precomputed_weights=None,
    )
    config = WorkerConfig(
        accelerator=builder.accelerator,
        start_bpm="BPH.13208",
        end_bpm="BPH.13208",
        magnet_range="$start/$end",
        corrector_strengths=None,
        tune_knobs_file=None,
    )

    payloads = builder.attach_global_weights([(data, config, 0)], num_batches=1)
    weights = payloads[0][0].precomputed_weights

    assert weights is not None
    assert np.allclose(weights.x, [[1.0]])
    assert np.allclose(weights.y, [[2.0 / 3.0]])
    assert np.allclose(weights.px, [[0.5]])
    assert np.allclose(weights.py, [[0.4]])


def test_make_tracking_data_freezes_arrays(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    df = pd.DataFrame(
        {
            "turn": [1],
            "name": ["BPH.13208"],
            "x": [1.0],
            "y": [0.0],
            "px": [0.1],
            "py": [0.0],
            "var_x": [1.0],
            "var_y": [np.inf],
            "var_px": [1.0],
            "var_py": [np.inf],
        }
    ).set_index(["turn", "name"])
    plan = WorkerObservationPlan(
        range_spec=WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPH.13208", sdir=1),
        file_idx=0,
        kick_plane="x",
        bpm_names=["BPH.13208"],
        bad_bpms=None,
    )

    data = builder.make_tracking_data(
        turn_batch=[1],
        file_turn_map={1: 0},
        plan=plan,
        machine_deltaps=[0.0],
        arrays_cache={0: builder.extract_arrays(df)},
        track_data={0: df},
        n_run_turns=1,
    )

    assert not data.position_comparisons.flags.writeable
    assert not data.momentum_comparisons.flags.writeable
