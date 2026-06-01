from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.accelerators import SPS
from aba_optimiser.training.worker_payloads import WorkerPayloadBuilder
from aba_optimiser.training.worker_setup import WorkerObservationPlan, WorkerRangeSpec
from aba_optimiser.workers.common import KickPlane, TrackingData, WorkerConfig

if TYPE_CHECKING:
    from pathlib import Path


def _make_builder(tmp_path: Path, all_bpms: list[str] | None = None) -> WorkerPayloadBuilder:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    accelerator = SPS(sequence_file=seq_file, kinetic_energy=450.0)
    accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    return WorkerPayloadBuilder(
        accelerator=accelerator,
        all_bpms=all_bpms or ["BPH.13208", "BPV.13308"],
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
        tracking_start_bpm="BPH.13208",
        tracking_end_bpm="BPH.13208",
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


def test_attach_global_weights_ignores_unused_momentum_channels_for_position_only(
    tmp_path: Path,
) -> None:
    builder = _make_builder(tmp_path, all_bpms=["BPH.13208", "BPV.13308"])
    data = TrackingData(
        position_comparisons=np.zeros((1, 1, 2), dtype=np.float64),
        momentum_comparisons=np.zeros((1, 1, 2), dtype=np.float64),
        position_variances=np.array([[[1e-8, 1e-8]]], dtype=np.float64),
        momentum_variances=np.array([[[1e-12, 1e-12]]], dtype=np.float64),
        init_coords=np.array([[1.0, 0.1, 1.0, 0.1, 0.0, 0.01]], dtype=np.float64),
        init_pts=np.array([0.01], dtype=np.float64),
        precomputed_weights=None,
    )
    config = WorkerConfig(
        accelerator=builder.accelerator,
        tracking_start_bpm="BPH.13208",
        tracking_end_bpm="BPV.13308",
        magnet_range="$start/$end",
        corrector_strengths=None,
        tune_knobs_file=None,
        kick_plane=KickPlane.XY,
    )

    payloads = builder.attach_global_weights(
        [(data, config, 0)],
        num_batches=1,
        optimise_momenta=False,
    )
    weights = payloads[0][0].precomputed_weights

    assert weights is not None
    assert np.allclose(weights.x, [[1.0]])
    assert np.allclose(weights.y, [[1.0]])
    assert np.allclose(weights.px, [[1e4]])
    assert np.allclose(weights.py, [[1e4]])


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
        kick_plane=KickPlane.X,
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


def test_get_observation_positions_uses_full_grid_offsets(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path, all_bpms=["BPH.1", "BPH.2", "BPH.3"])
    df = pd.DataFrame(
        {
            "turn": [1, 1, 1, 2, 2, 2],
            "name": ["BPH.1", "BPH.2", "BPH.3", "BPH.1", "BPH.2", "BPH.3"],
            "x": [0.0] * 6,
            "y": [0.0] * 6,
            "px": [0.0] * 6,
            "py": [0.0] * 6,
            "var_x": [1.0] * 6,
            "var_y": [1.0] * 6,
            "var_px": [1.0] * 6,
            "var_py": [1.0] * 6,
        }
    ).set_index(["turn", "name"])

    positions = builder.get_observation_positions(
        df=df,
        bpm_names=["BPH.2", "BPH.3", "BPH.1"],
        sdir=1,
        turn=1,
        n_run_turns=1,
    )

    assert positions.tolist() == [1, 2, 3]


def _make_multi_turn_df(n_turns: int = 4) -> pd.DataFrame:
    bpms = ["BPH.1", "BPH.2", "BPH.3"]
    rows = [(t, b) for t in range(1, n_turns + 1) for b in bpms]
    turns, names = zip(*rows)
    return pd.DataFrame(
        {
            "turn": list(turns),
            "name": list(names),
            "x": list(range(len(rows))),
            "y": [0.0] * len(rows),
            "px": [0.0] * len(rows),
            "py": [0.0] * len(rows),
            "var_x": [1.0] * len(rows),
            "var_y": [1.0] * len(rows),
            "var_px": [1.0] * len(rows),
            "var_py": [1.0] * len(rows),
        }
    ).set_index(["turn", "name"])


def test_get_observation_positions_batch_matches_single_turn_calls(tmp_path: Path) -> None:
    """Batch method must return the same positions as repeated single-turn calls."""
    builder = _make_builder(tmp_path, all_bpms=["BPH.1", "BPH.2", "BPH.3"])
    df = _make_multi_turn_df()
    bpm_names = ["BPH.2", "BPH.3", "BPH.1"]
    turns = [1, 2, 3, 4]

    batch_positions = builder.get_observation_positions_batch(
        df=df, bpm_names=bpm_names, sdir=1, turns=turns, n_run_turns=1
    )

    for i, turn in enumerate(turns):
        single = builder.get_observation_positions(
            df=df, bpm_names=bpm_names, sdir=1, turn=turn, n_run_turns=1
        )
        assert batch_positions[i].tolist() == single.tolist(), f"Mismatch at turn={turn}"


def test_get_observation_positions_batch_multi_run_turns_matches_single(tmp_path: Path) -> None:
    """Batch method must handle n_run_turns > 1 correctly."""
    builder = _make_builder(tmp_path, all_bpms=["BPH.1", "BPH.2", "BPH.3"])
    df = _make_multi_turn_df(n_turns=4)
    bpm_names = ["BPH.1", "BPH.2", "BPH.3"]
    turns = [1, 2]

    batch_positions = builder.get_observation_positions_batch(
        df=df, bpm_names=bpm_names, sdir=1, turns=turns, n_run_turns=2
    )

    for i, turn in enumerate(turns):
        single = builder.get_observation_positions(
            df=df, bpm_names=bpm_names, sdir=1, turn=turn, n_run_turns=2
        )
        assert batch_positions[i].tolist() == single.tolist(), f"Mismatch at turn={turn}"


def test_get_observation_positions_batch_backward_direction(tmp_path: Path) -> None:
    """sdir=-1 batch results must match per-turn single calls."""
    builder = _make_builder(tmp_path, all_bpms=["BPH.1", "BPH.2", "BPH.3"])
    # For sdir=-1 we need enough turns in the dataframe that turn-1 exists.
    df = _make_multi_turn_df(n_turns=4)
    bpm_names = ["BPH.3", "BPH.2", "BPH.1"]  # reversed for backward traversal
    turns = [2, 3, 4]

    batch_positions = builder.get_observation_positions_batch(
        df=df, bpm_names=bpm_names, sdir=-1, turns=turns, n_run_turns=1
    )

    for i, turn in enumerate(turns):
        single = builder.get_observation_positions(
            df=df, bpm_names=bpm_names, sdir=-1, turn=turn, n_run_turns=1
        )
        assert batch_positions[i].tolist() == single.tolist(), f"Mismatch at turn={turn}"


def test_make_worker_payload_multi_turn_batch_matches_single_turn_calls(tmp_path: Path) -> None:
    """Vectorised make_worker_payload must return same arrays as repeated single-turn calls."""
    all_bpms = ["BPH.1", "BPH.2", "BPH.3"]
    builder = _make_builder(tmp_path, all_bpms=all_bpms)
    # All values start at 1 so no row is all-zero (which would trigger validation).
    df = pd.DataFrame(
        {
            "turn": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "name": all_bpms * 3,
            "x": [float(i + 1) for i in range(9)],
            "y": [float(i + 1) * 0.1 for i in range(9)],
            "px": [float(i + 1) * 0.01 for i in range(9)],
            "py": [float(i + 1) * 0.001 for i in range(9)],
            "var_x": [1.0] * 9,
            "var_y": [2.0] * 9,
            "var_px": [3.0] * 9,
            "var_py": [4.0] * 9,
        }
    ).set_index(["turn", "name"])

    file_turn_map = {1: 0, 2: 0, 3: 0}
    start_bpm = "BPH.1"
    end_bpm = "BPH.3"
    sdir = 1
    bpm_names = ["BPH.1", "BPH.2", "BPH.3"]
    kick_plane = KickPlane.X
    machine_deltaps = [0.0]
    arrays_cache = {0: builder.extract_arrays(df)}
    track_data = {0: df}
    n_run_turns = 1

    pos_batch, mom_batch, pv_batch, mv_batch, ic_batch, pts_batch = builder.make_worker_payload(
        turn_batch=[1, 2, 3],
        file_turn_map=file_turn_map,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        sdir=sdir,
        bpm_names=bpm_names,
        kick_plane=kick_plane,
        machine_deltaps=machine_deltaps,
        arrays_cache=arrays_cache,
        track_data=track_data,
        n_run_turns=n_run_turns,
    )

    for i, turn in enumerate([1, 2, 3]):
        pos_s, mom_s, pv_s, mv_s, ic_s, pts_s = builder.make_worker_payload(
            turn_batch=[turn],
            file_turn_map=file_turn_map,
            start_bpm=start_bpm,
            end_bpm=end_bpm,
            sdir=sdir,
            bpm_names=bpm_names,
            kick_plane=kick_plane,
            machine_deltaps=machine_deltaps,
            arrays_cache=arrays_cache,
            track_data=track_data,
            n_run_turns=n_run_turns,
        )
        assert np.allclose(pos_batch[i], pos_s[0]), f"pos mismatch at turn {turn}"
        assert np.allclose(mom_batch[i], mom_s[0]), f"mom mismatch at turn {turn}"
        assert np.allclose(ic_batch[i], ic_s[0]), f"init_coords mismatch at turn {turn}"
        assert np.allclose(pts_batch[i], pts_s[0]), f"pts mismatch at turn {turn}"
