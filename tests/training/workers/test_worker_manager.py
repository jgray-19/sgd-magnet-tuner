from __future__ import annotations

import multiprocessing
import threading
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.workers.manager import WorkerManager
from aba_optimiser.training.workers.screening import OutlierScreener
from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata
from aba_optimiser.training.workers.validation import (
    payload_track_count,
    split_validation_payloads,
)
from aba_optimiser.workers import TrackingData, WorkerConfig
from aba_optimiser.workers.common import KickPlane

if TYPE_CHECKING:
    from pathlib import Path


class _FakeConn:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.sent: list[dict[str, object]] = []
        self._responses = responses

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def recv(self) -> dict[str, object]:
        return self._responses.pop(0)


class _FakeChannels:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.sent: list[object] = []

    def send_all(self, payload: object) -> None:
        self.sent.append(payload)

    def recv_all(self) -> list[object]:
        return self._responses


class _FakeWorker:
    def __init__(self) -> None:
        self.join_calls = 0
        self.terminate_calls = 0
        self.exitcode = 0
        self.pid = 1234

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        self.terminate_calls += 1


class _ConnThatMustNotPoll:
    def send(self, payload: object) -> None:
        self.sent = payload

    def poll(self, timeout: float | None = None) -> bool:
        raise AssertionError(f"Validation cleanup should not poll for payloads, got {timeout}")

    def recv(self) -> object:
        raise AssertionError("Validation cleanup should not receive a final payload")


def _start_pipe_worker(child_conn: multiprocessing.connection.Connection, responses: list[object]) -> threading.Thread:
    """Start a daemon thread that acts as a fake worker over a real pipe.

    The thread receives one binary message per response (via recv_bytes), then sends
    the corresponding response back via send(). This matches the WorkerChannels
    protocol where the parent sends via send_bytes and receives via recv.
    """
    def _run() -> None:
        for response in responses:
            child_conn.recv_bytes()
            child_conn.send(response)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _make_sps(tmp_path: Path) -> SPS:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    return SPS(sequence_file=seq_file, kinetic_energy=450.0)


def _make_manager(
    tmp_path: Path,
    *,
    n_data_points: dict[tuple[str, str], int] | None = None,
    all_bpms: list[str] | None = None,
) -> WorkerManager:
    return WorkerManager(
        n_data_points=n_data_points or {},
        ybpm="BPV.13308",
        magnet_range="$start/$end",
        fixed_start="BPH.13208",
        fixed_end="BPV.20108",
        accelerator=_make_sps(tmp_path),
        interface_options_per_file=[
            {"corrector_strengths": tmp_path / "correctors.tfs", "tune_knobs_file": tmp_path / "tune_knobs.txt"}
        ],
        all_bpms=all_bpms or ["BPH.13208", "BPV.13308", "BPH.13608", "BPV.20108"],
    )


def _make_track_df(all_bpms: list[str], turns: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for turn_idx, turn in enumerate(turns, start=1):
        for bpm_idx, name in enumerate(all_bpms, start=1):
            is_h = name.startswith("BPH")
            rows.append(
                {
                    "turn": turn,
                    "name": name,
                    "x": float(10 * turn_idx + bpm_idx) if is_h else 0.0,
                    "y": 0.0 if is_h else float(100 * turn_idx + bpm_idx),
                    "px": float(20 * turn_idx + bpm_idx) if is_h else 0.0,
                    "py": 0.0 if is_h else float(200 * turn_idx + bpm_idx),
                    "var_x": 1.0 if is_h else np.inf,
                    "var_y": np.inf if is_h else 1.0,
                    "var_px": 1.0 if is_h else np.inf,
                    "var_py": np.inf if is_h else 1.0,
                }
            )
    return pd.DataFrame(rows).set_index(["turn", "name"])


def _make_payload(
    accelerator: SPS,
    *,
    start_bpm: str,
    end_bpm: str,
    sdir: int,
    kick_plane: str,
    file_idx: int,
    n_tracks: int,
    n_points: int,
) -> tuple[TrackingData, WorkerConfig, int]:
    data = TrackingData(
        position_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        momentum_comparisons=np.zeros((n_tracks, n_points, 2), dtype=np.float64),
        position_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        momentum_variances=np.ones((n_tracks, n_points, 2), dtype=np.float64),
        init_coords=np.zeros((n_tracks, 6), dtype=np.float64),
        init_pts=np.zeros((n_tracks,), dtype=np.float64),
        precomputed_weights=None,
    )
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm=start_bpm,
        tracking_end_bpm=end_bpm,
        magnet_range="$start/$end",
        interface_options={},
        sdir=sdir,
        kick_plane=kick_plane,
    )
    return data, config, file_idx


def test_create_worker_payloads_multi_turn_creates_forward_and_backward_workers(
    tmp_path: Path,
) -> None:
    all_bpms = ["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308"]
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPH.13008"): 3,
        },
        all_bpms=all_bpms,
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    df = _make_track_df(all_bpms, [1, 2, 3])
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=2,
        num_batches=1,
        optimise_momenta=False,
        run_arc_by_arc=False,
        n_run_turns=1,
    )

    payloads = manager.create_worker_payloads(
        track_data={0: df},
        turn_batches=[[2]],
        file_turn_map={1: 0, 2: 0, 3: 0},
        start_bpms=["BPH.13208"],
        end_bpms=[],
        simulation_config=simulation_config,
        machine_deltaps=[0.0],
    )

    # Single-plane machines build same-plane ranges: the x-plane start spawns a
    # forward and backward worker that both observe the x-plane, ending at the
    # previous x-plane BPM.
    assert len(payloads) == 2
    assert [
        (
            config.tracking_start_bpm,
            config.tracking_end_bpm,
            config.sdir,
            config.kick_plane,
        )
        for _, config, _ in payloads
    ] == [
        ("BPH.13208", "BPH.13008", 1, "x"),
        ("BPH.13208", "BPH.13008", -1, "x"),
    ]

    forward_data = payloads[0][0]
    backward_data = payloads[1][0]
    assert np.isfinite(forward_data.position_variances[0, :, 0]).any()
    assert not np.isfinite(forward_data.position_variances[0, :, 1]).any()
    assert np.isfinite(backward_data.position_variances[0, :, 0]).any()
    assert not np.isfinite(backward_data.position_variances[0, :, 1]).any()


def test_create_worker_payloads_multi_turn_supports_mixed_start_planes(tmp_path: Path) -> None:
    all_bpms = ["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308"]
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPH.13008"): 3,
            ("BPV.13308", "BPV.13108"): 3,
        },
        all_bpms=all_bpms,
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    df = _make_track_df(all_bpms, [1, 2, 3])
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=4,
        num_batches=1,
        optimise_momenta=False,
        run_arc_by_arc=False,
        n_run_turns=1,
    )

    payloads = manager.create_worker_payloads(
        track_data={0: df},
        turn_batches=[[2]],
        file_turn_map={2: 0},
        start_bpms=["BPH.13208", "BPV.13308"],
        end_bpms=[],
        simulation_config=simulation_config,
        machine_deltaps=[0.0],
    )

    # Each single-plane start spawns a forward and backward worker confined to
    # its own plane, so the x-plane and y-plane starts give four workers total.
    assert len(payloads) == 4

    payload_by_key = {
        (config.tracking_start_bpm, config.tracking_end_bpm, config.sdir): (data, config)
        for data, config, _ in payloads
    }
    forward_h, forward_h_config = payload_by_key[("BPH.13208", "BPH.13008", 1)]
    backward_h, backward_h_config = payload_by_key[("BPH.13208", "BPH.13008", -1)]
    forward_v, forward_v_config = payload_by_key[("BPV.13308", "BPV.13108", 1)]
    backward_v, backward_v_config = payload_by_key[("BPV.13308", "BPV.13108", -1)]

    assert forward_h_config.kick_plane == "x"
    assert backward_h_config.kick_plane == "x"
    assert forward_v_config.kick_plane == "y"
    assert backward_v_config.kick_plane == "y"
    assert np.isfinite(forward_h.position_variances[0, :, 0]).any()
    assert not np.isfinite(forward_h.position_variances[0, :, 1]).any()
    assert np.isfinite(backward_h.position_variances[0, :, 0]).any()
    assert not np.isfinite(backward_h.position_variances[0, :, 1]).any()
    assert not np.isfinite(forward_v.position_variances[0, :, 0]).any()
    assert np.isfinite(forward_v.position_variances[0, :, 1]).any()
    assert not np.isfinite(backward_v.position_variances[0, :, 0]).any()
    assert np.isfinite(backward_v.position_variances[0, :, 1]).any()


def test_create_worker_payloads_arc_by_arc_uses_configured_fixed_pairs(tmp_path: Path) -> None:
    all_bpms = ["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308"]
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPH.13008"): 3,
        },
        all_bpms=all_bpms,
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    df = _make_track_df(all_bpms, [1, 2, 3])
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=2,
        num_batches=1,
        optimise_momenta=False,
        run_arc_by_arc=True,
    )

    payloads = manager.create_worker_payloads(
        track_data={0: df},
        turn_batches=[[2]],
        file_turn_map={2: 0},
        start_bpms=["BPH.13208"],
        end_bpms=["BPH.13008"],
        simulation_config=simulation_config,
        machine_deltaps=[0.0],
    )

    # Single-plane arc-by-arc ranges pair the same-plane start/end BPMs as the
    # fixed forward/backward boundaries, keeping both workers on the x-plane.
    assert [
        (
            config.tracking_start_bpm,
            config.tracking_end_bpm,
            config.sdir,
            config.kick_plane,
        )
        for _, config, _ in payloads
    ] == [
        ("BPH.13208", "BPH.13008", 1, "x"),
        ("BPH.13208", "BPH.13008", -1, "x"),
    ]


def test_create_worker_payloads_assigns_per_file_artifacts_from_file_turn_map(tmp_path: Path) -> None:
    manager = _make_manager(
        tmp_path,
        n_data_points={("BPH.13208", "BPV.13108"): 3},
        all_bpms=["BPV.13108", "BPH.13208", "BPV.13308"],
    )
    manager.interface_options_per_file = [
        {"corrector_strengths": tmp_path / "corr0.tfs", "tune_knobs_file": tmp_path / "knobs0.txt"},
        {"corrector_strengths": tmp_path / "corr1.tfs", "tune_knobs_file": tmp_path / "knobs1.txt"},
    ]

    payloads = manager.create_worker_payloads(
        track_data={
            0: _make_track_df(manager.all_bpms, [1, 2, 3]),
            1: _make_track_df(manager.all_bpms, [201, 202, 203]),
        },
        turn_batches=[[2], [202]],
        file_turn_map={2: 0, 202: 1},
        start_bpms=["BPH.13208"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=2,
            num_batches=1,
            optimise_momenta=False,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        machine_deltaps=[0.0, 1e-3],
    )

    assert [file_idx for _, _, file_idx in payloads] == [0, 1, 0, 1]
    assert [config.interface_options for _, config, _ in payloads] == [
        {"corrector_strengths": tmp_path / "corr0.tfs", "tune_knobs_file": tmp_path / "knobs0.txt"},
        {"corrector_strengths": tmp_path / "corr1.tfs", "tune_knobs_file": tmp_path / "knobs1.txt"},
        {"corrector_strengths": tmp_path / "corr0.tfs", "tune_knobs_file": tmp_path / "knobs0.txt"},
        {"corrector_strengths": tmp_path / "corr1.tfs", "tune_knobs_file": tmp_path / "knobs1.txt"},
    ]

    init_pts = [float(data.init_pts[0]) for data, _, _ in payloads]
    assert init_pts[0] == init_pts[2]
    assert init_pts[1] == init_pts[3]
    assert init_pts[0] != init_pts[1]


def test_build_bpm_masks_from_diagnostics_aggregates_multi_turn_losses(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    manager.worker_metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        )
    ]

    masks = OutlierScreener(manager.payload_builder).build_bpm_masks_from_diagnostics(
        diagnostics=[
            {
                "worker_id": 0,
                "loss_per_bpm": [1.0, 50.0, 1.0, 50.0],
            }
        ],
        worker_metadata=manager.worker_metadata,
        bpm_sigma_threshold=0.5,
    )

    assert len(masks) == 1
    assert masks[0].tolist() == [True, False]


def test_apply_screening_actions_expands_masks_across_turns(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    conn_a = _FakeConn([{"status": "ok"}])
    conn_b = _FakeConn([{"status": "ok"}])
    manager.parent_conns = [conn_a, conn_b]  # ty:ignore[invalid-assignment]
    manager.worker_metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        ),
        WorkerRuntimeMetadata(
            worker_id=1,
            file_idx=1,
            start_bpm="BPV.13308",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=1,
            bpm_names=["BPV.13308", "BPV.20108"],
        ),
    ]

    OutlierScreener(manager.payload_builder).apply_screening_actions(
        parent_conns=manager.parent_conns,
        worker_metadata=manager.worker_metadata,
        bpm_masks=[np.array([True, False]), np.array([False, True])],
        worker_disabled=[False, True],
    )

    assert conn_a.sent == [
        {
            "cmd": "apply_mask",
            "keep_bpm_mask": [True, False, True, False],
            "disable_worker": False,
        }
    ]
    assert conn_b.sent == [
        {
            "cmd": "apply_mask",
            "keep_bpm_mask": [False, True],
            "disable_worker": True,
        }
    ]

def test_summarise_screening_losses_logs_pre_and_projected_loss(tmp_path: Path, caplog) -> None:
    manager = _make_manager(tmp_path)
    manager.worker_metadata = [
        WorkerRuntimeMetadata(
            worker_id=0,
            file_idx=0,
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane=KickPlane.XY,
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        )
    ]

    with caplog.at_level("INFO"):
        OutlierScreener(manager.payload_builder).summarise_screening_losses(
            diagnostics=[{"worker_id": 0, "loss_per_bpm": [1.0, 9.0, 1.0, 9.0]}],
            bpm_masks=[np.array([True, False])],
            worker_disabled=[False],
            worker_metadata=manager.worker_metadata,
        )

    assert "Pre-screening loss summary" in caplog.text
    assert "Projected post-screening loss summary" in caplog.text
    assert "total=2.000000e+01" in caplog.text
    assert "total=2.000000e+00" in caplog.text


def test_split_validation_payloads_covers_multiple_ranges_when_available(
    tmp_path: Path,
) -> None:
    accelerator = _make_sps(tmp_path)
    payloads = [
        _make_payload(
            accelerator,
            start_bpm="BPH.13008",
            end_bpm="BPH.13408",
            sdir=1,
            kick_plane="x",
            file_idx=0,
            n_tracks=5,
            n_points=120,
        ),
        _make_payload(
            accelerator,
            start_bpm="BPH.13008",
            end_bpm="BPH.13408",
            sdir=-1,
            kick_plane="x",
            file_idx=0,
            n_tracks=5,
            n_points=120,
        ),
        _make_payload(
            accelerator,
            start_bpm="BPH.14008",
            end_bpm="BPH.14408",
            sdir=1,
            kick_plane="x",
            file_idx=0,
            n_tracks=30,
            n_points=60,
        ),
        _make_payload(
            accelerator,
            start_bpm="BPH.14008",
            end_bpm="BPH.14408",
            sdir=-1,
            kick_plane="x",
            file_idx=0,
            n_tracks=30,
            n_points=60,
        ),
    ]

    split = split_validation_payloads(payloads)
    training_payloads = split.training_payloads
    validation_payloads = split.validation_payloads
    duplicated = split.duplicated_validation_payload

    assert duplicated is False
    assert {
        (p[1].tracking_start_bpm, p[1].tracking_end_bpm) for p in validation_payloads
    } == {
        ("BPH.13008", "BPH.13408"),
        ("BPH.14008", "BPH.14408"),
    }
    assert {p[1].sdir for p in validation_payloads} == {1, -1}
    assert len(training_payloads) > 0

    validation_tracks = sum(payload_track_count(payload) for payload in validation_payloads)
    training_tracks = sum(payload_track_count(payload) for payload in training_payloads)
    assert validation_tracks * 10 >= training_tracks


def test_split_validation_payloads_pairs_opposite_directions_with_mixed_planes(
    tmp_path: Path,
) -> None:
    accelerator = _make_sps(tmp_path)
    payloads = [
        _make_payload(
            accelerator,
            start_bpm="BPH.13208",
            end_bpm="BPV.13108",
            sdir=1,
            kick_plane="x",
            file_idx=0,
            n_tracks=5,
            n_points=100,
        ),
        _make_payload(
            accelerator,
            start_bpm="BPH.13208",
            end_bpm="BPV.13108",
            sdir=-1,
            kick_plane="y",
            file_idx=0,
            n_tracks=5,
            n_points=100,
        ),
        _make_payload(
            accelerator,
            start_bpm="BPH.14008",
            end_bpm="BPV.13908",
            sdir=1,
            kick_plane="x",
            file_idx=0,
            n_tracks=5,
            n_points=50,
        ),
    ]

    split = split_validation_payloads(payloads)

    assert len(split.validation_payloads) == 3
    assert {(p[1].sdir, p[1].kick_plane) for p in split.validation_payloads} == {
        (1, "x"),
        (-1, "y"),
    }
    assert {
        (p[1].tracking_start_bpm, p[1].tracking_end_bpm)
        for p in split.validation_payloads
    } == {
        ("BPH.13208", "BPV.13108"),
        ("BPH.14008", "BPV.13908"),
    }


def test_split_validation_payloads_spreads_across_sorted_range_groups(
    tmp_path: Path,
) -> None:
    accelerator = _make_sps(tmp_path)
    payloads = [
        _make_payload(accelerator, start_bpm="BPH.10008", end_bpm="BPH.10408", sdir=1, kick_plane="x", file_idx=0, n_tracks=5, n_points=150),
        _make_payload(accelerator, start_bpm="BPH.10008", end_bpm="BPH.10408", sdir=-1, kick_plane="x", file_idx=0, n_tracks=5, n_points=150),
        _make_payload(accelerator, start_bpm="BPH.11008", end_bpm="BPH.11408", sdir=1, kick_plane="x", file_idx=0, n_tracks=5, n_points=130),
        _make_payload(accelerator, start_bpm="BPH.11008", end_bpm="BPH.11408", sdir=-1, kick_plane="x", file_idx=0, n_tracks=5, n_points=130),
        _make_payload(accelerator, start_bpm="BPH.12008", end_bpm="BPH.12408", sdir=1, kick_plane="x", file_idx=0, n_tracks=5, n_points=110),
        _make_payload(accelerator, start_bpm="BPH.12008", end_bpm="BPH.12408", sdir=-1, kick_plane="x", file_idx=0, n_tracks=5, n_points=110),
        _make_payload(accelerator, start_bpm="BPH.13008", end_bpm="BPH.13408", sdir=1, kick_plane="x", file_idx=0, n_tracks=5, n_points=90),
        _make_payload(accelerator, start_bpm="BPH.13008", end_bpm="BPH.13408", sdir=-1, kick_plane="x", file_idx=0, n_tracks=5, n_points=90),
        _make_payload(accelerator, start_bpm="BPH.14008", end_bpm="BPH.14408", sdir=1, kick_plane="x", file_idx=0, n_tracks=5, n_points=70),
        _make_payload(accelerator, start_bpm="BPH.14008", end_bpm="BPH.14408", sdir=-1, kick_plane="x", file_idx=0, n_tracks=5, n_points=70),
    ]

    split = split_validation_payloads(payloads)

    assert {
        (p[1].tracking_start_bpm, p[1].tracking_end_bpm)
        for p in split.validation_payloads
    } == {
        ("BPH.10008", "BPH.10408"),
        ("BPH.12008", "BPH.12408"),
        ("BPH.14008", "BPH.14408"),
    }
    assert {p[1].sdir for p in split.validation_payloads} == {1, -1}


def test_termination_and_hessian_parallel_uses_broadcast_shutdown(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    manager.channels = _FakeChannels(
        [
            np.eye(2, dtype=np.float64),
            2.0 * np.eye(2, dtype=np.float64),
        ]
    )
    manager.workers = [_FakeWorker(), _FakeWorker()]  # type: ignore[assignment]
    manager.validation_workers = []
    manager.validation_parent_conns = []
    manager.validation_channels = None

    total = manager.termination_and_hessian(2, parallelism=True)

    np.testing.assert_allclose(total, 3.0 * np.eye(2, dtype=np.float64))
    assert manager.channels.sent == [(None, None)]
    assert [worker.join_calls for worker in manager.workers] == [1, 1]


def test_terminate_workers_kills_training_and_validation_workers(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    training = [_FakeWorker(), _FakeWorker()]
    validation = [_FakeWorker()]
    manager.workers = training  # type: ignore[assignment]
    manager.validation_workers = validation  # type: ignore[assignment]
    manager.parent_conns = []
    manager.validation_parent_conns = []
    manager.channels = None
    manager.validation_channels = None

    manager.terminate_workers()

    for worker in (*training, *validation):
        assert worker.terminate_calls == 1
        assert worker.join_calls == 1


def test_stop_validation_workers_does_not_wait_for_final_payload(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    worker = _FakeWorker()
    conn = _ConnThatMustNotPoll()
    manager.validation_channels = None
    manager.validation_workers = [worker]  # type: ignore[list-item]
    manager.validation_parent_conns = [conn]  # type: ignore[list-item]

    manager._stop_validation_workers()

    assert conn.sent == (None, None)
    assert worker.join_calls == 1


def test_termination_and_hessian_serial_stops_workers_one_by_one(tmp_path: Path) -> None:
    parent_a, child_a = multiprocessing.Pipe()
    parent_b, child_b = multiprocessing.Pipe()
    _start_pipe_worker(child_a, [np.eye(2, dtype=np.float64)])
    _start_pipe_worker(child_b, [2.0 * np.eye(2, dtype=np.float64)])

    manager = _make_manager(tmp_path)
    manager.parent_conns = [parent_a, parent_b]
    manager.workers = [_FakeWorker(), _FakeWorker()]  # type: ignore[assignment]
    manager.validation_workers = []
    manager.validation_parent_conns = []
    manager.validation_channels = None

    total = manager.termination_and_hessian(2, parallelism=False)

    np.testing.assert_allclose(total, 3.0 * np.eye(2, dtype=np.float64))
    assert [worker.join_calls for worker in manager.workers] == [1, 1]


def test_termination_and_hessian_disables_hessian_before_shutdown(tmp_path: Path) -> None:
    parent_conn, child_conn = multiprocessing.Pipe()
    # Two messages: ack for set_hessian_mode, then the hessian on termination
    _start_pipe_worker(child_conn, [{"worker_id": 0, "status": "ok"}, np.zeros((2, 2), dtype=np.float64)])

    manager = _make_manager(tmp_path)
    manager.parent_conns = [parent_conn]
    manager.workers = [_FakeWorker()]  # type: ignore[assignment]
    manager.validation_workers = []
    manager.validation_parent_conns = []
    manager.validation_channels = None

    total = manager.termination_and_hessian(2, estimate_hessian=False, parallelism=False)

    np.testing.assert_allclose(total, np.zeros((2, 2), dtype=np.float64))
    assert manager.workers[0].join_calls == 1


def test_termination_and_hessian_batched_limits_concurrent_shutdowns(tmp_path: Path) -> None:
    pairs = [multiprocessing.Pipe() for _ in range(3)]
    hessians = [float(i + 1) * np.eye(2, dtype=np.float64) for i in range(3)]
    for (_, child), hessian in zip(pairs, hessians):
        _start_pipe_worker(child, [hessian])

    manager = _make_manager(tmp_path)
    manager.parent_conns = [parent for parent, _ in pairs]
    manager.workers = [_FakeWorker(), _FakeWorker(), _FakeWorker()]  # type: ignore[assignment]
    manager.validation_workers = []
    manager.validation_parent_conns = []
    manager.validation_channels = None

    total = manager.termination_and_hessian(2, parallelism=2)

    np.testing.assert_allclose(total, 6.0 * np.eye(2, dtype=np.float64))
    assert [worker.join_calls for worker in manager.workers] == [1, 1, 1]
