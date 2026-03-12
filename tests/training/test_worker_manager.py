from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.worker_manager import WorkerManager
from aba_optimiser.training.worker_setup import WorkerRuntimeMetadata

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


def _make_sps(tmp_path: Path) -> SPS:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    return SPS(sequence_file=seq_file, beam_energy=450.0)


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
        corrector_strengths_files=[tmp_path / "correctors.tfs"],
        tune_knobs_files=[tmp_path / "tune_knobs.txt"],
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


def test_create_worker_payloads_multi_turn_creates_forward_and_backward_workers(
    tmp_path: Path,
) -> None:
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPV.13108"): 3,
        },
        all_bpms=["BPV.13108", "BPH.13208", "BPV.13308"],
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    df = pd.DataFrame(
        {
            "turn": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "name": [
                "BPV.13108",
                "BPH.13208",
                "BPV.13308",
                "BPV.13108",
                "BPH.13208",
                "BPV.13308",
                "BPV.13108",
                "BPH.13208",
                "BPV.13308",
            ],
            "x": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            "y": [3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 7.0, 0.0, 8.0],
            "px": [0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0],
            "py": [0.3, 0.0, 0.4, 0.5, 0.0, 0.6, 0.7, 0.0, 0.8],
            "var_x": [np.inf, 1.0, np.inf, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf],
            "var_y": [1.0, np.inf, 1.0, 1.0, np.inf, 1.0, 1.0, np.inf, 1.0],
            "var_px": [np.inf, 1.0, np.inf, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf],
            "var_py": [1.0, np.inf, 1.0, 1.0, np.inf, 1.0, 1.0, np.inf, 1.0],
        }
    ).set_index(["turn", "name"])
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

    assert len(payloads) == 2
    assert [
        (config.start_bpm, config.end_bpm, config.sdir, config.kick_plane)
        for _, config, _ in payloads
    ] == [
        ("BPH.13208", "BPV.13108", 1, "x"),
        ("BPH.13208", "BPV.13108", -1, "y"),
    ]

    forward_data = payloads[0][0]
    backward_data = payloads[1][0]
    assert np.isfinite(forward_data.position_variances[0, :, 0]).any()
    assert not np.isfinite(forward_data.position_variances[0, :, 1]).any()
    assert not np.isfinite(backward_data.position_variances[0, :, 0]).any()
    assert np.isfinite(backward_data.position_variances[0, :, 1]).any()


def test_create_worker_payloads_multi_turn_supports_mixed_start_planes(tmp_path: Path) -> None:
    all_bpms = ["BPV.13108", "BPH.13208", "BPV.13308"]
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPV.13108"): 3,
            ("BPV.13308", "BPH.13208"): 3,
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

    assert len(payloads) == 4

    payload_by_key = {
        (config.start_bpm, config.end_bpm, config.sdir): (data, config)
        for data, config, _ in payloads
    }
    forward_h, forward_h_config = payload_by_key[("BPH.13208", "BPV.13108", 1)]
    backward_h, backward_h_config = payload_by_key[("BPH.13208", "BPV.13108", -1)]
    forward_v, forward_v_config = payload_by_key[("BPV.13308", "BPH.13208", 1)]
    backward_v, backward_v_config = payload_by_key[("BPV.13308", "BPH.13208", -1)]

    assert forward_h_config.kick_plane == "x"
    assert backward_h_config.kick_plane == "y"
    assert forward_v_config.kick_plane == "y"
    assert backward_v_config.kick_plane == "x"
    assert np.isfinite(forward_h.position_variances[0, :, 0]).any()
    assert not np.isfinite(forward_h.position_variances[0, :, 1]).any()
    assert not np.isfinite(backward_h.position_variances[0, :, 0]).any()
    assert np.isfinite(backward_h.position_variances[0, :, 1]).any()
    assert not np.isfinite(forward_v.position_variances[0, :, 0]).any()
    assert np.isfinite(forward_v.position_variances[0, :, 1]).any()
    assert np.isfinite(backward_v.position_variances[0, :, 0]).any()
    assert not np.isfinite(backward_v.position_variances[0, :, 1]).any()


def test_create_worker_payloads_arc_by_arc_uses_configured_fixed_pairs(tmp_path: Path) -> None:
    all_bpms = ["BPH.13208", "BPV.13308", "BPV.20108"]
    manager = _make_manager(
        tmp_path,
        n_data_points={
            ("BPH.13208", "BPV.20108"): 3,
            ("BPH.13208", "BPV.13308"): 2,
        },
        all_bpms=all_bpms,
    )
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
        end_bpms=["BPV.13308"],
        simulation_config=simulation_config,
        machine_deltaps=[0.0],
    )

    assert [
        (config.start_bpm, config.end_bpm, config.sdir, config.kick_plane)
        for _, config, _ in payloads
    ] == [
        ("BPH.13208", "BPV.20108", 1, "x"),
        ("BPH.13208", "BPV.13308", -1, "y"),
    ]


def test_create_worker_payloads_assigns_per_file_artifacts_from_file_turn_map(tmp_path: Path) -> None:
    manager = _make_manager(
        tmp_path,
        n_data_points={("BPH.13208", "BPV.13108"): 3},
        all_bpms=["BPV.13108", "BPH.13208", "BPV.13308"],
    )
    manager.corrector_strengths_files = [tmp_path / "corr0.tfs", tmp_path / "corr1.tfs"]
    manager.tune_knobs_files = [tmp_path / "knobs0.txt", tmp_path / "knobs1.txt"]

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
    assert [config.corrector_strengths for _, config, _ in payloads] == [
        tmp_path / "corr0.tfs",
        tmp_path / "corr1.tfs",
        tmp_path / "corr0.tfs",
        tmp_path / "corr1.tfs",
    ]
    assert [config.tune_knobs_file for _, config, _ in payloads] == [
        tmp_path / "knobs0.txt",
        tmp_path / "knobs1.txt",
        tmp_path / "knobs0.txt",
        tmp_path / "knobs1.txt",
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
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane="xy",
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        )
    ]

    masks = manager._build_bpm_masks_from_diagnostics(
        diagnostics=[
            {
                "worker_id": 0,
                "loss_per_bpm": [1.0, 50.0, 1.0, 50.0],
            }
        ],
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
            start_bpm="BPH.13208",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane="xy",
            n_run_turns=2,
            bpm_names=["BPH.13208", "BPH.13608"],
        ),
        WorkerRuntimeMetadata(
            worker_id=1,
            start_bpm="BPV.13308",
            end_bpm="BPV.20108",
            sdir=1,
            kick_plane="xy",
            n_run_turns=1,
            bpm_names=["BPV.13308", "BPV.20108"],
        ),
    ]

    manager._apply_screening_actions(
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
