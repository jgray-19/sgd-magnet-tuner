from __future__ import annotations

import numpy as np
import pandas as pd

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.worker_manager import WorkerManager
from aba_optimiser.workers.tracking import TrackingWorker
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker


def _make_sps(tmp_path) -> SPS:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    return SPS(sequence_file=seq_file, beam_energy=450.0)


def _make_manager(tmp_path, *, file_kick_planes: dict[int, str] | None = None) -> WorkerManager:
    manager = WorkerManager(
        n_data_points={("BPH.13208", "BPV.13108"): 3, ("BPV.13308", "BPH.13208"): 3},
        ybpm="BPV.13308",
        magnet_range="$start/$end",
        fixed_start="BPH.13208",
        fixed_end="BPV.20108",
        accelerator=_make_sps(tmp_path),
        corrector_strengths_files=[tmp_path / "correctors.tfs"],
        tune_knobs_files=[tmp_path / "tune_knobs.txt"],
        all_bpms=["BPV.13108", "BPH.13208", "BPV.13308"],
        file_kick_planes=file_kick_planes,
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    return manager


def _make_track_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn": [1, 1, 1],
            "name": ["BPV.13108", "BPH.13208", "BPV.13308"],
            "x": [0.0, 1.0, 0.0],
            "y": [3.0, 0.0, 4.0],
            "px": [0.0, 0.1, 0.0],
            "py": [0.3, 0.0, 0.4],
            "var_x": [np.inf, 1.0, np.inf],
            "var_y": [1.0, np.inf, 1.0],
            "var_px": [np.inf, 1.0, np.inf],
            "var_py": [1.0, np.inf, 1.0],
        }
    ).set_index(["turn", "name"])


def test_infer_kick_plane_classifies_single_and_dual_plane_files() -> None:
    x_df = pd.DataFrame(
        {
            "x": [0.0, 3.0, -2.0],
            "px": [0.0, 0.2, -0.1],
            "y": [1.0, 1.0, 1.0],
            "py": [0.5, 0.5, 0.5],
        }
    )
    y_df = pd.DataFrame(
        {
            "x": [2.0, 2.0, 2.0],
            "px": [0.1, 0.1, 0.1],
            "y": [0.0, 4.0, -3.0],
            "py": [0.0, 0.3, -0.2],
        }
    )
    xy_df = pd.DataFrame(
        {
            "x": [0.0, 3.0, -2.0],
            "px": [0.0, 0.2, -0.1],
            "y": [0.0, 4.0, -3.0],
            "py": [0.0, 0.3, -0.2],
        }
    )

    assert DataManager.infer_kick_plane(x_df) == "x"
    assert DataManager.infer_kick_plane(y_df) == "y"
    assert DataManager.infer_kick_plane(xy_df) == "xy"


def test_create_worker_payloads_skips_single_plane_file_for_mismatched_start_plane(tmp_path) -> None:
    manager = WorkerManager(
        n_data_points={("BPH.13208", "BPV.13108"): 2},
        ybpm="BPV.13108",
        magnet_range="$start/$end",
        fixed_start="BPH.13208",
        fixed_end="BPV.13108",
        accelerator=_make_sps(tmp_path),
        corrector_strengths_files=[tmp_path / "correctors.tfs"],
        tune_knobs_files=[tmp_path / "tune_knobs.txt"],
        all_bpms=["BPV.13108", "BPH.13208"],
        file_kick_planes={0: "x"},
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    payloads = manager.create_worker_payloads(
        track_data={
            0: pd.DataFrame(
                {
                    "turn": [1, 1, 2, 2, 3, 3],
                    "name": ["BPV.13108", "BPH.13208"] * 3,
                    "x": [0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
                    "y": [3.0, 0.0, 4.0, 0.0, 5.0, 0.0],
                    "px": [0.0, 0.1, 0.0, 0.2, 0.0, 0.3],
                    "py": [0.3, 0.0, 0.4, 0.0, 0.5, 0.0],
                    "var_x": [np.inf, 1.0, np.inf, 1.0, np.inf, 1.0],
                    "var_y": [1.0, np.inf, 1.0, np.inf, 1.0, np.inf],
                    "var_px": [np.inf, 1.0, np.inf, 1.0, np.inf, 1.0],
                    "var_py": [1.0, np.inf, 1.0, np.inf, 1.0, np.inf],
                }
            ).set_index(["turn", "name"])
        },
        turn_batches=[[2]],
        file_turn_map={1: 0, 2: 0, 3: 0},
        start_bpms=["BPH.13208"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=2,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        machine_deltaps=[0.0],
    )

    assert [(config.start_bpm, config.kick_plane) for _, config, _ in payloads] == [
        ("BPH.13208", "x"),
    ]


def test_create_worker_payloads_keeps_single_plane_file_for_dual_plane_bpm(tmp_path) -> None:
    manager = WorkerManager(
        n_data_points={("BPH.13208", "BPV.13108"): 2},
        ybpm="BPV.13108",
        magnet_range="$start/$end",
        fixed_start="BPH.13208",
        fixed_end="BPV.13108",
        accelerator=_make_sps(tmp_path),
        corrector_strengths_files=[tmp_path / "correctors.tfs"],
        tune_knobs_files=[tmp_path / "tune_knobs.txt"],
        all_bpms=["BPV.13108", "BPH.13208"],
        file_kick_planes={0: "x"},
    )
    manager.accelerator.infer_monitor_plane = lambda bpm: "HV"  # type: ignore[method-assign]
    payloads = manager.create_worker_payloads(
        track_data={
            0: pd.DataFrame(
                {
                    "turn": [1, 1, 2, 2, 3, 3],
                    "name": ["BPV.13108", "BPH.13208"] * 3,
                    "x": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                    "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "px": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                    "py": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "var_x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "var_y": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "var_px": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    "var_py": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                }
            ).set_index(["turn", "name"])
        },
        turn_batches=[[2]],
        file_turn_map={1: 0, 2: 0, 3: 0},
        start_bpms=["BPH.13208"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=2,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        machine_deltaps=[0.0],
    )

    assert [(config.start_bpm, config.kick_plane) for _, config, _ in payloads] == [
        ("BPH.13208", "x"),
        ("BPH.13208", "x"),
    ]


def test_select_worker_class_reuses_generic_tracking_workers(tmp_path) -> None:
    manager = _make_manager(tmp_path)

    assert manager._select_worker_class("xy", True) is TrackingWorker
    assert manager._select_worker_class("xy", False) is PositionOnlyTrackingWorker
    assert manager._select_worker_class("x", True) is TrackingWorker
    assert manager._select_worker_class("y", False) is PositionOnlyTrackingWorker
