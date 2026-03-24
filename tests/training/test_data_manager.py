from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.data_manager import (
    DataManager,
    _boundary_turns_for_track,
    _distribute_target_batches_by_file,
    _get_range_spec_plan,
    _group_turns_by_file,
)


def _make_track_df(turns: list[int], bpm_name: str = "BPM.1") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn": turns,
            "name": [bpm_name] * len(turns),
            "x": [0.0] * len(turns),
            "px": [0.0] * len(turns),
            "y": [0.0] * len(turns),
            "py": [0.0] * len(turns),
            "var_x": [1.0] * len(turns),
            "var_y": [1.0] * len(turns),
            "var_px": [1.0] * len(turns),
            "var_py": [1.0] * len(turns),
        }
    ).set_index(["turn", "name"])


def test_prepare_turn_batches_treats_tracks_per_worker_as_a_max(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "aba_optimiser.training.data_manager.random.shuffle",
        lambda turns: None,
    )

    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            tracks_per_worker=4,
            num_workers=6,
            num_batches=2,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        measurement_files=["file0.parquet"],
        num_bunches=1,
        flattop_turns=12,
    )
    data_manager.track_data = {0: _make_track_df(list(range(12)))}
    data_manager.available_turns = list(range(12))
    data_manager.file_map = dict.fromkeys(range(12), 0)

    data_manager.prepare_turn_batches(
        SimpleNamespace(
            start_bpms=["BPM.1"],
            end_bpms=[],
        )  # ty:ignore[invalid-argument-type]
    )

    assert sorted(len(batch) for batch in data_manager.turn_batches) == [2, 4, 4]
    assert len(data_manager.turn_batches) == 3
    assert all(len(batch) <= 4 for batch in data_manager.turn_batches)


def test_prepare_turn_batches_keeps_partial_batches_per_file(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "aba_optimiser.training.data_manager.random.shuffle",
        lambda turns: None,
    )

    file0_turns = list(range(7))
    file1_turns = list(range(100, 107))
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            tracks_per_worker=4,
            num_workers=8,
            num_batches=2,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        measurement_files=["file0.parquet", "file1.parquet"],
        num_bunches=1,
        flattop_turns=7,
    )
    data_manager.track_data = {
        0: _make_track_df(file0_turns),
        1: _make_track_df(file1_turns),
    }
    data_manager.available_turns = file0_turns + file1_turns
    data_manager.file_map = dict.fromkeys(file0_turns, 0) | dict.fromkeys(file1_turns, 1)

    data_manager.prepare_turn_batches(
        SimpleNamespace(
            start_bpms=["BPM.1"],
            end_bpms=[],
        )  # ty:ignore[invalid-argument-type]
    )

    assert sorted(len(batch) for batch in data_manager.turn_batches) == [1, 1, 4, 4]
    assert [data_manager.file_map[batch[0]] for batch in data_manager.turn_batches] == [
        0,
        1,
        0,
        1,
    ]
    assert all(
        len({data_manager.file_map[turn] for turn in batch}) == 1
        for batch in data_manager.turn_batches
    )


def test_prepare_turn_batches_respects_worker_cap_when_capacity_is_higher(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "aba_optimiser.training.data_manager.random.shuffle",
        lambda turns: None,
    )

    total_turns = 302
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            tracks_per_worker=5,
            num_workers=60,
            num_batches=2,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
        measurement_files=["file0.parquet"],
        num_bunches=1,
        flattop_turns=total_turns,
    )
    data_manager.track_data = {0: _make_track_df(list(range(total_turns)))}
    data_manager.available_turns = list(range(total_turns))
    data_manager.file_map = dict.fromkeys(range(total_turns), 0)

    data_manager.prepare_turn_batches(
        SimpleNamespace(
            start_bpms=["BPM.1"],
            end_bpms=[],
        )  # ty:ignore[invalid-argument-type]
    )

    assert len(data_manager.turn_batches) == 30
    assert all(len(batch) <= 5 for batch in data_manager.turn_batches)
    assert len(data_manager.turn_batches) * 2 == 60




def test_prepare_turn_batches_honours_explicit_num_batches_floor(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "aba_optimiser.training.data_manager.random.shuffle",
        lambda turns: None,
    )

    total_turns = 12
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            tracks_per_worker=5,
            num_workers=1,
            num_batches=3,
            run_arc_by_arc=True,
            use_fixed_bpm=True,
            n_run_turns=1,
        ),
        measurement_files=["file0.parquet"],
        num_bunches=1,
        flattop_turns=total_turns,
    )
    data_manager.track_data = {0: _make_track_df(list(range(total_turns)))}
    data_manager.available_turns = list(range(total_turns))
    data_manager.file_map = dict.fromkeys(range(total_turns), 0)

    data_manager.prepare_turn_batches(
        SimpleNamespace(
            start_bpms=["BPM.1", "BPM.2"],
            end_bpms=["BPM.3", "BPM.4"],
        )  # ty:ignore[invalid-argument-type]
    )

    assert len(data_manager.turn_batches) == 3
    assert sorted(len(batch) for batch in data_manager.turn_batches) == [3, 3, 4]

def test_get_total_turns_uses_real_batch_sizes() -> None:
    data_manager = DataManager(
        bpms_in_range=["BPM.1"],
        all_bpms=["BPM.1"],
        simulation_config=SimulationConfig(
            tracks_per_worker=5,
            num_workers=2,
            num_batches=2,
        ),
        measurement_files=["file0.parquet"],
        num_bunches=1,
        flattop_turns=10,
    )
    data_manager.turn_batches = [[1, 2, 3, 4, 5], [6, 7]]

    assert data_manager.get_total_turns() == 6


def test_get_range_spec_plan_modes() -> None:
    assert _get_range_spec_plan(
        run_arc_by_arc=False,
        use_fixed_bpm=False,
        num_starts=3,
        num_ends=5,
    ) == (6, "2 directions x 3 start BPMs")

    assert _get_range_spec_plan(
        run_arc_by_arc=True,
        use_fixed_bpm=True,
        num_starts=2,
        num_ends=4,
    ) == (6, "fixed pairs (2 starts + 4 ends)")

    assert _get_range_spec_plan(
        run_arc_by_arc=True,
        use_fixed_bpm=False,
        num_starts=2,
        num_ends=3,
    ) == (12, "2 directions x 2 starts x 3 ends")


def test_turn_capacity_counting_helpers() -> None:
    available_turns = [1, 2, 3, 10, 11]
    file_map = {1: 0, 2: 0, 3: 0, 10: 1, 11: 1}
    grouped = _group_turns_by_file(available_turns, file_map)

    assert grouped == {0: [1, 2, 3], 1: [10, 11]}
    assert sum((len(turns) + 2 - 1) // 2 for turns in grouped.values()) == 3


def test_distribute_target_batches_by_file_balances_when_needed() -> None:
    turns_by_file = {
        0: list(range(8)),
        1: list(range(100, 108)),
    }
    targets, use_balanced, effective = _distribute_target_batches_by_file(
        turns_by_file,
        tracks_per_worker=5,
        num_batches=6,
    )

    assert use_balanced is True
    assert effective == 6
    assert targets == {0: 3, 1: 3}


def test_boundary_turns_for_track_short_track() -> None:
    assert _boundary_turns_for_track([1, 2], margin=1) == [1, 2]


def test_boundary_turns_for_track_long_track() -> None:
    assert _boundary_turns_for_track([1, 2, 3, 4, 5], margin=1) == [1, 5]
