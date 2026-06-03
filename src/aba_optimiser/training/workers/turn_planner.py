"""Turn-batch planning for tracking workers."""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from aba_optimiser.training.config.tracking import TrackingPlan

    ShuffleTurns = Callable[[list[int]], None]

LOGGER = logging.getLogger(__name__)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ceil(numerator / denominator) for positive integers."""
    return (numerator + denominator - 1) // denominator


def group_turns_by_file(
    available_turns: list[int], file_map: dict[int, int]
) -> dict[int, list[int]]:
    """Group available turns by measurement file index."""
    turns_by_file: dict[int, list[int]] = {}
    for turn in available_turns:
        turns_by_file.setdefault(file_map[turn], []).append(turn)
    return turns_by_file


def _distribute_target_batches_by_file(
    turns_by_file: dict[int, list[int]],
    tracks_per_worker: int,
    num_turn_batches: int,
    per_worker_batches: int,
) -> tuple[dict[int, int], bool, int]:
    """Allocate target turn-batch counts per file for worker assignment."""
    file_ids = sorted(turns_by_file.keys())
    target_batches_by_file = dict.fromkeys(file_ids, 0)

    if per_worker_batches <= 0:
        raise ValueError(f"per_worker_batches must be positive, got {per_worker_batches}")
    if tracks_per_worker < per_worker_batches:
        raise ValueError(
            f"tracks_per_worker={tracks_per_worker} must be >= per_worker_batches={per_worker_batches}"
        )

    batches_to_assign = max(0, num_turn_batches)
    total_available_batches = sum(
        len(turns_by_file[file_idx]) // per_worker_batches for file_idx in file_ids
    )
    batches_to_assign = min(batches_to_assign, total_available_batches)

    while batches_to_assign > 0:
        assigned_in_round = False
        for file_idx in file_ids:
            if batches_to_assign == 0:
                break
            if (
                target_batches_by_file[file_idx]
                < len(turns_by_file[file_idx]) // per_worker_batches
            ):
                target_batches_by_file[file_idx] += 1
                batches_to_assign -= 1
                assigned_in_round = True
        if not assigned_in_round:
            break

    min_needed_per_file = {
        file_idx: _ceil_div(
            len(turns_by_file[file_idx]),
            (tracks_per_worker // per_worker_batches) * per_worker_batches,
        )
        for file_idx in file_ids
    }
    use_balanced_sizing = any(
        target_batches_by_file[file_idx] > min_needed_per_file[file_idx]
        for file_idx in file_ids
    )
    effective_num_batches = sum(target_batches_by_file.values())
    return target_batches_by_file, use_balanced_sizing, effective_num_batches


def _get_range_spec_plan(
    *,
    run_arc_by_arc: bool,
    use_fixed_bpm: bool,
    num_starts: int,
    num_ends: int,
) -> tuple[int, str]:
    """Return (range_specs_per_batch, description) for the given planning mode.

    Thin wrapper around ``TrackingPlan.range_specs_per_batch`` for use in tests
    and helper code that does not have a ``TrackingPlan`` instance.
    """
    from aba_optimiser.training.config.tracking import ArcByArcTrackingPlan

    return ArcByArcTrackingPlan().range_specs_per_batch(
        run_arc_by_arc=run_arc_by_arc,
        use_fixed_bpm=use_fixed_bpm,
        num_starts=num_starts,
        num_ends=num_ends,
    )


@dataclass(frozen=True)
class WorkerTurnBatchPlan:
    """Concrete turn-batch plan plus logging metadata."""

    turn_batches: list[list[int]]
    range_specs_per_batch: int
    range_specs_desc: str


class WorkerTurnPlanner:
    """Compute worker turn batches from available turns and tracking mode."""

    def __init__(
        self,
        tracking_plan: TrackingPlan,
        simulation_config,
        *,
        shuffle_turns: ShuffleTurns | None = None,
    ) -> None:
        """Create a planner.

        Args:
            tracking_plan: Tracking-mode policy for BPM/range expansion.
            simulation_config: Worker and batching configuration.
            shuffle_turns: Optional in-place turn ordering strategy. Defaults to
                ``random.shuffle`` and can be overridden for deterministic tests.
        """
        self.tracking_plan = tracking_plan
        self.simulation_config = simulation_config
        self.shuffle_turns = shuffle_turns if shuffle_turns is not None else random.shuffle

    def build_turn_batches(
        self,
        *,
        available_turns: list[int],
        file_map: dict[int, int],
        num_files: int,
        num_starts: int,
        num_ends: int,
    ) -> WorkerTurnBatchPlan:
        """Plan batches and log the resulting worker allocation."""
        turns_by_file = group_turns_by_file(available_turns, file_map)
        tracks_per_worker = self.simulation_config.tracks_per_worker
        num_workers = self.simulation_config.num_workers
        range_specs_per_batch, range_specs_desc = self.tracking_plan.range_specs_per_batch(
            run_arc_by_arc=self.simulation_config.run_arc_by_arc,
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            num_starts=num_starts,
            num_ends=num_ends,
        )
        worker_turn_batches = max(1, num_workers // max(1, range_specs_per_batch))
        max_batches_by_turn_capacity = sum(
            _ceil_div(len(turns), tracks_per_worker) for turns in turns_by_file.values()
        )
        num_turn_batches = min(worker_turn_batches, max_batches_by_turn_capacity)

        planned_workers = num_turn_batches * range_specs_per_batch
        limiting_factor = (
            "data capacity"
            if max_batches_by_turn_capacity < worker_turn_batches
            else "num_workers cap"
        )
        LOGGER.info(
            "Worker planning: requested=%d workers, range_specs_per_batch=%d (%s), starts=%d, ends=%d",
            num_workers,
            range_specs_per_batch,
            range_specs_desc,
            num_starts,
            num_ends,
        )
        LOGGER.info(
            "Turn-batch planning: "
            "num_workers_cap→%d batches, "
            "data_capacity→%d batches (%d files x up to ceil(turns/%d) each), "
            "selected=%d batches [limited by %s]",
            worker_turn_batches,
            max_batches_by_turn_capacity,
            num_files,
            tracks_per_worker,
            num_turn_batches,
            limiting_factor,
        )
        LOGGER.info(
            "Planned workers: %d turn batches x %d range specs = %d workers "
            "(num_batches=%d is MAD-internal sub-batching, does not affect worker count)",
            num_turn_batches,
            range_specs_per_batch,
            planned_workers,
            self.simulation_config.num_batches,
        )

        return WorkerTurnBatchPlan(
            turn_batches=self._materialise_turn_batches(
                turns_by_file,
                num_turn_batches,
                tracks_per_worker,
                per_worker_batches=1,
            ),
            range_specs_per_batch=range_specs_per_batch,
            range_specs_desc=range_specs_desc,
        )

    def _materialise_turn_batches(
        self,
        turns_by_file: dict[int, list[int]],
        num_turn_batches: int,
        tracks_per_worker: int,
        per_worker_batches: int,
    ) -> list[list[int]]:
        """Materialise worker turn batches from grouped turns."""
        for turns in turns_by_file.values():
            self.shuffle_turns(turns)

        target_batches_by_file, use_balanced_sizing, effective_num_batches = (
            _distribute_target_batches_by_file(
                turns_by_file,
                tracks_per_worker,
                num_turn_batches,
                per_worker_batches,
            )
        )
        if effective_num_batches != num_turn_batches:
            LOGGER.warning(
                "Could only allocate %d/%d worker turn batches across files while keeping >=1 turn per batch",
                effective_num_batches,
                num_turn_batches,
            )
            num_turn_batches = effective_num_batches

        turn_batches: list[list[int]] = []
        file_queue = deque(sorted(target_batches_by_file.keys()))

        for _ in range(num_turn_batches):
            if not file_queue:
                LOGGER.warning(
                    "Only created %d/%d worker turn batches",
                    len(turn_batches),
                    num_turn_batches,
                )
                break

            file_idx: int | None = None
            for _ in range(len(file_queue)):
                candidate = file_queue.popleft()
                if turns_by_file.get(candidate) and target_batches_by_file[candidate] > 0:
                    file_idx = candidate
                    break

            if file_idx is None:
                LOGGER.warning(
                    "Only created %d/%d worker turn batches",
                    len(turn_batches),
                    num_turn_batches,
                )
                break

            turns_left = len(turns_by_file[file_idx])
            batches_left_for_file = target_batches_by_file[file_idx]
            max_turns_per_batch = (tracks_per_worker // per_worker_batches) * per_worker_batches
            if use_balanced_sizing:
                batch_size = per_worker_batches * _ceil_div(
                    _ceil_div(turns_left, per_worker_batches),
                    batches_left_for_file,
                )
                batch_size = min(batch_size, max_turns_per_batch)
                max_assignable = turns_left - per_worker_batches * (batches_left_for_file - 1)
                batch_size = min(
                    batch_size,
                    (max_assignable // per_worker_batches) * per_worker_batches,
                )
                batch_size = max(per_worker_batches, batch_size)
            else:
                batch_size = min(
                    max_turns_per_batch,
                    (turns_left // per_worker_batches) * per_worker_batches,
                )

            mad_num_batches = self.simulation_config.num_batches
            if batch_size >= mad_num_batches:
                batch_size = (batch_size // mad_num_batches) * mad_num_batches

            batch = turns_by_file[file_idx][:batch_size]
            turns_by_file[file_idx] = turns_by_file[file_idx][batch_size:]
            target_batches_by_file[file_idx] -= 1
            turn_batches.append(batch)

            if target_batches_by_file[file_idx] > 0 and turns_by_file[file_idx]:
                file_queue.append(file_idx)

        return turn_batches
