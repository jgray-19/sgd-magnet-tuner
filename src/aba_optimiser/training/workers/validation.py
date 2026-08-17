"""Validation payload selection policy for tracking workers.

This module keeps validation split heuristics separate from process orchestration
so that `WorkerManager` stays focused on worker lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import logging

    from aba_optimiser.workers import TrackingData, WorkerConfig


WorkerPayload: TypeAlias = tuple["TrackingData", "WorkerConfig", int]


def payload_track_count(payload: WorkerPayload) -> int:
    """Return number of tracked turns represented by one payload."""
    data, _config, _file_idx = payload
    return int(data.init_coords.shape[0])


def payload_range_points(payload: WorkerPayload) -> int:
    """Return number of BPM points (including repeated run turns) in one payload."""
    data, _config, _file_idx = payload
    return int(data.position_comparisons.shape[1])


@dataclass(frozen=True)
class ValidationSplitResult:
    """Result of pairing training payloads with held-out validation payloads.

    ``training_payloads`` and ``validation_payloads`` are built from disjoint turns
    (the validation turns were removed from training upstream in ``DataManager``),
    so validation loss is a genuine out-of-sample signal.
    """

    training_payloads: list[WorkerPayload]
    validation_payloads: list[WorkerPayload]


@dataclass(frozen=True)
class _ValidationGroup:
    """Grouped validation candidates for one file/range combination."""

    file_idx: int
    start_bpm: str
    end_bpm: str
    primary_indices: list[int]
    total_tracks: int
    max_range_points: int


def _validation_min_groups(num_groups: int) -> int:
    """Return the minimum number of distinct groups to cover in validation."""
    return max(0, num_groups)


def _spread_positions(length: int, count: int) -> list[int]:
    """Select deterministic positions spread across a sorted sequence."""
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]

    positions: list[int] = []
    for idx in range(count):
        pos = round(idx * (length - 1) / (count - 1))
        min_remaining = count - idx - 1
        max_pos = length - min_remaining - 1
        if positions and pos <= positions[-1]:
            pos = positions[-1] + 1
        positions.append(min(pos, max_pos))
    return positions


def _build_validation_groups(
    payloads: list[WorkerPayload],
    track_counts: dict[int, int],
    range_points: dict[int, int],
) -> list[_ValidationGroup]:
    """Collapse payloads into file/range groups while preserving direction pairing."""
    grouped: dict[tuple[int, str, str], dict[int, list[int]]] = {}

    for idx, payload in enumerate(payloads):
        _data, config, file_idx = payload
        key = (file_idx, config.tracking_start_bpm, config.tracking_end_bpm)
        grouped.setdefault(key, {}).setdefault(int(config.sdir), []).append(idx)

    groups: list[_ValidationGroup] = []
    for file_idx, start_bpm, end_bpm in sorted(grouped):
        per_dir = grouped[(file_idx, start_bpm, end_bpm)]
        primary_indices: list[int] = []

        for sdir in (1, -1):
            dir_indices = sorted(
                per_dir.get(sdir, []),
                key=lambda idx: (track_counts[idx], range_points[idx]),
                reverse=True,
            )
            if dir_indices:
                # One representative payload per direction is enough for coverage.
                primary_indices.append(dir_indices[0])

        if not primary_indices:
            continue

        group_indices = [idx for indices in per_dir.values() for idx in indices]
        groups.append(
            _ValidationGroup(
                file_idx=file_idx,
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                primary_indices=primary_indices,
                total_tracks=sum(track_counts[idx] for idx in group_indices),
                max_range_points=max(range_points[idx] for idx in group_indices),
            )
        )

    groups.sort(
        key=lambda group: (
            group.max_range_points,
            group.total_tracks,
            group.file_idx,
            group.start_bpm,
            group.end_bpm,
        ),
        reverse=True,
    )
    return groups


def split_validation_payloads(
    training_payloads: list[WorkerPayload],
    validation_candidates: list[WorkerPayload],
    logger: logging.Logger | None = None,
) -> ValidationSplitResult:
    """Retain all held-out validation candidates.

    ``validation_candidates`` are built from turns that were removed from training
    upstream, so every candidate is genuinely out-of-sample. Validation must cover
    every held-out file/range/direction/plane candidate; otherwise a small ACD
    run can silently skip one momentum file and report a non-representative loss.

    ``training_payloads`` is returned unchanged.
    """
    if not training_payloads:
        raise ValueError("No training worker payloads were created")
    if not validation_candidates:
        return ValidationSplitResult(training_payloads, [])

    range_points = {
        idx: payload_range_points(payload) for idx, payload in enumerate(validation_candidates)
    }
    track_counts = {
        idx: payload_track_count(payload) for idx, payload in enumerate(validation_candidates)
    }
    groups = _build_validation_groups(validation_candidates, track_counts, range_points)
    selected_indices = list(range(len(validation_candidates)))
    validation_payloads = list(validation_candidates)

    if logger is not None:
        selected_ranges = {
            (
                validation_candidates[idx][2],
                validation_candidates[idx][1].tracking_start_bpm,
                validation_candidates[idx][1].tracking_end_bpm,
            )
            for idx in selected_indices
        }
        logger.info(
            "Validation selection: candidates=%d, selected=%d payloads, tracks=%d, "
            "covered_ranges=%d/%d",
            len(validation_candidates),
            len(selected_indices),
            sum(track_counts[idx] for idx in selected_indices),
            len(selected_ranges),
            len(groups),
        )

    return ValidationSplitResult(
        training_payloads=training_payloads,
        validation_payloads=validation_payloads,
    )
