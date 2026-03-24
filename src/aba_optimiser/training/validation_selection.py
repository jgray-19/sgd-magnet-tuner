"""Validation payload selection policy for tracking workers.

This module keeps validation split heuristics separate from process orchestration
so that `WorkerManager` stays focused on worker lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
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


def _validation_target_tracks(payloads: list[WorkerPayload]) -> int:
    """Minimum validation tracks such that val >= 10% of training tracks."""
    total_tracks = sum(payload_track_count(payload) for payload in payloads)
    # val >= 0.1 * (total - val)  ->  val >= total / 11
    return max(1, ceil(total_tracks / 11.0))


@dataclass(frozen=True)
class ValidationSplitResult:
    """Result of selecting validation payloads from worker payloads."""

    training_payloads: list[WorkerPayload]
    validation_payloads: list[WorkerPayload]
    duplicated_validation_payload: bool


@dataclass(frozen=True)
class _ValidationGroup:
    """Grouped validation candidates for one file/range combination."""

    file_idx: int
    start_bpm: str
    end_bpm: str
    primary_indices: list[int]
    extra_indices: list[int]
    total_tracks: int
    max_range_points: int


def _validation_min_groups(num_groups: int) -> int:
    """Return the minimum number of distinct groups to cover in validation."""
    if num_groups <= 1:
        return 1
    return min(num_groups, max(2, ceil(sqrt(num_groups))))


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
        key = (file_idx, config.start_bpm, config.end_bpm)
        grouped.setdefault(key, {}).setdefault(int(config.sdir), []).append(idx)

    groups: list[_ValidationGroup] = []
    for file_idx, start_bpm, end_bpm in sorted(grouped):
        per_dir = grouped[(file_idx, start_bpm, end_bpm)]
        primary_indices: list[int] = []
        extra_indices: list[int] = []

        for sdir in (1, -1):
            dir_indices = sorted(
                per_dir.get(sdir, []),
                key=lambda idx: (track_counts[idx], range_points[idx]),
                reverse=True,
            )
            if dir_indices:
                primary_indices.append(dir_indices[0])
                extra_indices.extend(dir_indices[1:])

        if not primary_indices:
            continue

        group_indices = [idx for indices in per_dir.values() for idx in indices]
        groups.append(
            _ValidationGroup(
                file_idx=file_idx,
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                primary_indices=primary_indices,
                extra_indices=sorted(
                    extra_indices,
                    key=lambda idx: (track_counts[idx], range_points[idx]),
                    reverse=True,
                ),
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
    payloads: list[WorkerPayload],
    logger: logging.Logger | None = None,
) -> ValidationSplitResult:
    """Select validation payloads to cover the training distribution.

    Selection rules:
    - group payloads by file plus BPM range so opposite directions stay paired
    - include both `sdir=+1` and `sdir=-1` for a selected range when available
    - cover multiple ranges, not just the single largest one
    - keep adding payloads until validation reaches at least 10% of training tracks
    - validation payloads are selected from training payloads (not held out)
    """
    if not payloads:
        raise ValueError("No worker payloads were created")
    if len(payloads) == 1:
        return ValidationSplitResult(payloads, [payloads[0]], True)

    range_points = {idx: payload_range_points(payload) for idx, payload in enumerate(payloads)}
    track_counts = {idx: payload_track_count(payload) for idx, payload in enumerate(payloads)}
    groups = _build_validation_groups(payloads, track_counts, range_points)

    target_tracks = _validation_target_tracks(payloads)
    min_groups = _validation_min_groups(len(groups))

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    selected_tracks = 0

    def add_payload(idx: int) -> None:
        nonlocal selected_tracks
        if idx in selected_set:
            return
        selected_indices.append(idx)
        selected_set.add(idx)
        selected_tracks += track_counts[idx]

    selected_group_positions = set(_spread_positions(len(groups), min_groups))

    for group_idx, group in enumerate(groups):
        if group_idx not in selected_group_positions:
            continue
        for idx in group.primary_indices:
            add_payload(idx)

    if selected_tracks < target_tracks:
        for group_idx, group in enumerate(groups):
            if group_idx in selected_group_positions:
                continue
            for idx in group.primary_indices:
                add_payload(idx)
            if selected_tracks >= target_tracks:
                break

    if selected_tracks < target_tracks:
        for group in groups:
            for idx in group.extra_indices:
                add_payload(idx)
                if selected_tracks >= target_tracks:
                    break
            if selected_tracks >= target_tracks:
                break

    if logger is not None:
        selected_ranges = {
            (payloads[idx][2], payloads[idx][1].start_bpm, payloads[idx][1].end_bpm)
            for idx in selected_indices
        }
        logger.info(
            "Validation selection: payloads=%d, tracks=%d (target>=%d), covered_ranges=%d/%d",
            len(selected_indices),
            selected_tracks,
            target_tracks,
            len(selected_ranges),
            len(groups),
        )

    validation_payloads = [payload for idx, payload in enumerate(payloads) if idx in selected_set]
    training_payloads = list(payloads)
    duplicated_validation_payload = len(payloads) == 1

    return ValidationSplitResult(
        training_payloads=training_payloads,
        validation_payloads=validation_payloads,
        duplicated_validation_payload=duplicated_validation_payload,
    )
