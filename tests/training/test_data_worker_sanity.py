"""Fast sanity checks for data/worker utility logic."""

from __future__ import annotations

import numpy as np

from aba_optimiser.training.data_manager import _distribute_target_batches_by_file
from aba_optimiser.training.worker_payloads import WorkerPayloadBuilder


def test_distribute_target_batches_spreads_across_files() -> None:
    """Batch allocator should not starve files when budget allows."""
    turns_by_file = {
        0: list(range(20)),
        1: list(range(20, 40)),
    }

    target_batches_by_file, _use_balanced, effective = _distribute_target_batches_by_file(
        turns_by_file=turns_by_file,
        tracks_per_worker=10,
        num_batches=4,
    )

    assert effective == 4
    assert target_batches_by_file[0] >= 1
    assert target_batches_by_file[1] >= 1


def test_diagnostic_loss_per_bpm_reduces_multiturn() -> None:
    """Per-point losses should collapse to per-BPM sums over turns."""
    bpm_names = ["bpm1", "bpm2", "bpm3"]
    n_run_turns = 2
    # turn 0: [1,2,3], turn 1: [10,20,30]
    loss_per_point = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float64)

    reduced = WorkerPayloadBuilder.diagnostic_loss_per_bpm(
        loss_per_point=loss_per_point,
        bpm_names=bpm_names,
        n_run_turns=n_run_turns,
        worker_id=0,
    )

    assert np.allclose(reduced, np.array([11.0, 22.0, 33.0], dtype=np.float64))
