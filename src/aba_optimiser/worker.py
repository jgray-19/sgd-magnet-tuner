from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.arc_by_arc_worker import ArcByArcWorker
from aba_optimiser.config import RUN_ARC_BY_ARC
from aba_optimiser.ring_worker import RingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.base_worker import BaseWorker


# Utility functions from original worker
def fold_blocks_to_per_bpm(x: np.ndarray, nbpms: int, nblocks: int) -> np.ndarray:
    """
    Reshape the last dim (nbpms*nblocks) into (nblocks, nbpms).
    Works on arrays with any leading dims; only the last axis is touched.
    """
    x = np.asarray(x)
    *lead, last = x.shape
    assert last == nbpms * nblocks, f"expected last dim {nbpms * nblocks}, got {last}"
    return x.reshape(*lead, nblocks, nbpms)  # (..., nblocks, nbpms)


def reorder_bpm_last(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply BPM permutation on the last axis."""
    return np.take(x, perm, axis=-1)


# Backward compatibility: provide the original Worker class
# This selects the appropriate worker based on configuration
def build_worker(
    conn: Connection,
    worker_id: int,
    x_comparisons: np.ndarray,
    y_comparisons: np.ndarray,
    init_coords: list[list[float]],
    start_bpm: str,
) -> BaseWorker:
    """
    Factory function that returns the appropriate worker type based on configuration.

    Args:
        conn: Connection for inter-process communication
        worker_id: Unique identifier for this worker
    x_comparisons: Array of shape (num_particles, n_data_points)
    y_comparisons: Array of shape (num_particles, n_data_points)
    init_coords: List of initial coordinates for each particle
        start_bpm: Starting BPM name

    Returns:
        Worker instance of appropriate type
    """
    if RUN_ARC_BY_ARC:
        return ArcByArcWorker(
            conn, worker_id, x_comparisons, y_comparisons, init_coords, start_bpm
        )
    return RingWorker(
        conn, worker_id, x_comparisons, y_comparisons, init_coords, start_bpm
    )


# Export the individual worker classes and utility functions for explicit use
__all__ = [
    "build_worker",
]
