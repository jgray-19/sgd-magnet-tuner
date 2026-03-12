"""Compatibility wrappers for single-plane tracking workers."""

from __future__ import annotations

from aba_optimiser.workers.tracking import TrackingWorker
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker


class SinglePlaneTrackingWorker(TrackingWorker):
    """Backward-compatible alias for a single-plane momentum worker."""


class SinglePlanePositionOnlyTrackingWorker(PositionOnlyTrackingWorker):
    """Backward-compatible alias for a single-plane position-only worker."""
