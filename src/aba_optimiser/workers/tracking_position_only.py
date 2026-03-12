"""Position-only tracking worker specialization."""

from __future__ import annotations

from aba_optimiser.workers.tracking import TrackingWorker


class PositionOnlyTrackingWorker(TrackingWorker):
    """Tracking worker that compares only x/y position observables."""

    observables = ("x", "y")
    include_momentum = False
    hessian_weight_order = ("x", "y")
