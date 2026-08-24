"""Worker processes for distributed tracking and optimisation workloads.

This package provides worker process implementations for parallel computation
of accelerator physics simulations and optimisations. Workers communicate with
the main process via pipes and compute gradients and loss functions.

Available Workers:
    - TrackingWorker: Particle tracking (supports 'multi-turn' and 'arc-by-arc' modes)
    - PositionOnlyTrackingWorker: Position-only tracking (no momentum)
    - ClosedTwissWorker: Parametric closed-twiss matching (orbit, beta, phase,
      dispersion) via twiss/cofind and normal-form optical functions

Data Structures:
    - TrackingData: Input data for tracking workers
    - ClosedTwissData: Measured closed-twiss observables for closed-twiss workers
    - Observable: One measured observable family and its per-point variances
    - WorkerConfig: Configuration for all worker types
"""

from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.closed_twiss import ClosedTwissWorker
from aba_optimiser.workers.common import (
    ClosedTwissData,
    Observable,
    ObservableKind,
    PrecomputedTrackingWeights,
    TrackingData,
    WeightProcessor,
    WorkerConfig,
)
from aba_optimiser.workers.tracking import TrackingWorker
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker
from aba_optimiser.workers.tracking_validation import (
    PositionOnlyValidationTrackingWorker,
    ValidationTrackingWorker,
)

__all__ = [
    # Abstract base
    "AbstractWorker",
    # Worker implementations
    "TrackingWorker",
    "PositionOnlyTrackingWorker",
    "ValidationTrackingWorker",
    "PositionOnlyValidationTrackingWorker",
    "ClosedTwissWorker",
    # Data structures
    "TrackingData",
    "ClosedTwissData",
    "Observable",
    "ObservableKind",
    "WorkerConfig",
    "PrecomputedTrackingWeights",
    # Utilities
    "WeightProcessor",
]
