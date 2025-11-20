"""Worker processes for distributed tracking and optimisation workloads.

This package provides worker process implementations for parallel computation
of accelerator physics simulations and optimizations. Workers communicate with
the main process via pipes and compute gradients and loss functions.

Available Workers:
    - TrackingWorker: Particle tracking (supports 'multi-turn' and 'arc-by-arc' modes)
    - OpticsWorker: Optics function (beta, dispersion) computations

Data Structures:
    - TrackingData: Input data for tracking workers
    - OpticsData: Input data for optics workers
    - WorkerConfig: Configuration for all worker types
"""

from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import OpticsData, TrackingData, WeightProcessor, WorkerConfig
from aba_optimiser.workers.optics import OpticsWorker
from aba_optimiser.workers.tracking_worker import TrackingWorker

__all__ = [
    # Abstract base
    "AbstractWorker",
    # Worker implementations
    "TrackingWorker",
    "OpticsWorker",
    # Data structures
    "TrackingData",
    "OpticsData",
    "WorkerConfig",
    # Utilities
    "WeightProcessor",
]
