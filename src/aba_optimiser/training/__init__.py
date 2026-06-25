"""Training loops and utilities for magnet knob optimisation.

Coordinated loops orchestrate gradient evaluation, learning rate scheduling,
and checkpointing for the optimisation workflow.
"""

from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.config import (
    ArcByArcTrackingPlan,
    CheckpointConfig,
    ConfigurationManager,
    FullRingBpmTrackingPlan,
    KickerConfig,
    KickerTrackingPlan,
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
    TrackingPlan,
    WorkerRangeSpec,
    build_tracking_plan,
    create_arc_measurement_config,
)
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.optimisation.loop import OptimisationLoop
from aba_optimiser.training.optimisation.scheduler import LRScheduler
from aba_optimiser.training.utils import (
    bpm_supports_both_planes,
    bpm_supports_plane,
    extract_bpm_range_names,
    filter_bad_bpms,
    find_common_bpms,
    load_tfs_files,
    normalise_true_strengths,
)
from aba_optimiser.training.workers.lifecycle import WorkerLifecycleManager
from aba_optimiser.training.workers.manager import WorkerManager

__all__ = [
    "BaseController",
    "ArcByArcTrackingPlan",
    "CheckpointConfig",
    "ConfigurationManager",
    "Controller",
    "FullRingBpmTrackingPlan",
    "KickerConfig",
    "KickerTrackingPlan",
    "MeasurementConfig",
    "MeasurementDetails",
    "OutputConfig",
    "SequenceConfig",
    "TrackingPlan",
    "WorkerRangeSpec",
    "build_tracking_plan",
    "create_arc_measurement_config",
    "DataManager",
    "OptimisationLoop",
    "LRScheduler",
    "WorkerLifecycleManager",
    "WorkerManager",
    "bpm_supports_both_planes",
    "bpm_supports_plane",
    "filter_bad_bpms",
    "normalise_true_strengths",
    "extract_bpm_range_names",
    "find_common_bpms",
    "load_tfs_files",
]
