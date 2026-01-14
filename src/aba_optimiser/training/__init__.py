"""Training loops and utilities for magnet knob optimisation.

Coordinated loops orchestrate gradient evaluation, learning rate scheduling,
and checkpointing for the optimisation workflow.
"""

from aba_optimiser.training.base_controller import BaseController, LHCControllerMixin
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.controller import Controller, LHCController
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig
from aba_optimiser.training.controller_helpers import (
    create_arc_bpm_config,
    create_arc_measurement_config,
)
from aba_optimiser.training.data_manager import DataManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.training.scheduler import LRScheduler
from aba_optimiser.training.utils import (
    extract_bpm_range_names,
    filter_bad_bpms,
    find_common_bpms,
    load_tfs_files,
    normalize_true_strengths,
)
from aba_optimiser.training.worker_lifecycle import WorkerLifecycleManager
from aba_optimiser.training.worker_manager import WorkerManager

__all__ = [
    "BaseController",
    "LHCControllerMixin",
    "ConfigurationManager",
    "Controller",
    "LHCController",
    "BPMConfig",
    "MeasurementConfig",
    "SequenceConfig",
    "create_arc_bpm_config",
    "create_arc_measurement_config",
    "DataManager",
    "OptimisationLoop",
    "ResultManager",
    "LRScheduler",
    "WorkerLifecycleManager",
    "WorkerManager",
    "filter_bad_bpms",
    "normalize_true_strengths",
    "extract_bpm_range_names",
    "find_common_bpms",
    "load_tfs_files",
]
