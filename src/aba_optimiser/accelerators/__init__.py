"""Accelerator abstraction for encapsulating machine-specific parameters.

This module provides base classes and implementations for different accelerators,
reducing parameter passing and making it easy to add new machines.
"""

from aba_optimiser.accelerators.base import Accelerator, KnobSpec
from aba_optimiser.accelerators.lhc import LHC
from aba_optimiser.accelerators.psb import PSB
from aba_optimiser.accelerators.sps import SPS

__all__ = ["Accelerator", "KnobSpec", "LHC", "PSB", "SPS"]
