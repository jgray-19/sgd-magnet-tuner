"""Accelerator abstraction for encapsulating machine-specific parameters.

This module provides base classes and implementations for different accelerators,
reducing parameter passing and making it easy to add new machines.
"""

from aba_optimiser.accelerators.base import Accelerator
from aba_optimiser.accelerators.lhc import LHC

__all__ = ["Accelerator", "LHC"]
