"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- CoreMadInterface: Shared minimal functionality from pymadng-utils
- AbaMadInterface: Repository-specific helper extensions
- GenericMadInterface: Generic setup functionality
- GradientDescentMadInterface: Abstract base for gradient descent optimization
- LHCOptimisationMadInterface: LHC-specific optimization interface
"""

from pymadng_utils.mad import CoreMadInterface

from .aba_mad_interface import (
    AbaMadInterface,
)
from .dispatch import get_mad_interface
from .lhc_optimising_interface import LHCOptimisationMadInterface
from .optimising_mad_interface import GenericMadInterface, GradientDescentMadInterface

__all__ = [
    "CoreMadInterface",
    "AbaMadInterface",
    "GenericMadInterface",
    "GradientDescentMadInterface",
    "LHCOptimisationMadInterface",
    "get_mad_interface",
]
