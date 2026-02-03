"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- BaseMadInterface: Core functionality without automatic setup
- GenericMadInterface: Generic setup functionality
- GradientDescentMadInterface: Abstract base for gradient descent optimization
- LHCOptimisationMadInterface: LHC-specific optimization interface
"""

from .base_mad_interface import BaseMadInterface
from .dispatch import get_mad_interface
from .lhc_optimising_interface import LHCOptimisationMadInterface
from .optimising_mad_interface import GenericMadInterface, GradientDescentMadInterface

__all__ = [
    "BaseMadInterface",
    "GenericMadInterface",
    "GradientDescentMadInterface",
    "LHCOptimisationMadInterface",
    "get_mad_interface",
]
