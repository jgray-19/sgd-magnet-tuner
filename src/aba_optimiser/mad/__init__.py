"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- BaseMadInterface: Core functionality without automatic setup
- optimisationMadInterface: For accelerator optimisation workflows
"""

from .base_mad_interface import BaseMadInterface
from .optimising_mad_interface import OptimisationMadInterface

__all__ = [
    "BaseMadInterface",
    "OptimisationMadInterface",
]
