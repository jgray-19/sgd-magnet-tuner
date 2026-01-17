"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- BaseMadInterface: Core functionality without automatic setup
- optimisationMadInterface: For accelerator optimisation workflows
- TrackingMadInterface: Lightweight interface for tracking simulations
"""

from .base_mad_interface import BaseMadInterface
from .optimising_mad_interface import OptimisationMadInterface
from .tracking_interface import (
    TrackingMadInterface,
)

__all__ = [
    "BaseMadInterface",
    "OptimisationMadInterface",
    "TrackingMadInterface",
]
