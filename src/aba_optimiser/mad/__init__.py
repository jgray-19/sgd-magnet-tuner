"""
MAD-NG interface modules for accelerator simulation.

This package provides different interfaces for working with MAD-NG:
- BaseMadInterface: Core functionality without automatic setup
- OptimizationMadInterface: For accelerator optimization workflows
- TrackingMadInterface: Lightweight interface for tracking simulations
"""

from .base_mad_interface import BaseMadInterface
from .mad_interface import OptimizationMadInterface
from .tracking_interface import (
    TrackingMadInterface,
    create_tracking_interface,
)

__all__ = [
    "BaseMadInterface",
    "OptimizationMadInterface",
    "TrackingMadInterface",
    "create_tracking_interface",
]
