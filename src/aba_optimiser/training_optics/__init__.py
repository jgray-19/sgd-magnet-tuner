"""Optics optimization training module for beta function optimization.

This module provides specialised controllers and managers for optics optimization
using quadrupole strengths to fit measured beta functions across the whole ring.
"""

from aba_optimiser.training_optics.controller import LHCOpticsController, OpticsController

__all__ = [
    "OpticsController",
    "LHCOpticsController",
]
