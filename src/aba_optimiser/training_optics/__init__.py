"""Optics optimisation training module for beta function optimisation.

This module provides specialised controllers and managers for optics optimisation
using quadrupole strengths to fit measured beta functions across the whole ring.
"""

from aba_optimiser.training_optics.controller import OpticsController

__all__ = [
    "OpticsController",
]
