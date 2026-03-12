"""Accelerator abstraction for encapsulating machine-specific parameters.

This module provides base classes and implementations for different accelerators,
reducing parameter passing and making it easy to add new machines.
"""

from aba_optimiser.accelerators.base import Accelerator
from aba_optimiser.accelerators.lhc import LHC
from aba_optimiser.accelerators.psb import PSB
from aba_optimiser.accelerators.sps import SPS

__all__ = ["Accelerator", "LHC", "PSB", "SPS", "instantiate_accelerator_from"]


def instantiate_accelerator_from(
    accelerator: Accelerator,
    **kwargs,
) -> Accelerator:
    """Create a new accelerator instance based on an existing one.

    This factory function extracts parameters from an existing accelerator and creates
    a new instance of the same type, with optional parameter overrides.

    Args:
        accelerator: The accelerator instance to base the new one on
        **kwargs: Optional parameter overrides (e.g., beam_energy=7000.0, beam=2)

    Returns:
        A new Accelerator instance of the same type with merged parameters

    Examples:
        >>> lhc_b1 = LHC(beam=1, sequence_file="lhc.seq", beam_energy=6800)
        >>> # Create a copy with different beam energy
        >>> lhc_b1_7tev = instantiate_accelerator_from(lhc_b1, beam_energy=7000)
        >>> # Create LHC beam 2 from beam 1
        >>> lhc_b2 = instantiate_accelerator_from(lhc_b1, beam=2)
        >>> # Create SPS variant with different optimization targets
        >>> sps_opt = instantiate_accelerator_from(sps, optimise_bends=True)
    """
    # Common parameters for all accelerators
    common_params = {
        "sequence_file": kwargs.get("sequence_file", accelerator.sequence_file),
        "beam_energy": kwargs.get("beam_energy", accelerator.beam_energy),
        "optimise_energy": kwargs.get("optimise_energy", accelerator.optimise_energy),
        "optimise_quadrupoles": kwargs.get(
            "optimise_quadrupoles", accelerator.optimise_quadrupoles
        ),
        "optimise_sextupoles": kwargs.get("optimise_sextupoles", accelerator.optimise_sextupoles),
    }

    # LHC-specific instantiation
    if isinstance(accelerator, LHC):
        params = {
            **common_params,
            "beam": kwargs.get("beam", accelerator.beam),
            "optimise_correctors": kwargs.get(
                "optimise_correctors", accelerator.optimise_correctors
            ),
            "optimise_bends": kwargs.get("optimise_bends", accelerator.optimise_bends),
            "normalise_bends": kwargs.get("normalise_bends", accelerator.normalise_bends),
            "optimise_other_quadrupoles": kwargs.get(
                "optimise_other_quadrupoles", accelerator.optimise_other_quadrupoles
            ),
            "optimise_quad_dx": kwargs.get("optimise_quad_dx", accelerator.optimise_quad_dx),
        }
        return LHC(**params)

    # SPS-specific instantiation
    if isinstance(accelerator, SPS):
        params = {
            **common_params,
            "custom_knobs_to_optimise": kwargs.get(
                "custom_knobs_to_optimise", accelerator.custom_knobs_to_optimise
            ),
        }
        return SPS(**params)

    if isinstance(accelerator, PSB):
        params = {
            **common_params,
            "ring": kwargs.get("ring", accelerator.ring),
            "custom_knobs_to_optimise": kwargs.get(
                "custom_knobs_to_optimise", accelerator.custom_knobs_to_optimise
            ),
        }
        return PSB(**params)

    # Fallback for other accelerator types
    raise ValueError(f"Unknown accelerator type: {type(accelerator)}")
