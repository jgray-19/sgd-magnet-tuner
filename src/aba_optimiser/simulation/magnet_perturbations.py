"""
Magnet perturbation utilities for accelerator simulations.

This module provides functions for applying noise to different types of magnets
(quadrupoles, sextupoles, dipoles) and managing their strength perturbations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tfs

from aba_optimiser.config import BEND_ERROR_FILE

if TYPE_CHECKING:
    from pymadng import MAD

logger = logging.getLogger(__name__)


def apply_magnet_perturbations(
    mad: MAD, rel_k1_std_dev: float = 1e-4, seed: int = 42, magnet_type: str | list[str] = "all"
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Apply perturbations to magnets in the MAD sequence.

    Args:
        mad: MAD instance
        rel_k1_std_dev: Relative standard deviation for K1 perturbations
        seed: Random seed for reproducibility
        magnet_type: Magnet types to apply perturbations to. Can be "all" or a list like ["q", "s"] ("q" for quadrupoles, "s" for sextupoles, "d" for dipoles)
    Returns:
        Tuple of (magnet_strengths, true_strengths)
    """
    rng = np.random.default_rng(seed)
    bend_errors_table = tfs.read(BEND_ERROR_FILE)
    bend_errors_dict = bend_errors_table["K0L"].to_dict()  # ty:ignore[unresolved-attribute]
    logger.info("Scanning sequence for quadrupoles and sextupoles")
    magnet_strengths = {}
    true_strengths = {}
    num_bends = 0
    num_quads = 0
    num_sexts = 0

    if isinstance(magnet_type, str):
        magnet_types = list("qsd") if magnet_type == "all" else list(magnet_type)
    else:
        magnet_types = magnet_type

    dodip_err = "d" in magnet_types
    doquad_err = "q" in magnet_types
    dosext_err = "s" in magnet_types

    for elm in mad.loaded_sequence:
        # Dipoles
        if dodip_err and elm.kind == "sbend" and elm.k0 != 0 and elm.name[:3] == "MB.":
            if elm.name not in bend_errors_dict:
                raise ValueError(
                    f"Bend error for {elm.name} not found in {BEND_ERROR_FILE}"
                )
            k0l_error = bend_errors_dict[elm.name]
            elm.k0 += k0l_error / elm.l
            magnet_strengths[elm.name + ".k0"] = elm.k0
            true_strengths[elm.name] = elm.k0
            num_bends += 1
            # if elm.kind == "rbend" and elm.k0 != 0:
            # elm.k0 = elm.k0 + rng.normal(0, abs(elm.k0 * 1e-4))
            # magnet_strengths[elm.name + ".k0"] = elm.k0
            # true_strengths[elm.name] = elm.k0
            # num_bends += 1

        # Quadrupoles
        if doquad_err and elm.kind == "quadrupole" and elm.k1 != 0:
            if elm.name[:3] == "MQ.":
                elm.k1 = elm.k1 + rng.normal(0, abs(elm.k1 * rel_k1_std_dev))
                magnet_strengths[elm.name + ".k1"] = elm.k1
                true_strengths[elm.name] = elm.k1
                num_quads += 1
            # elif elm.name[:3] != "MQT":
            #     elm.k1 = elm.k1 + rng.normal(0, abs(elm.k1 * 1e-4))
            #     magnet_strengths[elm.name + ".k1"] = elm.k1
            #     true_strengths[elm.name] = elm.k1
            #     num_quads += 1

        # Sextupoles
        elif dosext_err and elm.kind == "sextupole" and elm.k2 != 0 and elm.name[:3] == "MS.":
            elm.k2 = elm.k2 + rng.normal(0, abs(elm.k2 * 1e-4))
            magnet_strengths[elm.name + ".k2"] = elm.k2
            true_strengths[elm.name] = elm.k2
            num_sexts += 1

    logger.info(
        f"Found {num_bends} dipoles, {num_quads} quadrupoles, {num_sexts} sextupoles"
    )
    if num_bends > 0:
        logger.info(f"Applied bend errors from file to {num_bends} dipoles")

    if num_quads > 0:
        logger.info(
            f"Applied relative K1 noise with std dev: {rel_k1_std_dev} to {num_quads} quadrupoles"
        )

    if num_sexts > 0:
        logger.info(
            f"Applied absolute K2 noise with std dev: 1e-4 to {num_sexts} sextupoles"
        )

    return magnet_strengths, true_strengths
