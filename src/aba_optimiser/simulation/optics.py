"""
Beam optics calculation utilities for accelerator simulations.

This module provides functions for twiss calculations, beta beating analysis,
tune matching, and orbit correction procedures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import tfs

logger = logging.getLogger(__name__)


def calculate_beta_beating(
    changed_tws: pd.DataFrame, initial_tws: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate beta beating between two twiss calculations.

    Args:
        changed_tws: Twiss after changes
        initial_tws: Initial reference twiss

    Returns:
        Tuple of (beta11_beating, beta22_beating) as pandas Series
    """
    beta11_beating = (changed_tws["beta11"] - initial_tws["beta11"]) / initial_tws["beta11"]
    beta22_beating = (changed_tws["beta22"] - initial_tws["beta22"]) / initial_tws["beta22"]

    logger.info(
        f"Beta11 beating: {beta11_beating.mean() * 100:.2f}% ± {beta11_beating.std() * 100:.2f}%"
    )
    logger.info(
        f"Beta22 beating: {beta22_beating.mean() * 100:.2f}% ± {beta22_beating.std() * 100:.2f}%"
    )

    return beta11_beating, beta22_beating


def run_initial_twiss_analysis(
    changed_tws: tfs.TfsDataFrame, initial_tws: tfs.TfsDataFrame
) -> tfs.TfsDataFrame:
    """
    Run twiss after magnet perturbations and analyze beta beating.

    Args:
        mad: MAD instance with perturbed magnets
        initial_tws: Initial reference twiss

    Returns:
        Twiss dataframe after perturbations
    """
    logger.info("Running twiss calculation after quadrupole strength adjustments")
    beta11_beating, beta22_beating = calculate_beta_beating(changed_tws, initial_tws)

    logger.info(
        f"Old tunes: {initial_tws.q1:.4f}, {initial_tws.q2:.4f}. "
        f"New tunes: {changed_tws.q1:.4f}, {changed_tws.q2:.4f}"
        f" with beta11 beating {beta11_beating.mean() * 100:.2f}% ± {beta11_beating.std() * 100:.2f}%"
        f" and beta22 beating {beta22_beating.mean() * 100:.2f}% ± {beta22_beating.std() * 100:.2f}%"
    )

    return changed_tws
