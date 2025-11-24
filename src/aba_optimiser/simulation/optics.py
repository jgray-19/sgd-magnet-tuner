"""
Beam optics calculation utilities for accelerator simulations.

This module provides functions for twiss calculations, beta beating analysis,
tune matching, and orbit correction procedures.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import tfs
    from pymadng import MAD

logger = logging.getLogger(__name__)


def calculate_beta_beating(
    changed_tws: tfs.TfsDataFrame, initial_tws: tfs.TfsDataFrame
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


def match_tunes(mad: MAD, target_qx: float, target_qy: float, deltap: float) -> dict[str, float]:
    """
    Match tunes to target values using MAD.

    Args:
        mad: MAD instance
        target_qx: Target horizontal tune
        target_qy: Target vertical tune

    Returns:
        Dictionary of matched tune knobs
    """
    logger.info("Starting tune matching for initial conditions")

    mad["result"] = mad.match(
        command=rf"\ -> twiss{{sequence=MADX[SEQ_NAME], deltap={deltap:.16e}}}",
        variables=[
            {"var": "'MADX.dqx_b1_op'", "name": "'dQx.b1_op'"},
            {"var": "'MADX.dqy_b1_op'", "name": "'dQy.b1_op'"},
        ],
        equalities=[
            {"expr": f"\\t -> t.q1-(62+{target_qx})", "name": "'q1'"},
            {"expr": f"\\t -> t.q2-(60+{target_qy})", "name": "'q2'"},
        ],
        objective={"fmin": 1e-8},
        info=2,
    )

    # Store matched tunes in Python variables
    matched_tunes = {key: mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")}
    logger.info(f"Matched tune knobs: {matched_tunes}")

    return matched_tunes


def perform_orbit_correction(
    mad: MAD,
    machine_deltap: float,
    target_qx: float,
    target_qy: float,
    corrector_file: Path,
    beam: int = 1,
) -> None:
    """
    Perform orbit correction and tune rematching with off-momentum twiss.

    Args:
        mad: MAD instance
        machine_deltap: Machine momentum deviation
        target_qx: Target horizontal tune
        target_qy: Target vertical tune
        corrector_file: Path to save corrector strengths
    """
    logger.info(f"Setting machine deltap: {machine_deltap}")
    mad["machine_deltap"] = machine_deltap
    mad["qx"] = target_qx
    mad["qy"] = target_qy
    mad["correct_file"] = str(corrector_file.absolute())

    logger.info(f"Starting orbit correction with corrector file: {corrector_file}")
    mad.send(rf"""
local correct, option in MAD

io.write("*** orbit correction using off momentum twiss\n")
local tbl = twiss {{ sequence=loaded_sequence, deltap=machine_deltap }}

! Increase file numerical formatting
local fmt = option.numfmt ; option.numfmt = "% -.16e"
correct {{ sequence=loaded_sequence, model=tbl, method="micado", info=1, plane="x" }} :write(correct_file)
option.numfmt = fmt ! restore formatting

io.write("*** rematching tunes for off-momentum twiss\n")
match {{
  command := twiss {{sequence=loaded_sequence, observe=1, deltap=machine_deltap}},
  variables = {{ rtol=1e-6, -- 1 ppm
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{ tol = 1e-10,
    {{ expr = \t -> t.q1-62-qx, name='q1' }},
    {{ expr = \t -> t.q2-60-qy, name='q2' }},
  }},
  info=2
}}
""")
    logger.info("Orbit correction and tune rematching completed")

    # Store matched tunes in Python variables
    matched_tunes = {
        f"dqx.b{beam}_op": mad[f"MADX['dqx_b{beam}_op']"],
        f"dqy.b{beam}_op": mad[f"MADX['dqy_b{beam}_op']"],
    }
    logger.info(f"Matched tune knobs: {matched_tunes}")

    return matched_tunes


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
