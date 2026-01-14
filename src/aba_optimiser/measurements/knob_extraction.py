"""
Knob Extraction from NXCALS - Extension Module
-----------------------------------------------

This module extends omc3.machine_data_extraction.nxcals_knobs with additional magnet type extraction
functions not yet available in OMC3.

It imports the core functionality from OMC3 and adds extraction functions for:
- MB (Main Dipole) magnets
- MS (Sextupole) magnets
- MQ (Main Quadrupole) magnets
- MCB (Orbit Corrector) magnets

For MQT extraction and core functionality, use the functions from omc3.machine_data_extraction.nxcals_knobs directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omc3.machine_data_extraction.lsa_utils import calc_k_from_iref
from omc3.machine_data_extraction.mqt_extraction import generate_mqt_names, get_mqt_vals
from omc3.machine_data_extraction.nxcals_knobs import (
    NXCALSResult,
    get_energy,
    get_knob_vals,
    get_raw_vars,
    map_pc_name_to_madx,
    strip_i_meas,
)

if TYPE_CHECKING:
    from datetime import datetime

    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# Re-export commonly used items for backward compatibility
__all__ = [
    "NXCALSResult",
    "get_knob_vals",
    "get_raw_vars",
    "get_energy",
    "calc_k_from_iref",
    "strip_i_meas",
    "map_pc_name_to_madx",
    "generate_mqt_names",
    "get_mqt_vals",
    "get_mb_vals",
    "get_ms_vals",
    "get_mq_vals",
    "get_mcb_vals",
]


# Additional Magnet Type Extraction Functions ----------------------------------
# These are not yet in OMC3, so we define them here


def get_mb_vals(
    spark: SparkSession, time: datetime, beam: int, energy: float | None = None
) -> list[NXCALSResult]:
    """
    Retrieve MB (Main Dipole) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MB power converters,
    calculates the corresponding K-values (integrated dipole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).
        energy (float | None): Beam energy in GeV. If None, retrieves from NXCALS.

    Returns:
        list[NXCALSResult]: List of NXCALSResult objects containing the MAD-X knob names,
            K-values, and timestamps.

    Raises:
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    patterns = ["RPTE.UA%.RB%.A%:I_MEAS"]
    return get_knob_vals(
        spark, time, beam, patterns, expected_knobs=None, log_prefix="MB: ", energy=energy
    )


def get_ms_vals(
    spark: SparkSession, time: datetime, beam: int, energy: float | None = None
) -> list[NXCALSResult]:
    """
    Retrieve MS (Sextupole) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MS power converters,
    calculates the corresponding K-values (integrated sextupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).
        energy (float | None): Beam energy in GeV. If None, retrieves from NXCALS.

    Returns:
        list[NXCALSResult]: List of NXCALSResult objects containing the MAD-X knob names,
            K-values, and timestamps.

    Raises:
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    patterns = [f"%.RS%B{beam}:I_MEAS"]  # All sextupoles for the beam
    return get_knob_vals(
        spark, time, beam, patterns, expected_knobs=None, log_prefix="MS: ", energy=energy
    )


def get_mq_vals(
    spark: SparkSession, time: datetime, beam: int, energy: float | None = None
) -> list[NXCALSResult]:
    """
    Retrieve MQ (Main Quadrupole) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MQ power converters,
    calculates the corresponding K-values (integrated quadrupole strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).
        energy (float | None): Beam energy in GeV. If None, retrieves from NXCALS.

    Returns:
        list[NXCALSResult]: List of NXCALSResult objects containing the MAD-X knob names,
            K-values, and timestamps.

    Raises:
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    patterns = [
        "%.RQ%.A%:I_MEAS",  # All arc quadrupoles
    ]
    knob_list = get_knob_vals(
        spark,
        time,
        beam,
        patterns,
        expected_knobs=None,
        log_prefix="MQ: ",
        delta_days=0.1,
        energy=energy,
    )
    # Remove all the b2 magnets for beam 1 and vice versa
    return [knob for knob in knob_list if not knob.name.endswith(f"b{3 - beam}")]


def get_mcb_vals(
    spark: SparkSession, time: datetime, beam: int, energy: float | None = None
) -> list[NXCALSResult]:
    """
    Retrieve MCB (Orbit Corrector) knob values from NXCALS for a specific time and beam.

    This function queries NXCALS for current measurements of MCB power converters,
    calculates the corresponding K-values (integrated corrector strengths) using LSA,
    and returns them in MAD-X format with timestamps.

    Args:
        spark (SparkSession): Active Spark session for NXCALS queries.
        time (datetime): The timestamp for which to retrieve the data (timezone-aware recommended).
        beam (int): The beam number (1 or 2).
        energy (float | None): Beam energy in GeV. If None, retrieves from NXCALS.

    Returns:
        list[NXCALSResult]: List of NXCALSResult objects containing the MAD-X knob names,
            K-values, and timestamps.

    Raises:
        RuntimeError: If no data is found in NXCALS or LSA calculations fail.
    """
    beam_pattern = f"%RCB%B{beam}:I_MEAS"
    both_pattern = "RPMBB%RCBX%:I_MEAS"
    patterns = [beam_pattern, both_pattern]

    return get_knob_vals(
        spark,
        time,
        beam,
        patterns,
        expected_knobs=None,
        log_prefix="MCB: ",
        delta_days=0.25,
        energy=energy,
    )
