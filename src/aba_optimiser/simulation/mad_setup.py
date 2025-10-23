"""
MAD-NG setup and initialization utilities for accelerator simulations.

This module provides high-level setup functions that use the MAD interfaces
for consistency across the codebase. Prefer using the interface classes directly
for new code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.mad.tracking_interface import TrackingMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)


def create_mad_interface(
    debug: bool = False, stdout: str | None = None, redirect_stderr: bool = False
) -> BaseMadInterface:
    """
    Create a new BaseMadInterface instance.

    Args:
        debug: Enable debug mode
        stdout: Redirect stdout to file (optional)
        redirect_stderr: Redirect stderr

    Returns:
        Initialised MAD interface
    """
    logger.debug("Creating new MAD interface")
    kwargs = {"debug": debug, "redirect_stderr": redirect_stderr}
    if stdout is not None:
        kwargs["stdout"] = stdout
    return BaseMadInterface(**kwargs)


def select_bpms(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Select only BPM rows from a TFS DataFrame.

    Args:
        df: TFS DataFrame containing element data

    Returns:
        Filtered DataFrame with only BPM elements
    """
    if df.index.name == "name":
        return df[df.index.str.match(r"^BPM\.\d\d.*")]
    return df[df["name"].str.match(r"^BPM\.\d\d.*")]


def setup_tracking_interface(
    sequence_file: Path,
    seq_name: str,
    beam_energy: float,
    matched_tunes: dict,
    true_strengths: dict,
    enable_logging: bool = True,
) -> TrackingMadInterface:
    """
    Set up a TrackingMadInterface for tracking simulations.

    Args:
        sequence_file: Path to sequence file
        seq_name: Sequence name
        beam_energy: Beam energy in GeV
        matched_tunes: Dictionary of matched tune knobs
        true_strengths: Dictionary of magnet strengths
        enable_logging: Enable MAD logging

    Returns:
        Configured tracking interface
    """
    # Create tracking interface with basic setup
    interface = TrackingMadInterface(enable_logging=enable_logging)
    interface.setup_for_tracking(sequence_file, seq_name, beam_energy)

    # Set tune knobs
    for key, val in matched_tunes.items():
        interface.set_variables(f"MADX['{key}']", val)

    # Set magnet strengths using interface method
    interface.set_magnet_strengths(true_strengths)

    return interface
