"""
Tracking interface for beam dynamics simulations.

This module provides a specialised interface for tracking simulations that builds
on the base MAD interface without unnecessary optimization setup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base_mad_interface import BaseMadInterface

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


class TrackingMadInterface(BaseMadInterface):
    """
    Specialised MAD interface for tracking simulations.

    This class provides tracking-specific functionality while keeping
    the interface lightweight and focused.
    """

    def __init__(self, enable_logging: bool = True, **mad_kwargs):
        """
        Initialise tracking MAD interface.

        Args:
            enable_logging: Whether to enable MAD output logging
            **mad_kwargs: Additional keyword arguments for MAD
        """
        if enable_logging:
            mad_kwargs.setdefault("stdout", "mad_stdout.log")
            mad_kwargs.setdefault("redirect_stderr", True)
            mad_kwargs.setdefault("debug", True)
        else:
            mad_kwargs.setdefault("stdout", "/dev/null")
            mad_kwargs.setdefault("redirect_stderr", True)

        super().__init__(**mad_kwargs)
        LOGGER.debug("Initialised tracking MAD interface")

    def setup_for_tracking(
        self,
        sequence_file: str | Path,
        seq_name: str,
        beam_energy: float,
        observe_pattern: str = "BPM",
    ) -> None:
        """
        Complete setup for tracking simulations.

        Args:
            sequence_file: Path to sequence file
            seq_name: Sequence name
            beam_energy: Beam energy in GeV
            element_range: Element range for cycling (optional)
            cycle_to_start: Whether to cycle to start of range
        """
        LOGGER.info(f"Setting up tracking for {seq_name}")

        # Load sequence and set beam
        self.load_sequence(sequence_file, seq_name)
        self.setup_beam(beam_energy)

        # Configure observation for BPMs
        self.observe_elements(observe_pattern)
