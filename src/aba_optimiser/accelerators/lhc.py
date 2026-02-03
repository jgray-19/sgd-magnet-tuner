"""LHC-specific accelerator implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator
from aba_optimiser.physics.lhc_bends import normalise_lhcbend_magnets

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

class LHC(Accelerator):
    """Large Hadron Collider accelerator configuration.

    This class encapsulates LHC-specific parameters like beam numbers,
    default BPMs, and sequence file locations.
    """

    # LHC magnet patterns as class constants
    PATTERN_MAIN_BEND = "MB%."
    PATTERN_RBEND = "MB[RXWAL]%w*%."
    PATTERN_MAIN_QUAD = "MQ%."
    PATTERN_CORRECTOR = "MCB"

    def __init__(
        self,
        beam: int,
        sequence_file: Path | str,
        beam_energy: float = 6800.0,

        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,

        # LHC-specific bend control
        optimise_energy: bool = False,
        optimise_correctors: bool = False,

        optimise_bends: bool = False,
        normalise_bends: bool | None = None,
    ):
        """Initialise LHC accelerator for a specific beam.

        Args:
            beam: Beam number (1 or 2)
            sequence_file: Path to sequence file
            beam_energy: Beam energy in GeV
            optimise_quadrupoles: Whether to optimise quadrupoles
            optimise_sextupoles: Whether to optimise sextupoles

            optimise_bends: Whether to optimise dipole bends
            normalise_bends: Whether to normalise bend strengths
            optimise_correctors: Whether to optimise corrector magnets
            optimise_energy: Whether to optimise beam energy

        Raises:
            ValueError: If an invalid beam number is provided
        """
        if beam not in (1, 2):
            raise ValueError(f"LHC beam must be 1 or 2, got {beam}")

        self.beam = beam

        # Store LHC-specific optimisation flags
        self.optimise_bends = optimise_bends
        if normalise_bends is None:
            normalise_bends = optimise_bends
        self.normalise_bends = normalise_bends
        self.optimise_correctors = optimise_correctors


        # Initialise base Accelerator
        super().__init__(
            sequence_file=sequence_file,
            beam_energy=beam_energy,
            seq_name=f"lhcb{beam}",
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
        )

    def get_seq_name(self) -> str:
        """Return the sequence name for this LHC beam."""
        return f"lhcb{self.beam}"

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled.

        Returns:
            True if at least one optimisation type is enabled
        """
        return (
            self.optimise_bends
            or self.optimise_quadrupoles
            or self.optimise_sextupoles
            or self.optimise_correctors
            or self.optimise_energy
        )

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this LHC accelerator."""
        targets = []
        if self.optimise_bends:
            targets.append("bends")
        if self.optimise_quadrupoles:
            targets.append("quadrupoles")
        if self.optimise_sextupoles:
            targets.append("sextupoles")
        if self.optimise_correctors:
            targets.append("correctors")
        if self.optimise_energy:
            targets.append("beam energy")
        if targets:
            LOGGER.info(f"Optimisation targets: {', '.join(targets)}")
        else:
            LOGGER.info("No optimisation targets set.")

    def get_bend_lengths(self, mad_iface) -> dict[str, float] | None:
        """Return LHC bend lengths when bend normalisation is enabled."""
        if not (self.optimise_bends and self.normalise_bends):
            return None
        return getattr(mad_iface, "bend_lengths", None)

    def normalise_true_strengths(
        self, true_strengths: dict[str, float], bend_lengths: dict[str, float] | None
    ) -> dict[str, float]:
        """Normalize LHC bend strengths when applicable."""
        if self.optimise_bends and bend_lengths:
            return normalise_lhcbend_magnets(true_strengths, bend_lengths)
        return true_strengths

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool]]:
        """Return LHC-specific knob specifications.

        Returns:
            List of (kind, attribute, pattern, zero_check) tuples defining
            all possible knobs that can be created for LHC optimization.
        """
        return [
            ("sbend", "k0", self.PATTERN_MAIN_BEND, True),
            ("rbend", "k0", self.PATTERN_RBEND, True),
            ("quadrupole", "k1", self.PATTERN_MAIN_QUAD, True),
            ("hkicker", "kick", self.PATTERN_CORRECTOR, False),
        ]
