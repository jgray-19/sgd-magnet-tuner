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
    PATTERN_QUAD_NON_TUNE = "MQ[MYX]"  # Explicitly not MQT or MQ.

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
        optimise_other_quadrupoles: bool = False,
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
        self.optimise_other_quadrupoles = optimise_other_quadrupoles


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
            or self.optimise_other_quadrupoles
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
            targets.append("main quadrupoles")
        if self.optimise_other_quadrupoles:
            targets.append("other quadrupoles")
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

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return LHC-specific knob specifications.

        Returns:
            List of (kind, attribute, pattern, zero_check, optimise_flag) tuples defining
            all possible knobs that can be created for LHC optimization.
        """
        return [
            ("sbend", "k0", self.PATTERN_MAIN_BEND, True, self.optimise_bends),
            ("rbend", "k0", self.PATTERN_RBEND, True, self.optimise_bends),
            ("quadrupole", "k1", self.PATTERN_MAIN_QUAD, True, self.optimise_quadrupoles),
            ("quadrupole", "k1", self.PATTERN_QUAD_NON_TUNE, True, self.optimise_other_quadrupoles),
            ("hkicker", "kick", self.PATTERN_CORRECTOR, False, self.optimise_correctors),
        ]

    def parse_bad_bpm_specification(self, bad_bpm_spec: str) -> tuple[str, str | None]:
        """Parse LHC bad BPM specification into BPM name and optional plane.

        LHC naming convention:
        - No suffix: dual-plane BPM (e.g., "BPM.14L1.B1")
        - .H suffix: horizontal plane only (e.g., "BPM.14L1.B1.H")
        - .V suffix: vertical plane only (e.g., "BPM.14L1.B1.V")

        Args:
            bad_bpm_spec: Bad BPM specification

        Returns:
            Tuple of (bpm_base_name, plane) where plane is "H", "V", or None for both planes
        """
        if bad_bpm_spec.endswith(".H"):
            return bad_bpm_spec[:-2], "H"
        if bad_bpm_spec.endswith(".V"):
            return bad_bpm_spec[:-2], "V"
        return bad_bpm_spec, None

    def get_bpm_plane_mask(
        self, bpm_list: list[str], bad_bpms: list[str]
    ) -> tuple[list[bool], list[bool]]:
        """Generate masks for LHC BPM planes considering single-plane and bad BPMs.

        LHC-specific implementation that handles:
        1. Single-plane BPMs (those with .H or .V suffix)
        2. Bad BPM filtering per plane
        3. Dual-plane BPMs (standard naming without suffix)

        Logic:
        - For dual-plane BPMs:
          - If "BPM.X.H" is in bad_bpms → mask horizontal plane
          - If "BPM.X.V" is in bad_bpms → mask vertical plane
          - If "BPM.X" is in bad_bpms → mask both planes
        - For single-plane BPMs:
          - "BPM.X.H" measures only horizontal → mask vertical
          - "BPM.X.V" measures only vertical → mask horizontal
          - If the BPM appears in bad_bpms, mask its measuring plane too

        Args:
            bpm_list: List of BPM names (can include .H or .V suffixes)
            bad_bpms: List of bad BPM specifications (with optional plane suffixes)

        Returns:
            Tuple of (h_mask, v_mask) where True means the plane should be used
        """
        # Build a dict of bad planes per BPM: {base_name -> set of bad planes}
        bad_bpm_dict: dict[str, set[str]] = {}
        for bad_spec in bad_bpms:
            base_name, plane = self.parse_bad_bpm_specification(bad_spec)
            bad_planes = bad_bpm_dict.setdefault(base_name, set())
            if plane is None:
                bad_planes.update(["H", "V"])
            else:
                bad_planes.add(plane)

        # Generate masks for each BPM
        h_mask = []
        v_mask = []
        for bpm_name in bpm_list:
            base_name, plane_suffix = self.parse_bad_bpm_specification(bpm_name)
            bad_planes = bad_bpm_dict.get(base_name, set())

            # A plane is measured if BPM doesn't exclude it, and not bad
            h_mask.append(plane_suffix != "V" and "H" not in bad_planes)
            v_mask.append(plane_suffix != "H" and "V" not in bad_planes)

        return h_mask, v_mask
