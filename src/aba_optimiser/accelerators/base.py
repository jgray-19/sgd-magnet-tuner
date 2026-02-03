"""Base accelerator class defining the interface for all accelerators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class Accelerator(ABC):
    """Abstract base class for accelerator definitions.

    This class encapsulates all machine-specific configuration and provides
    a factory method for creating MAD interfaces, eliminating the need to
    pass many individual parameters through multiple layers.
    """

    def __init__(
        self,
        sequence_file: Path | str,
        beam_energy: float,
        seq_name: str | None = None,
        optimise_energy: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise base accelerator.

        Args:
            sequence_file: Path to the sequence file
            beam_energy: Beam energy in GeV
            seq_name: Sequence name (auto-detected if None)
        """
        self.sequence_file = Path(sequence_file)
        self.beam_energy = beam_energy
        self.seq_name = seq_name
        self.optimise_energy = optimise_energy
        self.optimise_quadrupoles = optimise_quadrupoles
        self.optimise_sextupoles = optimise_sextupoles
        self.custom_knobs_to_optimise = custom_knobs_to_optimise

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled.

        Returns:
            True if at least one optimisation type is enabled
        """
        return (
            self.optimise_energy
            or self.optimise_quadrupoles
            or self.optimise_sextupoles
            or self.custom_knobs_to_optimise is not None
        )

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this accelerator."""
        targets = []
        if self.optimise_quadrupoles:
            targets.append("quadrupoles")
        if self.optimise_sextupoles:
            targets.append("sextupoles")
        if self.optimise_energy:
            targets.append("beam energy")
        if self.custom_knobs_to_optimise is not None:
            targets.append(f"custom knobs: {self.custom_knobs_to_optimise}")
        if targets:
            LOGGER.info(f"Optimisation targets: {', '.join(targets)}")
        else:
            LOGGER.info("No optimisation targets set.")

    def get_bend_lengths(self, mad_iface) -> dict[str, float] | None:
        """Return bend lengths required for accelerator-specific normalisation.

        Args:
            mad_iface: MAD interface instance used for model setup

        Returns:
            Dictionary of bend lengths or None if not applicable
        """
        return None

    def normalise_true_strengths(
        self, true_strengths: dict[str, float], bend_lengths: dict[str, float] | None
    ) -> dict[str, float]:
        """Apply accelerator-specific normalisation to true strengths.

        Args:
            true_strengths: Dictionary of true magnet strengths
            bend_lengths: Bend lengths for normalisation (if applicable)

        Returns:
            Normalised strengths dictionary (default: unchanged)
        """
        return true_strengths

    def format_result_knob_names(self, knob_names: list[str]) -> list[str]:
        """Format knob names for result reporting.

        Args:
            knob_names: Knob names as used in optimisation

        Returns:
            Knob names adjusted for reporting (default: unchanged)
        """
        if not self.optimise_energy:
            return list(knob_names)

        formatted = list(knob_names)
        if "pt" in formatted:
            formatted.remove("pt")
            formatted.append("deltap")
        return formatted

    @abstractmethod
    def get_seq_name(self) -> str:
        """Return the sequence name for this accelerator.

        Returns:
            Sequence name to use in MAD
        """
        pass

    @abstractmethod
    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool]]:
        """Return the knob specifications supported by this accelerator.

        Returns:
            List of (kind, attribute, pattern, zero_check) tuples defining
            all possible knobs that can be created for this accelerator.
        """
        pass

    def get_bpm_plane_mask(
        self, bpm_list: list[str], bad_bpms: list[str]
    ) -> tuple[list[bool], list[bool]]:
        """Generate masks indicating which BPMs measure which planes.

        This method identifies single-plane vs dual-plane BPMs and determines which
        planes should be masked out based on bad BPM filtering. The masks are used
        to set appropriate weights (0 or NaN) for planes that don't have measurements.

        Args:
            bpm_list: List of BPM names to generate masks for
            bad_bpms: List of bad BPM specifications (can include plane suffixes)

        Returns:
            Tuple of (h_mask, v_mask) where each mask is a list of booleans.
            True indicates the plane should be used, False indicates it should be masked.

        Note:
            Default implementation assumes all BPMs are dual-plane and returns all True.
            Accelerator-specific subclasses should override this method to handle
            single-plane BPMs and plane-specific bad BPM filtering.
        """
        # Default: all BPMs measure both planes
        h_mask = [True] * len(bpm_list)
        v_mask = [True] * len(bpm_list)
        # Set all bad BPMs to False in both masks
        return h_mask, v_mask

    def parse_bad_bpm_specification(self, bad_bpm_spec: str) -> tuple[str, str | None]:
        """Parse a bad BPM specification into BPM name and optional plane.

        Args:
            bad_bpm_spec: Bad BPM specification (e.g., "BPM.14L1.B1" or "BPM.14L1.B1.H")

        Returns:
            Tuple of (bpm_base_name, plane) where plane is "H", "V", or None for both planes

        Note:
            Default implementation assumes no plane suffixes (dual-plane only).
            Override in accelerator-specific subclasses for different naming conventions.
        """
        # Default: no plane suffix, all BPMs are dual-plane
        return bad_bpm_spec, None
