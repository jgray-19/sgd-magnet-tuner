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
