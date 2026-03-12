"""Base accelerator class defining the interface for all accelerators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface


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
        bpm_pattern: str,
        optimise_energy: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise base accelerator.

        Args:
            sequence_file: Path to the sequence file
            beam_energy: Beam energy in GeV
            bpm_pattern: Pattern for identifying BPMs in the sequence
        """
        self.sequence_file = Path(sequence_file)
        self.beam_energy = beam_energy
        self.bpm_pattern = bpm_pattern
        self.optimise_energy = optimise_energy
        self.optimise_quadrupoles = optimise_quadrupoles
        self.optimise_sextupoles = optimise_sextupoles
        self.custom_knobs_to_optimise = custom_knobs_to_optimise
        # Accelerator-owned state populated during model setup (if applicable).
        self.bend_lengths: dict[str, float] | None = None

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled."""
        return any(
            (
                self.optimise_quadrupoles,
                self.optimise_sextupoles,
                self.optimise_energy,
                bool(self.custom_knobs_to_optimise),
            )
        )

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this accelerator."""
        targets: list[str] = []
        if self.optimise_quadrupoles:
            targets.append("quadrupoles")
        if self.optimise_sextupoles:
            targets.append("sextupoles")
        if self.optimise_energy:
            targets.append("beam energy")
        if self.custom_knobs_to_optimise:
            targets.append(f"custom knobs: {self.custom_knobs_to_optimise}")
        if targets:
            LOGGER.info(f"Optimisation targets: {', '.join(targets)}")
        else:
            LOGGER.info("No optimisation targets set.")

    def get_bend_lengths(self) -> dict[str, float] | None:
        """Return bend lengths required for accelerator-specific normalisation.

        Returns:
            Dictionary of bend lengths or None if not applicable
        """
        return self.bend_lengths

    def normalise_true_strengths(
        self,
        true_strengths: dict[str, float],
        bend_lengths: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Apply accelerator-specific normalisation to true strengths.

        Args:
            true_strengths: Dictionary of true magnet strengths
            bend_lengths: Bend lengths for normalisation (optional). If None, uses
                accelerator-owned ``self.bend_lengths``.

        Returns:
            Normalised strengths dictionary (default: unchanged)
        """
        _ = bend_lengths
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

    @property
    @abstractmethod
    def seq_name(self) -> str:
        """Return the sequence name for this accelerator.

        Returns:
            Sequence name to use in MAD
        """
        ...

    @abstractmethod
    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return the knob specifications supported by this accelerator.

        Returns:
            List of (kind, attribute, pattern, zero_check) tuples defining
            all possible knobs that can be created for this accelerator.
        """
        pass

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        selected_specs: list[tuple[str, str, str, bool]],
    ) -> None:
        """Prepare accelerator-specific MAD state before knob creation."""
        _ = mad_iface, selected_specs

    def get_mad_attr_specs(self) -> dict[str, dict[str, str]]:
        """Return accelerator-specific attr name/value expressions for knob creation."""
        return {}

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return per-family override metadata keyed by family code d/q/s."""
        return {}

    @staticmethod
    @abstractmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from BPM name."""
        pass

    @abstractmethod
    def get_tune_variables(self) -> tuple[str, str]:
        """Return (qx_knob, qy_knob) MAD-X variable names for tune matching."""
        pass

    @abstractmethod
    def get_tune_integers(self) -> tuple[int, int]:
        """Return (qx_integer, qy_integer) used for full tune targets."""
        pass
