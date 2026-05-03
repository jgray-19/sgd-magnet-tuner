"""Base accelerator class defining the interface for all accelerators."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pymadng_utils.accelerators.base import Accelerator as BaseAccelerator

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
    from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface


class Accelerator(BaseAccelerator, ABC):
    """Abstract base class for accelerator definitions.

    This class encapsulates all machine-specific configuration and provides
    a factory method for creating MAD interfaces, eliminating the need to
    pass many individual parameters through multiple layers.
    """

    def __init__(
        self,
        sequence_file: Path | str,
        pc: float,
        bpm_pattern: str,
        optimise_energy: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_quad_dx: bool = False,
        optimise_quad_dy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise base accelerator.

        Args:
            sequence_file: Path to the sequence file
            pc: Particle momentum in GeV/c
            bpm_pattern: Pattern for identifying BPMs in the sequence
        """
        # Call to super() now
        super().__init__(sequence_file=sequence_file, pc=pc, bpm_pattern=bpm_pattern)
        self.optimise_energy = optimise_energy
        self.optimise_quadrupoles = optimise_quadrupoles
        self.optimise_sextupoles = optimise_sextupoles
        self.optimise_quad_dx = optimise_quad_dx
        self.optimise_quad_dy = optimise_quad_dy
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
                self.optimise_quad_dx,
                self.optimise_quad_dy,
                bool(self.custom_knobs_to_optimise),
            )
        )

    @property
    @abstractmethod
    def tune_variables(self) -> tuple[str, str]:
        """Return the names of the horizontal and vertical tune variables."""
        pass

    @property
    @abstractmethod
    def tune_integers(self) -> tuple[int, int]:
        """Return the integer tune values."""
        pass

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this accelerator."""
        targets: list[str] = []
        if self.optimise_quadrupoles:
            targets.append("quadrupoles")
        if self.optimise_sextupoles:
            targets.append("sextupoles")
        if self.optimise_energy:
            targets.append("beam energy")
        if self.optimise_quad_dx:
            targets.append("quadrupole horizontal offsets")
        if self.optimise_quad_dy:
            targets.append("quadrupole vertical offsets")
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

    @abstractmethod
    def get_supported_knob_specs(self) -> list[tuple[str, str, str, str | None, bool]]:
        """Return the knob specifications supported by this accelerator.

        Returns:
            List of (kind, attribute, pattern, nonzero_attr, optimise_flag) tuples defining
            all possible knobs that can be created for this accelerator.
        """
        pass

    @property
    def quadrupole_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return quadrupole misalignment patterns keyed by attribute name."""
        return {}

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        selected_specs: list[tuple[str, str, str, str | None]],
    ) -> None:
        """Prepare accelerator-specific MAD state before knob creation."""
        del selected_specs
        quadrupole_dx_patterns = (
            self.quadrupole_misalignment_patterns.get("dx", ())
            if self.optimise_quad_dx
            else ()
        )
        quadrupole_dy_patterns = (
            self.quadrupole_misalignment_patterns.get("dy", ())
            if self.optimise_quad_dy
            else ()
        )
        if not quadrupole_dx_patterns and not quadrupole_dy_patterns:
            return

        mad_iface.mad.send(f"""
        local tblcat in MAD.utility
        local dx_patterns = {mad_iface.py_name}:recv()
        local dy_patterns = {mad_iface.py_name}:recv()
        local patterns = tblcat(dx_patterns, dy_patterns)
        for i, e in loaded_sequence:siter(magnet_range) do
            if e.kind == "quadrupole" then
                for _, pattern in ipairs(patterns) do
                    if string.match(e.name, pattern) then
                        e.dx = e.dx or 0
                        e.dy = e.dy or 0
                        e.misalign = MAD.typeid.deferred{{dx =\\->e.dx, dy =\\->e.dy}}
                        break
                    end
                end
            end
        end
        """)
        mad_iface.mad.send(quadrupole_dx_patterns).send(quadrupole_dy_patterns)

    @abstractmethod
    def apply_accelerator_specific_errors(self, mad_iface: AbaMadInterface) -> None:
        """Apply accelerator-specific model errors to the loaded MAD sequence."""
        pass

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
