"""Base accelerator class defining the interface for all accelerators."""

from __future__ import annotations

import logging
import re
import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

from pymadng_utils.accelerators.base import Accelerator as BaseAccelerator

LOGGER = logging.getLogger(__name__)
_INDEXED_RESULT_MULTIPOLE_RE = re.compile(r"\.(knl|ksl)\[(\d+)\]$")
_TILT_SEED = 1e-9  # Should discuss this with Laurent.
_MISALIGN_DEFER_CREATE = """\ne.misalign = MAD.typeid.deferred{dx =\\->e.dx, dy =\\->e.dy}"""
_KNOB_PREPARATION = {
    "dx": "e.dx = (e.dx or 0)" + _MISALIGN_DEFER_CREATE,
    "dy": "e.dy = (e.dy or 0)" + _MISALIGN_DEFER_CREATE,
    # MAD-NG drops a rotation whose scalar angle is zero (mad_dynmap.cpp:345),
    # silently zeroing the knob's Jacobian column. Seeding above ``minang``
    # (1e-10 rad) keeps the derivative; the knob inherits the seed as its value.
    "tilt": f"e.tilt = (e.tilt or 0) + {_TILT_SEED:.15e}",
}

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface


class KnobSpec(NamedTuple):
    """Specification for a single optimisable knob type."""

    kind: str
    attribute: str
    pattern: str
    nonzero_attr: str | None
    enabled: bool
    label: str


class Accelerator(BaseAccelerator, ABC):
    """Abstract base class for accelerator definitions.

    This class encapsulates all machine-specific configuration and provides
    a factory method for creating MAD interfaces, eliminating the need to
    pass many individual parameters through multiple layers.
    """

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float,
        bpm_pattern: str,
        particle: str = "proton",
        optimise_energy: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_quad_dx: bool = False,
        optimise_quad_dy: bool = False,
        optimise_quad_tilt: bool = False,
        optimise_bpm_dx: bool = False,
        optimise_bpm_dy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
        **kwargs,
    ):
        """Initialise base accelerator.

        Args:
            sequence_file: Path to the sequence file
            kinetic_energy: Particle kinetic energy in GeV
            bpm_pattern: Pattern for identifying BPMs in the sequence
        """
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            particle=particle,
            **kwargs,
        )
        self.optimise_energy = optimise_energy
        self.optimise_quadrupoles = optimise_quadrupoles
        self.optimise_sextupoles = optimise_sextupoles
        self.optimise_quad_dx = optimise_quad_dx
        self.optimise_quad_dy = optimise_quad_dy
        self.optimise_quad_tilt = optimise_quad_tilt
        self.optimise_bpm_dx = optimise_bpm_dx
        self.optimise_bpm_dy = optimise_bpm_dy
        self.custom_knobs_to_optimise = custom_knobs_to_optimise
        if self.custom_knobs_to_optimise is not None:
            legacy = [
                knob
                for knob in self.custom_knobs_to_optimise
                if knob.endswith(".dk0") or knob.endswith(".dk1") or knob.endswith(".dk2")
            ]
            if legacy:
                raise ValueError(
                    "Legacy dknl knob names are not supported; use '.dk0l', '.dk1l', or '.dk2l': "
                    + ", ".join(legacy)
                )

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled."""
        return (
            any(s.enabled for s in self.get_supported_knob_specs())
            or self.optimise_energy
            or bool(self.custom_knobs_to_optimise)
        )

    @property
    def ac_dipole_name(self) -> str:
        """Return the AC-dipole exciter name for machines that define one."""
        raise NotImplementedError(f"{type(self).__name__} does not define an AC-dipole exciter")

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
        # Use an ordered-dict trick to deduplicate labels while preserving insertion order.
        seen: dict[str, None] = {}
        for spec in self.get_supported_knob_specs():
            if spec.enabled:
                seen[spec.label] = None
        if self.optimise_energy:
            seen["beam energy"] = None
        if self.custom_knobs_to_optimise:
            seen[f"custom knobs: {self.custom_knobs_to_optimise}"] = None
        if seen:
            LOGGER.info("Optimisation targets: %s", ", ".join(seen))
        else:
            LOGGER.info("No optimisation targets set.")

    @abstractmethod
    def copy_with(self, **overrides) -> Accelerator:
        """Return a new instance of the same type with selected parameters overridden."""
        pass

    def get_bend_lengths(self) -> dict[str, float] | None:
        """Return bend lengths required for accelerator-specific normalisation."""
        return None

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
        formatted = []
        for knob_name in knob_names:
            match = _INDEXED_RESULT_MULTIPOLE_RE.search(knob_name)
            if match is not None:
                table, index_str = match.groups()
                order = int(index_str) - 1
                suffix = f".dk{order}l" if table == "knl" else f".dk{order}sl"
                knob_name = _INDEXED_RESULT_MULTIPOLE_RE.sub(suffix, knob_name)
            formatted.append(knob_name)

        if not self.optimise_energy:
            return formatted

        return formatted

    def format_result_knobs(self, knobs: dict[str, float]) -> dict[str, float]:
        """Map optimisation-space knob values to user-facing result names."""
        names = self.format_result_knob_names(list(knobs))
        return dict(zip(names, knobs.values(), strict=True))

    @abstractmethod
    def get_supported_knob_specs(self) -> list[KnobSpec]:
        """Return the knob specifications supported by this accelerator."""
        pass

    @property
    def quadrupole_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return quadrupole misalignment patterns keyed by attribute name."""
        return {}

    @property
    def bpm_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return BPM/monitor misalignment patterns keyed by attribute name."""
        return {}

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        selected_specs: list[tuple[str, str, str, str | None]],
    ) -> None:
        """Run each selected attribute's ``_KNOB_PREPARATION`` before knob creation."""
        grouped: dict[tuple[str, str], list[str]] = {}
        for kind, attr, pattern, _nonzero_attr in selected_specs:
            body = _KNOB_PREPARATION.get(attr)
            if body is not None:
                grouped.setdefault((kind, body), []).append(pattern)

        for (element_kind, body), patterns in grouped.items():
            self._prepare_matching_elements(
                mad_iface, element_kind, tuple(dict.fromkeys(patterns)), body
            )

    def _prepare_matching_elements(
        self,
        mad_iface: GradientDescentMadInterface,
        element_kind: str,
        patterns: tuple[str, ...],
        body: str,
    ) -> None:
        """Run Lua ``body`` once per element of ``element_kind`` matching ``patterns``.

        Args:
            mad_iface: Interface owning the MAD-NG process holding ``loaded_sequence``
            element_kind: MAD-NG element kind to match (e.g. ``"quadrupole"``)
            patterns: Element name patterns to match
            body: Lua statements, seeing the matched element as ``e``
        """
        mad_iface.mad.send(f"""
        local element_kind = {mad_iface.py_name}:recv()
        local patterns = {mad_iface.py_name}:recv()
        for i, e in loaded_sequence:siter(magnet_range) do
            if e.kind == element_kind then
                for _, pattern in ipairs(patterns) do
                    if string.match(e.name, pattern) then
{textwrap.indent(body, " " * 24)}
                        break
                    end
                end
            end
        end
        """)
        mad_iface.mad.send(element_kind).send(patterns)

    def get_mad_attr_spec(self, kind: str, attribute: str) -> dict[str, str]:
        """Return accelerator-specific expressions for one element attribute."""
        del kind, attribute
        return {}

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return per-family override metadata keyed by family code d/q/s."""
        return {}

    @staticmethod
    @abstractmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from BPM name."""
        pass
