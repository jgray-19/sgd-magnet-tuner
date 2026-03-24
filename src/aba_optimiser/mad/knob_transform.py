"""Knob-space conversion helpers for optimisation workflows.

This module keeps conversion policy between optimisation-space dknl knobs and
user-facing absolute strengths in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KnobSpaceTransform:
    """Translate between optimisation-space knobs and absolute strengths."""

    dknl_knob_to_absolute: dict[str, str]
    absolute_to_dknl_knob: dict[str, str]
    dknl_knob_base_strength: dict[str, float]
    dknl_knob_length: dict[str, float]

    def __post_init__(self) -> None:
        for knob, length in self.dknl_knob_length.items():
            if length == 0.0:
                raise ValueError(f"Zero length is invalid for dknl knob {knob!r}")

    @classmethod
    def empty(cls) -> KnobSpaceTransform:
        """Return an empty transform that performs identity mappings."""
        return cls({}, {}, {}, {})

    @staticmethod
    def dknl_attr_from_knob(knob_name: str) -> str | None:
        """Return the absolute strength attribute for a dknl knob suffix."""
        if knob_name.endswith(".dk0"):
            return "k0"
        if knob_name.endswith(".dk1"):
            return "k1"
        if knob_name.endswith(".dk2"):
            return "k2"
        return None

    @classmethod
    def absolute_name_from_dknl_knob(cls, knob_name: str) -> str | None:
        """Return absolute strength name (e.g. ``*.k1``) for ``*.dk1`` knobs."""
        attr = cls.dknl_attr_from_knob(knob_name)
        if attr is None:
            return None
        base_name = knob_name.rsplit(".", 1)[0]
        return f"{base_name}.{attr}"

    def absolute_to_optimisation_knobs(self, knob_values: dict[str, float]) -> dict[str, float]:
        """Convert absolute strengths (k0/k1/k2) to optimisation knobs (dk*)."""
        converted: dict[str, float] = {}
        for name, value in knob_values.items():
            dknl_name = self.absolute_to_dknl_knob.get(name)
            if dknl_name is None:
                converted[name] = float(value)
                continue
            base = self.dknl_knob_base_strength[dknl_name]
            length = self.dknl_knob_length[dknl_name]
            converted[dknl_name] = float(value - base) * length
        return converted

    def optimisation_to_absolute_knobs(self, knob_values: dict[str, float]) -> dict[str, float]:
        """Convert optimisation knobs (dk*) back to absolute strengths."""
        converted: dict[str, float] = {}
        for name, value in knob_values.items():
            absolute_name = self.dknl_knob_to_absolute.get(name)
            if absolute_name is None:
                converted[name] = float(value)
                continue
            base = self.dknl_knob_base_strength[name]
            length = self.dknl_knob_length[name]
            converted[absolute_name] = base + float(value) / length
        return converted

    def format_knob_names_for_output(self, knob_names: list[str]) -> list[str]:
        """Map ``*.dk*`` names to ``*.k*`` names for output only."""
        return [self.dknl_knob_to_absolute.get(knob, knob) for knob in knob_names]

    def convert_uncertainties_to_absolute(
        self,
        knob_names: list[str],
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Convert optimisation-space uncertainties to absolute-strength scale."""
        converted = np.asarray(uncertainties, dtype=np.float64).copy()
        for idx, knob in enumerate(knob_names):
            length = self.dknl_knob_length.get(knob)
            if length is None:
                continue
            converted[idx] = converted[idx] / length
        return converted

    def optimisation_to_absolute_affine(
        self,
        knob_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return affine map arrays for converting optimisation -> absolute.

        Returns:
            tuple[offsets, dabs_dopt] such that:
                absolute = offsets + optimisation * dabs_dopt
        """
        offsets = np.zeros(len(knob_names), dtype=np.float64)
        dabs_dopt = np.ones(len(knob_names), dtype=np.float64)

        for idx, knob in enumerate(knob_names):
            length = self.dknl_knob_length.get(knob)
            if length is None:
                continue
            offsets[idx] = self.dknl_knob_base_strength[knob]
            dabs_dopt[idx] = 1.0 / length

        return offsets, dabs_dopt
