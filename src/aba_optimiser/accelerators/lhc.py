"""LHC-specific accelerator implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng_utils.accelerators.lhc import LHC as BaseLHC  # noqa: N811

from aba_optimiser.accelerators.base import Accelerator, KnobSpec
from aba_optimiser.accelerators.magnet_grouping import normalise_lhcbend_magnets

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface


class LHC(BaseLHC, Accelerator):
    """Large Hadron Collider accelerator configuration.

    This class encapsulates LHC-specific parameters like beam numbers,
    default BPMs, and sequence file locations.
    """

    # LHC magnet patterns as class constants
    PATTERN_MAIN_BEND = "MB%."
    PATTERN_RBEND = "MB[RXWAL]%w*%."
    PATTERN_MAIN_QUAD = "MQ%."
    PATTERN_CORRECTOR = "MCB"
    PATTERN_QUAD_NON_TUNE = "MQ[^.TSY]"  # Explicitly not MQS, MQT or MQ., but still quadrupoles
    PATTERN_QUAD_DISPLACEMENT_Y = (
        "MQ[^ST]"  # Triplet quadrupoles and main quads with potential vertical misalignments
    )
    PATTERN_QUAD_DISPLACEMENT_X = (
        "MQ[^.TSYM]"  # Triplet quadrupoles, warm quads with potential horizontal misalignments
    )
    PATTERN_SEXTUPOLE = "MSS?%."
    QUAD_ERROR_TABLE = {
        "MQ.": 18e-4,
        "MQM": 12e-4,
        "MQY": 8e-4,
        "MQX": 10e-4,
        "MQW": 15e-4,
        # "MQT": 75e-4,
    }
    BPM_PATTERN = "^BPM.*$"

    def __init__(
        self,
        beam: int,
        sequence_file: Path | str,
        kinetic_energy: float = 6800.0,
        particle: str = "proton",
        bpm_pattern: str = BPM_PATTERN,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_energy: bool = False,
        # LHC-specific control
        optimise_correctors: bool = False,
        optimise_bends: bool = False,
        normalise_bends: bool | None = None,
        optimise_other_quadrupoles: bool = False,
        optimise_quad_dx: bool = False,
        optimise_quad_dy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise LHC accelerator for a specific beam.

        Args:
            beam: Beam number (1 or 2)
            sequence_file: Path to sequence file
            kinetic_energy: Particle kinetic energy in GeV
            optimise_quadrupoles: Whether to optimise quadrupoles
            optimise_sextupoles: Whether to optimise sextupoles
            bpm_pattern: Pattern for identifying BPMs in the sequence
            optimise_bends: Whether to optimise dipole bends
            normalise_bends: Whether to normalise bend strengths
            optimise_correctors: Whether to optimise corrector magnets
            optimise_energy: Whether to optimise beam energy

        Raises:
            ValueError: If an invalid beam number is provided
        """
        if beam not in (1, 2):
            raise ValueError(f"LHC beam must be 1 or 2, got {beam}")

        super().__init__(
            beam=beam,
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            particle=particle,
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
            optimise_quad_dx=optimise_quad_dx,
            optimise_quad_dy=optimise_quad_dy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )
        # LHC-specific optimisation flags not handled by any parent
        self.optimise_bends = optimise_bends
        if normalise_bends is None:
            normalise_bends = optimise_bends
        self.normalise_bends = normalise_bends
        self.optimise_correctors = optimise_correctors
        self.optimise_other_quadrupoles = optimise_other_quadrupoles
        self.bend_lengths: dict[str, float] | None = None

    def copy_with(self, **overrides) -> LHC:
        """Return a new LHC instance with selected parameters overridden."""
        o = overrides
        return LHC(
            beam=o.get("beam", self.beam),
            sequence_file=o.get("sequence_file", self.sequence_file),
            kinetic_energy=o.get("kinetic_energy", self.kinetic_energy),
            particle=o.get("particle", self.particle),
            bpm_pattern=o.get("bpm_pattern", self.bpm_pattern),
            optimise_energy=o.get("optimise_energy", self.optimise_energy),
            optimise_quadrupoles=o.get("optimise_quadrupoles", self.optimise_quadrupoles),
            optimise_sextupoles=o.get("optimise_sextupoles", self.optimise_sextupoles),
            optimise_correctors=o.get("optimise_correctors", self.optimise_correctors),
            optimise_bends=o.get("optimise_bends", self.optimise_bends),
            normalise_bends=o.get("normalise_bends", self.normalise_bends),
            optimise_other_quadrupoles=o.get("optimise_other_quadrupoles", self.optimise_other_quadrupoles),
            optimise_quad_dx=o.get("optimise_quad_dx", self.optimise_quad_dx),
            optimise_quad_dy=o.get("optimise_quad_dy", self.optimise_quad_dy),
            custom_knobs_to_optimise=o.get("custom_knobs_to_optimise", self.custom_knobs_to_optimise),
        )

    def get_bend_lengths(self) -> dict[str, float] | None:
        """Return LHC bend lengths when bend normalisation is enabled."""
        if not (self.optimise_bends and self.normalise_bends):
            return None
        return self.bend_lengths

    def normalise_true_strengths(
        self,
        true_strengths: dict[str, float],
        bend_lengths: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Normalise LHC bend strengths when applicable."""
        if bend_lengths is None:
            bend_lengths = self.bend_lengths
        if self.optimise_bends and bend_lengths:
            return normalise_lhcbend_magnets(true_strengths, bend_lengths)
        return true_strengths

    def get_supported_knob_specs(self) -> list[KnobSpec]:
        """Return LHC-specific knob specifications."""
        # fmt: off
        specs = [
            KnobSpec("sbend",      "k0",   self.PATTERN_MAIN_BEND,      "k0", self.optimise_bends,              "bends"),
            KnobSpec("rbend",      "k0",   self.PATTERN_RBEND,          "k0", self.optimise_bends,              "bends"),
            KnobSpec("quadrupole", "k1",   self.PATTERN_MAIN_QUAD,      "k1", self.optimise_quadrupoles,        "main quadrupoles"),
            KnobSpec("quadrupole", "k1",   self.PATTERN_QUAD_NON_TUNE,  "k1", self.optimise_other_quadrupoles,  "other quadrupoles"),
            KnobSpec("sextupole",  "k2",   self.PATTERN_SEXTUPOLE,      "k2", self.optimise_sextupoles,         "sextupoles"),
            KnobSpec("hkicker",    "kick", self.PATTERN_CORRECTOR,      None, self.optimise_correctors,         "correctors"),
            KnobSpec("vkicker",    "kick", self.PATTERN_CORRECTOR,      None, self.optimise_correctors,         "correctors"),
        ]
        # fmt: on
        label_map = {"dx": "quadrupole horizontal offsets", "dy": "quadrupole vertical offsets"}
        for attr, patterns in self.quadrupole_misalignment_patterns.items():
            for pattern in patterns:
                specs.append(KnobSpec("quadrupole", attr, pattern, "k1", getattr(self, f"optimise_quad_{attr}"), label_map[attr]))
        return specs

    @property
    def quadrupole_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return LHC quadrupole patterns eligible for misalignment knobs."""
        return {
            "dx": (self.PATTERN_QUAD_DISPLACEMENT_X,),
            "dy": (self.PATTERN_QUAD_DISPLACEMENT_Y,),
        }

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        selected_specs: list[tuple[str, str, str, str | None]],
    ) -> None:
        """Prepare LHC-specific MAD state for knob creation."""
        super().prepare_mad_for_knob_creation(mad_iface, selected_specs)
        if self.optimise_bends and self.normalise_bends:
            mad_iface.mad.send(f"""
            bend_dict = {{}}
            bend_lengths = {{}}
            for i, e in loaded_sequence:siter(magnet_range) do
                if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                    bend_dict[e.name .. ".dk0l"] = e.k0
                    bend_lengths[e.name .. ".dk0l"] = e.l
                end
            end
            {mad_iface.py_name}:send(bend_dict, true)
            {mad_iface.py_name}:send(bend_lengths, true)
            bend_dict = {mad_iface.py_name}:recv()
            """)
            true_strengths_dict: dict[str, float] = mad_iface.mad.recv()
            self.bend_lengths = mad_iface.mad.recv()
            normalised_names = normalise_lhcbend_magnets(true_strengths_dict, self.bend_lengths)
            mad_iface.mad.send(normalised_names)

    def get_mad_attr_spec(self, kind: str, attribute: str) -> dict[str, str]:
        """Return LHC-specific attr naming/value expressions."""
        if not self.normalise_bends or kind != "sbend" or attribute != "k0":
            return {}
        return {
            "name_expr": 'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".dk0l"',
            "mad_value": "bend_dict[k_str_name]",
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """LHC BPMs measure both planes simultaneously."""
        del bpm_name
        return "HV"

    def get_perturbation_families(self) -> dict[str, dict[str, float | str | dict]]:
        """Return perturbation-family metadata for LHC."""
        return {
            "d": {
                "default_rel_std": 1e-4,
                "pattern": "MB\\.",
            },
            "q": {
                "relative_error_table": self.QUAD_ERROR_TABLE,
            },
            "s": {
                "default_rel_std": 1e-4,
                "pattern": "MS\\.",
            },
        }
