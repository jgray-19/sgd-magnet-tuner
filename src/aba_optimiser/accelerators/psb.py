"""PSB-specific accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymadng_utils.accelerators.psb import PSB as BasePSB  # noqa: N811

from aba_optimiser.accelerators.base import Accelerator, KnobSpec

if TYPE_CHECKING:
    from pathlib import Path


PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV = 0.160


class PSB(BasePSB, Accelerator):
    """Proton Synchrotron Booster accelerator configuration."""

    PATTERN_SBENDS = r"^BR%.BHZ%d+$"
    PATTERN_RBENDS = r"^BR%.BSW%d+L%d+%.%d+$"
    PATTERN_QUADRUPOLE = "^BR%.Q[FD][OE]%d+$"
    PATTERN_SEXTUPOLE = r"^BR%d+%.XNO%d+L1$"
    PATTERN_SKEW_SEXTUPOLE = r"^BR%d+%.XSK[26]L4$"
    PATTERN_CORRECTOR_H = r"^B[RE]%d+%.DHZ%d+L%d+$"
    PATTERN_CORRECTOR_V = r"^B[RE]%d+%.DVT%d+L%d+$"
    BEND_PERTURBATION_PATTERN = r"(?i)^BR\.(?:BHZ\d+|BSW\d+L\d+\.\d+)$"
    QUAD_PERTURBATION_PATTERN = r"(?i)^BR\.Q(?:FO\d+|DE\d+)$"
    BPM_PATTERN_TEMPLATE = "^BR{ring}%.BPM%d+L{ring}$"

    def __init__(
        self,
        ring: int,
        sequence_file: Path | str,
        kinetic_energy: float = PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV,
        particle: str = "proton",
        bpm_pattern: str | None = None,
        optimise_bends: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_quad_dy: bool = False,
        optimise_quad_dx: bool = False,
        optimise_sextupoles: bool = False,
        optimise_correctors: bool = False,
        optimise_energy: bool = False,
        optimise_bpm_dx: bool = False,
        optimise_bpm_dy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise PSB accelerator for a specific ring."""
        if ring not in (1, 2, 3, 4):
            raise ValueError(f"PSB ring must be 1, 2, 3, or 4, got {ring}")

        super().__init__(
            ring=ring,
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern or self.BPM_PATTERN_TEMPLATE.format(ring=ring),
            particle=particle,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_quad_dy=optimise_quad_dy,
            optimise_quad_dx=optimise_quad_dx,
            optimise_sextupoles=optimise_sextupoles,
            optimise_energy=optimise_energy,
            optimise_bpm_dx=optimise_bpm_dx,
            optimise_bpm_dy=optimise_bpm_dy,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )
        # PSB-specific optimisation flags not handled by any parent
        self.optimise_bends = optimise_bends
        self.optimise_correctors = optimise_correctors

    def copy_with(self, **overrides) -> PSB:
        """Return a new PSB instance with selected parameters overridden."""
        o = overrides
        return PSB(
            ring=o.get("ring", self.ring),
            sequence_file=o.get("sequence_file", self.sequence_file),
            kinetic_energy=o.get("kinetic_energy", self.kinetic_energy),
            particle=o.get("particle", self.particle),
            bpm_pattern=o.get("bpm_pattern", self.bpm_pattern),
            optimise_energy=o.get("optimise_energy", self.optimise_energy),
            optimise_quadrupoles=o.get("optimise_quadrupoles", self.optimise_quadrupoles),
            optimise_sextupoles=o.get("optimise_sextupoles", self.optimise_sextupoles),
            optimise_bends=o.get("optimise_bends", self.optimise_bends),
            optimise_correctors=o.get("optimise_correctors", self.optimise_correctors),
            optimise_quad_dx=o.get("optimise_quad_dx", self.optimise_quad_dx),
            optimise_quad_dy=o.get("optimise_quad_dy", self.optimise_quad_dy),
            optimise_bpm_dx=o.get("optimise_bpm_dx", self.optimise_bpm_dx),
            optimise_bpm_dy=o.get("optimise_bpm_dy", self.optimise_bpm_dy),
            custom_knobs_to_optimise=o.get("custom_knobs_to_optimise", self.custom_knobs_to_optimise),
        )

    @property
    def seq_name(self) -> str:
        """Return the sequence name for the selected PSB ring."""
        return f"psb{self.ring}"

    def get_supported_knob_specs(self) -> list[KnobSpec]:
        """Return the PSB knob specifications currently supported."""
        bpm_pattern = self.BPM_PATTERN_TEMPLATE.format(ring=self.ring)
        # fmt: off
        return [
            KnobSpec("quadrupole", "k1",      self.PATTERN_QUADRUPOLE,    "k1", self.optimise_quadrupoles, "quadrupoles"),
            KnobSpec("sbend",      "k0",      self.PATTERN_SBENDS,        "k0", self.optimise_bends,       "bends"),
            KnobSpec("rbend",      "k0",      self.PATTERN_RBENDS,        "k0", self.optimise_bends,       "bends"),
            KnobSpec("multipole",  "knl[3]",  self.PATTERN_SEXTUPOLE,     "knl[3]", self.optimise_sextupoles,  "sextupoles"),
            KnobSpec("multipole",  "ksl[3]",  self.PATTERN_SKEW_SEXTUPOLE,"ksl[3]", self.optimise_sextupoles,  "skew sextupoles"),
            KnobSpec("hkicker",    "kick",    self.PATTERN_CORRECTOR_H,   None, self.optimise_correctors,  "correctors"),
            KnobSpec("vkicker",    "kick",    self.PATTERN_CORRECTOR_V,   None, self.optimise_correctors,  "correctors"),
            KnobSpec("quadrupole", "dy",      self.PATTERN_QUADRUPOLE,    "k1", self.optimise_quad_dy,     "quadrupole vertical offsets"),
            KnobSpec("quadrupole", "dx",      self.PATTERN_QUADRUPOLE,    "k1", self.optimise_quad_dx,     "quadrupole horizontal offsets"),
            KnobSpec("monitor",    "dx",      bpm_pattern,                None, self.optimise_bpm_dx,      "BPM horizontal offsets"),
            KnobSpec("monitor",    "dy",      bpm_pattern,                None, self.optimise_bpm_dy,      "BPM vertical offsets"),
        ]
        # fmt: on

    @property
    def quadrupole_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return PSB quadrupole patterns eligible for misalignment knobs."""
        return {
            "dx": (self.PATTERN_QUADRUPOLE,),
            "dy": (self.PATTERN_QUADRUPOLE,),
        }

    @property
    def bpm_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return PSB BPM patterns eligible for misalignment knobs."""
        bpm_pattern = self.BPM_PATTERN_TEMPLATE.format(ring=self.ring)
        return {
            "dx": (bpm_pattern,),
            "dy": (bpm_pattern,),
        }

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation metadata for PSB ring bends and QFO/QDE quadrupoles."""
        return {
            "d": {
                "default_rel_std": 8e-4,
                "pattern": self.BEND_PERTURBATION_PATTERN,
            },
            "q": {
                "default_rel_std": 2e-3,
                "pattern": self.QUAD_PERTURBATION_PATTERN,
            },
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from PSB monitor names, including ACD markers."""
        name = bpm_name.upper()
        if any(token in name for token in (".BPM", ".BWS", ".BPP", ".BPT")):
            return "HV"
        if name.endswith("_AFTER") or name.endswith("_BEFORE"):
            return "HV"
        raise ValueError(f"Unsupported PSB monitor name for plane inference: {bpm_name}")
