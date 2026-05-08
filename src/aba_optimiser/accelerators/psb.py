"""PSB-specific accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV = 0.160


class PSB(Accelerator):
    """Proton Synchrotron Booster accelerator configuration."""

    PATTERN_SBENDS = r"^BR%.BHZ%d+$"
    PATTERN_RBENDS = r"^BR%.BSW%d+L%d+%.%d+$"
    PATTERN_QUADRUPOLE = "^BR%.Q[FD][OE]%d+$"
    PATTERN_SEXTUPOLE = r"^BR%d+%.XNO[49]L1$"
    PATTERN_SKEW_SEXTUPOLE = r"^BR%d+%.XSK[26]L4$"
    PATTERN_CORRECTOR_H = r"^B[RE]%d+%.DHZ%d+L%d+$"
    PATTERN_CORRECTOR_V = r"^B[RE]%d+%.DVT%d+L%d+$"
    QUAD_PERTURBATION_PATTERN = r"^BR\.Q(?:FO\d+|DE\d+)$"
    BPM_PATTERN_TEMPLATE = "^BR{ring}%.BPM"

    def __init__(
        self,
        ring: int,
        sequence_file: Path | str,
        kinetic_energy: float = PSB_FLAT_BOTTOM_KINETIC_ENERGY_GEV,
        bpm_pattern: str | None = None,
        optimise_bends: bool = False,
        optimise_quadrupoles: bool = False,
        optimise_quad_dy: bool = False,
        optimise_quad_dx: bool = False,
        optimise_sextupoles: bool = False,
        optimise_correctors: bool = False,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise PSB accelerator for a specific ring."""
        if ring not in (1, 2, 3, 4):
            raise ValueError(f"PSB ring must be 1, 2, 3, or 4, got {ring}")

        self.ring = ring
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern or self.BPM_PATTERN_TEMPLATE.format(ring=ring),
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
            optimise_quad_dy=optimise_quad_dy,
            optimise_quad_dx=optimise_quad_dx,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )
        self.optimise_bends = optimise_bends
        self.optimise_correctors = optimise_correctors

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled."""
        return super().has_any_optimisation() or self.optimise_bends or self.optimise_correctors

    @property
    def seq_name(self) -> str:
        """Return the sequence name for the selected PSB ring."""
        return f"psb{self.ring}"

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, str | None, bool]]:
        """Return the PSB knob specifications currently supported.

        Returns:
            List of (kind, attribute, pattern, nonzero_attr, optimise_flag) tuples defining
            all possible knobs that can be created for this accelerator.
        """
        # fmt: off
        return [
            ("quadrupole", "k1", self.PATTERN_QUADRUPOLE, "k1", self.optimise_quadrupoles),

            # Bends
            ("sbend", "k0", self.PATTERN_SBENDS, "k0", self.optimise_bends),
            ("rbend", "k0", self.PATTERN_RBENDS, "k0", self.optimise_bends),

            # Sextupoles
            ("multipole", "knl[3]", self.PATTERN_SEXTUPOLE, None, self.optimise_sextupoles),
            ("multipole", "ksl[3]", self.PATTERN_SKEW_SEXTUPOLE, None, self.optimise_sextupoles),

            # Correctors
            ("hkicker", "kick", self.PATTERN_CORRECTOR_H, None, self.optimise_correctors),
            ("vkicker", "kick", self.PATTERN_CORRECTOR_V, None, self.optimise_correctors),

            # Quadrupole misalignments
            ("quadrupole", "dy", self.PATTERN_QUADRUPOLE, "k1", self.optimise_quad_dy),
            ("quadrupole", "dx", self.PATTERN_QUADRUPOLE, "k1", self.optimise_quad_dx),
        ]
        # fmt: on

    @property
    def quadrupole_misalignment_patterns(self) -> dict[str, tuple[str, ...]]:
        """Return PSB quadrupole patterns eligible for misalignment knobs."""
        return {
            "dx": (self.PATTERN_QUADRUPOLE,),
            "dy": (self.PATTERN_QUADRUPOLE,),
        }

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation metadata for PSB quadrupoles."""
        return {
            "q": {
                "default_rel_std": 2e-4,
                "pattern": self.QUAD_PERTURBATION_PATTERN,
            },
        }

    def apply_accelerator_specific_errors(self, mad_iface: AbaMadInterface) -> None:
        """PSB has no accelerator-specific startup error tables."""
        del mad_iface

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        local = bpm_name.split(".", 1)[-1]
        if local.upper().startswith("B"):
            return "HV"
        raise ValueError(f"Unsupported PSB monitor name: {bpm_name}")

    def get_ac_dipole_marker(self) -> str:
        return "HACMAP"  # "VACMAP" is also valid

    @property
    def ac_dipole_location(self) -> tuple[str, float]:
        return self.get_ac_dipole_marker(), 0.0

    def get_exciter_bpm(
        self,
        plane: str,
        common_bpms: list[str] | None = None,
    ) -> tuple[str, str] | None:
        """Return the two BPMs adjacent to the PSB exciter."""
        del plane, common_bpms
        return f"BR{self.ring}.BPM2L3", f"BR{self.ring}.BPM3L3"

    @property
    def tune_variables(self) -> tuple[str, str]:
        """Return PSB tune variable names."""
        return "kBRQF", "kBRQD"

    @property
    def tune_integers(self) -> tuple[int, int]:
        """Return PSB integer tunes."""
        return 4, 4
