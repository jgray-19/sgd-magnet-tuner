"""PSB-specific accelerator implementation."""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator

if TYPE_CHECKING:
    from pathlib import Path


PROTON_MASS_GEV = 0.9382720813
PSB_FLAT_BOTTOM_MOMENTUM_GEV = 0.571
PSB_FLAT_BOTTOM_TOTAL_ENERGY_GEV = sqrt(
    PSB_FLAT_BOTTOM_MOMENTUM_GEV**2 + PROTON_MASS_GEV**2
)


class PSB(Accelerator):
    """Proton Synchrotron Booster accelerator configuration."""

    PATTERN_QUADRUPOLE = "^BR%.Q(FO%d+|DE%d+)$"
    QUAD_PERTURBATION_PATTERN = r"^BR\.Q(?:FO\d+|DE\d+)$"
    BPM_PATTERN_TEMPLATE = "^BR{ring}%.BPM"

    def __init__(
        self,
        ring: int,
        sequence_file: Path | str,
        beam_energy: float = PSB_FLAT_BOTTOM_TOTAL_ENERGY_GEV,
        bpm_pattern: str | None = None,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise PSB accelerator for a specific ring."""
        if ring not in (1, 2, 3, 4):
            raise ValueError(f"PSB ring must be 1, 2, 3, or 4, got {ring}")

        self.ring = ring

        super().__init__(
            sequence_file=sequence_file,
            beam_energy=beam_energy,
            bpm_pattern=bpm_pattern or self.BPM_PATTERN_TEMPLATE.format(ring=ring),
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )

    @property
    def seq_name(self) -> str:
        """Return the sequence name for the selected PSB ring."""
        return f"psb{self.ring}"

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return the PSB knob specifications currently supported."""
        return [
            (
                "quadrupole",
                "k1",
                self.PATTERN_QUADRUPOLE,
                True,
                self.optimise_quadrupoles,
            ),
        ]

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation metadata for PSB quadrupoles."""
        return {
            "q": {
                "default_rel_std": 2e-4,
                "pattern": self.QUAD_PERTURBATION_PATTERN,
            },
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from PSB monitor names."""
        name = bpm_name.upper()
        if any(token in name for token in (".BPM", ".BWS", ".BPP", ".BPT")):
            return "HV"
        raise ValueError(f"Unsupported PSB monitor name for plane inference: {bpm_name}")

    def get_tune_variables(self) -> tuple[str, str]:
        """Return PSB tune variable names."""
        return "kBRQF", "kBRQD"

    def get_tune_integers(self) -> tuple[int, int]:
        """Return PSB integer tunes."""
        return 4, 4
