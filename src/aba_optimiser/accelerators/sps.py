"""SPS-specific accelerator implementation with generic optimisation targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator, KnobSpec

if TYPE_CHECKING:
    from pathlib import Path


class SPS(Accelerator):
    """Super Proton Synchrotron accelerator configuration.

    This class intentionally exposes only generic optimisation categories,
    without LHC-specific options (bends, correctors, quadrupole displacements).
    """

    # Restrict to main SPS families:
    # quadrupoles QF/QD/QFA/QDA, sextupoles LSF/LSD.
    PATTERN_QUADRUPOLE = "^Q[FD]A?%."
    PATTERN_SEXTUPOLE = "^LS[FD]A?%."
    BPM_PATTERN = "^BP[HV]%."

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float = 450.0,
        bpm_pattern: str = BPM_PATTERN,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise SPS accelerator.

        Args:
            sequence_file: Path to sequence file.
            kinetic_energy: Particle kinetic energy in GeV.
            seq_name: Sequence name to use in MAD.
            optimise_quadrupoles: Whether to optimise quadrupole strengths.
            optimise_sextupoles: Whether to optimise sextupole strengths.
            optimise_energy: Whether to optimise beam energy.
            custom_knobs_to_optimise: Optional explicit knob whitelist.
        """
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern,
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )

    def copy_with(self, **overrides) -> SPS:
        """Return a new SPS instance with selected parameters overridden."""
        o = overrides
        return SPS(
            sequence_file=o.get("sequence_file", self.sequence_file),
            kinetic_energy=o.get("kinetic_energy", self.kinetic_energy),
            bpm_pattern=o.get("bpm_pattern", self.bpm_pattern),
            optimise_energy=o.get("optimise_energy", self.optimise_energy),
            optimise_quadrupoles=o.get("optimise_quadrupoles", self.optimise_quadrupoles),
            optimise_sextupoles=o.get("optimise_sextupoles", self.optimise_sextupoles),
            custom_knobs_to_optimise=o.get(
                "custom_knobs_to_optimise", self.custom_knobs_to_optimise
            ),
        )

    @property
    def seq_name(self) -> str:
        """Return the sequence name for SPS."""
        return "sps"

    def get_supported_knob_specs(self) -> list[KnobSpec]:
        """Return generic SPS knob specifications."""
        return [
            KnobSpec(
                "quadrupole",
                "k1",
                self.PATTERN_QUADRUPOLE,
                "k1",
                self.optimise_quadrupoles,
                "quadrupoles",
            ),
            KnobSpec(
                "sextupole",
                "k2",
                self.PATTERN_SEXTUPOLE,
                "k2",
                self.optimise_sextupoles,
                "sextupoles",
            ),
        ]

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Return perturbation-family metadata for SPS main families."""
        #https://cds.cern.ch/record/66887/files/LABII-MA-Int-75-2.pdf?version=1
        return {
            "d": {
                "default_rel_std": 2e-5,
                "pattern": self.PATTERN_SEXTUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
            "q": {
                "default_rel_std": 2e-4,
                "pattern": self.PATTERN_QUADRUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
            "s": {
                "default_rel_std": 10e-4,
                "pattern": self.PATTERN_SEXTUPOLE.replace("%", "\\"),  # Change from lua to regex pattern
            },
        }

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from SPS BPM family name."""
        name = bpm_name.upper()
        if name.startswith("BPH"):
            return "H"
        if name.startswith("BPV"):
            return "V"
        raise ValueError(f"Unsupported SPS BPM name for plane inference: {bpm_name}")

    def get_ac_dipole_marker(self) -> str:
        """SPS does not use the LHC AC-dipole exciter model."""
        raise NotImplementedError("SPS does not define an AC-dipole exciter marker")

    @property
    def ac_dipole_name(self) -> str:
        """SPS does not use an AC-dipole exciter model."""
        raise NotImplementedError("SPS does not define an AC-dipole exciter")

    @property
    def ac_dipole_location(self) -> tuple[str, float]:
        """SPS does not use the LHC AC-dipole exciter model."""
        raise NotImplementedError("SPS does not define an AC-dipole exciter location")

    # def get_exciter_bpm(
    #     self,
    # ) -> tuple[str, str]:
    #     """SPS does not define an upstream-style AC-dipole exciter BPM."""
    #     raise NotImplementedError("SPS does not define an AC-dipole exciter BPM")

    @property
    def tune_variables(self) -> tuple[str, str]:
        """Return SPS tune variable names."""
        return "kqf", "kqd"

    @property
    def tune_integers(self) -> tuple[int, int]:
        """Return SPS integer tunes."""
        return 20, 20
