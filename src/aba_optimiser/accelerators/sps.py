"""SPS-specific accelerator implementation with generic optimisation targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator

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
        beam_energy: float = 450.0,
        bpm_pattern: str = BPM_PATTERN,
        optimise_quadrupoles: bool = False,
        optimise_sextupoles: bool = False,
        optimise_energy: bool = False,
        custom_knobs_to_optimise: list[str] | None = None,
    ):
        """Initialise SPS accelerator.

        Args:
            sequence_file: Path to sequence file.
            beam_energy: Beam energy in GeV.
            seq_name: Sequence name to use in MAD.
            optimise_quadrupoles: Whether to optimise quadrupole strengths.
            optimise_sextupoles: Whether to optimise sextupole strengths.
            optimise_energy: Whether to optimise beam energy.
            custom_knobs_to_optimise: Optional explicit knob whitelist.
        """
        super().__init__(
            sequence_file=sequence_file,
            beam_energy=beam_energy,
            bpm_pattern=bpm_pattern,
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
            custom_knobs_to_optimise=custom_knobs_to_optimise,
        )

    @property
    def seq_name(self) -> str:
        """Return the sequence name for SPS."""
        return "sps"

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return generic SPS knob specifications.

        Returns:
            List of (kind, attribute, pattern, zero_check, optimise_flag) tuples.
        """
        return [
            ("quadrupole", "k1", self.PATTERN_QUADRUPOLE, True, self.optimise_quadrupoles),
            ("sextupole", "k2", self.PATTERN_SEXTUPOLE, True, self.optimise_sextupoles),
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


    def get_tune_variables(self) -> tuple[str, str]:
        """Return SPS tune variable names."""
        return "kqf", "kqd"

    def get_tune_integers(self) -> tuple[int, int]:
        """Return SPS integer tunes."""
        return 20, 20
