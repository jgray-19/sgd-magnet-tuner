"""LHC-specific accelerator implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aba_optimiser.accelerators.base import Accelerator
from aba_optimiser.physics.lhc_bends import normalise_lhcbend_magnets

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface


class LHC(Accelerator):
    """Large Hadron Collider accelerator configuration.

    This class encapsulates LHC-specific parameters like beam numbers,
    default BPMs, and sequence file locations.
    """

    # LHC magnet patterns as class constants
    PATTERN_MAIN_BEND = "MB%."
    PATTERN_RBEND = "MB[RXWAL]%w*%."
    PATTERN_MAIN_QUAD = "MQ%."
    PATTERN_CORRECTOR = "MCB"
    PATTERN_QUAD_NON_TUNE = "MQ[^T.]"  # Explicitly not MQT or MQ., but still quadrupoles
    PATTERN_QUAD_DISPLACEMENT = "MQX*%."  # "MQX.*[LR][15]"
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
        beam_energy: float = 6800.0,
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
    ):
        """Initialise LHC accelerator for a specific beam.

        Args:
            beam: Beam number (1 or 2)
            sequence_file: Path to sequence file
            beam_energy: Beam energy in GeV
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

        self.beam = beam

        # Store LHC-specific optimisation flags
        self.optimise_bends = optimise_bends
        if normalise_bends is None:
            normalise_bends = optimise_bends
        self.normalise_bends = normalise_bends
        self.optimise_correctors = optimise_correctors
        self.optimise_other_quadrupoles = optimise_other_quadrupoles
        self.optimise_quad_dx = optimise_quad_dx
        self.optimise_quad_dy = optimise_quad_dy


        # Initialise base Accelerator
        super().__init__(
            sequence_file=sequence_file,
            beam_energy=beam_energy,
            bpm_pattern=bpm_pattern,
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_sextupoles=optimise_sextupoles,
        )

    @property
    def seq_name(self) -> str:
        """Return the sequence name for this LHC beam."""
        return f"lhcb{self.beam}"

    def has_any_optimisation(self) -> bool:
        """Check if any optimisation is enabled.

        Returns:
            True if at least one optimisation type is enabled
        """
        return (
            self.optimise_bends
            or self.optimise_quadrupoles
            or self.optimise_other_quadrupoles
            or self.optimise_sextupoles
            or self.optimise_correctors
            or self.optimise_energy
            or self.optimise_quad_dx
            or self.optimise_quad_dy
        )

    def log_optimisation_targets(self) -> None:
        """Log the optimisation targets for this LHC accelerator."""
        targets = []
        if self.optimise_bends:
            targets.append("bends")
        if self.optimise_quadrupoles:
            targets.append("main quadrupoles")
        if self.optimise_other_quadrupoles:
            targets.append("other quadrupoles")
        if self.optimise_sextupoles:
            targets.append("sextupoles")
        if self.optimise_correctors:
            targets.append("correctors")
        if self.optimise_energy:
            targets.append("beam energy")
        if self.optimise_quad_dx:
            targets.append("quadrupole horizontal offsets")
        if self.optimise_quad_dy:
            targets.append("quadrupole vertical offsets")
        if targets:
            LOGGER.info(f"Optimisation targets: {', '.join(targets)}")
        else:
            LOGGER.info("No optimisation targets set.")

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

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return LHC-specific knob specifications.

        Returns:
            List of (kind, attribute, pattern, zero_check, optimise_flag) tuples defining
            all possible knobs that can be created for LHC optimization.
        """
        return [
            ("sbend", "k0", self.PATTERN_MAIN_BEND, True, self.optimise_bends),
            # ("rbend", "k0", self.PATTERN_RBEND, True, self.optimise_bends),
            ("quadrupole", "k1", self.PATTERN_MAIN_QUAD, True, self.optimise_quadrupoles),
            ("quadrupole", "k1", self.PATTERN_QUAD_NON_TUNE, True, self.optimise_other_quadrupoles),
            ("hkicker", "kick", self.PATTERN_CORRECTOR, False, self.optimise_correctors),
            ("vkicker", "kick", self.PATTERN_CORRECTOR, False, self.optimise_correctors),
            ("quadrupole", "dx", self.PATTERN_QUAD_DISPLACEMENT, False, self.optimise_quad_dx),
            ("quadrupole", "dy", self.PATTERN_QUAD_DISPLACEMENT, False, self.optimise_quad_dy),
        ]

    def prepare_mad_for_knob_creation(
        self,
        mad_iface: GradientDescentMadInterface,
        selected_specs: list[tuple[str, str, str, bool]],
    ) -> None:
        """Prepare LHC-specific MAD state for knob creation."""
        del selected_specs
        if self.optimise_bends and self.normalise_bends:
            mad_iface.mad.send(f"""
            bend_dict = {{}}
            bend_lengths = {{}}
            for i, e in loaded_sequence:siter(magnet_range) do
                if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                    bend_dict[e.name .. ".k0"] = e.k0
                    bend_lengths[e.name .. ".k0"] = e.l
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

        if self.optimise_quad_dx or self.optimise_quad_dy:
            insert_qx_qy_str = ""
            if self.optimise_quad_dx:
                insert_qx_qy_str += "e.dx = e.dx or 1e-6\n"
            if self.optimise_quad_dy:
                insert_qx_qy_str += "e.dy = e.dy or 1e-6\n"
            mad_iface.mad.send(f"""
            for i, e in loaded_sequence:siter(magnet_range) do
                if e.kind == "quadrupole" and string.match(e.name, "{self.PATTERN_QUAD_DISPLACEMENT}") then
                    {insert_qx_qy_str}
                    e.misalign = MAD.typeid.deferred{{dx =\\->e.dx, dy =\\->e.dy}}
                end
            end
            """)

    def get_mad_attr_specs(self) -> dict[str, dict[str, str]]:
        """Return LHC-specific attr naming/value expressions."""
        if not self.normalise_bends:
            return {}
        return {
            "sbend": {
                "name_expr": 'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".k0"',
                "mad_value": "bend_dict[k_str_name]",
            },
            "rbend": {
                "name_expr": 'string.gsub(e.name, "(MB[RXWAL]%w*%.)([A-G]?)([0-9]+[LR][1-8].*)", "%1%3") .. (e.k0 >= 0 and "_p" or "_n") .. ".k0"',
                "mad_value": "bend_dict[k_str_name]",
            },
        }

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

    def get_tune_variables(self) -> tuple[str, str]:
        """Return LHC operational tune knob names."""
        return f"dqx_b{self.beam}_op", f"dqy_b{self.beam}_op"

    def get_tune_integers(self) -> tuple[int, int]:
        """Return LHC integer tunes."""
        return 62, 60

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        """Infer measurement plane from LHC BPM family name."""
        del bpm_name
        return "HV"
