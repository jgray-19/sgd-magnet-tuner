"""LHC-specific MAD interface with LHC element patterns.

This module provides the LHC-specific GradientDescentMadInterface that defines
LHC magnet patterns as Lua regex variables, avoiding hardcoded strings
throughout the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.optimising_mad_interface import (
    BPM_PATTERN,
    GradientDescentMadInterface,
)
from aba_optimiser.physics.lhc_bends import normalise_lhcbend_magnets

if TYPE_CHECKING:
    from pathlib import Path


class LHCOptimisationMadInterface(GradientDescentMadInterface):
    """LHC-specific MAD interface with consolidated magnet patterns.

    This interface defines LHC element identification patterns as Lua regex
    variables that are set globally in MAD, ensuring consistency across
    all LHC-specific operations.

    Patterns defined:
    - PATTERN_MAIN_BEND: Main bending magnets (MB.*)
    - PATTERN_RBEND: Rectangular bends (MB[RXWAL]...)
    - PATTERN_MAIN_QUAD: Main quadrupoles (MQ.*)
    - PATTERN_CORRECTOR: Orbit correctors (MCB)
    """

    # Bend knob name extraction pattern
    BEND_KNOB_NAME_PATTERN = (
        'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".k0"'
    )

    def __init__(
        self,
        accelerator: LHC,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bad_bpms: list[str] | None = None,
        corrector_strengths: Path | None = None,
        tune_knobs_file: Path | None = None,
        start_bpm: str | None = None,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
    ):
        """Initialize LHC-specific MAD interface with gradient descent optimization.

        Args:
            magnet_range: Magnet range for optimization
            bpm_range: BPM range for optimization
            bad_bpms: List of bad BPMs to exclude
            corrector_strengths: Path to corrector strengths file
            tune_knobs_file: Path to tune knobs file
            start_bpm: Starting BPM
            beam_energy: Beam energy in GeV
            py_name: Python variable name in MAD
            debug: Enable debug mode
            mad_logfile: Path to MAD log file
            discard_mad_output: Whether to discard MAD output
        """
        super().__init__(
            accelerator,
            magnet_range,
            bpm_range,
            BPM_PATTERN,
            bad_bpms,
            corrector_strengths,
            tune_knobs_file,
            start_bpm,
            py_name,
            debug,
            mad_logfile,
            discard_mad_output,
        )

    def get_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """
        Get LHC-specific knob specifications for gradient descent optimization.

        Returns:
            List of (kind, attribute, pattern, zero_check) tuples defining
            all possible knobs that can be created for LHC optimization.
            Subclasses or configs will select which ones to use.
        """
        return self.accelerator.get_supported_knob_specs()

    def _prepare_knob_data(
        self, selected_specs: list[tuple[str, str, str, bool]]
    ) -> None:
        """Prepare bend normalization data when bends are selected."""
        assert isinstance(self.accelerator, LHC)
        if self.accelerator.optimise_bends and self.accelerator.normalise_bends:
            self._make_bend_dict()

    def _get_attr_specs(self) -> dict[str, dict[str, str]]:
        """Provide LHC-specific attribute naming/value expressions."""
        assert isinstance(self.accelerator, LHC)
        if self.accelerator.normalise_bends:
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
        return {}

    def _make_bend_dict(self) -> None:
        """Create and normalise the bend strength dictionary for LHC."""
        self.mad.send(f"""
        bend_dict = {{}}
        bend_lengths = {{}}
        for i, e in loaded_sequence:siter(magnet_range) do
            if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                bend_dict[e.name .. ".k0"] = e.k0
                bend_lengths[e.name .. ".k0"] = e.l ! Length is `l` attribute
            end
        end
        {self.py_name}:send(bend_dict, true)
        {self.py_name}:send(bend_lengths, true)
        bend_dict = {self.py_name}:recv()
        """)
        true_strengths_dict: dict[str, float] = self.mad.recv()
        self.bend_lengths: dict[str, float] = self.mad.recv()
        normalised_names = normalise_lhcbend_magnets(true_strengths_dict, self.bend_lengths)
        self.mad.send(normalised_names)
