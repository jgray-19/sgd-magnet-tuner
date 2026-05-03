"""Repository-specific MAD-NG interface extensions.

This module builds on shared classes from ``pymadng-utils``:
- ``CoreMadInterface``: minimal common API.
- ``AcDipoleMadInterface``: optional AC dipole extension.

``AbaMadInterface`` adds repository-only helper methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng_utils.mad.knob_mad_interface import KnobMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator

logger = logging.getLogger(__name__)

_MAGNET_STRENGTH_SUFFIXES = {".k0", ".k1", ".k2", ".kick"}
_DKNL_INDEX_BY_ATTR_LUA = {"k0": 1, "k1": 2, "k2": 3}
_DKNL_STRENGTH_ATTRS = frozenset(_DKNL_INDEX_BY_ATTR_LUA)


class AbaMadInterface(KnobMadInterface):
    """Repository-local extension of ``KnobMadInterface`` with helper utilities."""

    def __init__(self, accelerator: Accelerator, **kwargs):
        super().__init__(accelerator=accelerator, **kwargs)
        self.accelerator = accelerator
        self.accelerator.apply_accelerator_specific_errors(self)

    def _add_deferred_dknl(self, element_name: str) -> None:
        """If the dknl attribute for an element is empty and not deferred, add a deferred table to allow storing perturbations."""
        self.mad.send(f"""
if not MAD.typeid.is_deferred(loaded_sequence['{element_name}'].dknl) then
    loaded_sequence['{element_name}'].dknl = MAD.typeid.deferred {{0.0, 0.0, 0.0, 0.0}}
end
        """)

    def _set_dknl_component(self, element_name: str, attr: str, delta_strength: float) -> None:
        """Store a strength delta in one dknl component."""
        dknl_index = _DKNL_INDEX_BY_ATTR_LUA[attr]
        if float(self.mad[f"loaded_sequence['{element_name}'].l"]) == 0.0:
            raise ValueError(f"Cannot set dknl delta for element {element_name} with zero length")

        self._add_deferred_dknl(element_name)

        self.mad.send(f"""
loaded_sequence['{element_name}'].dknl[{dknl_index}] = {self.py_name}:recv() * loaded_sequence['{element_name}'].l
        """)
        self.mad.send(delta_strength)

    def _get_effective_element_strength(self, element_name: str, attr: str) -> float:
        """Return the effective element strength, including dknl perturbations."""
        if attr not in _DKNL_STRENGTH_ATTRS:
            return self.mad[f"loaded_sequence['{element_name}'].{attr}"]

        if self.mad[f"loaded_sequence['{element_name}'].l"] == 0.0:
            raise ValueError(
                f"Cannot get effective strength for element {element_name} with zero length"
            )

        # This will pass if the dknl table is empty (not created yet)
        if len(self.mad.loaded_sequence[element_name].dknl) == 0:
            return self.mad[f"loaded_sequence['{element_name}'].{attr}"]

        dknl_index = _DKNL_INDEX_BY_ATTR_LUA[attr]
        self.mad.send(f"""
local l, dknl, {attr} in loaded_sequence['{element_name}']
{self.py_name}:send({attr} + dknl[{dknl_index}] / l)
        """)
        return self.mad.recv()

    def _get_base_element_strength(self, element_name: str, attr: str) -> float:
        """Return the underlying element strength without any dknl perturbation applied."""
        return float(self.mad[f"loaded_sequence['{element_name}'].{attr}"])

    def _set_effective_element_strength(
        self, element_name: str, attr: str, target_strength: float
    ) -> None:
        """Set an element strength, routing k0/k1/k2 through dknl."""
        if attr not in _DKNL_STRENGTH_ATTRS:
            self.mad[f"loaded_sequence['{element_name}'].{attr}"] = target_strength
            return

        base_strength = float(self.mad[f"loaded_sequence['{element_name}'].{attr}"])
        self._set_dknl_component(element_name, attr, float(target_strength) - base_strength)

    def set_magnet_strengths(self, strengths: dict[str, float]) -> None:
        """Set magnet strengths, storing quadrupole updates in dknl."""
        logger.debug(f"Setting {len(strengths)} magnet strengths")
        direct_variables: dict[str, float] = {}

        for name, strength in strengths.items():
            if not any(name.endswith(suffix) for suffix in _MAGNET_STRENGTH_SUFFIXES):
                raise ValueError(
                    f"Magnet name '{name}' must end with one of {_MAGNET_STRENGTH_SUFFIXES}"
                )

            magnet_name, attr = name.rsplit(".", 1)
            if attr in _DKNL_STRENGTH_ATTRS:
                self._set_effective_element_strength(magnet_name, attr, strength)
            else:
                direct_variables[f"loaded_sequence['{magnet_name}'].{attr}"] = strength

        if direct_variables:
            self.set_variables(**direct_variables)

    def get_magnet_strengths(self, names: list[str]) -> dict[str, float]:
        """Get effective magnet strengths, including quadrupole dknl perturbations."""
        strengths: dict[str, float] = {}
        for name in names:
            magnet_name, attr = name.rsplit(".", 1)
            strengths[name] = self._get_effective_element_strength(magnet_name, attr)
        return strengths

    def get_base_magnet_strengths(self, names: list[str]) -> dict[str, float]:
        """Get underlying magnet strengths without any dknl perturbation applied."""
        strengths: dict[str, float] = {}
        for name in names:
            magnet_name, attr = name.rsplit(".", 1)
            strengths[name] = self._get_base_element_strength(magnet_name, attr)
        return strengths

    def observe_bpms(
        self, bpm_pattern: str | None = None, bad_bpms: list[str] | None = None
    ) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        if bpm_pattern is None:
            bpm_pattern = self.accelerator.bpm_pattern
        super().observe_bpms(bpm_pattern, bad_bpms)

    def get_bpm_list(self, bpm_range: str) -> tuple[list[str], list[str]]:
        """Get list of observed BPM names in the sequence and in a range."""
        logger.debug(f"Getting BPM list for range: {bpm_range}")

        get_bpms_mad = f"""
        local all_bpms = {{}}
        local bpm_in_range = {{}}
        for _, elm in loaded_sequence:iter() do
            if elm:is_observed() then
                table.insert(all_bpms, elm.name)
            end
        end
        for _, elm in loaded_sequence:iter("{bpm_range}") do
            if elm:is_observed() then
                table.insert(bpm_in_range, elm.name)
            end
        end
        {self.py_name}:send(all_bpms, true)
        {self.py_name}:send(bpm_in_range, true)
        """
        self.mad.send(get_bpms_mad)
        all_bpms = self.mad.receive()
        bpms_in_range = self.mad.receive()

        # It is possible that the first and last BPM are the same physical BPM if you have a marker BPM at the start/end of the sequence.
        # In that case, we should remove the duplicate to avoid confusion.
        if all_bpms[0] == all_bpms[-1] and len(all_bpms) > 1:
            all_bpms = all_bpms[:-1]
        bpms_in_range = [bpm for bpm in all_bpms if bpm in bpms_in_range]
        logger.debug(f"Found {len(bpms_in_range)} BPMs in range {bpm_range}")
        return all_bpms, bpms_in_range

    def pt2dp(self, pt: float) -> float:
        """Convert transverse momentum to delta p/p."""
        self.mad.send(
            f"{self.py_name}:send(MAD.gphys.pt2dp({self.py_name}:recv(), loaded_sequence.beam.beta))"
        )
        self.mad.send(pt)
        return self.mad.recv()

    def dp2pt(self, dp: float) -> float:
        """Convert delta p/p to transverse momentum."""
        self.mad.send(
            f"{self.py_name}:send(MAD.gphys.dp2pt({self.py_name}:recv(), loaded_sequence.beam.beta))"
        )
        self.mad.send(dp)
        return self.mad.recv()
