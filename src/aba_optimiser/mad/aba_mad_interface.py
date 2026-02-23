"""Repository-specific MAD-NG interface extensions.

This module builds on shared classes from ``pymadng-utils``:
- ``CoreMadInterface``: minimal common API.
- ``AcDipoleMadInterface``: optional AC dipole extension.

``AbaMadInterface`` adds repository-only helper methods.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pymadng_utils.mad import CoreMadInterface

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class AbaMadInterface(CoreMadInterface):
    """Repository-local extension of ``CoreMadInterface`` with helper utilities."""

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
        bpms_in_range = [bpm for bpm in all_bpms if bpm in bpms_in_range]
        logger.debug(f"Found {len(bpms_in_range)} BPMs in range {bpm_range}")
        return all_bpms, bpms_in_range

    def set_magnet_strengths(self, strengths: dict[str, float]) -> None:
        """Set magnet strengths using standardised naming conventions."""
        suffixes = {".k0", ".k1", ".k2", ".kick"}
        logger.debug(f"Setting {len(strengths)} magnet strengths")

        variables_to_set = {}
        for name, strength in strengths.items():
            if not any(suffix in name for suffix in suffixes):
                raise ValueError(f"Magnet name '{name}' must end with one of {suffixes}")
            magnet_name, var = name.rsplit(".", 1)
            variables_to_set[f"MADX['{magnet_name}'].{var}"] = strength

        self.set_variables(**variables_to_set)

    def apply_corrector_strengths(self, corrector_table: pd.DataFrame) -> None:
        """Apply corrector strengths from a table to MAD sequence."""
        logger.debug(f"Applying corrector strengths to {len(corrector_table)} elements")

        mappings = {
            "hkicker": [("kick", "hkick")],
            "vkicker": [("kick", "vkick")],
            "tkicker": [("hkick", "hkick"), ("vkick", "vkick")],
        }

        for _, row in corrector_table.iterrows():
            ename = row["ename"]
            try:
                element = self.mad.loaded_sequence[ename]
                kind = element.kind
            except KeyError:
                logger.warning(f"Element {ename} not found in loaded sequence")
                continue

            if kind in mappings:
                for attr, col in mappings[kind]:
                    if col in row.index:
                        self.mad.send(f"loaded_sequence['{ename}'].{attr} = {self.py_name}:recv()")
                        self.mad.send(row[col])
                    else:
                        logger.warning(
                            f"Column '{col}' not found in corrector table for element {ename}"
                        )
            else:
                logger.warning(f"Element {ename} has unknown kind '{kind}'")

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
