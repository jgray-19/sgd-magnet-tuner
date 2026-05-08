"""Repository-specific MAD-NG interface extensions.

This module builds on shared classes from ``pymadng-utils``:
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, NamedTuple

from pymadng_utils.mad.knob_mad_interface import KnobMadInterface

if TYPE_CHECKING:
    from pymadng_utils.accelerators import Accelerator

logger = logging.getLogger(__name__)


class _MultipoleInfo(NamedTuple):
    dk_table: str  # MAD-NG table storing perturbations: "dknl" or "dksl"
    base_table: str  # MAD-NG table storing base strengths: "knl" or "ksl"
    index: int  # 1-based Lua index into the table
    dk_suffix: str  # knob name suffix used in MAD variables (e.g. "dk1l", "dk1sl")
    is_delta: bool  # True when attr is already a delta (e.g. dk1l), False for absolute (k1)


# Maximum multipole order supported (k0 through k{MAX_MULTIPOLE-1}).
# This also sets the size of the dknl/dksl deferred tables allocated in MAD-NG.
MAX_MULTIPOLE = 3

# Generated from MAX_MULTIPOLE: normal components use dknl/knl, skew use dksl/ksl.
def _build_multipole_attrs(max_order: int) -> dict[str, _MultipoleInfo]:
    attrs: dict[str, _MultipoleInfo] = {}
    for n in range(max_order):
        idx = n + 1  # MAD-NG tables are 1-based
        for dk_table, base_table, abs_attr, delta_attr in [
            ("dknl", "knl", f"k{n}",  f"dk{n}l"),
            ("dksl", "ksl", f"k{n}s", f"dk{n}sl"),
        ]:
            info_abs   = _MultipoleInfo(dk_table, base_table, idx, delta_attr, is_delta=False)
            info_delta = _MultipoleInfo(dk_table, base_table, idx, delta_attr, is_delta=True)
            attrs[abs_attr]   = info_abs
            attrs[delta_attr] = info_delta
    return attrs

_MULTIPOLE_ATTRS = _build_multipole_attrs(MAX_MULTIPOLE)

_MISALIGN_ATTRS = frozenset({"dx", "dy"})

_MAGNET_STRENGTH_SUFFIXES = (
    {f".{attr}" for attr in _MULTIPOLE_ATTRS} | {f".{attr}" for attr in _MISALIGN_ATTRS} | {".kick"}
)

# Re-exported for use in optimising_mad_interface.py
MULTIPOLE_ATTRS = _MULTIPOLE_ATTRS

_INDEXED_MULTIPOLE_RE = re.compile(r"^(knl|ksl)\[(\d+)\]$")


def indexed_multipole_attr_info(attr: str) -> _MultipoleInfo | None:
    """Return multipole metadata for indexed MAD attrs such as ``knl[3]`` or ``ksl[3]``."""
    match = _INDEXED_MULTIPOLE_RE.fullmatch(attr)
    if match is None:
        return None

    base_table, index_str = match.groups()
    index = int(index_str)
    if index < 1 or index > MAX_MULTIPOLE:
        return None

    order = index - 1
    base_attr = f"k{order}" if base_table == "knl" else f"k{order}s"
    return _MULTIPOLE_ATTRS[base_attr]


class AbaMadInterface(KnobMadInterface):
    """Repository-local extension of ``KnobMadInterface`` with helper utilities."""

    def __init__(self, accelerator: Accelerator, **kwargs):
        super().__init__(accelerator=accelerator, **kwargs)
        self.accelerator = accelerator
        self.accelerator.apply_accelerator_specific_errors(self)

    # --- multipole perturbation table helpers ---

    def _ensure_deferred_dk_table(self, element_name: str, dk_table: str) -> None:
        """Initialise the dknl/dksl table on an element if it is not yet deferred."""
        zeros = ", ".join(["0.0"] * MAX_MULTIPOLE)
        self.mad.send(f"""
if not MAD.typeid.is_deferred(loaded_sequence['{element_name}'].{dk_table}) then
    loaded_sequence['{element_name}'].{dk_table} = MAD.typeid.deferred {{{zeros}}}
end
        """)

    def _set_dk_component(self, element_name: str, info: _MultipoleInfo, delta: float) -> None:
        """Write a delta strength into the correct dknl/dksl slot."""
        self._ensure_deferred_dk_table(element_name, info.dk_table)
        self.mad.send(f"""
loaded_sequence['{element_name}'].{info.dk_table}[{info.index}] = {self.py_name}:recv()
        """)
        self.mad.send(delta)

    # --- misalignment helpers (unchanged, separate from multipole logic) ---

    def _set_misalignment(self, element_name: str, attr: str, value: float) -> None:
        """Set a misalignment value, preserving other misalignment attributes already set."""
        # The plain `if not mad[...]` truthiness test is unreliable here because pymadng
        # returns a MadRef object even for an unset table, which is always truthy.
        self.mad.send(f"{self.py_name}:send(loaded_sequence['{element_name}'].misalign, true)")
        misalign_dict = self.mad.recv()
        if not isinstance(misalign_dict, dict) or len(misalign_dict) == 0:
            self.mad[f"loaded_sequence['{element_name}'].misalign"] = []
        self.mad[f"loaded_sequence['{element_name}'].misalign.{attr}"] = value

    def _get_misalignment(self, element_name: str, attr: str) -> float:
        """Get a misalignment value, returning 0.0 if not set."""
        self.mad.send(f"{self.py_name}:send(loaded_sequence['{element_name}'].misalign, true)")
        misalign_dict = self.mad.recv()
        if not isinstance(misalign_dict, dict) or len(misalign_dict) == 0:
            return 0.0
        return float(misalign_dict.get(attr, 0.0))

    # --- generic element strength get/set ---

    def _get_effective_element_strength(self, element_name: str, attr: str) -> float:
        """Return the effective element strength, including any dknl/dksl perturbations."""
        if attr in _MISALIGN_ATTRS:
            return self._get_misalignment(element_name, attr)

        info = _MULTIPOLE_ATTRS.get(attr)
        if info is None:
            return self.mad[f"loaded_sequence['{element_name}'].{attr}"]

        # If the dk table hasn't been initialised yet, return the base attribute directly.
        if len(getattr(self.mad.loaded_sequence[element_name], info.dk_table)) == 0:
            return self.mad[f"loaded_sequence['{element_name}'].{attr}"]

        if info.is_delta:
            return float(
                self.mad[f"loaded_sequence['{element_name}'].{info.dk_table}[{info.index}]"]
            )

        # Absolute attrs: effective = base + perturbation
        self.mad.send(f"""
local {info.dk_table}, {attr} in loaded_sequence['{element_name}']
{self.py_name}:send({attr} + {info.dk_table}[{info.index}])
        """)
        return self.mad.recv()

    def _get_base_element_strength(self, element_name: str, attr: str) -> float:
        """Return the element strength ignoring any dknl/dksl perturbation."""
        if self.mad[f"loaded_sequence['{element_name}'].{attr}"] is not None:
            return float(self.mad[f"loaded_sequence['{element_name}'].{attr}"])
        info = _MULTIPOLE_ATTRS[attr]
        return float(self.mad[f"loaded_sequence['{element_name}'].{info.base_table}[{info.index}]"])

    def _set_effective_element_strength(self, element_name: str, attr: str, target: float) -> None:
        """Set an element strength, routing multipole updates through dknl/dksl."""
        info = _MULTIPOLE_ATTRS.get(attr)
        if info is None:
            self.mad[f"loaded_sequence['{element_name}'].{attr}"] = target
            return

        delta = (
            float(target)
            if info.is_delta
            else (float(target) - float(self.mad[f"loaded_sequence['{element_name}'].{attr}"]))
        )
        self._set_dk_component(element_name, info, delta)

    # --- public API ---

    def set_magnet_strengths(self, strengths: dict[str, float]) -> None:
        """Set magnet strengths, routing multipole updates through dknl/dksl."""
        logger.debug(f"Setting {len(strengths)} magnet strengths")
        direct_variables: dict[str, float] = {}

        for name, strength in strengths.items():
            if not any(name.endswith(suffix) for suffix in _MAGNET_STRENGTH_SUFFIXES):
                raise ValueError(
                    f"Magnet name '{name}' must end with one of {_MAGNET_STRENGTH_SUFFIXES}"
                )
            magnet_name, attr = name.rsplit(".", 1)
            if attr in _MISALIGN_ATTRS:
                self._set_misalignment(magnet_name, attr, strength)
            elif attr in _MULTIPOLE_ATTRS:
                self._set_effective_element_strength(magnet_name, attr, strength)
            else:
                direct_variables[f"loaded_sequence['{magnet_name}'].{attr}"] = strength

        if direct_variables:
            self.set_variables(**direct_variables)

    def get_magnet_strengths(self, names: list[str]) -> dict[str, float]:
        """Get effective magnet strengths, including any dknl/dksl perturbations."""
        return {name: self._get_effective_element_strength(*name.rsplit(".", 1)) for name in names}

    def get_base_magnet_strengths(self, names: list[str]) -> dict[str, float]:
        """Get underlying magnet strengths without any dknl/dksl perturbation."""
        return {name: self._get_base_element_strength(*name.rsplit(".", 1)) for name in names}

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

        # Remove duplicate if the first and last BPM are the same physical element.
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
