"""Repository-specific MAD-NG interface extensions.

This module builds on shared classes from ``pymadng-utils``:
- ``CoreMadInterface``: minimal common API.
- ``AcDipoleMadInterface``: optional AC dipole extension.

``AbaMadInterface`` adds repository-only helper methods.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from pymadng_utils.mad import KnobMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator

logger = logging.getLogger(__name__)

_MAGNET_STRENGTH_SUFFIXES = {".k0", ".k1", ".k2", ".kick"}
_DKNL_INDEX_BY_ATTR_LUA = {"k0": 1, "k1": 2, "k2": 3}
_DKNL_STRENGTH_ATTRS = frozenset(_DKNL_INDEX_BY_ATTR_LUA)

_PERTURBATION_BASE_SPECS: dict[str, dict[str, Any]] = {
    "d": {"kind": ("sbend", "rbend"), "attr": "k0", "dknl_index": 0},
    "q": {"kind": ("quadrupole",), "attr": "k1", "dknl_index": 1},
    "s": {"kind": ("sextupole",), "attr": "k2", "dknl_index": 2},
}


class AbaMadInterface(KnobMadInterface):
    """Repository-local extension of ``CoreMadInterface`` with helper utilities."""

    def __init__(self, accelerator: Accelerator, **kwargs):
        super().__init__(**kwargs)
        self.accelerator = accelerator
        self.load_sequence(self.accelerator.sequence_file, self.accelerator.seq_name)
        self.setup_beam(beam_energy=self.accelerator.beam_energy)

    #     self._add_dknl_attributes()

    # def _add_dknl_attributes(self) -> None:
    #     """Add dknl attributes to all elements in the sequence for storing perturbations."""
    #     self.mad.send("""
    #     n_added = 0
    #     for _, elm in loaded_sequence:iter() do
    #         if elm.dknl then
    #             loaded_sequence[elm.name].dknl = MAD.typeid.deferred {0.0, 0.0, 0.0, 0.0}
    #             n_added = n_added + 1
    #         end
    #     end
    #     """)
    #     logger.debug(f"Added dknl attributes to {self.mad.n_added} elements in the sequence")
    #     self.mad["n_added"] = None  # Clear the variable to avoid confusion later

    def _set_dknl_delta(self, element_name: str, attr: str, delta_strength: float) -> None:
        """Store a magnet strength delta in the matching dknl component."""
        self._set_dknl_component(
            element_name,
            attr,
            delta_strength,
        )

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

        dknl_index = _DKNL_INDEX_BY_ATTR_LUA[attr]
        self.mad.send(f"""
local l, dknl, {attr} in loaded_sequence['{element_name}']
{self.py_name}:send({attr} + dknl[{dknl_index}] / l)
        """)
        return self.mad.recv()

    def _set_effective_element_strength(
        self, element_name: str, attr: str, target_strength: float
    ) -> None:
        """Set an element strength, routing k0/k1/k2 through dknl."""
        if attr not in _DKNL_STRENGTH_ATTRS:
            self.mad[f"loaded_sequence['{element_name}'].{attr}"] = target_strength
            return

        base_strength = float(self.mad[f"loaded_sequence['{element_name}'].{attr}"])
        self._set_dknl_delta(element_name, attr, float(target_strength) - base_strength)

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

    def apply_magnet_perturbations(
        self,
        rel_error: float | None = 1e-4,
        seed: int = 42,
        magnet_type: str | list[str] = "all",
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Apply accelerator-specific perturbations via dknl to the loaded sequence."""
        if magnet_type == "all":
            requested = ["d", "q", "s"]
        else:
            requested = magnet_type if isinstance(magnet_type, list) else [magnet_type]
        if not requested:
            return {}, {}

        family_overrides = self.accelerator.get_perturbation_families()
        family_configs = [
            _PERTURBATION_BASE_SPECS[family] | family_overrides[family]
            for family in ("d", "q", "s")
            if family in requested and family in family_overrides
        ]
        if not family_configs:
            return {}, {}

        rng = np.random.default_rng(seed)
        magnet_strengths: dict[str, float] = {}
        true_strengths: dict[str, float] = {}

        for elm in self.mad.loaded_sequence:
            for family_config in family_configs:
                if elm.kind not in family_config["kind"]:
                    continue

                pattern = family_config.get("pattern")
                if pattern and not re.match(str(pattern), elm.name):
                    continue

                element_rel_error = self._resolve_relative_error(
                    family_config=family_config,
                    element_name=str(elm.name),
                    rel_error=rel_error,
                )
                if element_rel_error is None:
                    continue

                attr = str(family_config["attr"])
                strength_before = float(elm[attr])
                delta = float(rng.normal(0, abs(strength_before * element_rel_error)))
                strength_after = strength_before + delta

                if attr in _DKNL_STRENGTH_ATTRS:
                    self._set_dknl_delta(elm.name, attr, delta)
                else:
                    elm[attr] = strength_after

                magnet_strengths[f"{elm.name}.{attr}"] = strength_after
                true_strengths[str(elm.name)] = strength_after
                break

        return magnet_strengths, true_strengths

    def _resolve_relative_error(
        self,
        family_config: dict[str, Any],
        element_name: str,
        rel_error: float | None,
    ) -> float | None:
        """Resolve the relative error for one element from the global or family settings."""
        if rel_error is not None:
            return rel_error

        relative_error_table = family_config.get("relative_error_table")
        if isinstance(relative_error_table, dict):
            for prefix, rel_value in relative_error_table.items():
                if element_name.startswith(str(prefix)):
                    return float(rel_value)

        default_rel_std = family_config.get("default_rel_std")
        if default_rel_std is not None:
            return float(default_rel_std)

        if relative_error_table is not None:
            return None

        raise ValueError(
            f"Relative error not specified for family with kind {family_config['kind']}"
        )

    def match_tunes(
        self,
        target_qx: float,
        target_qy: float,
        deltap: float = 0.0,
    ) -> dict[str, float]:
        """Match tunes using a shared recipe, with accelerator-provided tune variable names."""
        qx_knob, qy_knob = self.accelerator.get_tune_variables()
        qx_int, qy_int = self.accelerator.get_tune_integers()
        self.mad["result"] = self.mad.match(
            command=rf"\ -> twiss{{sequence=loaded_sequence, deltap={deltap:.16e}}}",
            variables=[
                {"var": f"'MADX.{qx_knob}'", "name": f"'{qx_knob}'"},
                {"var": f"'MADX.{qy_knob}'", "name": f"'{qy_knob}'"},
            ],
            equalities=[
                {"expr": f"\\t -> t.q1-({qx_int}+{target_qx})", "name": "'q1'"},
                {"expr": f"\\t -> t.q2-({qy_int}+{target_qy})", "name": "'q2'"},
            ],
            objective={"fmin": 1e-8},
            info=2,
        )
        return {
            qx_knob: self.mad[f"MADX['{qx_knob}']"],
            qy_knob: self.mad[f"MADX['{qy_knob}']"],
        }

    def perform_orbit_correction(
        self,
        machine_deltap: float,
        target_qx: float,
        target_qy: float,
        corrector_file: Path | None,
        twiss_name: str = "zero_twiss",
    ) -> dict[str, float]:
        """Perform orbit correction and tune rematching with a shared MAD flow."""
        qx_knob, qy_knob = self.accelerator.get_tune_variables()
        qx_int, qy_int = self.accelerator.get_tune_integers()
        self.mad["machine_deltap"] = machine_deltap
        self.mad["correct_file"] = str(corrector_file.absolute()) if corrector_file else None

        self.mad.send(rf"""
local correct, option in MAD

io.write("*** orbit correction using off momentum twiss\n")
local tws_offmom = twiss {{ sequence=loaded_sequence, deltap=machine_deltap }}

! Increase file numerical formatting
local fmt = option.numfmt ; option.numfmt = "% -.16e"
local tbl = correct {{ sequence=loaded_sequence, model=tws_offmom, target={twiss_name}, method="svd", info=1, plane="x" }}
if correct_file then
    tbl:write(correct_file)
end
option.numfmt = fmt ! restore formatting

io.write("*** rematching tunes for off-momentum twiss\n")
match {{
  command := twiss {{sequence=loaded_sequence, observe=0, deltap=machine_deltap}},
  variables = {{ rtol=1e-4,
    {{ var = 'MADX.{qx_knob}', name='{qx_knob}' }},
    {{ var = 'MADX.{qy_knob}', name='{qy_knob}' }},
  }},
  equalities = {{ tol = 1e-6,
    {{ expr = \t -> t.q1-{qx_int + target_qx:.16e}, name='q1' }},
    {{ expr = \t -> t.q2-{qy_int + target_qy:.16e}, name='q2' }},
  }},
  objective = {{fmin = 1e-8}},
  info=2
}}

{self.py_name}:send("Complete")
""")
        self._check_mad_response(
            "Complete", "Error during MAD-NG orbit correction and tune matching"
        )
        return {
            qx_knob: self.mad[f"MADX['{qx_knob}']"],
            qy_knob: self.mad[f"MADX['{qy_knob}']"],
        }

    def _check_mad_response(self, expected: str, error_msg: str) -> None:
        """Check that the response from MAD-NG matches the expected value."""
        try:
            if (result := self.mad.recv()) != expected:
                raise RuntimeError(f"Unexpected response from MAD-NG: {result}. {error_msg}")
        except Exception as e:
            raise RuntimeError(error_msg) from e
