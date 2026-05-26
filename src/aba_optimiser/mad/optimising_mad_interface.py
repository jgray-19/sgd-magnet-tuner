"""High-level MAD-NG interfaces used by optimisation and worker code.

The classes in this module build on :mod:`aba_optimiser.mad.aba_mad_interface`
to provide a fully configured MAD-NG session for optimisation workflows. They
handle sequence loading, BPM observation setup, optional corrector/tune-knob
application, and knob discovery for gradient-based tuning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tfs
from pymadng_utils.io.utils import read_knobs

from .aba_mad_interface import (
    MAX_MULTIPOLE,
    MULTIPOLE_ATTRS,
    AbaMadInterface,
    indexed_multipole_attr_info,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pymadng_utils.accelerators import Accelerator as PyMadAccelerator

    from aba_optimiser.accelerators import Accelerator, KnobSpec

BPM_PATTERN = "^BPM"
LOGGER = logging.getLogger(__name__)

_CORRECTOR_ATTRS_BY_KIND: dict[str, tuple[tuple[str, str], ...]] = {
    "hkicker": (("kick", "hkick"),),
    "vkicker": (("kick", "vkick"),),
    "tkicker": (("hkick", "hkick"), ("vkick", "vkick")),
}


def _ensure_cycleable_start_element(
    iface: GenericMadInterface,
    start_bpm: str,
    observed_bpms: list[str],
) -> None:
    """Replace a non-BPM start element with a same-name marker for MAD cycling."""
    if start_bpm in observed_bpms:
        return
    iface.mad.send(f"""
correct_elm = MADX['{start_bpm}']
local new_elm = MAD.element.marker ('{start_bpm}') {{ at=loaded_sequence:upos(correct_elm) }}
local replaced = loaded_sequence:replace({{new_elm}}, '{start_bpm}')
MADX['{start_bpm}'] = new_elm
{iface.py_name}:send(replaced and #replaced or 0)
correct_elm = nil
    """)
    if iface.mad.recv() != 1:
        raise ValueError(f"Failed to replace start element {start_bpm} with a marker")
    LOGGER.info("Replaced non-BPM start element with cycle marker: %s", start_bpm)


def _absolute_name_from_dk_knob(knob_name: str) -> str | None:
    """Map a dknl/dksl perturbation knob name back to its absolute strength attribute."""
    for abs_attr, mp in MULTIPOLE_ATTRS.items():
        if not mp.is_delta and knob_name.endswith(f".{mp.dk_suffix}"):
            return knob_name[: -(len(mp.dk_suffix) + 1)] + f".{abs_attr}"
    return None


def _deferred_table_helpers() -> str:
    """Lua helpers that lazily initialise dknl/dksl tables the first time they are written."""
    zeros = ", ".join(["0.0"] * MAX_MULTIPOLE)
    return f"""
local function make_dknl_deferred_knob(e)
    if not MAD.typeid.is_deferred(loaded_sequence[e.name].dknl) then
        loaded_sequence[e.name].dknl = MAD.typeid.deferred {{{zeros}}}
    end
end
local function make_dksl_deferred_knob(e)
    if not MAD.typeid.is_deferred(loaded_sequence[e.name].dksl) then
        loaded_sequence[e.name].dksl = MAD.typeid.deferred {{{zeros}}}
    end
end
"""


class GenericMadInterface(AbaMadInterface):
    """
    Generic MAD interface for all setup tasks EXCEPT knob creation.

    This interface handles:
    - Loading sequences and setting up beams
    - Observing BPMs and configuring ranges
    - Applying corrector strengths and tune knobs
    - General MAD-NG operations

    Knob creation (for gradient descent) is handled by ``GradientDescentMadInterface``.

    This separation allows non-gradient-descent use cases to use this interface
    without unnecessary knob creation overhead.
    """

    def __init__(
        self,
        accelerator: PyMadAccelerator,
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
        stdout, redirect_stderr = self._resolve_mad_stdout(mad_logfile, discard_mad_output)

        super().__init__(
            accelerator=accelerator,
            stdout=stdout,
            redirect_stderr=redirect_stderr,
            py_name=py_name,
            debug=debug,
        )

        self.accelerator = accelerator
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range if bpm_range is not None else magnet_range
        self.knob_names: list[str] = []
        self.knob_name_set: set[str] = set()
        self.elem_spos: list[float] = []

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range

        self.observe_bpms(bad_bpms=bad_bpms)
        all_bpms, _ = self.get_bpm_list(self.bpm_range)
        self.make_all_monitors_thin(all_bpms)
        self.unobserve_all_elements()

        if start_bpm is not None:
            _ensure_cycleable_start_element(self, start_bpm, all_bpms)
            self.cycle_sequence(marker_name=start_bpm)
            LOGGER.info(f"Cycled sequence to start at BPM: {start_bpm}")
        else:
            LOGGER.info("Skipping sequence cycling (no start BPM provided)")

        # Setup observation and ranges
        self.observe_bpms(bad_bpms=bad_bpms)
        self.bpms_in_range, self.nbpms, self.all_bpms = self.count_bpms(self.bpm_range)
        observe_set = (
            self.all_bpms
            if start_bpm is not None and start_bpm not in self.bpms_in_range
            else self.bpms_in_range
        )
        self.observe_elements(observe_set)
        LOGGER.info("Restricted active observation set to %d BPMs", len(observe_set))

        # Apply corrector strengths if provided
        if corrector_strengths is not None:
            self._set_correctors(corrector_strengths)
        else:
            LOGGER.info("Skipping corrector strengths (not provided)")

        # Apply tune knobs if provided
        if tune_knobs_file is not None:
            self._set_tune_knobs(tune_knobs_file)
        else:
            LOGGER.info("Skipping tune knobs (not provided)")

    @staticmethod
    def _resolve_mad_stdout(
        mad_logfile: Path | None, discard_mad_output: bool
    ) -> tuple[Path | str | None, bool]:
        """Return (stdout, redirect_stderr) for the MAD process."""
        if mad_logfile is not None:
            if discard_mad_output:
                LOGGER.warning(
                    "MAD logfile specified, but discard_mad_output is True. MAD output will be logged."
                )
            LOGGER.info(f"MAD logfile set to: {mad_logfile.absolute()}")
            return mad_logfile, True
        if discard_mad_output:
            return "/dev/null", True
        return None, False

    def count_bpms(self, bpm_range: str) -> tuple[list[str], int, list[str]]:
        """Count the number of BPM elements in the specified range."""
        all_bpms, bpms_in_range = self.get_bpm_list(bpm_range)
        LOGGER.info(f"Counted {len(bpms_in_range)} BPMs in range: {bpm_range}")
        return bpms_in_range, len(bpms_in_range), all_bpms

    def make_all_monitors_thin(self, monitors: list[str]) -> None:
        """Replace monitor elements with markers in the specified BPM range."""
        for bpm in monitors:
            assert "monitor" in self.mad.MADX[bpm].kind, (
                f"Element {bpm} is not a monitor, cannot be made thin"
            )
            self.make_element_thin(bpm)
        LOGGER.info(
            f"Replaced {len(monitors)} monitor BPMs with markers in range: {self.bpm_range}"
        )

    def _sync_corrector_table_to_loaded_sequence(self, corrector_table: tfs.TfsDataFrame) -> None:
        """Mirror applied corrector strengths onto the tracked sequence copy."""
        synced = 0
        for row in corrector_table.itertuples():
            ename = getattr(row, "ename", None)
            if ename is None:
                raise ValueError("Corrector table is missing required column 'ename'")
            targets = _CORRECTOR_ATTRS_BY_KIND.get(getattr(row, "kind", None))  # ty:ignore[invalid-argument-type]
            if targets is None:
                continue
            for attr, col in targets:
                self.mad[f"loaded_sequence['{ename}'].{attr}"] = float(getattr(row, col))
            synced += 1
        if synced:
            LOGGER.info("Mirrored %d corrector strengths onto loaded_sequence", synced)

    def _set_correctors(self, corrector_strengths: Path) -> None:
        """Load corrector strengths from file and apply them to the sequence."""
        if not corrector_strengths.exists():
            LOGGER.warning(f"Corrector strengths file not found: {corrector_strengths}")
            return

        def _apply_from_tfs() -> None:
            corrector_table = tfs.read(corrector_strengths)
            required_cols = {"kind", "hkick", "hkick_old", "vkick", "vkick_old"}
            missing_cols = required_cols.difference(corrector_table.columns)
            if missing_cols:
                raise ValueError(
                    "TFS corrector table is missing required columns: "
                    + ", ".join(sorted(missing_cols))
                )
            non_monitors = corrector_table["kind"] != "monitor"
            corrector_table: tfs.TfsDataFrame = corrector_table[non_monitors]  # type: ignore[assignment, not-subscriptable]
            changed = (corrector_table["hkick"] != corrector_table["hkick_old"]) | (
                corrector_table["vkick"] != corrector_table["vkick_old"]
            )
            LOGGER.info(f"Applying {changed.sum()} non-zero corrector strengths from {corrector_strengths}")  # ty:ignore[unresolved-attribute]
            changed_table = corrector_table[changed]
            self.apply_corrector_strengths(changed_table)  # ty:ignore[invalid-argument-type]
            self._sync_corrector_table_to_loaded_sequence(changed_table)  # ty:ignore[invalid-argument-type]

        def _apply_from_knobs() -> None:
            knobs = read_knobs(corrector_strengths)
            for name, val in knobs.items():
                self.mad.send(f"MADX['{name}'] = {val}")
            LOGGER.info(f"Set {len(knobs)} corrector knobs from {corrector_strengths}")

        suffix = corrector_strengths.suffix.lower()
        parser_order = {
            ".tfs": [("tfs", _apply_from_tfs)],
            ".txt": [("knobs", _apply_from_knobs)],
        }.get(suffix, [("tfs", _apply_from_tfs), ("knobs", _apply_from_knobs)])

        parser_errors: list[tuple[str, Exception]] = []
        for parser_name, parser in parser_order:
            try:
                parser()
                break
            except (tfs.TfsFormatError, ValueError, KeyError, TypeError, OSError) as exc:
                parser_errors.append((parser_name, exc))
        else:
            details = "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in parser_errors)
            raise ValueError(
                f"Failed to apply corrector strengths from {corrector_strengths}. "
                f"Parsers attempted: {details}"
            ) from parser_errors[-1][1]

        self.mad.send(f"{self.py_name}:send('done')")
        self._check_mad_response(
            "done", f"Failed to apply corrector strengths from {corrector_strengths}"
        )

    def _set_tune_knobs(self, tune_knobs_file: Path) -> None:
        """Load and set predefined tune knobs from file."""
        tune_knobs = read_knobs(tune_knobs_file)
        # Get existing tune knob names in MAD
        prev = self.mad.recv_vars(*[f"MADX['{name}']" for name in tune_knobs])
        for name, val in tune_knobs.items():
            self.mad.send(f"MADX['{name}'] = {val}")
        self.mad.send(f"{self.py_name}:send('done')")
        self._check_mad_response("done", f"Failed to set tune knobs from {tune_knobs_file}")
        LOGGER.debug(f"Previous tune knob values: {prev}")
        LOGGER.debug(f"Set tune knobs from {tune_knobs_file}: {len(tune_knobs)}")


class GradientDescentMadInterface(GenericMadInterface):
    """
    MAD interface for gradient descent optimisation.

    Extends GenericMadInterface with knob creation capabilities. Accelerator-specific
    behaviour (which knobs to create, naming, MAD preparation) is provided via
    accelerator hooks, keeping this interface generic.
    """

    accelerator: Accelerator  # narrows the parent's PyMadAccelerator type

    def __init__(
        self,
        accelerator: Accelerator,
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
        super().__init__(
            accelerator,
            magnet_range,
            bpm_range,
            bad_bpms,
            corrector_strengths,
            tune_knobs_file,
            start_bpm,
            py_name,
            debug,
            mad_logfile,
            discard_mad_output,
        )

        if accelerator.has_any_optimisation():
            self._make_adj_knobs()
        else:
            LOGGER.warning(
                "Gradient descent optimisation interface initialised without any optimisation enabled."
                "\nUse GenericMadInterface if no optimisation is required."
            )

    def get_knob_specs(self) -> list[KnobSpec]:
        """
        Return all knob specifications supported by this accelerator.

        Returns:
            List of KnobSpec named tuples with fields:
            - kind: MAD element kind (e.g., "sbend", "quadrupole", "hkicker")
            - attribute: MAD element attribute (e.g., "k0", "k1", "kick")
            - pattern: Regex pattern to match element names
            - nonzero_attr: Optional MAD attribute that must be nonzero for a knob to be created
            - enabled: Whether this spec is currently enabled
            - label: Human-readable label for logging
        """
        return self.accelerator.get_supported_knob_specs()

    def _filter_knob_specs(
        self, all_specs: list[KnobSpec]
    ) -> list[tuple[str, str, str, str | None]]:
        """Keep only specs enabled by the accelerator's optimise_* flags."""
        return [
            (spec.kind, spec.attribute, spec.pattern, spec.nonzero_attr)
            for spec in all_specs
            if spec.enabled
        ]

    def _build_attr_block(self, attr_conditions: list[tuple[str, str, str]]) -> str:
        """
        Build the MAD-NG Lua block that assigns a deferred knob variable to each matched element.

        Each entry in attr_conditions is (kind, attr, lua_condition). For multipole
        attrs (k1, k1s, dk1l, …) the knob routes through dknl/dksl so the base
        strength is never mutated. For direct attrs (kick, dx, …) the element
        attribute itself becomes a deferred variable.
        """
        attr_specs = self.accelerator.get_mad_attr_specs()
        lines: list[str] = []

        for kind, attr, condition in attr_conditions:
            spec = attr_specs.get(kind, {})
            mp = MULTIPOLE_ATTRS.get(attr) or indexed_multipole_attr_info(attr)

            # The knob variable name in MAD (e.g. "MQXA.1R1.dk1l")
            default_name_expr = (
                f'e.name .. ".{mp.dk_suffix}"' if mp is not None else f'e.name .. ".{attr}"'
            )
            name_expr = spec.get("name_expr", default_name_expr)
            mad_value = spec.get("mad_value", f"e.{attr}")

            tmpl = [
                f"if {condition} then",
                f"    local k_str_name = {name_expr}",
                "    loaded_sequence[k_str_name] = loaded_sequence[k_str_name] or 0.0",
            ]
            if mp is not None:
                tmpl += [
                    f"    make_{mp.dk_table}_deferred_knob(e)",
                    f"    loaded_sequence[e.name].{mp.dk_table}[{mp.index}] = \\->loaded_sequence[k_str_name]",
                ]
            else:
                tmpl += [
                    f"    loaded_sequence[k_str_name] = {mad_value}",
                    f"    loaded_sequence[e.name].{attr} = \\->loaded_sequence[k_str_name]",
                ]
            tmpl.append(f"    store_knobs(k_str_name, {mad_value}, '{attr}', s)")
            tmpl.append("end")
            lines.extend(f"    {line}" for line in tmpl)

        return "\n".join(lines) if lines else "        -- no attributes selected"

    def get_absolute_knob_values(self, knob_names: list[str]) -> dict[str, float]:
        """Return underlying absolute strengths for optimisation knob names.

        For ``*.dk0l/*.dk1l/*.dk2l`` (and skew equivalents) this returns the base
        element strength before any perturbation. For direct knobs it returns the
        current value from the sequence.
        """
        absolute_names = [_absolute_name_from_dk_knob(knob) or knob for knob in knob_names]
        return self.get_base_magnet_strengths(absolute_names)

    def _make_adj_knobs(self) -> None:
        """Create deferred-strength knobs in MAD for all elements matching the knob specs."""
        filtered_specs = self._filter_knob_specs(self.get_knob_specs())
        self.accelerator.prepare_mad_for_knob_creation(self, filtered_specs)

        attr_block = ""
        if filtered_specs:
            attr_conditions = [
                (
                    kind, attr,
                    f'(e.kind == "{kind}" {"and e." + nonzero_attr + " ~=0 " if nonzero_attr else ""}and e.name:match("{pattern}"))',
                )
                for kind, attr, pattern, nonzero_attr in filtered_specs
            ]
            attr_block = f"""
local function store_knobs(k_str_name, mad_value, attr, spos)
    if not used[k_str_name] then
        used[k_str_name] = true  ! deduplicate (e.g. bends with shared k0)
        table.insert(knob_names, k_str_name)
        table.insert(spos_list, spos)
    end
end
{_deferred_table_helpers()}
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
{self._build_attr_block(attr_conditions)}
end
"""

        energy_block = ""
        if self.accelerator.optimise_energy:
            energy_block = """
loaded_sequence['pt'] = loaded_sequence['pt'] or 1e-6
table.insert(knob_names, "pt")
"""

        self.mad.send(f"""
local knob_names = {{}}
local spos_list = {{}}
local used = {{}}
{attr_block}
{energy_block}
coord_names = {{"x", "px", "y", "py", "t", "pt"}}
{self.py_name}:send(knob_names, true)
{self.py_name}:send(spos_list, true)
        """)
        knob_names_all: list[str] = self.mad.recv()
        elem_spos_all: list[float] = self.mad.recv()

        # Optionally restrict to a user-specified subset of knobs
        if self.accelerator.custom_knobs_to_optimise is not None:
            keep = set(self.accelerator.custom_knobs_to_optimise)
            pairs = [(k, s) for k, s in zip(knob_names_all, elem_spos_all) if k in keep]
            knob_names_all, elem_spos_all = (list(x) for x in zip(*pairs)) if pairs else ([], [])
            LOGGER.info(f"Filtered to {len(knob_names_all)} knobs based on custom_knobs_to_optimise")

        self.knob_names = knob_names_all
        self.elem_spos = elem_spos_all
        self.knob_name_set = set(self.knob_names)
        self.mad["knob_names"] = self.knob_names

        if self.elem_spos:
            LOGGER.info(
                f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}"
            )
            LOGGER.debug(f"Knob names: {self.knob_names}")
        else:
            LOGGER.info("No knobs created")

    def receive_knob_values(self) -> np.ndarray:
        """Retrieve the current values of all knobs from the MAD-NG session."""
        var_names = [f"loaded_sequence['{k}']" for k in self.knob_names]
        values = self.mad.recv_vars(*var_names)
        # recv_vars returns a scalar when only one variable is requested
        if len(self.knob_names) == 1:
            values = [values]
        return np.array(values, dtype=float)

    def update_knob_values(self, knob_values: dict[str, float]) -> None:
        """Update knob values in the MAD-NG session."""
        unknown = [n for n in knob_values if n not in self.knob_name_set]
        if unknown:
            raise ValueError(
                "Unknown knob names supplied to update_knob_values: "
                + ", ".join(unknown[:10])
                + ("..." if len(unknown) > 10 else "")
            )
        commands = [f"loaded_sequence['{n}'] = {v}" for n, v in knob_values.items()]
        if commands:
            self.mad.send("\n".join(commands))
        LOGGER.info(f"Updated {len(commands)} knobs from {len(knob_values)} provided")
