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
    _DKNL_INDEX_BY_ATTR_LUA,
    _DKNL_STRENGTH_ATTRS,
    AbaMadInterface,
)

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator

# BPM_PATTERN = "^BPM%.%d-%d.*"
BPM_PATTERN = "^BPM"
LOGGER = logging.getLogger(__name__)

# MAD code templates
MAKE_KNOBS_INIT_MAD = """
local knob_names = {}
local spos_list = {}
"""

MAKE_KNOBS_LOOP_MAD = """
local used = {{}}
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
    local k_str_name ! Define as nil for scope
{attr_block}
    if k_str_name and not used[k_str_name] then
        ! If knob was redefined and not used yet, then add it
        used[k_str_name] = true ! We assume that all the bends have the same k0 for [A-D].
        table.insert(knob_names, k_str_name)
        table.insert(spos_list, s)
    end
end
"""

MAKE_KNOBS_END_MAD = """
coord_names = {{"x", "px", "y", "py", "t", "pt"}}
{py_name}:send(knob_names, true)
{py_name}:send(spos_list, true)
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
        accelerator: Accelerator,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bad_bpms: list[str] | None = None,
        corrector_strengths: Path | None = None,
        tune_knobs_file: Path | None = None,
        start_bpm: str | None = None,
        replace_all_monitors_with_markers: bool = False,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
    ):
        """
        Initialise generic MAD interface with automatic setup.

        Args:
            accelerator: Accelerator instance providing configuration
            magnet_range: Range of magnets to include, e.g., "MARKER.1/MARKER.10"
            bpm_range: Range of BPMs to observe, e.g., "BPM.13R3.B1/BPM.12L4.B1"
            bpm_pattern: Pattern for BPM matching. Defaults to accelerator pattern.
            bad_bpms: List of bad BPMs to exclude from observation
            beam_energy: Beam energy in GeV
            corrector_strengths: Path to corrector strengths file, or None to skip
            seq_name: Name of the sequence to load
            tune_knobs_file: Path to tune knobs file, or None to skip
            py_name: Name used for Python-MAD communication
            debug: Enable debug mode. Defaults to False.
            mad_logfile: Path to MAD log file. Defaults to None.
            discard_mad_output: Whether to discard MAD-NG output to stdout.
        """
        # Configure MAD output
        stdout = None
        redirect_stderr = False
        if mad_logfile is not None:
            stdout = mad_logfile
            redirect_stderr = True
            if discard_mad_output:
                LOGGER.warning(
                    "MAD logfile specified, but discard_mad_output is True. MAD output will be logged."
                )
            LOGGER.info(f"MAD logfile set to: {mad_logfile.absolute()}")
        elif discard_mad_output:
            stdout = "/dev/null"
            redirect_stderr = True

        # Initialise base class
        super().__init__(
            accelerator=accelerator,
            stdout=stdout,
            redirect_stderr=redirect_stderr,
            py_name=py_name,
            debug=debug,
        )
        # Store setup attributes
        self.accelerator = accelerator
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range if bpm_range is not None else magnet_range

        # Type hints for attributes set during initialization
        self.nbpms: int
        self.all_bpms: list[str]
        self.knob_names: list[str] = []
        self.knob_name_set: set[str] = set()
        self.elem_spos: list[float] = []

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range

        nbpms = None
        if replace_all_monitors_with_markers:
            # Setup observation and ranges
            self.observe_bpms(accelerator.bpm_pattern, bad_bpms)
            bpms_in_range, nbpms, all_bpms = self.count_bpms(self.bpm_range)
            # Unobserve all BPMs first to avoid duplicates
            self.unobserve_elements([accelerator.bpm_pattern])
            for bpm in bpms_in_range:
                if "monitor" in self.mad.MADX[bpm].kind:
                    self.replace_with_marker(bpm)
            self.mad.send("loaded_sequence:update()")

        if start_bpm is not None:
            if self.mad.MADX[start_bpm].kind != "marker":
                start_bpm = self.install_marker(start_bpm, "marker_" + start_bpm)
            self.cycle_sequence(marker_name=start_bpm)
            LOGGER.info(f"Cycled sequence to start at BPM: {start_bpm}")
        else:
            LOGGER.info("Skipping sequence cycling (no start BPM provided)")

        # Setup observation and ranges
        self.observe_bpms(accelerator.bpm_pattern, bad_bpms)
        self.bpms_in_range, self.nbpms, self.all_bpms = self.count_bpms(self.bpm_range)
        if nbpms is not None and nbpms != self.nbpms:
            print(
                f"I had {nbpms} in range {bpms_in_range} but counted {self.nbpms} BPMs in {self.bpms_in_range}"
            )
            print(f"First bpm: {start_bpm}, range: {self.bpm_range}")
            raise ValueError(
                f"Number of BPMs in range {self.bpm_range} does not match expected count: {nbpms} != {self.nbpms}"
            )

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

    def count_bpms(self, bpm_range) -> tuple[list[str], int, list[str]]:
        """Count the number of BPM elements in the specified range."""
        all_bpms, bpms_in_range = self.get_bpm_list(bpm_range)
        nbpms = len(bpms_in_range)
        LOGGER.info(f"Counted {nbpms} BPMs in range: {bpm_range}")
        return bpms_in_range, nbpms, all_bpms

    def _set_correctors(self, corrector_strengths: Path) -> None:
        """Load corrector strengths from file and apply them to the sequence."""
        if not corrector_strengths.exists():
            LOGGER.warning(f"Corrector strengths file not found: {corrector_strengths}")
            return
        try:
            corrector_table = tfs.read(corrector_strengths)

            # Filter out monitor elements from the corrector table
            non_monitors = corrector_table["kind"] != "monitor"
            corrector_table: tfs.TfsDataFrame = corrector_table[non_monitors]  # type: ignore[assignment, not-subscriptable]

            # Log how many non-zero correctors are being applied
            changed = (corrector_table["hkick"] != corrector_table["hkick_old"]) | (
                corrector_table["vkick"] != corrector_table["vkick_old"]
            )
            LOGGER.info(
                f"Applying {changed.sum()} non-zero corrector strengths from {corrector_strengths}"  # ty:ignore[unresolved-attribute]
            )

            # Apply corrector strengths for non-zero correctors only
            self.apply_corrector_strengths(corrector_table[changed])  # ty:ignore[invalid-argument-type]
        except (tfs.TfsFormatError, UnboundLocalError) as e:
            LOGGER.warning(f"Error reading or applying corrector strengths: {e}, assuming knobs")
            knobs = read_knobs(corrector_strengths)
            for name, val in knobs.items():
                self.mad.send(f"MADX['{name}'] = {val}")

            LOGGER.info(f"Set {len(knobs)} corrector knobs from {corrector_strengths}")

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
    Generic MAD interface for gradient descent optimization.

    Extends GenericMadInterface with knob creation capabilities specific to gradient descent
    optimization.

    Accelerator-specific behavior is provided by accelerator hooks
    (knob specs, attr specs, MAD preparation), keeping this interface generic.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        bpm_pattern: str = BPM_PATTERN,
        bad_bpms: list[str] | None = None,
        corrector_strengths: Path | None = None,
        tune_knobs_file: Path | None = None,
        start_bpm: str | None = None,
        replace_all_monitors_with_markers: bool = True,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
    ):
        """
        Initialize gradient descent MAD interface with knob creation.

        Args:
            accelerator: Accelerator instance providing configuration
            magnet_range: Range of magnets to include
            bpm_range: Range of BPMs to observe
            bpm_pattern: Pattern for BPM matching. Defaults to accelerator pattern.
            bad_bpms: List of bad BPMs to exclude
            corrector_strengths: Path to corrector strengths file
            tune_knobs_file: Path to tune knobs file
            py_name: Name used for Python-MAD communication
            debug: Enable debug mode
            mad_logfile: Path to MAD log file
            discard_mad_output: Whether to discard MAD-NG output
        """
        super().__init__(
            accelerator,
            magnet_range,
            bpm_range,
            bad_bpms,
            corrector_strengths,
            tune_knobs_file,
            start_bpm,
            replace_all_monitors_with_markers,
            py_name,
            debug,
            mad_logfile,
            discard_mad_output,
        )

        if accelerator.has_any_optimisation():
            # Create gradient descent knobs using accelerator-provided specs
            self._make_adj_knobs()
        else:
            LOGGER.warning(
                "Gradient descent optimisation interface initialised without any optimisation enabled."
                "\nUse GenericMadInterface if no optimisation is required."
            )

    def get_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """
        Get the list of knob specifications for this accelerator.

        Returns the full list of supported knobs from the accelerator definition.

        Returns:
            List of (kind, attribute, pattern, zero_check) tuples:
            - kind: MAD element kind (e.g., "sbend", "quadrupole", "hkicker")
            - attribute: MAD element attribute (e.g., "k0", "k1", "kick")
            - pattern: Regex pattern to match element names
            - zero_check: Whether to exclude elements with zero attribute values
        """
        return self.accelerator.get_supported_knob_specs()

    def _filter_knob_specs(
        self, all_specs: list[tuple[str, str, str, bool, bool]]
    ) -> list[tuple[str, str, str, bool]]:
        """
        Filter knob specifications based on the accelerator's optimisation settings.

        Default filtering keeps only specs enabled by accelerator optimise_* flags.

        Args:
            all_specs: All available knob specifications from get_knob_specs()

        Returns:
            Filtered list of specs to actually create knobs for
        """
        return [
            (kind, attr, pattern, zero_check)
            for kind, attr, pattern, zero_check, optimise_flag in all_specs
            if optimise_flag
        ]

    def _prepare_knob_data(self, selected_specs: list[tuple[str, str, str, bool]]) -> None:
        """Prepare accelerator-specific MAD state required for knob creation."""
        self.accelerator.prepare_mad_for_knob_creation(self, selected_specs)

    def _get_attr_specs(self) -> dict[str, dict[str, str]]:
        """Return accelerator-specific attribute name/value expressions."""
        return self.accelerator.get_mad_attr_specs()

    def _build_attr_block(self, attr_conditions: list[tuple[str, str, str]]) -> str:
        """
        Build the MAD-NG Lua block that assigns deferred knobs for the selected attributes.

        attr_conditions carries the fully-qualified condition for each attribute (kind, name pattern,
        non-zero check) so the generated Lua `if` clauses mirror the same predicates used in the
        outer `element_condition`.

        This was created to handle generic attributes, and special cases like bend k0 naming,
        simplifying extension.
        """
        lines: list[str] = []
        attr_specs = self._get_attr_specs()

        for idx, (kind, attr, condition) in enumerate(attr_conditions):
            spec = attr_specs.get(kind, {})
            name_expr = spec.get("name_expr", f'e.name .. ".{attr}"')
            mad_value = spec.get("mad_value", f"e.{attr}")
            if attr in _DKNL_STRENGTH_ATTRS:
                dknl_index = _DKNL_INDEX_BY_ATTR_LUA[attr]
                preserved_lines = []
                deferred_entries = []
                for preserved_index in range(1, 5):
                    if preserved_index == dknl_index:
                        continue
                    preserved_name = f"dknl_{preserved_index}"
                    preserved_lines.append(
                        f"    local {preserved_name} = current_dknl[{preserved_index}]"
                    )
                    deferred_entries.append(
                        f"            [{preserved_index}] := {preserved_name},"
                    )
                deferred_entry_lines = "\n".join(deferred_entries)
                tmpl = [
                    f"{{prefix}} {condition} then",
                    f"    k_str_name = {name_expr}",
                    f"    local base_strength = {mad_value}",
                    "    local current_strength = base_strength",
                    "    local length = e.l",
                    "    if length ~= 0 then",
                    f"        current_strength = current_strength + e.dknl[{dknl_index}] / length",
                    "    end",
                    "    MADX[k_str_name] = current_strength",
                    "    local current_dknl = e.dknl",
                    *preserved_lines,
                    "    if length ~= 0 then",
                    "        e.dknl = MAD.typeid.deferred{{",
                    deferred_entry_lines,
                    f"            [{dknl_index}] := (MADX[k_str_name] - base_strength) * length,",
                    "        }}",
                    "    else",
                    "        e.dknl = MAD.typeid.deferred{{",
                    deferred_entry_lines,
                    f"            [{dknl_index}] := 0.0,",
                    "        }}",
                    "    end",
                ]
            else:
                tmpl = [
                    f"{{prefix}} {condition} then",
                    f"    k_str_name = {name_expr}",
                    f"    MADX[k_str_name] = {mad_value}",
                    f"    e.{attr} = \\->MADX[k_str_name]",
                ]
            prefix = "if" if idx == 0 else "elseif"
            for line in tmpl:
                lines.append(f"        {line.format(prefix=prefix)}")

        if not lines:
            lines.append("        -- no attributes selected")
        lines.append("        end")
        return "\n".join(lines)

    def _make_adj_knobs(self) -> None:
        """
        Create deferred-strength knobs for elements matching the knob specifications.

        Uses accelerator-provided knob specs to determine which knobs to create.
        """
        knob_specs = self.get_knob_specs()

        # Filter knob specs based on configuration
        filtered_specs = self._filter_knob_specs(knob_specs)

        self._prepare_knob_data(filtered_specs)

        mad_code = MAKE_KNOBS_INIT_MAD

        if filtered_specs:
            attr_conditions: list[tuple[str, str, str]] = []
            for kind, attr, pattern, zero_check in filtered_specs:
                zero_chk_str = f"and e.{attr} ~=0" if zero_check else ""
                condition = f'(e.kind == "{kind}" {zero_chk_str} and e.name:match("{pattern}"))'
                attr_conditions.append((kind, attr, condition))

            attr_block = self._build_attr_block(attr_conditions)
            loop_code = MAKE_KNOBS_LOOP_MAD.format(attr_block=attr_block)
            mad_code += loop_code

        # Add energy knob if needed
        if self.accelerator.optimise_energy:
            mad_code += 'table.insert(knob_names, "pt")\n'

        mad_code += MAKE_KNOBS_END_MAD.format(py_name=self.py_name)

        self.mad.send(mad_code)
        knob_names_all: list[str] = self.mad.recv()
        elem_spos_all: list[float] = self.mad.recv()

        # Filter knobs if optimise_knobs is specified
        if self.accelerator.custom_knobs_to_optimise is not None:
            original_count = len(knob_names_all)
            filtered_indices = [
                i
                for i, k in enumerate(knob_names_all)
                if k in self.accelerator.custom_knobs_to_optimise
            ]
            self.knob_names = [knob_names_all[i] for i in filtered_indices]
            self.elem_spos = [elem_spos_all[i] for i in filtered_indices]
            LOGGER.info(
                f"Filtered knobs from {original_count} to {len(self.knob_names)} based on custom_knobs_to_optimise"
            )
        else:
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
        """
        Retrieve the current values of knobs from the MAD-NG session.

        Returns:
            np.ndarray: Array of knob values in the same order as knob_names.
        """
        var_names = [f"MADX['{k}']" for k in self.knob_names]
        values = self.mad.recv_vars(*var_names)
        # Handle case where recv_vars returns a scalar for single knob
        if len(self.knob_names) == 1:
            values = [values]
        return np.array(values, dtype=float)

    def update_knob_values(self, knob_values: dict[str, float]) -> None:
        """
        Update the values of knobs in the MAD-NG session.

        Args:
            knob_values (dict[str, float]): Dictionary of knob names and their new values.
        """
        update_commands: list[str] = []
        for name, value in knob_values.items():
            if name in self.knob_name_set:
                LOGGER.info(f"Updating knob '{name}' to value {value}")
                update_commands.append(f"MADX['{name}'] = {value}")
            else:
                LOGGER.warning(f"Unknown knob '{name}' ignored")
        if update_commands:
            self.mad.send("\n".join(update_commands))
