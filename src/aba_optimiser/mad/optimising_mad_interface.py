import logging
from pathlib import Path

import numpy as np
import tfs

from aba_optimiser.config import BEAM_ENERGY, SimulationConfig
from aba_optimiser.io.utils import read_knobs

from .base_mad_interface import BaseMadInterface

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


class OptimisationMadInterface(BaseMadInterface):
    """
    Encapsulates communication with MAD-NG via pymadng.MAD.
    """

    def __init__(
        self,
        sequence_file: str | Path,
        seq_name: str | None = None,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        simulation_config: SimulationConfig | None = None,
        bpm_pattern: str = BPM_PATTERN,
        bad_bpms: list[str] | None = None,
        beam_energy: float = BEAM_ENERGY,
        corrector_strengths: Path | None = None,
        tune_knobs_file: Path | None = None,
        start_bpm: str | None = None,
        py_name: str = "py",
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
        optimise_knobs: list[str] | None = None,
    ):
        """
        Initialise optimisation MAD interface with automatic setup.

        Args:
            sequence_file: Path to the MAD-X sequence file
            magnet_range: Range of magnets to include, e.g., "MARKER.1/MARKER.10"
            bpm_range: Range of BPMs to observe, e.g., "BPM.13R3.B1/BPM.12L4.B1"
            simulation_config: Simulation configuration for which parameters to optimise
            bpm_pattern: Pattern for BPM matching
            bad_bpms: List of bad BPMs to exclude from observation
            beam_energy: Beam energy in GeV
            corrector_strengths: Path to corrector strengths file, or None to skip
            seq_name: Name of the sequence to load
            tune_knobs_file: Path to tune knobs file, or None to skip
            py_name: Name used for Python-MAD communication
            debug: bool = False, Enable debug mode. Defaults to False.
            mad_logfile: Path | None = None, Path to MAD log file. Defaults to None.
            discard_mad_output: bool = False, Whether to discard MAD-NG output to stdout.
            optimise_knobs: list[str] | None = None, List of knob names to optimise. If None, optimises all knobs.
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
            stdout=stdout, redirect_stderr=redirect_stderr, py_name=py_name, debug=debug
        )
        # Store optimisation-specific attributes
        self.sequence_file = sequence_file
        if seq_name is None:
            seq_name = Path(sequence_file).stem
        self.seq_name = seq_name
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range if bpm_range is not None else magnet_range
        self.bpm_pattern = bpm_pattern
        self.beam_energy = beam_energy
        self.seq_name = seq_name
        self.optimise_knobs = optimise_knobs

        # Type hints for attributes set during initialization
        self.nbpms: int
        self.knob_names: list[str]
        self.elem_spos: list[float]

        # Perform automatic setup using base class methods
        self.load_sequence(sequence_file, self.seq_name)
        self.setup_beam(self.beam_energy)

        if start_bpm is not None:
            marker_name = self.install_marker(start_bpm, "marker_" + start_bpm)
            self.cycle_sequence(marker_name=marker_name)
            LOGGER.info(f"Cycled sequence to start at BPM: {start_bpm}")
        else:
            LOGGER.info("Skipping sequence cycling (no start BPM provided)")

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range
        self.mad["bpm_pattern"] = self.bpm_pattern

        # Setup optimisation-specific functionality
        self._observe_bpms(bad_bpms)
        self.nbpms, self.all_bpms = self.count_bpms(self.bpm_range)

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

        # Once you have done all the setup, create the knobs, if required
        if simulation_config is not None:
            self._make_adj_knobs(simulation_config)
        else:
            LOGGER.info("Skipping knob creation (no simulation config provided)")

    def count_bpms(self, bpm_range) -> tuple[int, list[str]]:
        """Count the number of BPM elements in the specified range."""
        all_bpms = self.get_bpm_list(bpm_range)
        nbpms = len(all_bpms)
        LOGGER.info(f"Counted {nbpms} BPMs in range: {bpm_range}")
        return nbpms, all_bpms

    def _observe_bpms(self, bad_bpms: list[str] | None) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        self.observe_elements(self.bpm_pattern)
        LOGGER.info(f"Set up observation for BPMs matching pattern: {self.bpm_pattern}")
        if bad_bpms:
            self.unobserve_elements(bad_bpms)
            LOGGER.info(f"Set up observation for bad BPMs: {bad_bpms}")

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

        self.mad.send(f"{self.py_name}:send(true)")
        assert self.mad.recv(), "Failed to set corrector strengths"

    def _set_tune_knobs(self, tune_knobs_file: Path) -> None:
        """Load and set predefined tune knobs from file."""
        tune_knobs = read_knobs(tune_knobs_file)
        # Get existing tune knob names in MAD
        prev = self.mad.recv_vars(*[f"MADX['{name}']" for name in tune_knobs])

        for name, val in tune_knobs.items():
            self.mad.send(f"MADX['{name}'] = {val}")

        self.mad.send(f"{self.py_name}:send(true)")
        assert self.mad.recv(), "Failed to set tune knobs"

        LOGGER.debug(f"Previous tune knob values: {prev}")
        LOGGER.debug(f"Set tune knobs from {tune_knobs_file}: {len(tune_knobs)}")

    def _make_bend_dict(self) -> None:
        """
        Create a dictionary of the bend strengths in the current sequence.
        Includes both sbends and rbends for normalisation.
        """
        from aba_optimiser.physics.lhc_bends import normalise_lhcbend_magnets

        self.mad.send(f"""
        bend_dict = {{}}
        bend_lengths = {{}}
        for i, e in loaded_sequence:siter(magnet_range) do
            if (e.kind == "sbend" or e.kind == "rbend") and e.k0 ~= 0 then
                bend_dict[e.name .. ".k0"] = e.k0
                bend_lengths[e.name .. ".k0"] = e.l ! Length is `l` attribute
            end
        end
        ! print("Sending bend dictionary " .. MAD.tostring(bend_dict))
        {self.py_name}:send(bend_dict, true)
        {self.py_name}:send(bend_lengths, true)
        bend_dict = {self.py_name}:recv()
        ! print("Retrieved " .. MAD.tostring(bend_dict))
        """)
        true_strengths_dict: dict[str, float] = self.mad.recv()
        self.bend_lengths: dict[str, float] = self.mad.recv()
        normalised_names = normalise_lhcbend_magnets(true_strengths_dict, self.bend_lengths)
        self.mad.send(normalised_names)

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
        attr_specs = {
            "sbend": {
                "name_expr": 'string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".k0"',
                "mad_value": "bend_dict[k_str_name]",
            },
            "rbend": {
                "name_expr": 'string.gsub(e.name, "(MB[RXWAL]%w*%.)([A-G]?)([0-9]+[LR][1-8].*)", "%1%3") .. (e.k0 >= 0 and "_p" or "_n") .. ".k0"',
                "mad_value": "bend_dict[k_str_name]",
            },
        }

        for idx, (kind, attr, condition) in enumerate(attr_conditions):
            spec = attr_specs.get(kind, {})
            name_expr = spec.get("name_expr", f'e.name .. ".{attr}"')
            mad_value = spec.get("mad_value", f"e.{attr}")
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

    def _make_adj_knobs(self, simulation_config: SimulationConfig) -> None:
        """
        Create deferred-strength knobs for elements matching the optimisation settings.
        """
        if simulation_config.optimise_bends:
            self._make_bend_dict()
        mad_code = MAKE_KNOBS_INIT_MAD

        if (
            simulation_config.optimise_quadrupoles
            or simulation_config.optimise_bends
            or simulation_config.optimise_correctors
        ):
            attr_conditions: list[tuple[str, str, str]] = []
            for kind, attr, pattern, zero_check, flag in [
                ("sbend", "k0", "MB%.", True, simulation_config.optimise_bends),
                ("rbend", "k0", "MB[RXWAL]%w*%.", True, simulation_config.optimise_bends),
                ("quadrupole", "k1", "MQ%.", True, simulation_config.optimise_quadrupoles),
                ("hkicker", "kick", "MCB", False, simulation_config.optimise_correctors),
            ]:
                if flag:
                    zero_chk_str = f"and e.{attr} ~=0" if zero_check else ""
                    condition = f'(e.kind == "{kind}" {zero_chk_str} and e.name:match("{pattern}"))'
                    attr_conditions.append((kind, attr, condition))

            attr_block = self._build_attr_block(attr_conditions)
            loop_code = MAKE_KNOBS_LOOP_MAD.format(attr_block=attr_block)
            mad_code += loop_code

        if simulation_config.optimise_energy:
            mad_code += 'table.insert(knob_names, "pt")\n'

        mad_code += MAKE_KNOBS_END_MAD.format(py_name=self.py_name)

        self.mad.send(mad_code)
        knob_names_all: list[str] = self.mad.recv()
        elem_spos_all: list[float] = self.mad.recv()

        # Filter knobs if optimise_knobs is specified
        if self.optimise_knobs is not None:
            original_count = len(knob_names_all)
            filtered_indices = [i for i, k in enumerate(knob_names_all) if k in self.optimise_knobs]
            self.knob_names = [knob_names_all[i] for i in filtered_indices]
            self.elem_spos = [elem_spos_all[i] for i in filtered_indices]
            LOGGER.info(
                f"Filtered knobs from {original_count} to {len(self.knob_names)} based on optimise_knobs"
            )
        else:
            self.knob_names = knob_names_all
            self.elem_spos = elem_spos_all

        if self.elem_spos:
            LOGGER.info(
                f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}"
            )
            LOGGER.info(f"Knob names: {self.knob_names}")
        else:
            LOGGER.info("No knobs created. Just optimising energy")

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
        for name, value in knob_values.items():
            if name in self.knob_names:
                LOGGER.info(f"Updating knob '{name}' to value {value}")
                self.mad.send(f"MADX['{name}'] = {value}")
            else:
                LOGGER.warning(f"Unknown knob '{name}' ignored")
