import logging
from pathlib import Path

import numpy as np
import tfs

from aba_optimiser.config import BEAM_ENERGY, CORRECTOR_STRENGTHS, TUNE_KNOBS_FILE, SimulationConfig
from aba_optimiser.io.utils import read_knobs

from .base_mad_interface import BaseMadInterface

# BPM_PATTERN = "^BPM%.%d-%d.*"
BPM_PATTERN = "^BPM"
LOGGER = logging.getLogger(__name__)

# MAD code templates
COUNT_BPMS_MAD = """
local nbpms = 0
local bpm_names = {{}}
for _, elm in loaded_sequence:iter("{bpm_range}") do
    if elm:is_observed() then
        nbpms = nbpms + 1
        table.insert(bpm_names, elm.name)
    end
end
{py_name}:send(nbpms)
{py_name}:send(bpm_names, true)
"""

MAKE_KNOBS_INIT_MAD = """
local knob_names = {}
local spos_list = {}
"""

MAKE_KNOBS_LOOP_MAD = """
local used = {{}}
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
    if {element_condition} then
        local k_str_name
        if e.k0 and e.k0 ~= 0 then
            k_str_name = string.gsub(e.name, "(MB%.)([ABCD])([0-9]+[LR][1-8]%.B[12])", "%1%3") .. ".k0"
            MADX[k_str_name] = e.k0 ! Must not be 0.
            e.k0 = \\->MADX[k_str_name]
        elseif e.k1 and e.k1 ~= 0 then
            k_str_name = e.name .. ".k1"
            MADX[k_str_name] = e.k1 ! Must not be 0.
            e.k1 = \\->MADX[k_str_name]
        elseif e.k2 and e.k2 ~= 0 then
            k_str_name = e.name .. ".k2"
            MADX[k_str_name] = e.k2 ! Must not be 0.
            e.k2 = \\->MADX[k_str_name]
        end
        if k_str_name and not used[k_str_name] then
            used[k_str_name] = true ! We assume that all the bends have the same k0 for [A-D].
            table.insert(knob_names, k_str_name)
            table.insert(spos_list, s)
        end
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
        sequence_file: str,
        seq_name: str | None = None,
        magnet_range: str = "$start/$end",
        bpm_range: str | None = None,
        simulation_config: SimulationConfig | None = None,
        discard_mad_output: bool = False,
        bpm_pattern: str = BPM_PATTERN,
        bad_bpms: list[str] | None = None,
        beam_energy: float = BEAM_ENERGY,
        corrector_strengths: Path | None = CORRECTOR_STRENGTHS,
        tune_knobs_file: Path | None = TUNE_KNOBS_FILE,
        start_bpm: str | None = None,
        **kwargs,
    ):
        """
        Initialise optimization MAD interface with automatic setup.

        Args:
            sequence_file: Path to the MAD-X sequence file
            magnet_range: Range of magnets to include, e.g., "MARKER.1/MARKER.10"
            bpm_range: Range of BPMs to observe, e.g., "BPM.13R3.B1/BPM.12L4.B1"
            simulation_config: Simulation configuration for which parameters to optimise
            discard_mad_output: Whether to discard MAD-NG output to stdout
            bpm_pattern: Pattern for BPM matching
            bad_bpms: List of bad BPMs to exclude from observation
            beam_energy: Beam energy in GeV
            corrector_strengths: Path to corrector strengths file, or None to skip
            seq_name: Name of the sequence to load
            tune_knobs_file: Path to tune knobs file, or None to skip
            **kwargs: Additional keyword arguments for MAD initialization
        """
        # Configure MAD output
        if discard_mad_output:
            kwargs.setdefault("stdout", "/dev/null")
            kwargs.setdefault("redirect_stderr", True)

        # Initialise base class
        super().__init__(**kwargs)

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

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range
        self.mad["bpm_pattern"] = self.bpm_pattern

        if simulation_config is not None:
            self._make_adj_knobs(simulation_config)

        # Setup optimization-specific functionality
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

        # try:
        #     tws = self.run_twiss()
        #     LOGGER.info(f"Initial tunes: {tws.q1}, {tws.q2}")
        # except RuntimeError as e:
        #     LOGGER.error(f"Failed to run twiss after setup: {e}")

    def count_bpms(self, bpm_range) -> None:
        """Count the number of BPM elements in the specified range."""
        self.mad.send(COUNT_BPMS_MAD.format(bpm_range=bpm_range, py_name=self.py_name))
        nbpms = self.mad.recv()
        all_bpms = self.mad.recv()
        assert len(all_bpms) == nbpms, "Mismatch in counted BPMs and names list"
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
            corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

            # Log how many non-zero correctors are being applied
            nonzero = (corrector_table["hkick"] != 0) | (corrector_table["vkick"] != 0)
            LOGGER.info(f"Applying {nonzero.sum()} non-zero corrector strengths from {corrector_strengths}")

            # Apply corrector strengths for non-zero correctors only
            self.apply_corrector_strengths(corrector_table[nonzero])
        except (tfs.TfsFormatError, UnboundLocalError) as e:
            LOGGER.error(f"Error reading or applying corrector strengths: {e}, assuming knobs")
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

    def _make_adj_knobs(self, simulation_config: SimulationConfig) -> None:
        """
        Create deferred-strength knobs for elements matching the optimisation settings.
        """
        mad_code = MAKE_KNOBS_INIT_MAD

        if simulation_config.optimise_quadrupoles or simulation_config.optimise_bends:
            conditions = []
            for kind, attr, pattern, flag in [
                ("sbend", "k0", "MB%.", simulation_config.optimise_bends),
                ("quadrupole", "k1", "MQ%.", simulation_config.optimise_quadrupoles),
            ]:
                if flag:
                    conditions.append(f'(e.kind == "{kind}" and e.{attr} ~=0 and e.name:match("{pattern}"))')
            element_condition = " or ".join(conditions) if conditions else "false"

            loop_code = MAKE_KNOBS_LOOP_MAD.format(element_condition=element_condition)
            mad_code += loop_code

        if simulation_config.optimise_energy:
            mad_code += 'table.insert(knob_names, "pt")\n'

        mad_code += MAKE_KNOBS_END_MAD.format(py_name=self.py_name)

        self.mad.send(mad_code)
        self.knob_names: list[str] = self.mad.recv()
        self.elem_spos: list[float] = self.mad.recv()
        if self.elem_spos:
            LOGGER.info(f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}")
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
