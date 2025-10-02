from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tfs

from aba_optimiser.config import (
    BEAM_ENERGY,
    CORRECTOR_STRENGTHS,
    SEQ_NAME,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.io.utils import read_knobs

from .base_mad_interface import BaseMadInterface

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings


# BPM_PATTERN = "^BPM%.%d-%d.*"
BPM_PATTERN = "^BPM"
LOGGER = logging.getLogger(__name__)

# MAD code templates
COUNT_BPMS_MAD = """
local nbpms = 0
for _, elm in loaded_sequence:iter("{bpm_range}") do
    if elm.name:match(bpm_pattern) then
        nbpms = nbpms + 1
    end
end
{py_name}:send(nbpms)
"""

MAKE_KNOBS_INIT_MAD = """
local knob_names = {}
local spos_list = {}
"""

MAKE_KNOBS_LOOP_MAD = """
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
    if {element_condition} then
        local k_str_name = e.name .. "_k"
        if e.k0 and e.k0 ~= 0 then --and not (k_str_name:match("MB%.[AB][0-9]+R") or k_str_name:match("MB%.[BC][0-9]+L")) then
            -- k_str_name = k_str_name:gsub("^MB%.[AC]", "MB.")
            MADX[k_str_name] = e.k0 ! Must not be 0.
            e.k0 = \\->MADX[k_str_name]
        elseif e.k1 and e.k1 ~= 0 then
            MADX[k_str_name] = e.k1 ! Must not be 0.
            e.k1 = \\->MADX[k_str_name]
        elseif e.k2 and e.k2 ~= 0 then
            MADX[k_str_name] = e.k2 ! Must not be 0.
            e.k2 = \\->MADX[k_str_name]
        else
            goto continue
        end
        table.insert(knob_names, k_str_name)
        table.insert(spos_list, s)
        ::continue::
    end
end
"""

MAKE_KNOBS_END_MAD = """
table.insert(knob_names, "pt")
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
        magnet_range: str = "$start/$end",
        opt_settings: OptSettings = None,
        bpm_range: str | None = None,
        discard_mad_output: bool = True,
        bpm_pattern: str = BPM_PATTERN,
        use_real_strengths: bool = True,
        **kwargs,
    ):
        """
        Initialize optimization MAD interface with automatic setup.

        Args:
            sequence_file: Path to the MAD-X sequence file
            magnet_range: Range of magnets to include, e.g., "MARKER.1/MARKER.10"
            bpm_range: Range of BPMs to observe, e.g., "BPM.13R3.B1/BPM.12L4.B1"
            discard_mad_output: Whether to discard MAD-NG output to stdout
            bpm_pattern: Pattern for BPM matching
            use_real_strengths: Whether to use real corrector/knob strengths
            opt_settings: Optimization settings for adjustment knobs
            **kwargs: Additional keyword arguments for MAD initialization
        """
        # Configure MAD output
        if discard_mad_output:
            kwargs.setdefault("stdout", "/dev/null")
            kwargs.setdefault("redirect_stderr", True)

        # Initialize base class
        super().__init__(**kwargs)

        # Store optimization-specific attributes
        self.sequence_file = sequence_file
        self.magnet_range = magnet_range
        self.bpm_range = bpm_range if bpm_range is not None else magnet_range
        self.bpm_pattern = bpm_pattern

        # Type hints for attributes set during initialization
        self.nbpms: int
        self.knob_names: list[str]
        self.elem_spos: list[float]

        # Perform automatic setup using base class methods
        self.load_sequence(sequence_file, SEQ_NAME)
        self.setup_beam(BEAM_ENERGY)

        # Set MAD variables for ranges and patterns
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range
        self.mad["bpm_pattern"] = self.bpm_pattern

        # Setup optimization-specific functionality
        self.nbpms = self.count_bpms(self.bpm_range)
        self._observe_bpms()
        # if opt_settings.only_energy:
        self._set_correctors(use_real_strengths)
        self._set_tune_knobs(use_real_strengths)

        if opt_settings is not None:
            self._make_adj_knobs(opt_settings)

    def _apply_noise(
        self, value: float, rng: np.random.Generator | None = None
    ) -> float:
        """Apply Gaussian noise to a value if rng is provided."""
        if rng is None:
            return value
        return value + rng.normal(0, 1e-4 * abs(value))

    def count_bpms(self, bpm_range) -> None:
        """Count the number of BPM elements in the specified range."""
        self.mad.send(COUNT_BPMS_MAD.format(bpm_range=bpm_range, py_name=self.py_name))
        nbpms = self.mad.recv()
        LOGGER.info(f"Counted {nbpms} BPMs in range: {bpm_range}")
        return nbpms

    def _observe_bpms(self) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        self.observe_elements(self.bpm_pattern)
        LOGGER.info(f"Set up observation for BPMs matching pattern: {self.bpm_pattern}")

    def _set_correctors(self, use_real_strengths: bool | str) -> None:
        """Load corrector strengths from file and apply them to the sequence."""
        if not CORRECTOR_STRENGTHS.exists():
            LOGGER.warning(f"Corrector strengths file not found: {CORRECTOR_STRENGTHS}")
            return
        if use_real_strengths is False:
            LOGGER.info("Skipping setting correctors as use_real_strengths is False")
            return
        # if use_real_strengths == "noisy":
        #     LOGGER.info(
        #         "Adding noise to corrector strengths as use_real_strengths is 'noisy'"
        #     )
        #     rng = np.random.default_rng(seed=43)
        # else:
        #     rng = None
        corrector_table = tfs.read(CORRECTOR_STRENGTHS)

        # Filter out monitor elements from the corrector table
        corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

        # Log how many non-zero correctors are being applied
        nonzero = (corrector_table["hkick"] != 0) | (corrector_table["vkick"] != 0)
        LOGGER.info(
            f"Applying {nonzero.sum()} non-zero corrector strengths from {CORRECTOR_STRENGTHS}"
        )

        # Apply corrector strengths for non-zero correctors only
        self.apply_corrector_strengths(corrector_table[nonzero])

    def _set_tune_knobs(self, use_real_strengths: bool | str) -> None:
        """Load and set predefined tune knobs from file."""
        if use_real_strengths is False:
            LOGGER.info("Skipping setting tune knobs as use_real_strengths is False")
            return

        tune_knobs = read_knobs(TUNE_KNOBS_FILE)
        # rng = np.random.default_rng(seed=41) if use_real_strengths == "noisy" else None

        for name, val in tune_knobs.items():
            self.mad.send(f"MADX.{name} = {self._apply_noise(val)}")

        self.mad.send(f"{self.py_name}:send(true)")
        assert self.mad.recv(), "Failed to set tune knobs"

        LOGGER.info(f"Set tune knobs from {TUNE_KNOBS_FILE}: {tune_knobs}")

    def _make_adj_knobs(self, opt_settings: OptSettings) -> None:
        """
        Create deferred-strength knobs for elements matching the optimisation settings.
        """
        mad_code = MAKE_KNOBS_INIT_MAD

        if not opt_settings.only_energy:
            conditions = []
            for kind, attr, pattern, flag in [
                ("sbend", "k0", "MB%.", True),
                ("quadrupole", "k1", "MQ%.", opt_settings.optimise_quadrupoles),
                ("sextupole", "k2", "MS%.", opt_settings.optimise_sextupoles),
            ]:
                if flag:
                    conditions.append(
                        f'(e.kind == "{kind}" and e.{attr} ~=0 and e.name:match("{pattern}"))'
                    )
            element_condition = " or ".join(conditions) if conditions else "false"

            loop_code = MAKE_KNOBS_LOOP_MAD.format(element_condition=element_condition)
            mad_code += loop_code

        mad_code += MAKE_KNOBS_END_MAD.format(py_name=self.py_name)

        self.mad.send(mad_code)
        self.knob_names: list[str] = self.mad.recv()
        self.elem_spos: list[float] = self.mad.recv()
        if self.elem_spos:
            LOGGER.info(
                f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}"
            )
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
