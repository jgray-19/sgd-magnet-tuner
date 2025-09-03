from __future__ import annotations

import logging

import numpy as np
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BEAM_ENERGY,
    CORRECTOR_STRENGTHS,
    SEQ_NAME,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.io.utils import read_knobs

BPM_PATTERN = "^BPM%.%d-%d.*"
LOGGER = logging.getLogger(__name__)


class MadInterface:
    """
    Encapsulates communication with MAD-NG via pymadng.MAD.
    """

    def __init__(
        self,
        sequence_file: str,
        magnet_range: str,
        optimise_sextupoles: bool = False,
        bpm_range: str = None,
        discard_mad_output: bool = True,
        bpm_pattern: str = BPM_PATTERN,
        **kwargs,
    ):
        """
        Args:
            sequence_file (str): Path to the MAD-X sequence file.
            magnet_range (str): Range of magnets to include, e.g., "MARKER.1/MARKER.10".
            bpm_range (str): Range of BPMs to observe, e.g., "BPM.13R3.B1/BPM.12L4.B1".
            discard_mad_output (bool): Whether to discard MAD-NG output to stdout - will change the kwargs for pymadng.MAD.
            **kwargs: Additional keyword arguments for pymadng.MAD initialization.
        """
        self.sequence_file = sequence_file
        self.magnet_range = magnet_range

        if bpm_range is None:
            bpm_range = magnet_range
        self.bpm_range = bpm_range
        self.bpm_pattern = bpm_pattern

        if discard_mad_output:
            kwargs["stdout"] = "/dev/null"
            kwargs["redirect_stderr"] = True
        self.mad = MAD(**kwargs)
        self._load_sequence()

        self._count_bpms()
        self._observe_bpms()

        self._set_correctors()
        self._set_tune_knobs()
        self._make_adj_knobs(optimise_sextupoles)

    def _load_sequence(self) -> None:
        """Load the sequence and count BPMs in the specified range."""
        self.mad.send(f'MADX:load("{self.sequence_file}")')
        self.mad["magnet_range"] = self.magnet_range
        self.mad["bpm_range"] = self.bpm_range
        self.mad["bpm_pattern"] = self.bpm_pattern

        self.mad.send(f"loaded_sequence = MADX.{SEQ_NAME}")
        self.mad.send(
            f"loaded_sequence.beam = beam {{ particle = 'proton', energy = {BEAM_ENERGY} }}"
        )
        logging.info(f"Loaded sequence: {self.sequence_file}")

    def _count_bpms(self) -> None:
        """Count the number of BPM elements in the given range."""
        code = """
nbpms = 0
for _, elm in loaded_sequence:iter(bpm_range) do
    if elm.name:match(bpm_pattern) then
        nbpms = nbpms + 1
    end
end
py:send(nbpms)
"""
        self.mad.send(code)
        self.nbpms: int = self.mad.recv()
        logging.info(f"Counted {self.nbpms} BPMs in range: {self.bpm_range}")

    def _observe_bpms(self) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        code = """
local observed in MAD.element.flags
loaded_sequence:deselect(observed)
loaded_sequence:select(observed, {pattern=bpm_pattern})
py:send(true)
"""
        self.mad.send(code)
        assert self.mad.recv(), "Failed to set up BPM observation"
        logging.info(f"Set up observation for BPMs matching pattern: {BPM_PATTERN}")

    def _set_correctors(self) -> None:
        if not CORRECTOR_STRENGTHS.exists():
            LOGGER.warning(f"Corrector strengths file not found: {CORRECTOR_STRENGTHS}")
            return
        corrector_table = tfs.read(CORRECTOR_STRENGTHS)

        # Remove all rows that have monitor in the column kind
        corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

        # Loop through every row in the table
        for _, row in corrector_table.iterrows():
            self.mad.send(f"loaded_sequence['{row['ename']}'].hkick = py:recv()")
            self.mad.send(row["hkick"])
            self.mad.send(f"loaded_sequence['{row['ename']}'].vkick = py:recv()")
            self.mad.send(row["vkick"])

    def _set_tune_knobs(self) -> None:
        """Set up the MAD-NG session to include predefined tune knobs."""
        tune_knobs = read_knobs(TUNE_KNOBS_FILE)
        for name, val in tune_knobs.items():
            self.mad.send(f"MADX.{name} = {val}")

        self.mad.send("py:send(true)")
        assert self.mad.recv(), "Failed to set tune knobs"

        logging.info(f"Set tune knobs from {TUNE_KNOBS_FILE}: {tune_knobs}")

    def _make_adj_knobs(self, optimise_sextupoles: bool) -> None:
        """
        Create deferred-strength knobs for each group in elem_groups.
        If a group contains more than one element, they will share the same knob.
        """
        element_condition = (
            '(e.kind == "quadrupole" and e.k1 ~=0 and e.name:match("MQ%."))'
        )
        if optimise_sextupoles:
            element_condition += (
                ' or (e.kind == "sextupole" and e.k2 ~= 0 and e.name:match("MS%."))'
            )
        add_sext_knobs = ""
        if optimise_sextupoles:
            add_sext_knobs = """
            MADX[k_str_name] = e.k2 ! Must not be 0.
            e.k2 = \\->MADX[k_str_name]
            """

        self.mad.send(f"""
local knob_names = {{}}
local spos_list = {{}}
for i, e, s, ds in loaded_sequence:siter(magnet_range) do
    if {element_condition} then
        local k_str_name = e.name .. "_k"
        if e.k1 and e.k1 ~= 0 then
            MADX[k_str_name] = e.k1 ! Must not be 0.
            e.k1 = \\->MADX[k_str_name]
        elseif e.k2 and e.k2 ~= 0 then
            {add_sext_knobs}
        end
        table.insert(knob_names, k_str_name)
        table.insert(spos_list, s)
    end
end
table.insert(knob_names, "pt")
coord_names = {{"x", "px", "y", "py", "t", "pt"}}
py:send(knob_names, true)
py:send(spos_list, true)
""")
        self.knob_names: list[str] = self.mad.recv()
        self.elem_spos: list[float] = self.mad.recv()
        logging.info(
            f"Created {len(self.knob_names)} knobs from {self.elem_spos[0]} to {self.elem_spos[-1]}"
        )

    def receive_knob_values(self) -> np.ndarray:
        """
        Retrieve the current values of knobs from the MAD-NG session.

        Returns:
            np.ndarray: Array of knob values in the same order as knob_names.
        """
        var_names = [f"MADX['{k}']" for k in self.knob_names]
        values = self.mad.recv_vars(*var_names)
        return np.array(values, dtype=float)

    def run_twiss(self) -> tfs.TfsDataFrame:
        self.mad.send("""
tws = twiss{sequence=loaded_sequence, observe=1}
""")
        return self.mad.tws.to_df().set_index("name")

    def pt2dp(self, pt) -> float:
        self.mad.send("py:send(MAD.gphys.pt2dp(py:recv(), loaded_sequence.beam.beta))")
        self.mad.send(pt)
        return self.mad.recv()

    def dp2pt(self, dp) -> float:
        self.mad.send("py:send(MAD.gphys.dp2pt(py:recv(), loaded_sequence.beam.beta))")
        self.mad.send(dp)
        return self.mad.recv()

    def __del__(self):
        """Clean up the MAD-NG session."""
        if hasattr(self, "mad"):
            del self.mad
