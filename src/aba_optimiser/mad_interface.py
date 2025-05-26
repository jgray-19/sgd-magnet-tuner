import numpy as np
from pymadng import MAD
from aba_optimiser.config import BEAM_ENERGY, SEQ_NAME, TUNE_KNOBS_FILE, ELEM_NAMES_FILE
from aba_optimiser.utils import read_knobs, read_elem_names

class MadInterface:
    """
    Encapsulates communication with MAD-NG via pymadng.MAD.
    """

    def __init__(
        self,
        sequence_file: str,
        bpm_range: str,
        **kwargs,
    ):
        self.sequence_file = sequence_file
        self.bpm_range = bpm_range
        self.mad = MAD(**kwargs)
        self._load_sequence()

        self.nbpms = self._count_bpms()
        self._observe_bpms()

        self._set_tune_knobs()
        self.knob_names = self._make_adj_knobs()

    def _load_sequence(self) -> None:
        """Load the sequence and count BPMs in the specified range."""
        self.mad.send(f'MADX:load("{self.sequence_file}")')
        self.mad.send(f'range = "{self.bpm_range}"')
        self.mad.send(f"MADX.{SEQ_NAME}.beam = beam {{ particle = 'proton', energy = {BEAM_ENERGY} }}")

    def _count_bpms(self) -> int:
        """Count the number of BPM elements in the given range."""
        code = f"""
local {SEQ_NAME} in MADX
nbpms = 0
for _, elm in {SEQ_NAME}:iter(range) do
    if elm.name:match("BPM") then 
        nbpms = nbpms + 1 
    end
end
py:send(nbpms)
"""
        self.mad.send(code)
        return self.mad.recv()

    def _observe_bpms(self) -> None:
        """Set up the MAD-NG session to observe BPMs."""
        code = f"""
local {SEQ_NAME} in MADX
local observed in MAD.element.flags
{SEQ_NAME}:deselect(observed)
{SEQ_NAME}:select(observed, {{pattern="BPM"}})
"""
        self.mad.send(code)

    def _set_tune_knobs(self) -> None:
        """Set up the MAD-NG session to include predefined tune knobs."""
        tune_knobs = read_knobs(TUNE_KNOBS_FILE)
        for name, val in tune_knobs.items():
            before = self.mad.recv_vars(f"MADX.{name}")
            self.mad.send(f"MADX.{name} = {val}")
            print(f"Set {name} from {before} to {val}")

    def _make_adj_knobs(self) -> list[str]:
        """
        Create deferred-strength knobs for each group in elem_groups.
        If a group contains more than one element, they will share the same knob.
        """
        _, elem_groups = read_elem_names(ELEM_NAMES_FILE)
        # Send element groups and create knobs in MAD-NG
        self.mad.send("elem_groups = py:recv()").send(elem_groups)
        self.mad.send(f"""
local {SEQ_NAME} in MADX
knob_names = {{}}
for i, group in ipairs(elem_groups) do
    local base_elem = group[1]
    assert({SEQ_NAME}[base_elem], "Element " .. base_elem .. " not found")
    
    local k1_str_name = base_elem:gsub("%.[AB](%d+[RL]%d%.B%d)$", ".%1") .. "_k1"
    table.insert(knob_names, k1_str_name)
    -- Add k1s here if necessary
    
    {SEQ_NAME}[k1_str_name] = {SEQ_NAME}[base_elem].k1 or 0
    for _, elem in ipairs(group) do
        assert({SEQ_NAME}[elem], "Element " .. elem .. " not found")
        -- Set the shared knob for all elements in this group
        {SEQ_NAME}[elem].k1 = \\->{SEQ_NAME}[k1_str_name]
    end
end
coord_names = {{ "x", "px", "y", "py", "t", "pt" }}
py:send(knob_names)
""")
        knob_names = self.mad.recv()
        return knob_names

    def receive_knob_values(self) -> np.ndarray:
        """
        Retrieve the current values of knobs from the MAD-NG session.

        Returns:
            np.ndarray: Array of knob values in the same order as knob_names.
        """
        var_names = [f"MADX.{SEQ_NAME}['{k}']" for k in self.knob_names]
        values = self.mad.recv_vars(*var_names)
        return np.array(values, dtype=float)

    def update_knobs(self, knob_updates: dict[str, float]) -> None:
        """
        Update the knob strengths in the MAD-NG session.

        Args:
            knob_updates (dict[str, float]): Mapping from knob name to new value.
        """
        for name, val in knob_updates.items():
            self.mad.send(f"MADX.{SEQ_NAME}['{name}']:set0({val})")

    def get_element_positions(self) -> np.ndarray:
        """
        Retrieve the s-coordinate (longitudinal position) for each knob in knob_names.
        
        Returns:
            np.ndarray: Array of s-coordinates in the same order as knob_names.
        """
        base_names = [name.replace("_k1", "") for name in self.knob_names]
        self.mad.send(f"""
local knob_names = py:recv()
positions = {{}}
for i, elm, spos, ds in MADX.{SEQ_NAME}:iter() do
    base_name = elm.name:gsub("%.[AB](%d+[RL]%d%.B%d)$", ".%1")
    for _, name in ipairs(knob_names) do
        if base_name == name then
            positions[base_name] = spos
        end
    end
end
py:send(positions)
        """)
        self.mad.send(base_names)
        positions = self.mad.recv("positions") 
        # Convert the dictionary to a list of positions
        positions = [positions[name] for name in base_names]
        return np.array(positions, dtype=float)

    def __del__(self):
        """Clean up the MAD-NG session."""
        del self.mad
