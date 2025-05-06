# src/pymad_optimizer/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

from pathlib import Path

# Simulation parameters
MAX_EPOCHS        = int(1100)  # Total number of epochs for optimization
TRACKS_PER_WORKER = 200         # Number of tracks per worker
NUM_WORKERS       = 30          # Number of parallel worker processes
TOTAL_TRACKS      = TRACKS_PER_WORKER * NUM_WORKERS  # Total number of tracks

# Learning-rate schedule
WARMUP_EPOCHS    = 100         # Epochs for cosine warmup
DECAY_EPOCHS     = 1000        # Epoch at which cosine decay ends
WARMUP_LR_START  = 5e-6        # Initial learning rate at epoch 1
MAX_LR           = 7e-6        # Peak learning rate after warmup
MIN_LR           = 4e-6        # Final learning rate after decay

RAMP_UP_TURNS   = 500          # Number of turns to ramp up the beam
FLATTOP_TURNS   = 6000         # Number of turns to flatten the beam

if TOTAL_TRACKS > FLATTOP_TURNS:
    raise ValueError(
        f"Total number of tracks ({TOTAL_TRACKS}) must be less than the "
        f"number of turns in the flat top ({FLATTOP_TURNS})."
    )

# File paths
SEQUENCE_FILE    = Path("lhcb1.seq").absolute()                        # MAD-X sequence file
TRACK_DATA_FILE  = Path("data/track_data.tfs").absolute()              # Measurement TFS file
TRUE_STRENGTHS   = Path("data/true_strengths.txt").absolute()          # Ground-truth knob strengths
OUTPUT_KNOBS     = Path("data/final_knobs.txt").absolute()             # Where to write final strengths
KNOB_TABLE       = Path("data/knob_strengths_table.md").absolute()     # Markdown summary of results
ELEM_NAMES_FILE  = Path("data/elem_names.txt").absolute()            # File with element names

# Simulation specifics
BPM_RANGE        = "BPM.13R3.B1/BPM.13L4.B1"           # BPM selection range for tracking
BEAM_ENERGY      = 6800                                # Beam energy in GeV
SEQ_NAME        = "lhcb1"                             # Sequence name in MAD-X

# Instead of a single BPM_RANGE, define multiple overlapping windows:
start_bpms = [
    "BPM.13R3.B1",
    # "BPM.14R3.B1",
    # "BPM.21R3.B1",
]

end_bpms = [
    # "BPM.21L4.B1",
    # "BPM.14L4.B1",
    "BPM.13L4.B1",
]

# Create a list of tuples representing the start and end BPMs, it must be 
# every permutation of start and end BPMs
WINDOWS = [
    (start, end) for start in start_bpms for end in end_bpms
]
# Ensure that the start and end BPMs are not the same
for start, end in WINDOWS:
    if start == end:
        raise ValueError(f"Start and end BPMs cannot be the same: {start} == {end}")
    

"""
This has the current problem of no matter the number of files included in the simulation, 
I cannot reduce the error on the final result significantly. I am still limited to 5e-4. 
Potentially, I can get better results by doing the following method:
1. Optimise the Main Quadrupoles
# 2. Optimise the trim quadrupoles - might need to use other information other than x and y 
3. Optimise the skew quadrupoles ???
4. Optimise the sextupoles

S location as a degree of freedom?
Find out where individual errors might be reduced most. 


TODO:
- Look at adding errors to the sextupoles. 
- Look at adding two more simulations with off momentum errors.
- Look at understanding the uncertainty - mathematically, where it arises, and 
    which parameters reduce it and how.
- Think about tracking a distribution of particles using MAD-NG with a single 
    coordinate system.???

- think about how to pinpoint errors in the lattice, i.e. identify the largest 
    errors in the lattice
"""
