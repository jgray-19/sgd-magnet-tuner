# src/pymad_optimizer/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

from pathlib import Path

# Simulation parameters
MAX_EPOCHS        = int(5) # Total number of epochs for optimization
TRACKS_PER_WORKER = 2         # Number of tracks per worker
NUM_WORKERS       = 5          # Number of parallel worker processes
TOTAL_TRACKS      = TRACKS_PER_WORKER * NUM_WORKERS  # Total number of tracks

# Learning-rate schedule
WARMUP_EPOCHS    = 30          # Epochs for cosine warmup
DECAY_EPOCHS     = 1000         # Epoch at which cosine decay ends
WARMUP_LR_START  = 2e-8       # Initial learning rate at epoch 1
MAX_LR           = 1e-8        # Peak learning rate after warmup
MIN_LR           = 1e-9        # Final learning rate after decay
OPTIMISER_TYPE   = "adam"      # Optimiser type: "adam" or "amsgrad"
GRAD_NORM_ALPHA  = 0.7         # Gradient norm smoothing factor for smoothing loss
GRAD_PENALTY_COEFF = 1e-5   # Coefficient for gradient penalty

# Standard error of the noise
POSITION_STD_DEV = 1e-4   # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-6   # Standard deviation of the momentum noise
REL_K1_STD_DEV   = 1e-4   # Standard deviation of the K1 noise

# ACD parameters
RAMP_UP_TURNS   = 1_000        # Number of turns to ramp up the ACD
FLATTOP_TURNS   = 10_000       # Number of turns on the flat top
ACD_ON          = False        # Whether the ACD was used or not (Ignores the ramp up turns)

if TOTAL_TRACKS > FLATTOP_TURNS:
    raise ValueError(
        f"Total number of tracks ({TOTAL_TRACKS}) must be less than the "
        f"number of turns in the flat top ({FLATTOP_TURNS})."
    )

module_path = Path(__file__).absolute().parent.parent.parent
print(f"Current module path: {module_path}")
# File paths
SEQUENCE_FILE    = module_path / "mad_scripts/lhcb1.seq"        # MAD-X sequence file
TRACK_DATA_FILE  = module_path / "data/track_data.tfs"          # Measurement TFS file
NOISE_FILE       = module_path / "data/noise_data.feather"      # Noise TFS file
FILTERED_FILE    = module_path / "data/filtered_data.feather"   # Filtered TFS file
TRUE_STRENGTHS   = module_path / "data/true_strengths.txt"      # Ground-truth knob strengths
OUTPUT_KNOBS     = module_path / "data/final_knobs.txt"         # Where to write final strengths
KNOB_TABLE       = module_path / "data/knob_strengths_table.md" # Markdown summary of results
ELEM_NAMES_FILE  = module_path / "data/elem_names.txt"          # File with element names
TUNE_KNOBS_FILE  = module_path / "data/matched_tunes.txt"       # File with tune knobs

# Simulation specifics
BPM_RANGE        = "BPM.13R3.B1/BPM.13L4.B1"           # BPM selection range for tracking
# BPM_RANGE        = "BPM.11R3.B1/BPM.13L4.B1"           # BPM selection range for tracking
BEAM_ENERGY      = 6800                                # Beam energy in GeV
SEQ_NAME         = "lhcb1"                             # Sequence name in MAD-X (lowercase)
FILTER_DATA      = False                                # Whether to filter data with a Kalman filter
USE_NOISY_DATA   = True                                 # Whether to use noisy data for optimisation

# Instead of a single BPM_RANGE, define multiple overlapping windows:
start_bpms = [
    # 'BPM.11R3.B1',
    # 'BPM.12R3.B1',
    "BPM.13R3.B1",
    "BPM.14R3.B1",
]

end_bpms = [
    # 'BPM.13L4.B1',
    # 'BPM.13L4.B1',
    "BPM.13L4.B1",
    "BPM.13L4.B1",
    # "BPM.10L2.B1" # When doing further than the first BPM
]

# Create a list of tuples representing the start and end BPMs, it must be 
# every permutation of start and end BPMs
WINDOWS = [
    (start, end_bpms[i])
    for i, start in enumerate(start_bpms)
    # if (end != end_bpms[-1] or start == start_bpms[0]) and start != end
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
