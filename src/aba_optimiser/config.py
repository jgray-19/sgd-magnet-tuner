# src/pymad_optimizer/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

from pathlib import Path

# Simulation parameters
MAX_EPOCHS        = int(2000) # Total number of epochs for optimization
TRACKS_PER_WORKER = 217        # Number of tracks per worker
NUM_WORKERS       = 10          # Number of parallel worker processes
TOTAL_TRACKS      = TRACKS_PER_WORKER * NUM_WORKERS  # Total number of tracks

# Learning-rate schedule
WARMUP_EPOCHS    = 30          # Epochs for cosine warmup
DECAY_EPOCHS     = MAX_EPOCHS - WARMUP_EPOCHS         # Epoch at which cosine decay ends
WARMUP_LR_START  = 5e-9       # Initial learning rate at epoch 1
MAX_LR           = 5e-9      # Peak learning rate after warmup
MIN_LR           = 5e-9
OPTIMISER_TYPE   = "adam"      # Optimiser type: "adam" or "amsgrad"
GRAD_NORM_ALPHA  = 0.7         # Gradient norm smoothing factor for smoothing loss
GRAD_PENALTY_COEFF = 1e-5   # Coefficient for gradient penalty

MIN_FRACTION_MAX = 0.5  # Fraction of the maximum coordinate value that is the minimum to be considered
XY_MIN           = 1e-3
PXPY_MIN         = 15e-6
STD_CUT          = 1

# Standard error of the noise
POSITION_STD_DEV = 1e-5   # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-6   # Standard deviation of the momentum noise
REL_K1_STD_DEV   = 1e-2   # Standard deviation of the K1 noise

# ACD parameters
RAMP_UP_TURNS   = 1_000        # Number of turns to ramp up the ACD
FLATTOP_TURNS   = 50_000       # Number of turns on the flat top
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
BPM_RANGE        = "BPM.13R3.B1/BPM.12L5.B1"           # BPM selection range for tracking
BEAM_ENERGY      = 6800                                # Beam energy in GeV
SEQ_NAME         = "lhcb1"                             # Sequence name in MAD-X (lowercase)
FILTER_DATA      = True                               # Whether to filter data with a Kalman filter
USE_NOISY_DATA   = True                               # Whether to use noisy data for optimisation

X_BPM_START = "BPM.13R3.B1"  # Starting BPM for tracking
X_BPM_END = "BPM.13L5.B1"    # Ending BPM for tracking

Y_BPM_START = "BPM.14R3.B1"  # Starting BPM for tracking
Y_BPM_END = "BPM.12L5.B1"    # Ending BPM for tracking

WINDOWS = [
    (X_BPM_START, X_BPM_END),
    (Y_BPM_START, Y_BPM_END),
]    

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
