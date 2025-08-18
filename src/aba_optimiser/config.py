# src/pymad_optimizer/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

import logging
from pathlib import Path

# Simulation parameters
MAX_EPOCHS = 3000  # Total number of epochs for optimization
TRACKS_PER_WORKER = 1  # Number of tracks per worker
NUM_WORKERS = 1  # Number of parallel worker processes
TOTAL_TRACKS = TRACKS_PER_WORKER * NUM_WORKERS  # Total number of tracks

# Learning-rate schedule
WARMUP_EPOCHS = 200  # Epochs for cosine warmup
DECAY_EPOCHS = MAX_EPOCHS - WARMUP_EPOCHS  # Epoch at which cosine decay ends
WARMUP_LR_START = 1e-7  # Initial learning rate at epoch 1
MAX_LR = 5e-8  # Peak learning rate after warmup
MIN_LR = 5e-8
OPTIMISER_TYPE = "adam"  # or "adam" or "amsgrad" or "stoch_lbfgs_tr"
GRAD_NORM_ALPHA = 0.7  # Gradient norm smoothing factor for smoothing loss

# Standard error of the noise
POSITION_STD_DEV = 1e-6  # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-6  # Standard deviation of the momentum noise
REL_K1_STD_DEV = 1e-4  # Standard deviation of the K1 noise

RUN_ARC_BY_ARC = False
BPM_START_POINTS = [
    "BPM.13R3.B1",
    "BPM.14R3.B1",
    "BPM.15R3.B1",
]

N_RUN_TURNS = 1  # Number of turns to run the simulation for each track
OBSERVE_TURNS_FROM = 1  # Record from N turns
N_COMPARE_TURNS = N_RUN_TURNS - OBSERVE_TURNS_FROM + 1  # Number of turns to compare

# Tracking parameters
RAMP_UP_TURNS = 1_000  # Number of turns to ramp up the ACD
FLATTOP_TURNS = 2_000  # Number of turns on the flat top
NUM_TRACKS = 1  # Number of tracks of FLATTOP_TURNS, so total number of turns is FLATTOP_TURNS * NUM_TRACKS (asssuming acd is off)
ACD_ON = False  # Whether the ACD was used or not (Ignores the ramp up turns)

module_path = Path(__file__).absolute().parent.parent.parent
logger = logging.getLogger(__name__)
logger.info(f"Current module path: {module_path}")
# File paths
SEQUENCE_FILE = module_path / "mad_scripts/lhcb1.seq"  # MAD-X sequence file
TRACK_DATA_FILE = module_path / "data/track_data.tfs"  # Measurement TFS file
NOISE_FILE = module_path / "data/noise_data.feather"  # Noise TFS file
FILTERED_FILE = module_path / "data/filtered_data.feather"  # Filtered TFS file
KALMAN_FILE = module_path / "data/kalman_data.feather"  # Kalman-filtered TFS file
TRUE_STRENGTHS = module_path / "data/true_strengths.txt"  # Ground-truth knob strengths
OUTPUT_KNOBS = module_path / "data/final_knobs.txt"  # Where to write final strengths
KNOB_TABLE = module_path / "data/knob_strengths_table.md"  # Markdown summary of results
TUNE_KNOBS_FILE = module_path / "data/matched_tunes.txt"  # File with tune knobs
MAD_SCRIPTS_DIR = (
    module_path / "src" / "aba_optimiser" / "mad_scripts"
)  # Directory for MAD-NG scripts

# Simulation specifics
MAGNET_RANGE = "$start/$end"  # Magnet selection range for tracking
# MAGNET_RANGE = "BPM.13R3.B1/BPM.12L4.B1"
BEAM_ENERGY = 6800  # Beam energy in GeV
SEQ_NAME = "lhcb1"  # Sequence name in MAD-X (lowercase)
FILTER_DATA = False  # Whether to filter data with a Kalman filter
USE_NOISY_DATA = False  # Whether to use noisy data for optimisation

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
