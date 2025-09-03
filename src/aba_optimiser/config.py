# src/aba_optimiser/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# OPTIMISATION SETTINGS
# =============================================================================


@dataclass
class OptSettings:
    """Settings for optimisation simulations."""

    max_epochs: int
    tracks_per_worker: int
    num_workers: int
    num_batches: int
    warmup_epochs: int
    warmup_lr_start: float
    max_lr: float
    min_lr: float
    gradient_converged_value: float
    total_tracks: int = field(init=False)
    decay_epochs: int = field(init=False)

    def __post_init__(self):
        self.total_tracks = self.tracks_per_worker * self.num_workers
        self.decay_epochs = self.max_epochs - self.warmup_epochs


# Simulation parameters for dp/p optimisation
DPP_OPT_SETTINGS = OptSettings(
    max_epochs=500,
    tracks_per_worker=2000,
    num_workers=50,
    num_batches=50,
    warmup_epochs=10,
    warmup_lr_start=2e-9,
    max_lr=4e-7,
    min_lr=2e-7,
    gradient_converged_value=1e-8,
)

# Simulation parameters for quadrupole optimisation
QUAD_OPT_SETTINGS = OptSettings(
    max_epochs=10_000,
    tracks_per_worker=2000,
    num_workers=50,
    # num_workers=1,
    num_batches=50,
    warmup_epochs=10,
    warmup_lr_start=2e-9,
    max_lr=4e-7,
    min_lr=2e-7,
    gradient_converged_value=1e-8,
)

# Simulation parameters for sextupole optimisation
SEXT_OPT_SETTINGS = OptSettings(
    max_epochs=1000,
    tracks_per_worker=2000,
    num_workers=50,
    num_batches=20,
    warmup_epochs=10,
    warmup_lr_start=2e-7,
    max_lr=4e-7,
    min_lr=2e-7,
    gradient_converged_value=1e-9,
)

# Optimizer configuration
OPTIMISER_TYPE = "adam"  # Options: "adam", "amsgrad", "lbfgs"
GRAD_NORM_ALPHA = 0.7  # Gradient norm smoothing factor for smoothing loss

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

# Standard error of the noise
POSITION_STD_DEV = 1e-4  # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-6  # Standard deviation of the momentum noise
REL_K1_STD_DEV = 1e-3  # Standard deviation of the K1 noise
MACHINE_DELTAP = 2.2e-4  # The energy deviation of the machine from expected.

# =============================================================================
# BPM AND TRACKING SETTINGS
# =============================================================================

RUN_ARC_BY_ARC = True
BPM_START_POINTS = [
    # "BPM.9R4.B1",
    "BPM.10R4.B1",
    "BPM.11R4.B1",
    # "BPM.12R4.B1",
    # "BPM.13R4.B1",
    # "BPM.14R4.B1",
    # "BPM.15R4.B1",
    # "BPM.16R4.B1",
]

N_RUN_TURNS = 1  # Number of turns to run the simulation for each track
OBSERVE_TURNS_FROM = 1  # Record from N turns
N_COMPARE_TURNS = N_RUN_TURNS - OBSERVE_TURNS_FROM + 1  # Number of turns to compare

# Tracking parameters
RAMP_UP_TURNS = 1_000  # Number of turns to ramp up the ACD
FLATTOP_TURNS = 4_000  # Number of turns on the flat top
NUM_TRACKS = 30  # Number of tracks of FLATTOP_TURNS, so total number of turns is FLATTOP_TURNS * NUM_TRACKS (assuming acd is off)
ACD_ON = False  # Whether the ACD was used or not (Ignores the ramp up turns)
DELTAP = 3e-3

# =============================================================================
# SIMULATION SPECIFICS
# =============================================================================

MAGNET_RANGE = "BPM.11R4.B1/BPM.11L5.B1"
BEAM_ENERGY = 450  # Beam energy in GeV
PARTICLE_MASS = 938.27208816 * 1e-3  # [GeV] Proton energy-mass
SEQ_NAME = "lhcb1"  # Sequence name in MAD-X (lowercase)
FILTER_DATA = False  # Whether to filter data with a Kalman filter
USE_NOISY_DATA = False  # Whether to use noisy data for optimisation

# =============================================================================
# FILE PATHS
# =============================================================================

module_path = Path(__file__).absolute().parent.parent.parent
logger = logging.getLogger(__name__)
logger.info(f"Current module path: {module_path}")

# Data files
NO_NOISE_FILE = module_path / "data/track_data.parquet"  # Measurement Parquet file
NOISY_FILE = module_path / "data/noise_data.parquet"  # Noise Parquet file

EPLUS_NOISY_FILE = module_path / "data/eplus_data.parquet"  # E+ data file
EPLUS_NONOISE_FILE = module_path / "data/eplus_nonoise_data.parquet"

EMINUS_NOISY_FILE = module_path / "data/eminus_data.parquet"  # E- Noisy file
EMINUS_NONOISE_FILE = module_path / "data/eminus_nonoise_data.parquet"
FILTERED_FILE = module_path / "data/filtered_data.feather"  # Filtered TFS file
KALMAN_FILE = module_path / "data/kalman_data.feather"  # Kalman-filtered TFS file

# Other files
SEQUENCE_FILE = module_path / "mad_scripts/lhcb1.seq"  # MAD-X sequence file
TRUE_STRENGTHS = module_path / "data/true_strengths.txt"  # Ground-truth knob strengths
OUTPUT_KNOBS = module_path / "data/final_knobs.txt"  # Where to write final strengths
KNOB_TABLE = module_path / "data/knob_strengths_table.md"  # Markdown summary of results
TUNE_KNOBS_FILE = module_path / "data/matched_tunes.txt"
CORRECTOR_STRENGTHS = module_path / "data/corrector_strengths.tfs"
MAD_SCRIPTS_DIR = (
    module_path / "src" / "aba_optimiser" / "mad" / "mad_scripts"
)  # Directory for MAD-NG scripts

# =============================================================================
# TODO AND NOTES
# =============================================================================

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

- think about how to pinpoint errors in the lattice, i.e. identify the largest
    errors in the lattice
"""
