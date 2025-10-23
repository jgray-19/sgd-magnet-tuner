# src/aba_optimiser/config.py
"""
Configuration constants for the knob optimisation pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# OPTIMISATION SETTINGS
# =============================================================================

logger = logging.getLogger(__name__)


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
    optimiser_type: str = field(default="adam")  # Options: "adam", "amsgrad", "lbfgs"
    total_tracks: int = field(init=False)
    decay_epochs: int = field(init=False)

    only_energy: bool = False
    use_off_energy_data: bool = False
    optimise_quadrupoles: bool = field(default=False)
    optimise_sextupoles: bool = field(default=False)

    def __post_init__(self):
        self.total_tracks = self.tracks_per_worker * self.num_workers
        self.decay_epochs = self.max_epochs - self.warmup_epochs
        if self.optimise_sextupoles and not self.optimise_quadrupoles:
            logger.warning("Sextupoles cannot be optimised without quadrupoles.")
        # if self.optimiser_type == "lbfgs" and self.num_batches != 1:
        #     logger.warning("LBFGS optimiser requires num_batches=1; overriding.")
        #     self.num_batches = 1

        if self.optimise_sextupoles is True and self.use_off_energy_data is False:
            logger.warning(
                "Sextupole optimisation needs the use_off_energy_data flag set to True; overriding."
            )
            self.use_off_energy_data = True


# In the future, mode needs to be removed, instead it needs to be a flexible code that can set which parameters will be optimised on the fly.
# This includes quadrupoles, sextupoles and energy.

# Simulation parameters for dp/p optimisation
DPP_OPT_SETTINGS = OptSettings(
    # max_epochs=400,
    max_epochs=75,
    # For pre trimmed data
    # tracks_per_worker=447,
    # num_workers=59,
    # For post trimmed data
    tracks_per_worker=329,
    num_workers=60,
    num_batches=20,
    warmup_epochs=3,
    # num_batches=10,
    # warmup_epochs=2,
    # adam
    warmup_lr_start=1e-7,
    max_lr=3e-6,
    min_lr=3e-6,
    # lbfgs
    # warmup_lr_start=5e-7,
    # max_lr=1e0,
    # min_lr=1e0,
    # gradient_converged_value=1e-8,
    gradient_converged_value=3e-7,
    optimiser_type="adam",
    # optimiser_type="lbfgs",
    only_energy=True,
    use_off_energy_data=False,
)

# Simulation parameters for quadrupole optimisation
QUAD_OPT_SETTINGS = OptSettings(
    max_epochs=5000,
    tracks_per_worker=133,
    num_workers=60,
    # num_workers=1,
    # num_batches=1,
    num_batches=10,
    # Adam settings
    warmup_epochs=100,
    warmup_lr_start=2e-8,
    max_lr=1e-6,
    min_lr=5e-7,
    # LBFGS settings
    # warmup_epochs=20,
    # warmup_lr_start=1e-3,
    # max_lr=5e-2,
    # min_lr=5e-2,
    # Rest
    gradient_converged_value=1e-9,
    optimise_quadrupoles=True,
    optimiser_type="adam",
    # optimiser_type="lbfgs",
    use_off_energy_data=False,
)

# Simulation parameters for sextupole optimisation
SEXT_OPT_SETTINGS = OptSettings(
    max_epochs=1000,
    tracks_per_worker=2400,
    num_workers=50,
    num_batches=20,
    warmup_epochs=10,
    warmup_lr_start=2e-7,
    max_lr=4e-7,
    min_lr=2e-7,
    gradient_converged_value=1e-9,
    optimise_quadrupoles=True,
    optimise_sextupoles=True,
    use_off_energy_data=True,
)

# Optimiser configuration
# OPTIMISER_TYPE = "adam"  # Options: "adam", "amsgrad", "lbfgs"
# OPTIMISER_TYPE = "lbfgs"  # Options: "adam", "amsgrad", "lbfgs"
GRAD_NORM_ALPHA = 0.4  # Gradient norm smoothing factor for smoothing loss

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

# Standard error of the noise
POSITION_STD_DEV = 1e-4  # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-6  # Standard deviation of the momentum noise
REL_K1_STD_DEV = 1e-4  # Standard deviation of the K1 noise
MACHINE_DELTAP = -16e-5  # -11e-5  # The energy deviation of the machine from expected.
DELTAP = 1e-3

# =============================================================================
# BPM AND TRACKING SETTINGS
# =============================================================================

RUN_ARC_BY_ARC = True
BPM_START_POINTS = [
    "BPM.9R2.B1",
    "BPM.10R2.B1",
    "BPM.11R2.B1",
    "BPM.12R2.B1",
    "BPM.13R2.B1",
]

BPM_END_POINTS = [
    "BPM.9L3.B1",
    "BPM.10L3.B1",
    "BPM.11L3.B1",
    "BPM.12L3.B1",
    "BPM.13L3.B1",
]

# Whether to use different turns for each start BPM
DIFFERENT_TURNS_PER_RANGE = False

N_RUN_TURNS = 3  # Number of turns to run the simulation for each track
OBSERVE_TURNS_FROM = 1  # Record from N turns
N_COMPARE_TURNS = N_RUN_TURNS - OBSERVE_TURNS_FROM + 1  # Number of turns to compare

# Tracking parameters
RAMP_UP_TURNS = 1_000  # Number of turns to ramp up the ACD
FLATTOP_TURNS = 6_600  # Number of turns on the flat top
NUM_TRACKS = 3  # Number of tracks of FLATTOP_TURNS, so total number of turns is FLATTOP_TURNS * NUM_TRACKS (assuming acd is off)
TRACK_BATCH_SIZE = NUM_TRACKS  # Number of tracks to process in each batch to save RAM
ACD_ON = False  # Whether the ACD was used or not (Ignores the ramp up turns)
KICK_BOTH_PLANES = True  # Whether to kick in both planes or separately

# =============================================================================
# SIMULATION SPECIFICS
# =============================================================================

MAGNET_RANGE = "BPM.11R2.B1/BPM.11L3.B1"
BEAM_ENERGY = 6800  # Beam energy in GeV
PARTICLE_MASS = 938.27208816 * 1e-3  # [GeV] Proton energy-mass
SEQ_NAME = "lhcb1"  # Sequence name in MAD-X (lowercase)
CLEAN_DATA = True  # Whether to take the filtered data for optimisation (just weights as of 16/09)
USE_NOISY_DATA = True  # Whether to use noisy data for optimisation

# =============================================================================
# TRACKING BACKEND SELECTION
# =============================================================================

USE_XSUITE = False  # Whether to use xsuite for tracking instead of MAD-NG
BEAM_NUMBER = 1  # LHC beam number (1 or 2) for xsuite tracking
FILE_COLUMNS: tuple[str, ...] = (
    "name",
    "turn",
    "x",
    "px",
    "y",
    "py",
    "x_weight",
    "y_weight",
    "kick_plane",
)

# =============================================================================
# FILE PATHS
# =============================================================================

module_path = Path(__file__).absolute().parent.parent.parent
logger.info(f"Current module path: {module_path}")

# Data files
NO_NOISE_FILE = module_path / "data/track_data.parquet"  # Measurement Parquet file
NOISY_FILE = module_path / "data/noise_data.parquet"  # Noise Parquet file
CLEANED_FILE = module_path / "data/filtered_data.parquet"  # Filtered TFS file

EPLUS_NOISY_FILE = module_path / "data/eplus_data.parquet"  # E+ data file
EPLUS_NONOISE_FILE = module_path / "data/eplus_nonoise_data.parquet"
EPLUS_CLEANED_FILE = module_path / "data/eplus_filtered_data.parquet"

EMINUS_NOISY_FILE = module_path / "data/eminus_data.parquet"  # E- Noisy file
EMINUS_NONOISE_FILE = module_path / "data/eminus_nonoise_data.parquet"
EMINUS_CLEANED_FILE = module_path / "data/eminus_filtered_data.parquet"

KALMAN_FILE = module_path / "data/kalman_data.feather"  # Kalman-filtered TFS file

# MAD-NG scripts
MAD_SCRIPTS_DIR = (
    module_path / "src" / "aba_optimiser" / "mad" / "mad_scripts"
)  # Directory for MAD-NG scripts
TRACK_NO_KNOBS_INIT = MAD_SCRIPTS_DIR / "run_track_init_no_knobs.mad"
TRACK_INIT = MAD_SCRIPTS_DIR / "run_track_init.mad"
TRACK_SCRIPT = MAD_SCRIPTS_DIR / "run_track.mad"
HESSIAN_SCRIPT = MAD_SCRIPTS_DIR / "estimate_hessian.mad"


# Other files
# The MAD-X sequence file
SEQUENCE_FILE = module_path / "mad_scripts/lhcb1.seq"
# The xsuite JSON file for the LHC
XSUITE_JSON = module_path / "src" / "aba_optimiser" / "xsuite" / "lhcb1.json"
# Ground-truth knob strengths
TRUE_STRENGTHS_FILE = module_path / "data/true_strengths.txt"
# Where to write final strengths
OUTPUT_KNOBS = module_path / "data/final_knobs.txt"
# Markdown summary of results
KNOB_TABLE = module_path / "data/knob_strengths_table.txt"
# Matched tunes file
TUNE_KNOBS_FILE = module_path / "data/matched_tunes.txt"
# Corrector strengths file
CORRECTOR_STRENGTHS = module_path / "data/corrector_strengths.txt"
# Bend errors file
BEND_ERROR_FILE = module_path / "data/bend_errors.tfs"


# =============================================================================
# TODO AND NOTES
# =============================================================================

"""
This has the current problem of no matter the number of files included in the simulation,
I cannot reduce the error on the final result significantly. I am still limited to 5e-4.
Potentially, I can get better results by doing the following method:
1. Optimise the Main Quadrupoles
3. Optimise the skew quadrupoles ???

S location as a degree of freedom?
Find out where individual errors might be reduced most.

TODO:
- Look at adding two more simulations with off momentum errors.
- Look at understanding the uncertainty - mathematically, where it arises, and
    which parameters reduce it and how.
"""
