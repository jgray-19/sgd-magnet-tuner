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
class OptimiserConfig:
    """Configuration for gradient descent optimiser algorithms.

    Controls learning rates, convergence criteria, and optimiser type
    for the gradient descent training process.
    """

    max_epochs: int
    warmup_epochs: int
    warmup_lr_start: float
    max_lr: float
    min_lr: float
    gradient_converged_value: float
    optimiser_type: str = field(default="adam")  # Options: "adam", "amsgrad", "lbfgs"

    # Gradient smoothing for loss tracking
    grad_norm_alpha: float = field(default=0.4)

    # Computed fields
    decay_epochs: int = field(init=False)

    def __post_init__(self):
        self.decay_epochs = self.max_epochs - self.warmup_epochs


@dataclass
class SimulationConfig:
    """Configuration for simulation and worker process settings.

    Controls parallel worker distribution, tracking data, and which
    physical parameters (energy, quadrupoles, bends) to optimise.
    """

    tracks_per_worker: int
    num_workers: int
    num_batches: int

    # Which parameters to optimise
    optimise_energy: bool = field(default=True)
    optimise_quadrupoles: bool = field(default=False)
    optimise_bends: bool = field(default=False)

    # Whether to include momenta (px, py) in loss function
    # When False, only positions (x, y) are used for optimisation
    optimise_momenta: bool = field(default=True)

    # Worker mode: arc-by-arc vs whole ring
    run_arc_by_arc: bool = field(default=True)

    # Whether to use different turns for each BPM range
    different_turns_per_range: bool = field(default=False)

    # Whether to use fixed BPMs for start/end points
    # When True (default): Uses fixed reference approach - pairs by varying starts
    #                      with a fixed end, and varying ends with a fixed start.
    #                      Example: [A,B,C] x [X,Y,Z] -> [(A,Z), (B,Z), (C,Z), (A,X), (A,Y)]
    # When False: Creates ALL combinations (Cartesian product) of start and end BPMs.
    #             Every start BPM is paired with every end BPM.
    #             Example: [A,B,C] x [X,Y,Z] -> [(A,X), (A,Y), (A,Z), (B,X), (B,Y), (B,Z), (C,X), (C,Y), (C,Z)]
    #             This provides many more measurement combinations to constrain the fit.
    use_fixed_bpm: bool = field(default=True)

    # Logging level for worker processes (separate from main process)
    worker_logging_level: int = field(default=logging.WARNING)

    # Computed fields
    total_tracks: int = field(init=False)

    def __post_init__(self):
        self.total_tracks = self.tracks_per_worker * self.num_workers


# In the future, mode needs to be removed, instead it needs to be a flexible code that can set which parameters will be optimised on the fly.
# This includes bends, quadrupoles and energy.

# Optimiser configuration for dp/p optimisation
DPP_OPTIMISER_CONFIG = OptimiserConfig(
    # max_epochs=400,
    max_epochs=150,
    warmup_epochs=1,
    # num_batches=10,
    # warmup_epochs=2,
    # adam
    warmup_lr_start=4e-7,
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
)

# Simulation configuration for dp/p optimisation
DPP_SIMULATION_CONFIG = SimulationConfig(
    # For pre trimmed data
    # tracks_per_worker=447,
    # num_workers=59,
    # For post trimmed data
    tracks_per_worker=219,
    num_workers=60,
    num_batches=20,
    optimise_energy=True,
    optimise_quadrupoles=False,
    optimise_bends=False,
)

# Optimiser configuration for quadrupole optimisation
QUAD_OPTIMISER_CONFIG = OptimiserConfig(
    max_epochs=5000,
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
    optimiser_type="adam",
    # optimiser_type="lbfgs",
)

# Simulation configuration for quadrupole optimisation
QUAD_SIMULATION_CONFIG = SimulationConfig(
    tracks_per_worker=133,
    num_workers=60,
    # num_workers=1,
    # num_batches=1,
    num_batches=10,
    optimise_energy=False,
    optimise_quadrupoles=True,
    optimise_bends=False,
)

# =============================================================================
# NOISE PARAMETERS
# =============================================================================

# Standard error of the noise
POSITION_STD_DEV = 1e-5  # Standard deviation of the position noise
MOMENTUM_STD_DEV = 3e-7  # Standard deviation of the momentum noise
REL_K1_STD_DEV = 1e-4  # Standard deviation of the K1 noise
MACHINE_DELTAP = -16e-5  # -11e-5  # The energy deviation of the machine from expected.
DELTAP = 1e-3

# DEPRECATED: Pass as arguments to Controller and MAD interfaces
MAGNET_RANGE = "BPM.11R2.B1/BPM.11L3.B1"
BEAM_ENERGY = 6800  # Beam energy in GeV - pass as argument to Controller


# Physical constants
PARTICLE_MASS = 938.27208816 * 1e-3  # [GeV] Proton energy-mass

# MAD-X sequence name
LHCB1_SEQ_NAME = "lhcb1"  # Sequence name in MAD-X (lowercase)

# Global schema constant for data files (this is appropriate as a global)
FILE_COLUMNS: tuple[str, ...] = (
    "name",
    "turn",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
    "kick_plane",
)

# =============================================================================
# FILE PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
logger.info(f"Current project root: {PROJECT_ROOT}")

# Data files
NO_NOISE_FILE = PROJECT_ROOT / "data/track_data.parquet"  # Measurement Parquet file
NOISY_FILE = PROJECT_ROOT / "data/noise_data.parquet"  # Noise Parquet file
CLEANED_FILE = PROJECT_ROOT / "data/filtered_data.parquet"  # Filtered TFS file

# Other files
# Ground-truth knob strengths
TRUE_STRENGTHS_FILE = PROJECT_ROOT / "data/true_strengths.txt"
# Where to write final strengths
OUTPUT_KNOBS = PROJECT_ROOT / "data/final_knobs.txt"
# Markdown summary of results
KNOB_TABLE = PROJECT_ROOT / "data/knob_strengths_table.txt"
# Matched tunes file
TUNE_KNOBS_FILE = PROJECT_ROOT / "data/matched_tunes.txt"
# Corrector strengths file
CORRECTOR_STRENGTHS = PROJECT_ROOT / "data/corrector_strengths.txt"
# Bend errors file
BEND_ERROR_FILE = PROJECT_ROOT / "data/bend_errors.tfs"


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
