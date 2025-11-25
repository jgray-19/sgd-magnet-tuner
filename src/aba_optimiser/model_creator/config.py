"""Configuration constants for LHC model creation."""

from typing import Final

# Model configuration
MODIFIER: Final[str] = "R2025aRP_A18cmC18cmA10mL200cm_Flat.madx"
NAT_TUNES: Final[list[float]] = [0.28, 0.31]
DRV_TUNES: Final[list[float]] = [0.27, 0.322]
ENERGY: Final[int] = 6800
YEAR: Final[str] = "2025"
MADX_FILENAME: Final[str] = "job.create_model_nominal.madx"

# AC Dipole marker pattern - beam number will be formatted in
AC_MARKER_PATTERN: Final[str] = "MKQA.6L4.B{beam}"
AC_MARKER_OFFSET: Final[float] = 1.583 / 2

# Beam 4 configuration
LHC_KBUNCH: Final[int] = 1
LHC_NPART: Final[float] = 1.15e11

# Model data columns and headers
MODEL_STRENGTHS: Final[list[str]] = [
    "k1l",
    "k2l",
    "k3l",
    "k4l",
    "k5l",
    "k1sl",
    "k2sl",
    "k3sl",
    "k4sl",
    "k5sl",
]

MODEL_HEADER: Final[list[str]] = [
    "name",
    "type",
    "title",
    "origin",
    "date",
    "time",
    "refcol",
    "direction",
    "observe",
    "energy",
    "deltap",
    "length",
    "alfap",
    "q1",
    "q2",
    "q3",
    "dq1",
    "dq2",
    "dq3",
]

MODEL_COLUMNS: Final[list[str]] = [
    "name",
    "kind",
    "s",
    "betx",
    "alfx",
    "bety",
    "alfy",
    "mu1",
    "mu2",
    "x",
    "y",
    "dx",
    "dpx",
    "dy",
    "dpy",
    "r11",
    "r12",
    "r21",
    "r22",
]

# Tune matching tolerances
TUNE_MATCH_TOLERANCE: Final[float] = 1e-6
TUNE_MATCH_RTOL: Final[float] = 1e-6
TUNE_MATCH_FMIN: Final[float] = 1e-7

# Tune offsets for matching (absolute tunes are offset + fractional)
TUNE_Q1_OFFSET: Final[int] = 62
TUNE_Q2_OFFSET: Final[int] = 60
