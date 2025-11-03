# src/aba_optimiser/mad/scripts.py
"""
MAD-NG script path constants.
"""

from pathlib import Path

# Get the MAD directory path
mad_dir = Path(__file__).absolute().parent

# MAD-NG scripts directory
MAD_SCRIPTS_DIR = mad_dir / "mad_scripts"

# Script file paths
TRACK_NO_KNOBS_INIT = MAD_SCRIPTS_DIR / "run_track_init_no_knobs.mad"
TRACK_INIT = MAD_SCRIPTS_DIR / "run_track_init.mad"
TRACK_SCRIPT = MAD_SCRIPTS_DIR / "run_track.mad"
HESSIAN_SCRIPT = MAD_SCRIPTS_DIR / "estimate_hessian.mad"
