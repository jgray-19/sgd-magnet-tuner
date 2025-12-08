# src/aba_optimiser/mad/scripts.py
"""
MAD-NG script path constants.
"""

from pathlib import Path

# Get the MAD directory path
mad_dir = Path(__file__).absolute().parent

# MAD-NG scripts directory
MAD_SCRIPTS_DIR = mad_dir / "mad_scripts"

# Script file paths - Full tracking (position + momentum)
TRACK_INIT = MAD_SCRIPTS_DIR / "run_track_init.mad"
TRACK_SCRIPT = MAD_SCRIPTS_DIR / "run_track.mad"

# Script file paths - Position-only tracking (no momentum)
TRACK_INIT_POS_ONLY = MAD_SCRIPTS_DIR / "run_track_init_pos_only.mad"
TRACK_SCRIPT_POS_ONLY = MAD_SCRIPTS_DIR / "run_track_pos_only.mad"

# Hessian estimation scripts
HESSIAN_SCRIPT = MAD_SCRIPTS_DIR / "estimate_hessian.mad"
HESSIAN_SCRIPT_POS_ONLY = MAD_SCRIPTS_DIR / "estimate_hessian_pos_only.mad"

# Optics tracking scripts
TRACK_OPTICS_INIT = MAD_SCRIPTS_DIR / "run_optics_track_init.mad"
TRACK_OPTICS_SCRIPT = MAD_SCRIPTS_DIR / "run_optics_track.mad"
