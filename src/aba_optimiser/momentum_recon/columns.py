"""
Column name constants for momentum reconstruction.
"""

from __future__ import annotations

from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    DISPERSION,
    ERR,
    MOMENTUM_DISPERSION,
    ORBIT,
    PHASE_ADV,
)

# Measurement column mappings
MEASUREMENT_RENAME_MAPPING: dict[str, str] = {
    f"{BETA}X": "beta11",
    f"{BETA}Y": "beta22",
    f"{ALPHA}X": "alfa11",
    f"{ALPHA}Y": "alfa22",
    f"{PHASE_ADV}X": "mu1",
    f"{PHASE_ADV}Y": "mu2",
    f"{ORBIT}X": "x",
    f"{ORBIT}Y": "y",
    # Always include orbit errors!
    f"{ERR}{ORBIT}X": "x_err",
    f"{ERR}{ORBIT}Y": "y_err",
}

ERROR_RENAME_MAPPING: dict[str, str] = {
    f"{ERR}{BETA}X": "sqrt_betax_err",
    f"{ERR}{BETA}Y": "sqrt_betay_err",
    f"{ERR}{ALPHA}X": "alfax_err",
    f"{ERR}{ALPHA}Y": "alfay_err",
    f"{ERR}{PHASE_ADV}X": "mu1_err",
    f"{ERR}{PHASE_ADV}Y": "mu2_err",
}

DISPERSION_RENAME_MAPPING: dict[str, str] = {
    f"{DISPERSION}X": "dx",
    f"{DISPERSION}Y": "dy",
    f"{MOMENTUM_DISPERSION}X": "dpx",
    f"{MOMENTUM_DISPERSION}Y": "dpy",
}

ERROR_DISPERSION_RENAME_MAPPING: dict[str, str] = {
    f"{ERR}{DISPERSION}X": "dx_err",
    f"{ERR}{DISPERSION}Y": "dy_err",
    f"{ERR}{MOMENTUM_DISPERSION}X": "dpx_err",
    f"{ERR}{MOMENTUM_DISPERSION}Y": "dpy_err",
}

# Current BPM error columns to attach
CURRENT_BPM_ERRORS = [
    "sqrt_betax_err",
    "sqrt_betay_err",
    "alfax_err",
    "alfay_err",
    "dx_err",
    "dy_err",
    "dpx_err",
    "dpy_err",
]

# Neighbor BPM error mapping: (target_col, neighbor_suffix, source_col)
NEIGHBOR_BPM_ERROR_SPEC = [
    ("sqrt_betax_{}_err", "bpm_x_{}", "sqrt_betax_err"),
    ("sqrt_betay_{}_err", "bpm_y_{}", "sqrt_betay_err"),
    ("dx_{}_err", "bpm_x_{}", "dx_err"),
    ("dy_{}_err", "bpm_y_{}", "dy_err"),
]
