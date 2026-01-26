"""Helper functions and constants for squeeze step measurements and analysis.

This module centralizes common code used across multiple measurement and analysis scripts,
reducing duplication and ensuring consistency across the codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.model_creator.madx_utils import make_madx_sequence

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
DEFAULT_MEASUREMENT_DATE = "2025-04-27"
BETABEAT_DIR = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/")
BEAM_ENERGY = 6800.0  # GeV

MODEL_DIRS = {
    1: {
        "1.2m": "b1_120cm_injTunes",
        "1.2m_agc": "b1_120cm_injTunes",
        "1.05m": "b1_105cm_injTunes",
        "0.93m": "b2_93cm_injTunes",  # Double checked - this is correct (they accidentally wrote b2 in the folder name)
        "0.725m": "b1_72cm_injTunes",
        "0.6m": "b1_60cm_injTunes",
        "0.45m": "b1_44cm_flat_injTunes",
        "0.3m": "b1_30cm_flat_injTunes",
        "0.25m": "b1_24cm_flat_injTunes",
        "0.18m": "b1_18cm_flat_injTunes",
        "inj": "OMC3_LHCB1_2025_28m010_31p012",
    },
    2: {
        "1.2m": "b2_120cm_injTunes",
        "1.05m": "OMC3_LHCB2_105cm",
        "0.93m": "b2_93cm_injTunes",
        "0.725m": "b2_72cm_injTunes",
        "0.6m": "b2_60cm_injTunes",
        "0.45m": "b2_44cm_flat_injTunes",
        "0.3m": "b2_30cm_flat_injTunes",
        "0.25m": "b2_24cm_flat_injTunes",
        "0.18m": "b2_18cm_flat_injTunes",
        "inj": "OMC_LHCB2_2025_inj_28m010_31p012",
    },
}

ANALYSIS_DIRS = {
    1: {
        "1.2m": "2025-04-27_B1_120cm_injTunes_onOffMom",
        "1.2m_agc": "2025-04-27_B1_120cm_injTunes_onOffMom_afterGlobal",
        "inj": "2025-04-20_LHCB1_28m010_31p012_inj_onmom",
    },
    2: {
        "1.2m": "2025-04-27_B2_120cm_injTunes_onOffMom",
    },
}

MEASUREMENT_DATES = {
    "inj": "2025-04-20",
}


# ==================== HELPER FUNCTIONS ====================
def get_measurement_date(squeeze_step: str) -> str:
    """Get measurement date for a given squeeze step.

    Args:
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")

    Returns:
        Measurement date string (e.g., "2025-04-27")
    """
    return MEASUREMENT_DATES.get(squeeze_step, DEFAULT_MEASUREMENT_DATE)

def get_or_make_sequence(beam: int, model_dir: Path) -> Path:
    """Get cached sequence or generate new one.

    Args:
        beam: Beam number (1 or 2)
        model_dir: Path to model directory

    Returns:
        Path to sequence file
    """
    sequences_dir = PROJECT_ROOT / "sequences_from_models"
    sequences_dir.mkdir(exist_ok=True)
    seq_name = f"{model_dir.name}.seq"
    seq_path = sequences_dir / seq_name
    if seq_path.exists():
        logger.info(f"Using cached sequence: {seq_path}")
        return seq_path

    logger.info(f"Generating new sequence: {seq_path}")
    make_madx_sequence(beam, model_dir, seq_outdir=sequences_dir)
    generated = sequences_dir / f"lhcb{beam}_saved.seq"
    generated.rename(seq_path)
    return seq_path


def load_estimates(estimates_file: Path) -> dict[str, dict[str, float]] | dict[str, float]:
    """Load quadrupole estimates from file.

    Handles both formats:
    - Arc-based format (for plot_quad_diffs_and_phases): Arc X: <magnets>
    - Flat format (for check_arc1_optimisation): Arc 1: <magnets>

    Args:
        estimates_file: Path to file containing magnet estimates

    Returns:
        Dictionary mapping magnet names to k1 values or arcs to magnet dicts
    """
    estimates = {}
    current_arc = None

    with estimates_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("Arc"):
                current_arc = line.rstrip(":")  # remove trailing :
                if current_arc not in estimates:
                    estimates[current_arc] = {}
            elif line and current_arc:
                parts = line.split()
                if len(parts) == 2:
                    magnet, value = parts
                    estimates[current_arc][magnet] = float(value)

    if estimates:
        logger.info(f"Loaded {sum(len(v) for v in estimates.values())} magnet estimates from {estimates_file.name}")
    return estimates


def get_model_dir(beam: int, squeeze_step: str) -> Path:
    """Get model directory for a given beam and squeeze step.

    Args:
        beam: Beam number (1 or 2)
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")

    Returns:
        Path to model directory

    Raises:
        ValueError: If squeeze_step is not found for the beam
    """
    if squeeze_step not in MODEL_DIRS.get(beam, {}):
        raise ValueError(f"Model directory not defined for beam {beam}, squeeze_step {squeeze_step}")

    meas_date = get_measurement_date(squeeze_step)
    model_dir = BETABEAT_DIR / meas_date / f"LHCB{beam}/Models/" / MODEL_DIRS[beam][squeeze_step]
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    logger.info(f"Using model directory: {model_dir}")
    return model_dir


def get_analysis_dir(beam: int, squeeze_step: str) -> Path:
    """Get analysis directory for a given beam and squeeze step.

    Args:
        beam: Beam number (1 or 2)
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")

    Returns:
        Path to analysis directory

    Raises:
        ValueError: If squeeze_step is not found for the beam
    """
    if squeeze_step not in ANALYSIS_DIRS.get(beam, {}):
        raise ValueError(f"Analysis directory not defined for beam {beam}, squeeze_step {squeeze_step}")

    meas_date = get_measurement_date(squeeze_step)
    analysis_dir = BETABEAT_DIR / meas_date / f"LHCB{beam}/Results/" / ANALYSIS_DIRS[beam][squeeze_step]
    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory not found: {analysis_dir}")

    logger.info(f"Using analysis directory: {analysis_dir}")
    return analysis_dir


def get_results_dir(beam: int) -> Path:
    """Get results directory for a given beam.

    Args:
        beam: Beam number (1 or 2)

    Returns:
        Path to results directory
    """
    results_dir = PROJECT_ROOT / f"b{beam}_squeeze_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
