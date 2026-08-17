"""Environment configuration for LHC squeeze measurement workflows.

Path roots, per-beam model/analysis directory names, measurement dates and the
beam momentum, plus the directory-lookup helpers that resolve them. Kept as a
low-dependency leaf so ``squeeze.constants`` and ``squeeze.io`` can import it
without creating an import cycle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aba_optimiser.config import MEASUREMENTS_ARTIFACTS_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MEASUREMENT_DATE = "2025-04-27"
BETABEAT_DIR = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/")
PC = 6800.0  # GeV

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
        "inj_rdt": "OMC3_LHCB1_2025_inj_28m008_313p010",
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
        "inj_rdt": "2025_LHCB2_inj_028m008_0313p010",
    },
}

ANALYSIS_DIRS = {
    1: {
        "1.2m": "2025-04-27_B1_120cm_injTunes_onOffMom",
        "1.2m_agc": "2025-04-27_B1_120cm_injTunes_onOffMom_afterGlobal",
        "inj": "2025-04-20_LHCB1_28m010_31p012_inj_onmom",
        "inj_rdt": "LHCB1_inj_28m008_313p010_a3b3_RDTs_20-04-25",
    },
    2: {
        "1.2m": "2025-04-27_B2_120cm_injTunes_onOffMom",
        "inj_rdt": "17-29-02_ANALYSIS_highkicks_Injection",
    },
}

MEASUREMENT_DATES = {
    "inj": "2025-04-20",
    "inj_rdt": "2025-04-20",
}


def get_measurement_date(squeeze_step: str) -> str:
    """Return the measurement date for a squeeze step (e.g. "1.2m" -> "2025-04-27")."""
    return MEASUREMENT_DATES.get(squeeze_step, DEFAULT_MEASUREMENT_DATE)


def get_model_dir(beam: int, squeeze_step: str) -> Path:
    """Get model directory for a given beam and squeeze step.

    Raises:
        ValueError: If squeeze_step is not found for the beam, or the directory is absent.
    """
    if squeeze_step not in MODEL_DIRS.get(beam, {}):
        raise ValueError(
            f"Model directory not defined for beam {beam}, squeeze_step {squeeze_step}"
        )

    meas_date = get_measurement_date(squeeze_step)
    model_dir = BETABEAT_DIR / meas_date / f"LHCB{beam}/Models/" / MODEL_DIRS[beam][squeeze_step]
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    logger.info(f"Using model directory: {model_dir}")
    return model_dir


def get_analysis_dir(beam: int, squeeze_step: str) -> Path:
    """Get analysis directory for a given beam and squeeze step.

    Raises:
        ValueError: If squeeze_step is not found for the beam, or the directory is absent.
    """
    if squeeze_step not in ANALYSIS_DIRS.get(beam, {}):
        raise ValueError(
            f"Analysis directory not defined for beam {beam}, squeeze_step {squeeze_step}"
        )

    meas_date = get_measurement_date(squeeze_step)
    analysis_dir = (
        BETABEAT_DIR / meas_date / f"LHCB{beam}/Results/" / ANALYSIS_DIRS[beam][squeeze_step]
    )
    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory not found: {analysis_dir}")

    logger.info(f"Using analysis directory: {analysis_dir}")
    return analysis_dir


def get_results_dir(beam: int) -> Path:
    """Get (and create) the results directory for a given beam."""
    results_dir = MEASUREMENTS_ARTIFACTS_ROOT / "results" / f"b{beam}_squeeze_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir
