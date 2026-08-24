"""Integration tests for beta matching using MAD-NG differential algebra."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tfs

pytest.importorskip("tmom_recon")

from aba_optimiser.matching.matcher import BetaMatcher
from aba_optimiser.matching.matcher_config import MatcherConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

LOGGER = logging.getLogger(__name__)

def _rms_beta_beating(
    twiss: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[float, float]:
    """Return (rms_x, rms_y) beta beating fractions relative to model."""
    common = twiss.index.intersection(model.index)
    diff_x = (twiss.loc[common, "beta11"] - model.loc[common, "beta11"]) / model.loc[
        common, "beta11"
    ]
    diff_y = (twiss.loc[common, "beta22"] - model.loc[common, "beta22"]) / model.loc[
        common, "beta22"
    ]
    return float(np.sqrt(np.mean(diff_x**2))), float(np.sqrt(np.mean(diff_y**2)))


def _get_beta_correctors() -> list[str]:
    """Return the MQM_TOP beta corrector knobs from the omc3 package."""
    from pathlib import Path

    import omc3

    omc3_file = omc3.__file__
    assert omc3_file is not None, "omc3 package not properly installed"
    omc3_path = Path(omc3_file).parent
    knobs_file = (
        omc3_path
        / "model"
        / "accelerators"
        / "lhc"
        / "2025"
        / "correctors"
        / "correctors_b1"
        / "beta_correctors.json"
    )
    if not knobs_file.exists():
        pytest.skip(f"Beta correctors file not found: {knobs_file}")
    with knobs_file.open() as f:
        return json.load(f)["MQM_TOP"]


@pytest.mark.slow
# @pytest.mark.skip(reason="Possible issue with MAD-NG beta matching, needs investigation")
@pytest.mark.parametrize("match_method", ["lbfgs", "lm"])
def test_matcher_reduces_beta_beating(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface: AbaMadInterface,
    match_method: str,
) -> None:
    """After matching, RMS beta beating must be smaller than before matching."""
    # Nominal twiss (no errors) — used as the matching target
    loaded_interface.observe()
    model_twiss = loaded_interface.run_twiss()
    model_twiss_file = tmp_path / "model_twiss.tfs"
    tfs.write(model_twiss_file, model_twiss, save_index=True)

    # Apply quadrupole errors and measure beta beating before correction
    magnet_strengths, _ = loaded_interface.apply_magnet_perturbations(
        rel_error=1e-3, seed=42, magnet_type="q"
    )
    perturbed_twiss = loaded_interface.run_twiss()

    rms_x_before, rms_y_before = _rms_beta_beating(perturbed_twiss, model_twiss)
    assert rms_x_before > 1e-3, "Perturbation too small to be meaningful"
    assert rms_y_before > 1e-3, "Perturbation too small to be meaningful"

    # Get the actual perturbed strengths to pass to the matcher
    estimated_strengths = loaded_interface.get_magnet_strengths(list(magnet_strengths))
    knobs_list = _get_beta_correctors()
    matched_tunes = loaded_interface.match_tunes(target_qx=0.28, target_qy=0.31, deltap=0.0)

    config = MatcherConfig(
        model_twiss_file=model_twiss_file,
        estimated_strengths=estimated_strengths,
        knobs_list=knobs_list,
        tune_knobs=matched_tunes,
        sequence_file_path=seq_b1,
        magnet_range="$start/$end",
        kinetic_energy=6800.0,
        output_dir=tmp_path / "matcher_output",
    )
    matcher = BetaMatcher(config)

    final_knobs, _ = matcher.run_match(match_method)

    # Apply corrections back to the perturbed interface and measure improvement
    loaded_interface.set_madx_variables(**final_knobs)
    corrected_twiss = loaded_interface.run_twiss()

    rms_x_after, rms_y_after = _rms_beta_beating(corrected_twiss, model_twiss)
    LOGGER.info(
        "RMS beta beating before: %.4%% (X), %.4%% (Y); after: %.4%% (X), %.4%% (Y)",
        rms_x_before * 100,
        rms_y_before * 100,
        rms_x_after * 100,
        rms_y_after * 100,
    )

    assert rms_x_after < rms_x_before, (
        f"Beta-X beating did not improve: {rms_x_before:.4%} -> {rms_x_after:.4%}"
    )
    assert rms_y_after < rms_y_before, (
        f"Beta-Y beating did not improve: {rms_y_before:.4%} -> {rms_y_after:.4%}"
    )
