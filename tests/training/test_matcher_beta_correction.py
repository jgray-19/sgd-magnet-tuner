"""
Integration test for beta matching using estimated quadrupole strengths from controller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.matching.matcher import BetaMatcher
from aba_optimiser.matching.matcher_config import MatcherConfig
from tests.training.helpers import (
    generate_model_with_errors,
    get_twiss_without_errors,
)

if TYPE_CHECKING:
    import pandas as pd



def _plot_beta_beating_comparison(
    twiss_errs: pd.DataFrame,
    tws_no_err: pd.DataFrame,
    tws_corrected: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot beta beating before and after correction for debugging."""
    import matplotlib.pyplot as plt

    # Calculate beta beating before correction
    tws_errs_betax = (twiss_errs["beta11"] - tws_no_err["beta11"]) / tws_no_err["beta11"] * 100
    tws_errs_betay = (twiss_errs["beta22"] - tws_no_err["beta22"]) / tws_no_err["beta22"] * 100

    # Calculate beta beating after correction
    tws_corrected_betax = (
        (tws_corrected["beta11"] - tws_no_err["beta11"]) / tws_no_err["beta11"] * 100
    )
    tws_corrected_betay = (
        (tws_corrected["beta22"] - tws_no_err["beta22"]) / tws_no_err["beta22"] * 100
    )

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Beta X
    ax1.plot(twiss_errs["s"], tws_errs_betax, "r-", label="Before correction", linewidth=2)
    ax1.plot(tws_corrected["s"], tws_corrected_betax, "b-", label="After correction", linewidth=2)
    ax1.set_ylabel("Beta X beating (%)")
    ax1.set_title("Beta Beating Correction Results")
    ax1.grid(visible=True, alpha=0.3)
    ax1.legend()

    # Beta Y
    ax2.plot(twiss_errs["s"], tws_errs_betay, "r-", label="Before correction", linewidth=2)
    ax2.plot(tws_corrected["s"], tws_corrected_betay, "b-", label="After correction", linewidth=2)
    ax2.set_xlabel("s (m)")
    ax2.set_ylabel("Beta Y beating (%)")
    ax2.grid(visible=True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "beta_beating_correction.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"Beta beating plot saved to: {plot_file}")


@pytest.mark.slow
def test_matcher_beta_correction(
    tmp_path: Path,
    seq_b1: Path,
    estimated_strengths_file: Path,
    loaded_interface_with_beam: BaseMadInterface,
) -> None:
    """Test beta matching using estimated quadrupole strengths from controller."""
    # Generate model with errors for validation (same setup as controller)
    corrector_file = tmp_path / "corrector_track_off_magnet.tfs"
    magnet_strengths, matched_tunes, _ = generate_model_with_errors(
        loaded_interface_with_beam,
        sequence_file=seq_b1,
        dpp_value=0,
        magnet_range="$start/$end",
        corrector_file=corrector_file,
        beam=1,
        perturb_quads=True,
        perturb_bends=True,
    )
    # Select only BPMs from the twiss with errors
    loaded_interface_with_beam.observe_elements()
    twiss_errs = loaded_interface_with_beam.run_twiss(observe=1)  # Observe all elements

    # Read estimated strengths from file (written by test_quad_conv_with_errs)
    if not estimated_strengths_file.exists():
        pytest.skip(
            f"Estimated strengths file not found: {estimated_strengths_file}. Run test_quad_conv_with_errs first."
        )

    with estimated_strengths_file.open("r") as f:
        all_estimates: dict[str, float] = json.load(f)

    print(
        f"Loaded {len(all_estimates)} estimated quadrupole strengths from {estimated_strengths_file}"
    )

    # Get clean twiss as model twiss
    tws_no_err = get_twiss_without_errors(seq_b1, just_bpms=True)
    model_twiss_file = tmp_path / "model_twiss.tfs"
    tfs.write(model_twiss_file, tws_no_err, save_index=True)

    # Get beta correctors from omc3 package
    import omc3

    if omc3.__file__ is None:
        raise ValueError("omc3.__file__ is None")
    omc3_path = Path(omc3.__file__).parent
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

    # Read the knobs list from JSON
    with knobs_file.open("r") as f:
        knobs_data = json.load(f)
    knobs_list = knobs_data["MQM_TOP"]
    print(f"Using {len(knobs_list)} beta corrector knobs from {knobs_file}")

    matcher_config = MatcherConfig(
        model_twiss_file=model_twiss_file,
        estimated_strengths=all_estimates,
        knobs_list=knobs_list,
        tune_knobs=matched_tunes,
        sequence_file_path=seq_b1,
        seq_name="lhcb1",
        magnet_range="$start/$end",
        beam_energy=6800,
        output_dir=tmp_path / "matcher_output",
    )

    matcher = BetaMatcher(matcher_config, show_plots=False)
    # final_knobs, uncertainties = matcher.run_lbfgs_match()
    final_knobs, uncertainties = matcher.run_linear_match(n_steps=1, svd_cutoff=1e-6)

    # Check the tune knobs exist in the final knobs
    for tune_knob in matched_tunes:
        assert tune_knob in final_knobs, f"Tune knob {tune_knob} not found in final knobs"

    # Compute twiss with estimated strengths + final knobs using BaseMadInterface
    # This includes both beta and tune knobs
    new_interface = BaseMadInterface()
    new_interface.load_sequence(seq_b1, "lhcb1")
    new_interface.setup_beam(beam_energy=6800)
    new_interface.set_magnet_strengths(all_estimates)  # Apply estimated strengths
    new_interface.set_madx_variables(**final_knobs)  # Apply correction knobs
    new_interface.observe_elements()
    tws_corrected = new_interface.run_twiss(observe=1)  # Observe
    # If you want to test if the estimated strengths actually allow beta beating correction, use this.
    # loaded_interface_with_beam.set_madx_variables(**final_knobs)
    # tws_corrected = loaded_interface_with_beam.run_twiss(observe=1)  # Observe all elements

    # Plot beta beating comparison (uncomment to enable plotting)
    _plot_beta_beating_comparison(twiss_errs, tws_no_err, tws_corrected, tmp_path)

    # Check beta beating after correction
    # Compare to model twiss (tws_no_err)
    tws_corrected_betax = (tws_corrected["beta11"] - tws_no_err["beta11"]) / tws_no_err["beta11"]  # ty:ignore[unsupported-operator]
    tws_corrected_betay = (tws_corrected["beta22"] - tws_no_err["beta22"]) / tws_no_err["beta22"]  # ty:ignore[unsupported-operator]

    # Compute RMS beta beat
    rms_betax = (tws_corrected_betax.pow(2).mean()) ** 0.5
    rms_betay = (tws_corrected_betay.pow(2).mean()) ** 0.5

    print(f"RMS BetaX error after correction: {rms_betax * 100:.2f}%")
    print(f"RMS BetaY error after correction: {rms_betay * 100:.2f}%")

    # Assert that beta errors are reduced
    assert rms_betax < 1.1e-3, "RMS BetaX error exceeds 0.11% after beta matching"
    assert rms_betay < 1.1e-3, "RMS BetaY error exceeds 0.11% after beta matching"

    # Check tune correction
    target_q1 = tws_no_err.headers["q1"]
    target_q2 = tws_no_err.headers["q2"]
    corrected_q1 = tws_corrected.headers["q1"]
    corrected_q2 = tws_corrected.headers["q2"]
    print(f"Target Q1: {target_q1}, Corrected Q1: {corrected_q1}")
    print(f"Target Q2: {target_q2}, Corrected Q2: {corrected_q2}")
    assert abs(corrected_q1 - target_q1) < 1.1e-3, (
        f"Q1 error {abs(corrected_q1 - target_q1)} exceeds 1e-3"
    )
    assert abs(corrected_q2 - target_q2) < 1.1e-3, (
        f"Q2 error {abs(corrected_q2 - target_q2)} exceeds 1e-3"
    )

    # Check that the original beta beating was larger
    tws_errs_betax = (twiss_errs["beta11"] - tws_no_err["beta11"]) / tws_no_err["beta11"]  # ty:ignore[unsupported-operator]
    tws_errs_betay = (twiss_errs["beta22"] - tws_no_err["beta22"]) / tws_no_err["beta22"]  # ty:ignore[unsupported-operator]
    rms_errs_betax = (tws_errs_betax.pow(2).mean()) ** 0.5
    rms_errs_betay = (tws_errs_betay.pow(2).mean()) ** 0.5
    print(f"RMS BetaX error before correction: {rms_errs_betax * 100:.2f}%")
    print(f"RMS BetaY error before correction: {rms_errs_betay * 100:.2f}%")
    assert rms_errs_betax > 3e-3, "Original RMS BetaX errors were not significant"
    assert rms_errs_betay > 3e-3, "Original RMS BetaY errors were not significant"
