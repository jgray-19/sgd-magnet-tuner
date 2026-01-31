"""
Integration-style tests for dispersion estimation using tracking data.

This test validates that we can reproduce model dispersion at correctors
by tracking from nearby BPMs using measured optics parameters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tfs
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.model.constants import TWISS_ELEMENTS_DAT
from scipy import stats
from turn_by_turn import convert_to_tbt, write_tbt
from turn_by_turn.structures import TbtData

from aba_optimiser.dispersion.dispersion_estimation import estimate_corrector_dispersions
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.simulation.magnet_perturbations import apply_magnet_perturbations
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.acd import run_ac_dipole_tracking_with_particles
from aba_optimiser.xsuite.env import initialise_env

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.mad.base_mad_interface import BaseMadInterface

logger = logging.getLogger(__name__)


def _generate_nonoise_track(
    mad: BaseMadInterface,
    tmp_dir: Path,
    model_dir: Path,
    sequence_file: Path,
    flattop_turns: int,
    beam: Literal[1, 2] = 1,
    peturbed_magnets: str | None = "qd",
    magnet_range: str = "$start/$end",
) -> Path:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs.

    Args:
        mad: Loaded MAD interface with beam setup
        tmp_dir: Temporary directory for outputs
        model_dir: Directory containing model files
        sequence_file: Path to MAD-X sequence file
        flattop_turns: Number of turns for tracking
        beam: Beam number (1 or 2)
        magnet_types: Types of magnets to perturb ("q" for quadrupoles, "d" for dipoles, "s" for sextupoles)
        magnet_range: Range of magnets to perturb
    Returns:
        Path to analysis directory containing optics results
    """
    seq_name = f"lhcb{beam}"

    mad.mad["zero_twiss", "_"] = mad.mad.twiss(sequence="loaded_sequence")  # ty:ignore[invalid-assignment]
    # Apply magnet perturbations
    magnet_strengths = {}
    if peturbed_magnets is not None:
        magnet_strengths, _ = apply_magnet_perturbations(
            mad.mad,
            rel_k1_std_dev=1e-4,
            seed=42,
            magnet_type=peturbed_magnets,
        )

    # Create unique corrector file path based on destination
    corrector_file = tmp_dir / f"correctors_b{beam}.tfs"

    if beam == 2:
        mad.mad.MADX["dQx.b2_op"] = -4.13993e-06  # For 12cm beam 2, speed up convergence
        mad.mad.MADX["dQy.b2_op"] = -2.92699e-06  # For 12cm beam 2, speed up convergence

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
        beam=beam,
    )
    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,  # ty:ignore[invalid-argument-type]
        beam=beam,
        sequence_file=sequence_file,
        seq_name=seq_name,
    )

    # save the tune knobs to file with unique name
    tune_knobs_file = tmp_dir / f"tune_knobs_b{beam}.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    # Insert AC dipole and track with different momentum offsets
    acd_ramp = 1000
    driven_tunes = [0.27, 0.322]
    output_files = []
    action_list = [1e-10, 1e-10, 1e-10]
    angle_list = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    delta_values = [-5e-4, 0.0, 5e-4]

    for lag in [np.pi / 3]:
        monitored_line = run_ac_dipole_tracking_with_particles(
            line=env[seq_name],  # ty:ignore[not-subscriptable]
            beam=beam,
            ramp_turns=acd_ramp,
            flattop_turns=flattop_turns,
            driven_tunes=driven_tunes,
            lag=lag,
            bpm_pattern="bpm.*[^k]",
            action_list=action_list,
            angle_list=angle_list,
            delta_values=delta_values,
        )

        tbt_data = convert_to_tbt(monitored_line)
        for idx, matrix in enumerate(tbt_data.matrices):
            dpp = delta_values[idx]
            single_tbt = TbtData(
                matrices=[matrix],
                nturns=tbt_data.nturns,
                bunch_ids=[tbt_data.bunch_ids[idx] if tbt_data.bunch_ids else idx],
                meta=tbt_data.meta,
            )
            output_file = tmp_dir / f"track_result_b{beam}_{dpp}_{lag}.sdds"
            output_files.append(output_file)
            write_tbt(output_file, single_tbt, noise=1e-4)

    linfile_dir = tmp_dir / f"linfiles_b{beam}"
    hole_in_one_entrypoint(
        harpy=True,
        files=output_files,
        tbt_datatype="lhc",
        outputdir=linfile_dir,
        to_write=["lin", "spectra"],
        opposite_direction=beam == 2,  # Tracked beam 4, model beam 2!
        driven_excitation="acd",
        tunes=driven_tunes + [0.0],
        nattunes=[0.28, 0.31, 0.0],
        turn=[acd_ramp, 50e3],
        clean=True,
    )
    all_linfiles = list(linfile_dir.glob("*.linx"))
    all_files = [f.parent / f.name.strip(".linx") for f in all_linfiles]
    analysis_dir = tmp_dir / f"analysis_b{beam}"

    hole_in_one_entrypoint(
        optics=True,
        files=all_files,
        outputdir=analysis_dir,
        accel="lhc",
        beam=beam,
        model_dir=model_dir,
        year="2025",
        compensation="equation",
    )
    return analysis_dir


def _validate_dispersion_estimates(
    dispersion_df: pd.DataFrame,
    twiss_elements: pd.DataFrame,
    beam: int,
    tmp_dir: Path,
) -> None:
    """Validate dispersion estimates against model values by checking z-score statistics.

    Args:
        dispersion_df: DataFrame with estimated dispersions
        twiss_elements: DataFrame with model twiss values
        beam: Beam number for logging
        tmp_dir: Temporary directory for saving debug plots

    Raises:
        pytest.fail: If mean or std of z-scores are out of acceptable range
    """
    # Calculate z-scores
    z_scores = np.array(
        [
            (dispersion_df.loc[c, "DISPERSION"] - twiss_elements.loc[c, "DX"])
            / dispersion_df.loc[c, "STD"]
            for c in dispersion_df.index
        ]
    )

    # Plot z-scores for debugging
    plt.figure(figsize=(8, 6))
    plt.hist(z_scores, bins=20, alpha=0.7, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", label="Expected mean")
    plt.xlabel("Z-score")
    plt.ylabel("Frequency")
    plt.title(f"Z-scores Distribution for Beam {beam}")
    plt.legend()
    plt.grid(visible=True, alpha=0.3)
    plt.savefig(tmp_dir / f"z_scores_beam_{beam}.png")
    plt.close()

    # Compute z-score statistics
    n = len(z_scores)
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    # Log basic statistics
    logger.info(f"Beam {beam}: Number of correctors: {n}")
    logger.info(f"Beam {beam}: Mean z-score: {mean_z:.3f}")
    logger.info(f"Beam {beam}: Std z-score: {std_z:.3f}")
    ks_stat, p_value = stats.kstest(z_scores, "norm", args=(0, 1))
    logger.info(f"Beam {beam}: KS test statistic: {ks_stat:.3f}, p-value: {p_value:.3f}")

    # Check coverage within 3σ
    failures = []
    p_coverage = 2 * stats.norm.cdf(3) - 1
    tol_coverage = 3 * np.sqrt(p_coverage * (1 - p_coverage) / n)
    frac_coverage = np.mean(np.abs(z_scores) <= 3)
    logger.info(
        f"Beam {beam}: Fraction within |z| <= 3: {frac_coverage:.4f} (expected {p_coverage:.4f} ± {tol_coverage:.4f})"
    )
    if not (
        max(0, p_coverage - tol_coverage) <= frac_coverage <= min(1, p_coverage + tol_coverage)
    ):
        logger.error(f"Beam {beam}: Fraction within |z| <= 3 out of range")
        failures.append("coverage_3σ")

    # Check tails beyond 3σ
    p_tail = 1 - p_coverage
    tol_tail = 3 * np.sqrt(p_tail * (1 - p_tail) / n)
    frac_tail = np.mean(np.abs(z_scores) > 3)
    logger.info(
        f"Beam {beam}: Fraction with |z| > 3: {frac_tail:.4f} (expected {p_tail:.4f} ± {tol_tail:.4f})"
    )
    if frac_tail > p_tail + tol_tail:
        logger.error(f"Beam {beam}: Fraction with |z| > 3 exceeds threshold")
        failures.append("tail_3σ")

    if beam == 1:
        # Beam 2 has a signifcant mean bias - should be investigated when important.
        # Check mean bias
        mean_tol = 3 / np.sqrt(n)
        if abs(mean_z) > mean_tol:
            logger.error(f"Beam {beam}: |Mean z-score| ({abs(mean_z):.3f}) > {mean_tol:.3f}")
            failures.append("mean_bias")

    # Fail if any check fails
    if failures:
        pytest.fail(
            f"Beam {beam}: Z-score validation failed ({', '.join(failures)}). Mean: {mean_z:.3f}, Std: {std_z:.3f}"
        )


@pytest.mark.slow
def test_dispersion_b1(
    tmp_path: Path,
    seq_b1: Path,
    model_dir_b1: Path,
    loaded_interface_with_beam: BaseMadInterface,
) -> None:
    """Test that dispersion estimation reproduces model values at correctors.

    This test:
    1. Generates tracking data with off-momentum particles
    2. Analyzes the tracking to extract optics at BPMs
    3. Estimates dispersion at correctors by tracking from nearby BPMs
    4. Validates estimates match model within tolerance
    """
    beam = 1

    # Generate tracking data and analyze optics
    optics_dir = _generate_nonoise_track(
        loaded_interface_with_beam, tmp_path, model_dir_b1, seq_b1, 6600, beam, None
    )

    # Load model twiss for validation
    twiss_elements = tfs.read(model_dir_b1 / TWISS_ELEMENTS_DAT, index="NAME")

    # Estimate horizontal dispersion at all correctors
    dispersion_df, _statistics_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=seq_b1,
        model_dir=model_dir_b1,
        seq_name=f"lhcb{beam}",
        beam=beam,
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=50,
        plane="x",
    )

    # Validate estimates against model
    _validate_dispersion_estimates(dispersion_df, twiss_elements, beam, tmp_path)


@pytest.mark.slow
def test_dispersion_b2(
    tmp_path: Path,
    seq_b2: Path,
    model_dir_b2: Path,
    beam2_interface: BaseMadInterface,
) -> None:
    """Test that dispersion estimation reproduces model values at correctors.

    This test:
    1. Generates tracking data with off-momentum particles
    2. Analyzes the tracking to extract optics at BPMs
    3. Estimates dispersion at correctors by tracking from nearby BPMs
    4. Validates estimates match model within tolerance
    """

    beam = 2
    # Generate tracking data and analyze optics
    optics_dir = _generate_nonoise_track(
        beam2_interface, tmp_path, model_dir_b2, seq_b2, 6600, beam, None
    )

    # Load model twiss for validation
    twiss_elements = tfs.read(model_dir_b2 / TWISS_ELEMENTS_DAT, index="NAME")

    # Estimate horizontal dispersion at all correctors
    dispersion_df, _statistics_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=seq_b2,
        model_dir=model_dir_b2,
        seq_name=f"lhcb{beam}",
        beam=beam,
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=50,
        plane="x",
    )

    # Validate estimates against model
    _validate_dispersion_estimates(dispersion_df, twiss_elements, beam, tmp_path)
