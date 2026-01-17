#!/usr/bin/env python3
"""
Estimate delta-p (dpp) from corrector strengths using dispersion analysis.

This script estimates the momentum deviation (deltap) by:
1. Estimating horizontal dispersion at corrector locations using optics analysis data
2. Retrieving corrector strengths from NXCALS at specified times
3. Computing the total orbit length change from corrector kicks
4. Converting to momentum deviation using momentum compaction factor

The dispersion at correctors is estimated by propagating optics parameters
from nearby BPMs using MAD-NG differential algebra tracking, rather than
using model dispersion values.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import tfs
from nxcals.spark_session_builder import (  # pyright: ignore[reportMissingImports]
    get_or_create,
)
from uncertainties import ufloat

from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.dispersion.dispersion_estimation import estimate_corrector_dispersions
from aba_optimiser.measurements.knob_extraction import get_energy, get_mcb_vals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def convert_corr_strength_to_name(corr_strength: str) -> str:
    """Convert NXCALS corrector name to MAD-X/MAD-NG element name.

    Args:
        corr_strength: NXCALS corrector name (e.g., "acbch5.l1b1")

    Returns:
        MAD-X element name (e.g., "MCBCH.5L1.B1")

    Raises:
        ValueError: If corrector name format is not recognized
    """
    import re

    # Try beam-specific corrector format
    match = re.match(r"a(cb[a-z]*)([vh]y?)s?(\d+)\.([lr])(\d+)b(\d+)", corr_strength)
    if match:
        cb, plane, from_ip, leftright, ip, beam = match.groups()
        return f"M{cb.upper()}{plane.upper()}.{from_ip}{leftright.upper()}{ip}.B{beam}"
    # Try dual beam corrector format
    match = re.match(r"a(cbx)([vh])(\d+)\.([lr])(\d+)", corr_strength)
    if match:
        cbx, plane, from_ip, leftright, ip = match.groups()
        return f"M{cbx.upper()}{plane.upper()}.{from_ip}{leftright.upper()}{ip}"
    raise ValueError(f"Invalid corr_strength format: {corr_strength}")


def get_element_name_variants(elm_name: str) -> list[str]:
    """Get variants of an element name (base, A version, B version).

    Args:
        elm_name: Base element name (e.g., "MCBCH.5L1.B1")

    Returns:
        List of element name variants to try
    """
    split_name = elm_name.split(".")
    if len(split_name) < 3:
        # If name doesn't have enough parts, just return the base name
        return [elm_name]

    a_version = split_name[0] + ".A" + split_name[1] + "." + split_name[2]
    b_version = split_name[0] + ".B" + split_name[1] + "." + split_name[2]
    return [elm_name, a_version, b_version]


def get_dispersion_for_element(elm_name: str, dispersion_df: tfs.TfsDataFrame) -> float:
    """Get dispersion for an element, trying name variants if needed.

    Args:
        elm_name: Element name
        dispersion_df: DataFrame with corrector dispersions

    Returns:
        Dispersion value

    Raises:
        KeyError: If element not found in any variant
    """
    for variant in get_element_name_variants(elm_name):
        if variant in dispersion_df.index:
            return dispersion_df.loc[variant, "DISPERSION"]

    raise KeyError(f"Element {elm_name} not found in dispersion data, nor A/B versions")


def calculate_dpp_from_correctors(
    correctors: list[NXCalResult],
    dispersion_getter: callable,
    mom_compaction: float,
    ring_length: float,
    beam: int = 1,
) -> tuple[float, float]:
    """Calculate momentum deviation from corrector strengths and dispersion with uncertainty.

    The momentum deviation is computed as:
        dpp = -Σ(corrector_kick * dispersion) / (momentum_compaction * ring_length)

    Uncertainties are propagated from the dispersion uncertainties, ring
    circumference uncertainty (2 mm), and momentum compaction factor uncertainty
    (1%) using the uncertainties package.

    Args:
        correctors: List of corrector values from NXCALS
        dispersion_getter: Function that takes (elm_name, dispersion_data) and returns (disp_val, disp_std)
        mom_compaction: Momentum compaction factor (dimensionless)
        ring_length: Ring circumference in meters
        beam: Beam number (1 or 2) - affects kick sign

    Returns:
        Tuple of (dpp_value, dpp_uncertainty)

    Raises:
        KeyError: If a corrector is not found in dispersion data
    """
    delta_lengths = []
    for corrector in correctors:
        elm_name = convert_corr_strength_to_name(corrector.name)

        # Get dispersion with uncertainty using the provided getter function
        disp_val, disp_std = dispersion_getter(elm_name)

        if beam == 2:
            disp_val = -disp_val  # Reverse sign for beam 2 (opposite bending)

        # Create uncertain value for dispersion
        corrector_dispersion = ufloat(disp_val, disp_std)

        # Add 1e-4 relative error to corrector strength
        kick_uncertainty = 1e-4 * abs(corrector.value)
        corrector_kick = ufloat(corrector.value, kick_uncertainty)

        delta_length = corrector_kick * corrector_dispersion
        delta_lengths.append(delta_length)

    total_delta_length = sum(delta_lengths)
    # Add 2 mm uncertainty to ring circumference
    ring_length_uncertain = ufloat(ring_length, 0.002)
    # Add 1% uncertainty to momentum compaction factor
    mom_compaction_uncertain = ufloat(mom_compaction, 0.01 * abs(mom_compaction))
    dpp_uncertain = -1 * total_delta_length / (mom_compaction_uncertain * ring_length_uncertain)

    return dpp_uncertain.nominal_value, dpp_uncertain.std_dev


def get_dispersion_from_df(elm_name: str, dispersion_df: tfs.TfsDataFrame) -> tuple[float, float]:
    """Get dispersion value and uncertainty from dispersion DataFrame."""
    for variant in get_element_name_variants(elm_name):
        if variant in dispersion_df.index:
            disp_val = dispersion_df.loc[variant, "DISPERSION"]
            disp_std = dispersion_df.loc[variant, "STD"]
            return disp_val, disp_std
    raise KeyError(f"Element {elm_name} not found in dispersion data, nor A/B versions")


def get_dispersion_from_model(
    elm_name: str, twiss_elements: tfs.TfsDataFrame
) -> tuple[float, float]:
    """Get dispersion value from model twiss data (no uncertainty)."""
    for variant in get_element_name_variants(elm_name):
        if variant in twiss_elements.index:
            disp_val = twiss_elements.loc[variant, "DX"]  # Model dispersion column
            return disp_val, 0.0  # No uncertainty for model
    raise KeyError(f"Element {elm_name} not found in model twiss data, nor A/B versions")


def calculate_dpp_from_dispersion(
    correctors: list[NXCalResult],
    dispersion_df: tfs.TfsDataFrame,
    mom_compaction: float,
    ring_length: float,
    beam: int = 1,
) -> tuple[float, float]:
    """Calculate momentum deviation from corrector strengths and estimated dispersion."""
    return calculate_dpp_from_correctors(
        correctors,
        lambda elm: get_dispersion_from_df(elm, dispersion_df),
        mom_compaction,
        ring_length,
        beam,
    )


def calculate_dpp_from_model_dispersion(
    correctors: list[NXCalResult],
    twiss_elements: tfs.TfsDataFrame,
    mom_compaction: float,
    ring_length: float,
    beam: int = 1,
) -> tuple[float, float]:
    """Calculate momentum deviation using model dispersion from twiss_elements.dat."""
    return calculate_dpp_from_correctors(
        correctors,
        lambda elm: get_dispersion_from_model(elm, twiss_elements),
        mom_compaction,
        ring_length,
        beam,
    )


def estimate_dpp_for_times(
    beam: int,
    times_permil: dict[float, str],
    optics_dir: Path,
    model_dir: Path,
    sequence_file: Path,
    output_file: Path,
) -> None:
    """Estimate dpp for multiple times and save to JSON file.

    Args:
        beam: Beam number (1 or 2)
        times_permil: Dictionary mapping expected deltap (in per mil) to timestamp strings
        optics_dir: Directory containing optics analysis TFS files
        model_dir: Directory containing model twiss_elements.dat
        sequence_file: Path to MAD-X sequence file
        output_file: Path to output JSON file
    """
    logger.info(f"Starting dpp estimation for beam {beam}")

    # Step 1: Estimate dispersion at correctors using optics analysis data
    logger.info("Estimating horizontal dispersion at correctors")
    dispersion_df, _ = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=sequence_file,
        model_dir=model_dir,
        seq_name=f"lhcb{beam}",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=15,
        plane="x",
    )

    # Step 2: Load model parameters
    logger.info(f"Loading model parameters from {model_dir}")
    model = tfs.read(model_dir / "twiss_elements.dat", index="NAME")
    try:
        mom_compaction = model.headers["ALFA"]
    except KeyError:
        mom_compaction = model.headers["ALFAP"]
    ring_length = model.headers["LENGTH"]

    # Get list of horizontal kickers from model (KEYWORD == "HKICKER")
    try:
        horizontal_kickers = set(model[model["KEYWORD"] == "HKICKER"].index)
    except KeyError:
        horizontal_kickers = set(model[model["KIND"] == "hkicker"].index)

    # Step 3: Get corrector strengths from NXCALS and compute dpp
    spark = get_or_create()
    tz = ZoneInfo("UTC")
    results = {}

    for dpp_permil, start_str in times_permil.items():
        logger.info(f"Processing deltap = {dpp_permil} per mil (time: {start_str})")

        # Parse timestamp
        start_dt = datetime.strptime(start_str, "%Y_%m_%d@%H_%M_%S_%f").replace(tzinfo=tz)

        # Get corrector values from NXCALS
        mcb_vals = get_mcb_vals(spark, start_dt, beam=beam)
        logger.info(f"Retrieved {len(mcb_vals)} corrector values from NXCALS")

        # Filter to only horizontal correctors (check if element is HKICKER in model)
        horizontal_mcb_vals = []
        for corr in mcb_vals:
            try:
                elm_name = convert_corr_strength_to_name(corr.name)
                # Check if any name variant is in horizontal_kickers
                if any(
                    variant in horizontal_kickers for variant in get_element_name_variants(elm_name)
                ):
                    horizontal_mcb_vals.append(corr)
            except ValueError:
                # Skip correctors with unrecognized name format
                continue

        logger.info(f"Filtered to {len(horizontal_mcb_vals)} horizontal correctors (HKICKER)")

        # Calculate dpp using estimated dispersion with uncertainty
        dpp_est, dpp_est_err = calculate_dpp_from_dispersion(
            horizontal_mcb_vals, dispersion_df, mom_compaction, ring_length, beam=beam
        )

        # Calculate dpp using model dispersion
        dpp_model, dpp_model_err = calculate_dpp_from_model_dispersion(
            horizontal_mcb_vals, model, mom_compaction, ring_length, beam=beam
        )

        # Rescale to reference energy 6800 GeV
        energy, _ = get_energy(spark, start_dt)
        e_meas_est = energy * (1 + dpp_est)
        dpp_est_wrt_6800 = (e_meas_est - 6800) / 6800

        e_meas_model = energy * (1 + dpp_model)
        dpp_model_wrt_6800 = (e_meas_model - 6800) / 6800

        # Propagate uncertainty (energy uncertainty is negligible compared to dpp uncertainty)
        dpp_est_wrt_6800_err = energy / 6800 * dpp_est_err
        dpp_model_wrt_6800_err = energy / 6800 * dpp_model_err

        results[dpp_permil] = {
            "estimated_dispersion": {
                "value": dpp_est_wrt_6800,
                "uncertainty": dpp_est_wrt_6800_err,
            },
            "model_dispersion": {
                "value": dpp_model_wrt_6800,
                "uncertainty": dpp_model_wrt_6800_err,
            },
        }
        logger.info(
            f"  Estimated dpp (estimated dispersion): {dpp_est_wrt_6800:.6e} ± {dpp_est_wrt_6800_err:.6e}"
        )
        logger.info(
            f"  Estimated dpp (model dispersion): {dpp_model_wrt_6800:.6e} ± {dpp_model_wrt_6800_err:.6e}"
        )

    # Step 4: Save results to JSON
    logger.info(f"Saving results to {output_file}")
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Dpp estimation complete for beam {beam}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Estimate dpp from corrector strengths using dispersion analysis"
    )
    parser.add_argument(
        "--beam",
        type=int,
        choices=[1, 2],
        required=True,
        help="Beam number (1 or 2)",
    )
    parser.add_argument(
        "--optics-dir",
        type=Path,
        help="Directory containing optics analysis TFS files (beta_phase_x.tfs, etc.)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing model twiss_elements.dat",
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        help="Path to MAD-X sequence file (default: auto-detect from beam number)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to output JSON file (default: estimates_b{beam}.json in project root)",
    )

    args = parser.parse_args()

    # Auto-detect output file if not provided
    if args.output_file is None:
        results_dir = PROJECT_ROOT / f"b{args.beam}co_results"
        results_dir.mkdir(exist_ok=True)
        args.output_file = results_dir / f"estimates_b{args.beam}.json"
        logger.info(f"Using output file: {args.output_file}")

    # Define times for each beam (hardcoded from the original plot_results.py)
    meas_dir = PROJECT_ROOT
    optics_dir = (
        meas_dir / f"analysis_b{args.beam}_12cm" if args.optics_dir is None else args.optics_dir
    )
    model_dir = (
        meas_dir / f"models/lhcb{args.beam}_12cm" if args.model_dir is None else args.model_dir
    )
    if args.beam == 1:
        times_permil = {
            0: "2025_11_07@07_51_57_940",
            0.1: "2025_11_07@08_11_13_745",
            0.2: "2025_11_07@08_01_15_782",
            -0.1: "2025_11_07@08_15_44_795",
            -0.2: "2025_11_07@08_22_09_892",
        }
    else:
        times_permil = {
            0: "2025_11_07@07_38_44_035",
            0.2: "2025_11_07@07_51_18_881",
            0.1: "2025_11_07@08_03_44_894",
            -0.1: "2025_11_07@08_11_46_900",
            -0.2: "2025_11_07@08_18_27_900",
        }

    # Auto-detect sequence file if not provided
    if args.sequence_file is None:
        # args.sequence_file = PROJECT_ROOT / f"src/aba_optimiser/mad/sequences/lhcb{args.beam}.seq"
        args.sequence_file = model_dir / f"lhcb{args.beam}_saved.seq"
        logger.info(f"Using sequence file: {args.sequence_file}")


    # Run estimation
    estimate_dpp_for_times(
        beam=args.beam,
        times_permil=times_permil,
        optics_dir=optics_dir,
        model_dir=model_dir,
        sequence_file=args.sequence_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
