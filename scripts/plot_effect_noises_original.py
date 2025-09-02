"""
Noise Effect Analysis for SGD Magnet Tuner

This script analyzes the effects of various noise sources on beam tracking:
- Quadrupole strength errors
- Initial condition perturbations
- Combined effects

The analysis uses parallel processing to compute multiple samples and
generates comparison plots of the standard deviations.
"""

from __future__ import annotations

import argparse
import logging
from itertools import product, repeat
from typing import TYPE_CHECKING

import numpy as np
from pymadng import MAD
from tqdm.contrib.concurrent import process_map

from aba_optimiser.config import (
    BEAM_ENERGY,
    MAGNET_RANGE,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from scripts.plot_functions import (
    plot_error_bars_bpm_range,
    plot_std_log_comparison,
    show_plots,
)

if TYPE_CHECKING:
    import pandas as pd
    import tfs

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# These constants control the simulation behavior and analysis parameters

NTURNS = 1  # Number of tracking turns (single turn analysis)
NANGLES = 8  # Default number of initial angles to sample around the action circle
NUM_ERROR_SAMPLES = 30  # Number of Monte Carlo samples for error analysis
IC_XY_STD = 1e-4  # Standard deviation for initial condition position errors [m]
IC_PXPY_STD = 3e-6  # Standard deviation for initial condition momentum errors [rad]
ACTION0 = 6e-9  # Initial action for particle trajectory [m·rad]

# ============================================================================
# TRACKING COMMAND CONSTRUCTION
# ============================================================================


def build_track_command(
    df_twiss: tfs.TfsDataFrame,
    action: float,
    angle: float,
    nturns: int,
    start_err: bool = False,
) -> str:
    """
    Construct the MAD-NG track command string with given initial conditions and settings.

    This function converts physical parameters (action, angle) into initial coordinates
    for particle tracking using the Twiss parameters at the starting BPM.

    Args:
        df_twiss: Dataframe containing Twiss parameters (beta, alpha functions)
        action: Initial action in phase space [m·rad]
        angle: Initial angle in phase space [rad]
        nturns: Number of tracking turns
        start_err: Whether to add random initial condition errors

    Returns:
        MAD-NG track command string ready for execution
    """
    rng = np.random.default_rng()
    start_bpm = MAGNET_RANGE.split("/")[0]  # Extract starting BPM name from range

    # Get Twiss parameters at the starting BPM for coordinate transformation
    beta11 = df_twiss.loc[start_bpm, "beta11"]  # Horizontal beta function [m]
    beta22 = df_twiss.loc[start_bpm, "beta22"]  # Vertical beta function [m]
    alfa11 = df_twiss.loc[start_bpm, "alfa11"]  # Horizontal alpha function [-]
    alfa22 = df_twiss.loc[start_bpm, "alfa22"]  # Vertical alpha function [-]

    # Convert action-angle coordinates to physical coordinates
    # This transformation maps from normalized phase space to real coordinates
    cos0 = np.cos(angle)
    sin0 = np.sin(angle)

    # Calculate initial positions and momenta from action-angle variables
    # x = sqrt(2*J*beta) * cos(phi), px = -sqrt(2*J/beta) * (sin(phi) + alpha*cos(phi))
    x0 = np.sqrt(action * beta11) * cos0
    px0 = -np.sqrt(action / beta11) * (sin0 + alfa11 * cos0)
    y0 = np.sqrt(action * beta22) * cos0  # Same angle for both planes (round beam)
    py0 = -np.sqrt(action / beta22) * (sin0 + alfa22 * cos0)

    if start_err:
        # Add random perturbations to initial conditions to simulate measurement errors
        dx0, dy0 = rng.normal(0, IC_XY_STD, size=2)  # Position errors [m]
        dpx0, dpy0 = rng.normal(0, IC_PXPY_STD, size=2)  # Momentum errors [rad]
    else:
        # No additional errors for baseline tracking
        dx0, dpx0, dy0, dpy0 = 0.0, 0.0, 0.0, 0.0

    # Construct MAD-NG track command with initial coordinates
    return (
        f"trk, mflw = track{{sequence=MADX['{SEQ_NAME}'], "
        f"X0={{x={x0 + dx0:.6e}, px={px0 + dpx0:.6e}, y={y0 + dy0:.6e}, py={py0 + dpy0:.6e}, t=0, pt=0}}, "
        f"nturn={nturns}}}"
    )


# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================
# These functions are designed to work with process_map for parallel computation


def _compute_error_sample(
    i, matched_tunes, df_twiss, nturns, angle: float, start_err: bool = False
) -> tuple[pd.Series, pd.Series]:
    """
    Worker function: Run tracking with random quadrupole strength errors.

    This function simulates the effect of systematic quadrupole magnet errors
    on particle tracking. Each quadrupole's strength is perturbed by a random
    amount drawn from a normal distribution.

    Args:
        i: Sample index (for multiprocessing)
        matched_tunes: Dictionary of matched tune correction knobs
        df_twiss: Twiss dataframe with optical functions
        nturns: Number of tracking turns
        angle: Initial angle in phase space
        start_err: Whether to also include initial condition errors

    Returns:
        Tuple of (x_positions, y_positions) at BPMs after tracking
    """
    rng = np.random.default_rng()
    mad_i = _setup_mad()  # Create fresh MAD instance for this process

    # Apply the matched tune corrections
    for key, val in matched_tunes.items():
        mad_i[f"MADX['{key}']"] = val

    # Identify quadrupoles in the specified arc region
    seq = mad_i.MADX[SEQ_NAME]
    arc_start = df_twiss.loc[MAGNET_RANGE.split("/")[0], "s"]  # Start position [m]
    arc_end = df_twiss.loc[MAGNET_RANGE.split("/")[1], "s"]  # End position [m]

    # Filter quadrupoles: must be in arc, have nonzero strength, and be main quads
    quad_names = [
        elm.name
        for elm in seq
        if elm.kind == "quadrupole"  # Only quadrupole elements
        and elm.k1 != 0  # Must have nonzero strength
        and "MQ." in elm.name  # Only main quadrupoles (not trim/skew)
        and arc_start <= df_twiss.loc[elm.name, "s"] <= arc_end  # Within arc range
    ]

    # Apply random strength errors to each quadrupole
    # Each quad gets an independent random error proportional to its strength
    for name in quad_names:
        noise = rng.normal(scale=REL_K1_STD_DEV)  # Relative error (dimensionless)
        k1 = mad_i[f"MADX['{name}'].k1"]  # Current strength [m^-2]
        mad_i[f"MADX['{name}'].k1"] = k1 + noise * abs(k1)  # Apply relative error

    # Perform tracking with perturbed quadrupoles
    xi, yi = _get_track_end_positions(
        mad_i, matched_tunes, df_twiss, nturns=nturns, angle=angle, start_err=start_err
    )
    return xi, yi


def _compute_ic(
    i, matched_tunes, df_twiss, nturns, angle: float, action0
) -> tuple[pd.Series, pd.Series]:
    """
    Worker function: Run tracking with only initial condition perturbations.

    This function simulates the effect of measurement errors in the initial
    particle coordinates. The quadrupole strengths remain unchanged.

    Args:
        i: Sample index (for multiprocessing)
        matched_tunes: Dictionary of matched tune correction knobs
        df_twiss: Twiss dataframe with optical functions
        nturns: Number of tracking turns
        angle: Initial angle in phase space
        action0: Initial action value

    Returns:
        Tuple of (x_positions, y_positions) at BPMs after tracking
    """
    mad_p = _setup_mad()  # Create fresh MAD instance for this process

    # Perform tracking with initial condition errors only (no quad errors)
    xi, yi = _get_track_end_positions(
        mad_p,
        matched_tunes,
        df_twiss,
        nturns=nturns,
        angle=angle,
        action=action0,
        start_err=True,  # Enable initial condition errors
    )
    return xi, yi


# ============================================================================
# MAD-NG SETUP AND SIMULATION FUNCTIONS
# ============================================================================


def _setup_mad() -> MAD:
    """
    Initialize a MAD-NG process with sequence loaded, beam set, and BPMs selected.

    This function sets up a complete MAD-NG environment for beam tracking:
    1. Loads the accelerator sequence from file
    2. Defines the beam parameters (proton beam with specified energy)
    3. Configures observation points (BPMs only)
    4. Sets up the tracking start point with a marker

    Returns:
        Configured MAD instance ready for tracking simulations
    """
    # Create MAD-NG instance with logging enabled
    mad = MAD(stdout="mad_stdout.log", redirect_stderr=True, debug=True)

    # Load the accelerator sequence definition from the specified file
    mad.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
    seq = mad.MADX[SEQ_NAME]

    # Define beam properties for tracking
    seq.beam = mad.beam(particle='"proton"', energy=BEAM_ENERGY)

    # Configure observation points: only BPMs will record tracking data
    seq.deselect(mad.element.flags.observed)  # Clear all observation flags
    seq.select(mad.element.flags.observed, {"pattern": "'BPM'"})  # Select only BPMs

    # Install a marker at the starting BPM to define the tracking start point
    monitor_name = MAGNET_RANGE.split("/")[0]  # Extract first BPM name from range
    marker_name = mad.quote_strings(f"{monitor_name}_marker")
    seq.install(
        [
            mad.element.marker(
                **{"at": 0, "from": f'"{monitor_name}"', "name": marker_name}
            )
        ]
    )

    # Cycle the sequence to start tracking from the marker position
    seq.cycle(marker_name)

    return mad


def _match_tunes() -> tuple[dict[str, float], pd.DataFrame]:
    """
    Match the working point tunes and return the matched knob values and Twiss at start.

    This function performs tune matching to achieve the desired fractional tunes
    by adjusting the tune correction knobs. It uses MAD-NG's matching algorithm
    to find the knob settings that produce the target tunes.

    Returns:
        Tuple containing:
        - Dictionary of matched knob values (dqx_b1_op, dqy_b1_op)
        - Twiss dataframe with optical functions at all elements
    """
    mad = _setup_mad()
    mad["SEQ_NAME"] = SEQ_NAME
    mad["knob_range"] = MAGNET_RANGE

    # Target fractional tunes (the decimal part of the total tune)
    # Total tune = integer part + fractional part
    # For LHC: Qx ≈ 62.28, Qy ≈ 60.31
    tunes = [0.28, 0.31]  # [horizontal_fractional_tune, vertical_fractional_tune]

    # Perform tune matching using MAD-NG's optimization algorithm
    mad["result"] = mad.match(
        command=r"\ -> twiss{sequence=MADX[SEQ_NAME]}",  # Command to execute for each iteration
        variables=[
            # Variables to optimize (tune correction knobs)
            {"var": "'MADX.dqx_b1_op'", "name": "'dqx_b1_op'"},  # Horizontal tune knob
            {"var": "'MADX.dqy_b1_op'", "name": "'dqy_b1_op'"},  # Vertical tune knob
        ],
        equalities=[
            # Constraints to satisfy (target tune values)
            {
                "expr": f"\\t -> math.abs(t.q1)-(62+{tunes[0]})",
                "name": "'q1'",
            },  # Horizontal tune target
            {
                "expr": f"\\t -> math.abs(t.q2)-(60+{tunes[1]})",
                "name": "'q2'",
            },  # Vertical tune target
        ],
        objective={"fmin": 1e-18},  # Convergence criterion (very tight)
        info=2,  # Verbosity level for matching output
    )

    # Extract the matched knob values
    matched = {key: mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")}

    # Compute Twiss parameters with the matched tunes
    tw = mad.twiss(sequence=mad.MADX[SEQ_NAME])[0]
    df_twiss = tw.to_df()
    df_twiss.set_index("name", inplace=True)  # Use element names as index

    return matched, df_twiss


def _get_last_turn_data(mad: MAD) -> pd.DataFrame:
    """
    Retrieve the last-turn data from a MAD-NG tracking result.

    This function extracts the final positions from tracking results,
    filtering to keep only the last turn and the specified BPM range.

    Args:
        mad: MAD instance containing tracking results in mad["trk"]

    Returns:
        DataFrame with last-turn BPM data (columns: x, y) indexed by BPM name
    """
    trk = mad["trk"]  # Get tracking results object
    df = trk.to_df(columns=["name", "turn", "x", "y"])  # Convert to pandas DataFrame

    # Filter to last turn only and slice to the specified BPM range
    return (
        df[df["turn"] == df["turn"].max()]  # Keep only the final tracking turn
        .set_index("name")  # Use BPM names as index
        .loc[
            MAGNET_RANGE.split("/")[0] : MAGNET_RANGE.split("/")[1]
        ]  # Slice to specified BPM range
        .sort_index()  # Sort BPMs by name for consistent ordering
    )


def _get_track_end_positions(
    mad: MAD,
    matched_tunes: dict[str, float],
    df_twiss,
    nturns: int = 100,
    action: float = 6e-9,
    angle: float = 0.0,
    start_err: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Run tracking simulation and return final positions at BPMs.

    This is the main tracking function that:
    1. Applies the matched tune corrections
    2. Constructs and executes the tracking command
    3. Extracts the final BPM positions

    Args:
        mad: MAD instance configured for tracking
        matched_tunes: Dictionary of tune correction knob values
        df_twiss: Twiss dataframe with optical functions
        nturns: Number of tracking turns
        action: Initial action for particle trajectory [m·rad]
        angle: Initial angle in phase space [rad]
        start_err: Whether to include initial condition errors

    Returns:
        Tuple of (x_positions, y_positions) as pandas Series indexed by BPM name
    """
    # Apply the matched tune corrections to the lattice
    for key, val in matched_tunes.items():
        mad[f"MADX['{key}']"] = val

    # Construct and execute the tracking command
    trk_command = build_track_command(
        df_twiss, action, angle, nturns, start_err=start_err
    )
    mad.send(trk_command)  # Execute the tracking simulation

    # Extract final positions from tracking results
    df_last = _get_last_turn_data(mad)
    return df_last["x"], df_last["y"]


def _compute_baseline(angle, matched_tunes, df_twiss, nturns):
    """
    Compute baseline tracking (no errors) for a given initial angle.

    This function provides the reference trajectory against which
    error effects will be compared.

    Args:
        angle: Initial angle in phase space [rad]
        matched_tunes: Dictionary of tune correction knob values
        df_twiss: Twiss dataframe with optical functions
        nturns: Number of tracking turns

    Returns:
        Tuple of (angle, x_positions, y_positions) for this angle
    """
    mad = _setup_mad()  # Create fresh MAD instance

    # Perform tracking with no errors (reference case)
    xi, yi = _get_track_end_positions(
        mad, matched_tunes, df_twiss, nturns=nturns, angle=angle
    )
    return angle, xi, yi


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================


def main():
    """
    Main function that orchestrates the complete noise effect analysis.

    This function performs a comprehensive study of noise effects by:
    1. Setting up the simulation parameters and lattice
    2. Computing baseline trajectories (no errors)
    3. Computing trajectories with quadrupole errors
    4. Computing trajectories with initial condition errors
    5. Computing trajectories with combined errors
    6. Analyzing the statistical differences and generating plots
    """

    # ===== COMMAND LINE ARGUMENT PARSING =====
    parser = argparse.ArgumentParser(
        description="Analyze noise effects in particle beam tracking"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--nangles",
        type=int,
        default=NANGLES,
        help="Number of initial angles to sample around the action circle",
    )
    args = parser.parse_args()

    # Configure logging level based on command line arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - verbose output will be shown")
    else:
        logger.setLevel(logging.INFO)

    logger.info(
        f"Starting noise effect analysis: nturns={NTURNS}, "
        f"error_samples={NUM_ERROR_SAMPLES}, angles={args.nangles}"
    )

    # ===== SETUP AND BASELINE COMPUTATION =====
    # Match tunes to the desired working point and get optical functions
    logger.info("Matching tunes to working point...")
    matched_tunes, df_twiss = _match_tunes()

    # Sample angles uniformly around the action circle (0 to 2π)
    # This gives different initial phase space points with the same action
    angles = np.linspace(0, 2 * np.pi, args.nangles, endpoint=False)
    logger.info(f"Sampling {len(angles)} angles: {angles}")

    # Compute baseline trajectories (no errors) in parallel across all angles
    logger.info("Computing baseline trajectories (no errors)...")
    baseline_results = process_map(
        _compute_baseline,
        angles,  # Different angles to sample
        repeat(matched_tunes),  # Same tune settings for all
        repeat(df_twiss),  # Same optical functions for all
        repeat(NTURNS),  # Same number of turns for all
        desc="Baseline per angle",
        max_workers=None,  # Use all available CPU cores
    )

    # Reorganize results into a dictionary indexed by angle
    baseline = {ang: (xi, yi) for ang, xi, yi in baseline_results}

    # Use the first angle's results as reference for plotting
    baseline_x, baseline_y = baseline[angles[0]]
    bpm_names = baseline_x.index.tolist()  # Get list of BPM names in order
    assert baseline_x.index.equals(baseline_y.index), "X and Y BPM indices must match"

    # ===== QUADRUPOLE ERROR ANALYSIS =====
    logger.info("Computing trajectories with quadrupole strength errors...")

    # Create task list: all combinations of (sample_index, angle)
    # This creates NUM_ERROR_SAMPLES × num_angles total tasks
    quad_tasks = list(product(range(NUM_ERROR_SAMPLES), angles))
    idxs, angs = zip(*quad_tasks)  # Separate into parallel arrays

    # Execute all quadrupole error samples in parallel
    flat_quad = process_map(
        _compute_error_sample,
        idxs,  # Sample indices for reproducibility
        repeat(matched_tunes),  # Same tune settings for all
        repeat(df_twiss),  # Same optical functions for all
        repeat(NTURNS),  # Same number of turns for all
        angs,  # Different angles
        desc="Quadrupole error samples",
        max_workers=None,
    )

    # Reorganize flat results back into a nested structure by angle
    results_quad = {angle: [] for angle in angles}
    for (i, angle), res in zip(quad_tasks, flat_quad):
        results_quad[angle].append(res)

    # ===== INITIAL CONDITION ERROR ANALYSIS =====
    logger.info(
        f"Computing trajectories with initial condition errors: "
        f"samples={NUM_ERROR_SAMPLES}, position_std={IC_XY_STD}, momentum_std={IC_PXPY_STD}"
    )

    # Reuse the same task structure as quadrupole errors
    ic_tasks = quad_tasks  # Same (sample_index, angle) pairs
    idxs_ic, angs_ic = idxs, angs

    # Execute all initial condition error samples in parallel
    flat_ic = process_map(
        _compute_ic,
        idxs_ic,  # Sample indices for reproducibility
        repeat(matched_tunes),  # Same tune settings for all
        repeat(df_twiss),  # Same optical functions for all
        repeat(NTURNS),  # Same number of turns for all
        angs_ic,  # Different angles
        repeat(ACTION0),  # Same initial action for all
        desc="IC perturbations",
        max_workers=None,
    )

    # Reorganize results by angle
    results_ic = {angle: [] for angle in angles}
    for (i, angle), res in zip(ic_tasks, flat_ic):
        results_ic[angle].append(res)

    # ===== COMBINED ERROR ANALYSIS =====
    logger.info(
        "Computing trajectories with combined errors: "
        "quadrupole errors + initial condition errors"
    )

    # Use same task structure for consistency
    cmb_tasks = quad_tasks
    idxs_cmb, angs_cmb = idxs, angs

    # Execute combined error samples (both quadrupole and IC errors)
    flat_cmb = process_map(
        _compute_error_sample,
        idxs_cmb,  # Sample indices for reproducibility
        repeat(matched_tunes),  # Same tune settings for all
        repeat(df_twiss),  # Same optical functions for all
        repeat(NTURNS),  # Same number of turns for all
        angs_cmb,  # Different angles
        repeat(True),  # Enable initial condition errors (start_err=True)
        desc="Combined error samples",
        max_workers=None,
    )

    # Reorganize results by angle
    results_combined = {angle: [] for angle in angles}
    for (i, angle), res in zip(cmb_tasks, flat_cmb):
        results_combined[angle].append(res)

    # ===== STATISTICAL ANALYSIS AND PLOTTING =====
    logger.info("Analyzing statistical differences and generating plots...")

    # Get BPM positions ordered by their location along the accelerator
    s_positions = df_twiss.loc[bpm_names, "s"].to_numpy()  # Position along beamline [m]
    order = np.argsort(s_positions)  # Sort by position
    bpm_ord = [bpm_names[i] for i in order]  # BPM names in position order
    s_positions = s_positions[order]  # Sorted positions

    # ===== QUADRUPOLE ERROR ANALYSIS =====
    logger.info("Computing statistics for quadrupole strength errors...")

    # Compute differences from baseline for all quadrupole error samples
    # This gives us the displacement caused by magnet strength errors
    diffs_x = []
    diffs_y = []
    for angle in angles:
        bx, by = baseline[angle]  # Baseline positions for this angle
        for xi, yi in results_quad[angle]:  # All error samples for this angle
            # Calculate difference from baseline (error-induced displacement)
            diffs_x.append((xi[bpm_ord] - bx[bpm_ord]).to_numpy())
            diffs_y.append((yi[bpm_ord] - by[bpm_ord]).to_numpy())

    # Stack all differences into arrays: (n_samples × n_angles, n_bpms)
    diffs_x = np.stack(diffs_x, axis=0)
    diffs_y = np.stack(diffs_y, axis=0)

    # Compute standard deviation across all samples and angles
    # This represents the RMS error caused by quadrupole strength uncertainties
    std_x_quaderr = np.std(diffs_x, axis=0)
    std_y_quaderr = np.std(diffs_y, axis=0)

    # Generate and save quadrupole error plot
    fig_quad = plot_error_bars_bpm_range(
        s_positions,
        baseline_x[bpm_ord],
        std_x_quaderr,
        baseline_y[bpm_ord],
        std_y_quaderr,
        MAGNET_RANGE,
    )
    fig_quad.suptitle("Quadrupole Error Bars", fontsize=14)
    fig_quad.savefig("plots/error_bars_bpm_range.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Compute standard deviations for IC perturbations over all angles
    diffs_x = []
    diffs_y = []
    for angle in angles:
        bx, by = baseline[angle]
        for xi, yi in results_ic[angle]:
            diffs_x.append((xi[bpm_ord] - bx[bpm_ord]).to_numpy())
            diffs_y.append((yi[bpm_ord] - by[bpm_ord]).to_numpy())
    diffs_x = np.stack(diffs_x, axis=0)
    diffs_y = np.stack(diffs_y, axis=0)

    std_x_ic = np.std(diffs_x, axis=0)
    std_y_ic = np.std(diffs_y, axis=0)

    fig_ic = plot_error_bars_bpm_range(
        s_positions, baseline_x, std_x_ic, baseline_y, std_y_ic, MAGNET_RANGE
    )
    fig_ic.suptitle("IC Perturbation Error Bars", fontsize=14)
    fig_ic.savefig("plots/errorbar_comparison_ic.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Compute standard deviations for combined errors over all angles
    diffs_x = []
    diffs_y = []
    for angle in angles:
        bx, by = baseline[angle]
        for xi, yi in results_combined[angle]:
            diffs_x.append((xi[bpm_ord] - bx[bpm_ord]).to_numpy())
            diffs_y.append((yi[bpm_ord] - by[bpm_ord]).to_numpy())
    diffs_x = np.stack(diffs_x, axis=0)
    diffs_y = np.stack(diffs_y, axis=0)

    std_x_combined = np.std(diffs_x, axis=0)
    std_y_combined = np.std(diffs_y, axis=0)

    fig_combined = plot_error_bars_bpm_range(
        s_positions,
        baseline_x,
        std_x_combined,
        baseline_y,
        std_y_combined,
        MAGNET_RANGE,
    )
    fig_combined.suptitle("Combined Errors (Quadrupole + IC) Error Bars", fontsize=14)
    fig_combined.savefig(
        "plots/errorbar_comparison_combined.png", dpi=300, bbox_inches="tight"
    )
    # ----------------------------
    # Plot standard deviations on a logarithmic scale
    plot_std_log_comparison(
        s_positions,
        std_x_quaderr,
        std_y_quaderr,
        std_x_ic,
        std_y_ic,
        std_x_combined,
        std_y_combined,
    )

    show_plots()


if __name__ == "__main__":
    main()
