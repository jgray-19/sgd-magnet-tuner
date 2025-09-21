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
import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from tqdm.contrib.concurrent import process_map

from aba_optimiser.config import (
    BEAM_ENERGY,
    MAGNET_RANGE,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from aba_optimiser.mad.mad_interface import create_tracking_interface
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


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the noise effect simulation."""

    nturns: int = 500
    nangles: int = 8  # default number of initial angles
    num_error_samples: int = 30
    ic_xy_std: float = 1e-4  # Initial condition position standard deviation
    ic_pxpy_std: float = 3e-6  # Initial condition momentum standard deviation
    action0: float = 6e-9  # Initial action


class TrackingResult(NamedTuple):
    """Container for tracking simulation results."""

    x_positions: pd.Series
    y_positions: pd.Series


class NoiseAnalysisResults(NamedTuple):
    """Container for noise analysis results."""

    std_x: np.ndarray
    std_y: np.ndarray


def extract_base(name: str) -> str:
    """
    Reduce quadrupole name aliases to a common base.

    Args:
        name: Quadrupole element name

    Returns:
        Base name with reduced alias patterns
    """
    return re.sub(r"\.[AB](\d+[RL]\d\.B\d)$", r".\1", name)


def build_track_command(
    df_twiss: tfs.TfsDataFrame,
    action: float,
    angle: float,
    nturns: int,
    config: SimulationConfig,
    start_err: bool = False,
) -> str:
    """
    Construct the MAD-NG track command string with given initial conditions and settings.

    Args:
        df_twiss: Twiss dataframe containing optical functions
        action: Initial action value
        angle: Initial angle for action
        nturns: Number of tracking turns
        config: Simulation configuration parameters
        start_err: Whether to apply initial condition errors

    Returns:
        MAD-NG track command string
    """
    rng = np.random.default_rng()
    start_bpm = MAGNET_RANGE.split("/")[0]

    # Get optical functions at starting BPM
    beta11 = df_twiss.loc[start_bpm, "beta11"]
    beta22 = df_twiss.loc[start_bpm, "beta22"]
    alfa11 = df_twiss.loc[start_bpm, "alfa11"]
    alfa22 = df_twiss.loc[start_bpm, "alfa22"]

    # Calculate initial coordinates from action and angle
    cos0 = np.cos(angle)
    sin0 = np.sin(angle)
    x0 = np.sqrt(action * beta11) * cos0
    px0 = -np.sqrt(action / beta11) * (sin0 + alfa11 * cos0)
    y0 = np.sqrt(action * beta22) * cos0
    py0 = -np.sqrt(action / beta22) * (sin0 + alfa22 * cos0)

    # Apply random perturbations if requested
    if start_err:
        dx0, dy0 = rng.normal(0, config.ic_xy_std, size=2)
        dpx0, dpy0 = rng.normal(0, config.ic_pxpy_std, size=2)
    else:
        dx0, dpx0, dy0, dpy0 = 0.0, 0.0, 0.0, 0.0

    return (
        f"trk, mflw = track{{sequence=MADX['{SEQ_NAME}'], "
        f"X0={{x={x0 + dx0:.6e}, px={px0 + dpx0:.6e}, y={y0 + dy0:.6e}, py={py0 + dpy0:.6e}, t=0, pt=0}}, "
        f"nturn={nturns}}}"
    )


class MADSimulator:
    """Handles MAD-NG simulation setup and operations using consolidated interface."""

    def __init__(self):
        self._matched_tunes: dict[str, float] | None = None
        self._df_twiss: pd.DataFrame | None = None
        self._mad_interface = None

    def _get_interface(self):
        """Get or create the MAD interface."""
        if self._mad_interface is None:
            self._mad_interface = create_tracking_interface(
                SEQUENCE_FILE, SEQ_NAME, BEAM_ENERGY, MAGNET_RANGE, enable_logging=True
            )
        return self._mad_interface

    def match_tunes(self) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Match the working point tunes and return the matched knob values and Twiss data.

        Returns:
            Tuple of (matched tune parameters, Twiss dataframe)
        """
        if self._matched_tunes is not None and self._df_twiss is not None:
            return self._matched_tunes, self._df_twiss

        interface = self._get_interface()
        self._matched_tunes, self._df_twiss = interface.match_tunes()

        return self._matched_tunes, self._df_twiss

    def get_track_end_positions(
        self,
        matched_tunes: dict[str, float],
        df_twiss: pd.DataFrame,
        config: SimulationConfig,
        angle: float = 0.0,
        action: float | None = None,
        start_err: bool = False,
    ) -> TrackingResult:
        """
        Run tracking and return final positions at BPMs.

        Args:
            matched_tunes: Dictionary of matched tune parameters
            df_twiss: Twiss dataframe
            config: Simulation configuration
            angle: Initial angle
            action: Initial action (defaults to config.action0)
            start_err: Whether to apply initial condition errors

        Returns:
            TrackingResult with x and y positions
        """
        if action is None:
            action = config.action0

        interface = self._get_interface()
        x_pos, y_pos = interface.run_tracking(
            matched_tunes=matched_tunes,
            df_twiss=df_twiss,
            nturns=config.nturns,
            action=action,
            angle=angle,
            start_err=start_err,
            ic_xy_std=config.ic_xy_std,
            ic_pxpy_std=config.ic_pxpy_std,
        )
        return TrackingResult(x_pos, y_pos)


class NoiseAnalyzer:
    """Analyzes noise effects in beam tracking."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulator = MADSimulator()

    def compute_baseline(
        self, angle: float, matched_tunes: dict[str, float], df_twiss: pd.DataFrame
    ) -> tuple[float, pd.Series, pd.Series]:
        """
        Compute baseline tracking for a given angle.

        Args:
            angle: Initial angle
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe

        Returns:
            Tuple of (angle, x_positions, y_positions)
        """
        result = self.simulator.get_track_end_positions(
            matched_tunes, df_twiss, self.config, angle=angle
        )
        return angle, result.x_positions, result.y_positions

    def compute_error_sample(
        self,
        sample_idx: int,
        matched_tunes: dict[str, float],
        df_twiss: pd.DataFrame,
        angle: float,
        start_err: bool = False,
    ) -> TrackingResult:
        """
        Run tracking with random quadrupole errors and return positions.

        Args:
            sample_idx: Sample index (for reproducibility)
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe
            angle: Initial angle
            start_err: Whether to apply initial condition errors

        Returns:
            TrackingResult with x and y positions
        """
        # Create a new interface for this sample to avoid interference
        interface = create_tracking_interface(
            SEQUENCE_FILE, SEQ_NAME, BEAM_ENERGY, MAGNET_RANGE, enable_logging=False
        )

        # Get quadrupole names and apply errors
        quad_names = interface.get_quadrupole_names()
        interface.apply_magnet_errors(quad_names, REL_K1_STD_DEV, seed=sample_idx)

        # Run tracking
        x_pos, y_pos = interface.run_tracking(
            matched_tunes=matched_tunes,
            df_twiss=df_twiss,
            nturns=self.config.nturns,
            action=self.config.action0,
            angle=angle,
            start_err=start_err,
            ic_xy_std=self.config.ic_xy_std,
            ic_pxpy_std=self.config.ic_pxpy_std,
            seed=sample_idx,
        )
        return TrackingResult(x_pos, y_pos)

    def compute_ic_sample(
        self,
        sample_idx: int,
        matched_tunes: dict[str, float],
        df_twiss: pd.DataFrame,
        angle: float,
    ) -> TrackingResult:
        """
        Run tracking with initial condition perturbations and return positions.

        Args:
            sample_idx: Sample index
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe
            angle: Initial angle

        Returns:
            TrackingResult with x and y positions
        """
        return self.simulator.get_track_end_positions(
            matched_tunes, df_twiss, self.config, angle=angle, start_err=True
        )

    def compute_error_sample(
        self,
        sample_idx: int,
        matched_tunes: dict[str, float],
        df_twiss: pd.DataFrame,
        angle: float,
        start_err: bool = False,
    ) -> TrackingResult:
        """
        Run a single-turn track with random quadrupole errors and return positions.

        Args:
            sample_idx: Sample index (for reproducibility)
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe
            angle: Initial angle
            start_err: Whether to apply initial condition errors

        Returns:
            TrackingResult with x and y positions
        """
        rng = np.random.default_rng()
        mad = self.simulator.setup_mad()

        # Apply matched tunes
        for key, val in matched_tunes.items():
            mad[f"MADX['{key}']"] = val

        # Build quadrupole groups and apply errors
        seq = mad.MADX[SEQ_NAME]
        arc_start = df_twiss.loc[MAGNET_RANGE.split("/")[0], "s"]
        arc_end = df_twiss.loc[MAGNET_RANGE.split("/")[1], "s"]

        quad_names = [
            elm.name
            for elm in seq
            if elm.kind == "quadrupole"
            and elm.k1 != 0
            and "MQ." in elm.name
            and arc_start <= df_twiss.loc[elm.name, "s"] <= arc_end
        ]

        # Group quadrupoles by base name
        groups_all = {}
        for name in quad_names:
            base = extract_base(name)
            groups_all.setdefault(base, []).append(name)

        # Apply errors to each group
        for base, aliases in groups_all.items():
            noise = rng.normal(scale=REL_K1_STD_DEV)
            for name in aliases:
                k1 = mad[f"MADX['{name}'].k1"]
                mad[f"MADX['{name}'].k1"] = k1 + noise * abs(k1)

    def compute_standard_deviations(
        self,
        baseline: dict[float, tuple[pd.Series, pd.Series]],
        results: dict[float, list[TrackingResult]],
        bpm_names: list[str],
    ) -> NoiseAnalysisResults:
        """
        Compute standard deviations from baseline for all angles and samples.

        Args:
            baseline: Baseline results for each angle
            results: Noise results for each angle
            bpm_names: Ordered list of BPM names

        Returns:
            NoiseAnalysisResults with standard deviations
        """
        # Flatten all differences across angles and samples
        diffs_x = []
        diffs_y = []

        for angle in baseline:
            bx, by = baseline[angle]
            for result in results[angle]:
                diffs_x.append(
                    (result.x_positions[bpm_names] - bx[bpm_names]).to_numpy()
                )
                diffs_y.append(
                    (result.y_positions[bpm_names] - by[bpm_names]).to_numpy()
                )

        diffs_x = np.stack(diffs_x, axis=0)
        diffs_y = np.stack(diffs_y, axis=0)

        return NoiseAnalysisResults(
            std_x=np.std(diffs_x, axis=0), std_y=np.std(diffs_y, axis=0)
        )


# Multiprocessing wrapper functions
def _compute_baseline_wrapper(args):
    """Wrapper for multiprocessing baseline computation."""
    angle, matched_tunes, df_twiss, config = args
    analyzer = NoiseAnalyzer(config)
    return analyzer.compute_baseline(angle, matched_tunes, df_twiss)


def _compute_error_sample_wrapper(args):
    """Wrapper for multiprocessing error sample computation."""
    sample_idx, matched_tunes, df_twiss, angle, config, start_err = args
    analyzer = NoiseAnalyzer(config)
    result = analyzer.compute_error_sample(
        sample_idx, matched_tunes, df_twiss, angle, start_err
    )
    return (sample_idx, angle), result


def _compute_ic_sample_wrapper(args):
    """Wrapper for multiprocessing IC sample computation."""
    sample_idx, matched_tunes, df_twiss, angle, config = args
    analyzer = NoiseAnalyzer(config)
    result = analyzer.compute_ic_sample(sample_idx, matched_tunes, df_twiss, angle)
    return (sample_idx, angle), result


def save_error_bar_plot(
    s_positions: np.ndarray,
    baseline_x: pd.Series,
    baseline_y: pd.Series,
    std_x: np.ndarray,
    std_y: np.ndarray,
    title: str,
    filename: str,
) -> None:
    """
    Save an error bar plot with given parameters.

    Args:
        s_positions: BPM positions along the accelerator
        baseline_x: Baseline x positions
        baseline_y: Baseline y positions
        std_x: Standard deviation in x
        std_y: Standard deviation in y
        title: Plot title
        filename: Output filename
    """
    fig = plot_error_bars_bpm_range(
        s_positions, baseline_x, std_x, baseline_y, std_y, MAGNET_RANGE
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def main():
    """Main analysis function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze noise effects in beam tracking"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--nangles",
        type=int,
        default=8,
        help="Number of initial angles to sample",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=30,
        help="Number of error samples per angle",
    )
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logger.setLevel(logging.INFO)

    # Create configuration
    config = SimulationConfig(
        nangles=args.nangles,
        num_error_samples=args.nsamples,
    )

    logger.info(
        f"Starting analysis: nturns={config.nturns}, "
        f"nangles={config.nangles}, samples={config.num_error_samples}"
    )

    # Initialize analyzer and get baseline parameters
    analyzer = NoiseAnalyzer(config)
    matched_tunes, df_twiss = analyzer.simulator.match_tunes()

    # Sample angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, config.nangles, endpoint=False)

    # Compute baseline in parallel
    logger.info("Computing baseline tracks")
    baseline_args = [(angle, matched_tunes, df_twiss, config) for angle in angles]
    baseline_results = process_map(
        _compute_baseline_wrapper, baseline_args, desc="Baseline computation"
    )
    baseline = {ang: (xi, yi) for ang, xi, yi in baseline_results}

    # Use baseline of first angle for plotting reference
    baseline_x, baseline_y = baseline[angles[0]]
    bpm_names = baseline_x.index.tolist()

    # Get ordered BPM positions
    s_positions = df_twiss.loc[bpm_names, "s"].to_numpy()
    order = np.argsort(s_positions)
    bpm_names_ordered = [bpm_names[i] for i in order]
    s_positions = s_positions[order]

    # Compute quadrupole error samples in parallel
    logger.info("Computing quadrupole error samples")
    quad_tasks = list(product(range(config.num_error_samples), angles))
    quad_args = [
        (idx, matched_tunes, df_twiss, angle, config, False)
        for idx, angle in quad_tasks
    ]
    quad_flat = process_map(
        _compute_error_sample_wrapper, quad_args, desc="Quadrupole error samples"
    )

    # Reorganize results by angle
    results_quad = {angle: [] for angle in angles}
    for (idx, angle), result in quad_flat:
        results_quad[angle].append(result)

    # Compute IC perturbation samples in parallel
    logger.info("Computing IC perturbation samples")
    ic_args = [
        (idx, matched_tunes, df_twiss, angle, config) for idx, angle in quad_tasks
    ]
    ic_flat = process_map(
        _compute_ic_sample_wrapper, ic_args, desc="IC perturbation samples"
    )

    results_ic = {angle: [] for angle in angles}
    for (idx, angle), result in ic_flat:
        results_ic[angle].append(result)

    # Compute combined error samples in parallel
    logger.info("Computing combined error samples")
    combined_args = [
        (idx, matched_tunes, df_twiss, angle, config, True) for idx, angle in quad_tasks
    ]
    combined_flat = process_map(
        _compute_error_sample_wrapper, combined_args, desc="Combined error samples"
    )

    results_combined = {angle: [] for angle in angles}
    for (idx, angle), result in combined_flat:
        results_combined[angle].append(result)

    # Compute standard deviations for each noise type
    logger.info("Computing standard deviations")
    quad_analysis = analyzer.compute_standard_deviations(
        baseline, results_quad, bpm_names_ordered
    )
    ic_analysis = analyzer.compute_standard_deviations(
        baseline, results_ic, bpm_names_ordered
    )
    combined_analysis = analyzer.compute_standard_deviations(
        baseline, results_combined, bpm_names_ordered
    )

    # Generate and save plots
    logger.info("Generating plots")
    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        quad_analysis.std_x,
        quad_analysis.std_y,
        "Quadrupole Error Bars",
        "plots/error_bars_bpm_range.png",
    )

    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        ic_analysis.std_x,
        ic_analysis.std_y,
        "IC Perturbation Error Bars",
        "plots/errorbar_comparison_ic.png",
    )

    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        combined_analysis.std_x,
        combined_analysis.std_y,
        "Combined Errors (Quadrupole + IC) Error Bars",
        "plots/errorbar_comparison_combined.png",
    )  # Plot standard deviations on logarithmic scale
    plot_std_log_comparison(
        s_positions,
        quad_analysis.std_x,
        quad_analysis.std_y,
        ic_analysis.std_x,
        ic_analysis.std_y,
        combined_analysis.std_x,
        combined_analysis.std_y,
    )

    show_plots()
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
