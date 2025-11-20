#!/usr/bin/env python3
"""Run a dpp scan by invoking the local create_a34 workflow and the energy optimiser,
collect results and plot input vs output similar to the provided figure.

This script is intentionally defensive: it imports the `create_a34` function from
the local `scripts/create_a34.py` using importlib so it can set the expected
module-level `args` used inside that function.

Usage: python scripts/run_dpp_scan.py
"""

from __future__ import annotations

# Standard library imports
import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from aba_optimiser.config import REL_K1_STD_DEV

# Logger setup
logger = logging.getLogger("run_dpp_scan")

# Constants
SCALE_FACTOR = 1e4
FONT_SIZE = 16
Z_SCORE_THRESHOLD = 3
DEFAULT_FLATTOP_TURNS = 40_000
DEFAULT_NUM_TRACKS = 1
DEFAULT_TRACK_BATCH_SIZE = DEFAULT_NUM_TRACKS
MAX_RETRIES = 1
NUM_SCAN_POINTS = 3
SCAN_RANGE_MIN = -2e-4
SCAN_RANGE_MAX = 2e-4

# BPM and Magnet Configuration
# These define the ranges and points for the energy optimization scans
MAGNET_RANGES = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]

BPM_STARTS = [[f"BPM.{i}R{s}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]
BPM_END_POINTS = [[f"BPM.{i}L{s % 8 + 1}.B1" for i in [9, 10, 11, 12, 13]] for s in range(1, 9)]


@dataclass
class ScanResult:
    """Data class to hold results from a single dpp scan."""

    input_dpp: float
    per_repeat_raw_data: list[dict[str, Any]]
    per_repeat_extracted: list[float]
    per_repeat_unc: list[float]
    mean: float
    std: float


@dataclass
class ScanConfig:
    """Configuration for the dpp scan."""

    input_dpp_values: list[float]
    flattop_turns: int
    kick_both_planes: bool
    num_tracks: int
    rel_k1_std_dev: float
    track_batch_size: int
    use_xsuite: bool
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


def compute_weighted_mean_and_std(vals: list[float], uncs: list[float]) -> tuple[float, float]:
    """Compute weighted mean and standard deviation from values and uncertainties."""
    valid_indices = [
        i for i in range(len(vals)) if not (np.isnan(vals[i]) or np.isnan(uncs[i]) or uncs[i] <= 0)
    ]
    if not valid_indices:
        return float("nan"), float("nan")

    vals = [vals[i] for i in valid_indices]
    uncs = [uncs[i] for i in valid_indices]
    weights = [1.0 / (u**2) for u in uncs]
    sum_weights = sum(weights)
    weighted_mean = sum(v * w for v, w in zip(vals, weights)) / sum_weights
    # use std across arcs as the uncertainty
    std_val = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
    return float(weighted_mean), std_val


def compute_outlier_removed_stats(
    records: list[dict],
) -> tuple[list[float], list[float]]:
    """Perform z-score analysis on all data points and return outlier-removed means and stds.

    This function collects all per-repeat extracted values across all records,
    computes global z-scores to identify outliers, and then recomputes means and stds
    for each record excluding the outliers.
    """
    # Collect all valid data points: (record_index, value_index, value)
    all_points = []
    for record_idx, record in enumerate(records):
        extracted_values = record.get("per_repeat_extracted", [])
        for value_idx, value in enumerate(extracted_values):
            if not np.isnan(value):
                all_points.append((record_idx, value_idx, value))

    if not all_points:
        return [], []

    # Extract values for statistical analysis
    values = np.array([point[2] for point in all_points])
    global_mean = np.mean(values)
    global_std = np.std(values, ddof=1) if len(values) > 1 else 0

    # Compute z-scores and identify outliers
    z_scores = (values - global_mean) / global_std
    outlier_mask = np.abs(z_scores) > Z_SCORE_THRESHOLD
    logger.info(f"Found {np.sum(outlier_mask)} outliers out of {len(values)} total points")

    # Collect positions of outliers: set of (record_idx, value_idx)
    outlier_positions = set()
    for point_idx, (record_idx, value_idx, _) in enumerate(all_points):
        if outlier_mask[point_idx]:
            outlier_positions.add((record_idx, value_idx))

    # Recompute statistics per record, excluding outliers
    outlier_removed_means = []
    outlier_removed_stds = []
    for record_idx, record in enumerate(records):
        extracted_values = record.get("per_repeat_extracted", [])
        # Filter out NaN and outlier values
        valid_values = [
            value
            for value_idx, value in enumerate(extracted_values)
            if not np.isnan(value) and (record_idx, value_idx) not in outlier_positions
        ]
        if valid_values:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
        else:
            mean_val = np.nan
            std_val = np.nan
        outlier_removed_means.append(mean_val)
        outlier_removed_stds.append(std_val)

    return outlier_removed_means, outlier_removed_stds


def load_pickle_data(
    pickle_path: Path,
) -> tuple[list[float], list[float], list[float], list[dict]]:
    """Load and process data from pickle file.

    Returns:
        tuple: (inputs, means, stds, records) where records contain detailed data.
    """
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file {pickle_path} not found")

    try:
        with pickle_path.open("rb") as f:
            raw = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read pickle file {pickle_path}") from e

    records = [raw] if isinstance(raw, (dict, ScanResult)) else list(raw)

    dict_records = []
    points = []
    for rec in records:
        rec_dict = result_to_dict(rec) if isinstance(rec, ScanResult) else rec
        dict_records.append(rec_dict)

        try:
            inp = float(rec_dict.get("input", rec_dict.get("dpp", float("nan"))))
        except (TypeError, ValueError, AttributeError):
            inp = float("nan")

        mean = rec_dict.get("mean")
        std = rec_dict.get("std")

        # print every per_repeat value if available
        per = rec_dict.get("per_repeat")
        if per:
            for p in per:
                logger.info(f"Per repeat value: {p}")

        try:
            mean = float(mean)
        except (TypeError, ValueError):
            mean = float("nan")
        try:
            std = float(std)
        except (TypeError, ValueError):
            std = float("nan")

        points.append((inp, mean, std))

    points.sort(key=lambda t: t[0])
    xs = [p[0] for p in points]
    means = [p[1] for p in points]
    stds = [p[2] for p in points]

    return xs, means, stds, dict_records


def plot_results(x: list[float], y: list[float], yerr: list[float], out_path: Path):
    _plot_dpp_scan(x, y, yerr, out_path, "#1f77b4", "Calculated dpp from DLMN")


def plot_results_outlier_removed(x: list[float], y: list[float], yerr: list[float], out_path: Path):
    _plot_dpp_scan(x, y, yerr, out_path, "#dc143c", "Calculated dpp from DLMN (outliers removed)")


def _plot_dpp_scan(
    x: list[float],
    y: list[float],
    yerr: list[float],
    out_path: Path,
    marker_color: str,
    label: str,
):
    """Base plotting function for dpp scan results."""
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)

    with plt.rc_context(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": int(FONT_SIZE * 0.9),
            "ytick.labelsize": int(FONT_SIZE * 0.9),
            "legend.fontsize": int(FONT_SIZE * 0.9),
        }
    ):
        logging.info(f"Plotting dpp scan results to {out_path}")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.grid(which="both", linestyle="-", alpha=0.4)

        # Ideal line: y = x
        xlims = (float(np.nanmin(x)), float(np.nanmax(x)))
        xs = np.linspace(xlims[0], xlims[1], 100)
        ax.plot(
            xs * SCALE_FACTOR,
            xs * SCALE_FACTOR,
            linestyle="--",
            color="#ff8c00",
            label="Ideal (Input dpp = Calculated dpp)",
        )

        # Plot measured points with error bars
        ax.errorbar(
            x * SCALE_FACTOR,
            y * SCALE_FACTOR,
            yerr=yerr * SCALE_FACTOR,
            fmt="s",
            color=marker_color,
            label=label,
            elinewidth=2.0,
            capsize=5,
            markersize=6,
        )
        logger.info(f"Means: {y}")
        logger.info(f"Uncertainties: {yerr}")

        ax.set_ylim(-3.0, 3.0)
        ax.set_xlabel(r"$\mathrm{Input}\;\Delta p/p\; (\times 10^{-4})$")
        ax.set_ylabel(r"$\mathrm{Calculated}\;\Delta p/p\; (\times 10^{-4})$")
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved dpp scan plot to {out_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the dpp scan script."""
    parser = argparse.ArgumentParser(
        description="Run dpp scan using create_a34 and energy optimiser"
    )
    parser.add_argument("--out", default="plots/dpp_scan.png", help="Output plot path")
    parser.add_argument(
        "--from-pickle",
        action="store_true",
        help="Load results from runs/dpp_scan_results.pickle and plot without running any computation",
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help=(
            "Resume a long-running scan by loading any previously saved results from the pickle path. "
            "Only missing input points will be computed; a checkpoint is saved after each point."
        ),
    )
    parser.add_argument(
        "--pickle",
        default="runs/dpp_scan_results.pickle",
        help="Path to pickle file for reading/writing results and checkpoints",
    )
    return parser.parse_args()


def handle_pickle_mode(args: argparse.Namespace) -> None:
    """Handle loading and plotting from pickle file without running computations."""
    try:
        xs, means, stds, records = load_pickle_data(Path(args.pickle))
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return

    plot_results(xs, means, stds, Path(args.out))

    # Perform outlier analysis if data available
    if records and any("per_repeat_extracted" in rec for rec in records):
        outlier_removed_means, outlier_removed_stds = compute_outlier_removed_stats(records)
        if outlier_removed_means:
            outlier_plot_path = Path(args.out).parent / (
                Path(args.out).stem + "_outliers_removed.png"
            )
            plot_results_outlier_removed(
                xs, outlier_removed_means, outlier_removed_stds, outlier_plot_path
            )


def load_checkpoint(pickle_path: Path) -> dict[float, ScanResult]:
    """Load checkpoint data from pickle file and return a map of input_dpp to ScanResult."""
    prior_map: dict[float, ScanResult] = {}
    if pickle_path.exists():
        try:
            xs_ckpt, means_ckpt, stds_ckpt, recs_ckpt = load_pickle_data(pickle_path)
            # Reconstruct per-input records map for fast lookup
            for rec in recs_ckpt:
                key = float(rec.get("input", rec.get("dpp", float("nan"))))
                if not np.isnan(key):
                    prior_map[key] = ScanResult(
                        input_dpp=key,
                        per_repeat_raw_data=rec.get("per_repeat", []),
                        per_repeat_extracted=rec.get("per_repeat_extracted", []),
                        per_repeat_unc=rec.get("per_repeat_unc", []),
                        mean=rec.get("mean", float("nan")),
                        std=rec.get("std", float("nan")),
                    )
            logger.info("Loaded checkpoint with %d inputs from %s", len(prior_map), pickle_path)
        except (FileNotFoundError, RuntimeError):
            logger.exception("Failed to load checkpoint from %s; starting fresh", pickle_path)
    return prior_map


def run_single_dpp_scan(config: ScanConfig, input_dpp: float) -> ScanResult:
    """Run energy optimization for a single dpp value across multiple magnet ranges.

    Args:
        config: Scan configuration
        input_dpp: The delta p/p value to scan

    Returns:
        ScanResult with the results for this input dpp.
    """
    logger.info(f"Starting scan for input_dpp={input_dpp}")

    per_repeat_values: list[float] = []
    per_repeat_uncertainties: list[float] = []
    per_repeat_raw_data: list[dict] = []

    # Call create_a34 for this input to generate the machine with errors
    if not _run_create_a34(input_dpp, config):
        logger.error(f"create_a34 failed for input_dpp={input_dpp}, recording as NaN")
        return ScanResult(
            input_dpp=input_dpp,
            per_repeat_raw_data=[{"error": "create_a34_failed_after_retries"}],
            per_repeat_extracted=[float("nan")],
            per_repeat_unc=[float("nan")],
            mean=float("nan"),
            std=float("nan"),
        )

    # Run energy optimization for each magnet range
    for rep, magnet_range in enumerate(config.magnet_ranges):
        logger.debug(f"Repeat {rep + 1}/{len(config.magnet_ranges)} for input_dpp={input_dpp}")
        result = _run_energy_optimization(
            input_dpp, magnet_range, config.bpm_starts[rep], config.bpm_end_points[rep]
        )
        per_repeat_values.append(result["value"])
        per_repeat_uncertainties.append(result["uncertainty"])
        per_repeat_raw_data.append(result["raw_data"])

    # Compute weighted mean and uncertainty as std across arcs
    mean_val, std_val = compute_weighted_mean_and_std(per_repeat_values, per_repeat_uncertainties)

    logger.info(f"Completed scan for input_dpp={input_dpp}: mean={mean_val:.6e}, std={std_val:.6e}")
    return ScanResult(
        input_dpp=input_dpp,
        per_repeat_raw_data=per_repeat_raw_data,
        per_repeat_extracted=per_repeat_values,
        per_repeat_unc=per_repeat_uncertainties,
        mean=mean_val,
        std=std_val,
    )


def _run_create_a34(input_dpp: float, config: ScanConfig) -> bool:
    """Run create_a34 with retries. Returns True if successful."""
    for attempt in range(MAX_RETRIES):
        try:
            from create_a34 import create_a34

            create_a34(
                config.flattop_turns,
                config.kick_both_planes,
                input_dpp,
                config.num_tracks,
                config.rel_k1_std_dev,
                config.track_batch_size,
                config.use_xsuite,
            )
            return True
        except (RuntimeError, OSError, AssertionError) as e:
            logger.warning(
                "create_a34 failed for input_dpp=%s (attempt %d/%d): %s",
                input_dpp,
                attempt + 1,
                MAX_RETRIES,
                e,
            )
            if attempt == MAX_RETRIES - 1:
                logger.exception(
                    "create_a34 failed after %d attempts for input_dpp=%s",
                    MAX_RETRIES,
                    input_dpp,
                )
    return False


def _run_energy_optimization(
    input_dpp: float,
    magnet_range: str,
    bpm_starts: list[str],
    bpm_end_points: list[str],
) -> dict:
    """Run energy optimization for a single magnet range."""
    try:
        from aba_optimiser.config import DPP_OPTIMISER_CONFIG, DPP_SIMULATION_CONFIG
        from aba_optimiser.training.controller import Controller
        from aba_optimiser.training.controller_config import (
            BPMConfig,
            MeasurementConfig,
            SequenceConfig,
        )

        # Create config objects (note: this uses Controller directly, not LHCController)
        # You'll need to provide actual values for sequence_file_path, measurement_files, etc.
        sequence_config = SequenceConfig(
            sequence_file_path="path/to/sequence",  # TODO: Provide actual sequence file
            magnet_range=magnet_range,
        )

        measurement_config = MeasurementConfig(
            measurement_files="path/to/measurements",  # TODO: Provide actual measurement files
            machine_deltaps=input_dpp,
        )

        bpm_config = BPMConfig(
            start_points=bpm_starts,
            end_points=bpm_end_points,
        )

        energy_controller = Controller(
            optimiser_config=DPP_OPTIMISER_CONFIG,
            simulation_config=DPP_SIMULATION_CONFIG,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_config=bpm_config,
            show_plots=False,
        )
        energy_res, uncertainties = energy_controller.run()
        logger.debug("Energy optimiser result: deltap=%.6e", energy_res["deltap"])

        # Extract deltap and its reported uncertainty
        val = _safe_float_extract(energy_res.get("deltap"))
        unc = _safe_float_extract(uncertainties.get("deltap"))

        return {
            "value": val,
            "uncertainty": unc,
            "raw_data": {"result": energy_res, "uncertainties": uncertainties},
        }

    except AssertionError as e:
        logger.exception(
            "Energy optimiser failed for input_dpp=%s, magnet_range=%s",
            input_dpp,
            magnet_range,
        )
        return {
            "value": float("nan"),
            "uncertainty": float("nan"),
            "raw_data": {"error": "energy_opt_failed", "exception": str(e)},
        }


def _safe_float_extract(value: Any) -> float:
    """Safely extract a float value, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO)


def create_scan_config() -> ScanConfig:
    """Create the scan configuration."""
    input_dpp_values = np.linspace(SCAN_RANGE_MIN, SCAN_RANGE_MAX, NUM_SCAN_POINTS).tolist()

    return ScanConfig(
        input_dpp_values=input_dpp_values,
        flattop_turns=DEFAULT_FLATTOP_TURNS,
        kick_both_planes=True,
        num_tracks=DEFAULT_NUM_TRACKS,
        rel_k1_std_dev=REL_K1_STD_DEV,
        track_batch_size=DEFAULT_TRACK_BATCH_SIZE,
        use_xsuite=True,
        magnet_ranges=MAGNET_RANGES,
        bpm_starts=BPM_STARTS,
        bpm_end_points=BPM_END_POINTS,
    )


def log_scan_config(config: ScanConfig):
    """Log the scan configuration."""
    logger.info("Using BPM start points: %s", config.bpm_starts)
    logger.info("Using BPM end points: %s", config.bpm_end_points)
    logger.info("Using magnet ranges: %s", config.magnet_ranges)
    logger.info(
        "Scan range: %d points from %.1e to %.1e",
        len(config.input_dpp_values),
        SCAN_RANGE_MIN,
        SCAN_RANGE_MAX,
    )


def run_scan(args: argparse.Namespace, config: ScanConfig) -> list[ScanResult]:
    """Run the dpp scan, handling checkpoints and resuming."""
    pickle_path = Path(args.pickle)
    pickle_path.parent.mkdir(exist_ok=True, parents=True)

    # Load checkpoint if resuming
    prior_results = load_checkpoint(pickle_path) if args.from_checkpoint else {}

    results = []
    for idx, input_dpp in enumerate(config.input_dpp_values):
        logger.info(
            "Running scan %d/%d: input_dpp=%.6e",
            idx + 1,
            len(config.input_dpp_values),
            input_dpp,
        )

        # Check if we already have this result from checkpoint
        if args.from_checkpoint and input_dpp in prior_results:
            result = prior_results[input_dpp]
            logger.info("Using cached result for input_dpp=%.6e", input_dpp)
        else:
            result = run_single_dpp_scan(config, input_dpp)

        results.append(result)

        # Save checkpoint
        save_checkpoint(results, prior_results, pickle_path, args.from_checkpoint)

    # Merge with prior results if resuming
    if args.from_checkpoint and prior_results:
        final_results = merge_results_with_prior(results, prior_results)
    else:
        final_results = results

    return sorted(final_results, key=lambda r: r.input_dpp)


def merge_results_with_prior(
    results: list[ScanResult], prior_results: dict[float, ScanResult]
) -> list[ScanResult]:
    """Merge current results with prior checkpoint results."""
    combined = {r.input_dpp: r for r in results}
    for dpp, result in prior_results.items():
        combined.setdefault(dpp, result)
    return [combined[dpp] for dpp in sorted(combined.keys())]


def save_checkpoint(
    results: list[ScanResult],
    prior_results: dict[float, ScanResult],
    pickle_path: Path,
    from_checkpoint: bool,
):
    """Save current results as checkpoint."""
    try:
        combined = {r.input_dpp: r for r in results}
        if from_checkpoint:
            for dpp, result in prior_results.items():
                combined.setdefault(dpp, result)

        combined_list = [combined[dpp] for dpp in sorted(combined.keys())]
        with pickle_path.open("wb") as f:
            pickle.dump(combined_list, f)
        logger.info("Checkpoint saved to %s (%d points)", pickle_path, len(combined_list))
    except (OSError, pickle.PicklingError) as e:
        logger.exception("Failed to save checkpoint to %s: %s", pickle_path, e)


def process_results(results: list[ScanResult], args: argparse.Namespace):
    """Process results: compute outliers, save final data, and generate plots."""
    # Convert to dict format for compatibility
    raw_results = [result_to_dict(r) for r in results]

    # Perform outlier analysis
    outlier_removed_means, outlier_removed_stds = compute_outlier_removed_stats(raw_results)

    # Extract data for plotting
    xs = [r.input_dpp for r in results]
    means = [r.mean for r in results]
    stds = [r.std for r in results]

    # Save final results
    save_final_results(raw_results, Path(args.pickle))

    # Generate plots
    plot_results(xs, means, stds, Path(args.out))

    if outlier_removed_means:
        outlier_plot_path = Path(args.out).parent / (Path(args.out).stem + "_outliers_removed.png")
        plot_results_outlier_removed(
            xs, outlier_removed_means, outlier_removed_stds, outlier_plot_path
        )


def result_to_dict(result: ScanResult) -> dict:
    """Convert ScanResult to dictionary format for compatibility."""
    return {
        "input": result.input_dpp,
        "per_repeat": result.per_repeat_raw_data,
        "per_repeat_extracted": result.per_repeat_extracted,
        "per_repeat_unc": result.per_repeat_unc,
        "mean": result.mean,
        "std": result.std,
    }


def save_final_results(results: list[dict], pickle_path: Path):
    """Save final results to pickle file."""
    try:
        with pickle_path.open("wb") as f:
            pickle.dump(results, f)
        logger.info("Final results saved to %s", pickle_path)
    except (OSError, pickle.PicklingError) as e:
        logger.exception("Failed to save final results to %s: %s", pickle_path, e)


def main():
    """Main entry point for the dpp scan script."""
    args = parse_arguments()
    setup_logging()

    logger.info("Starting dpp scan script")

    # Handle pickle-only mode
    if args.from_pickle:
        handle_pickle_mode(args)
        return

    # Create scan configuration
    config = create_scan_config()
    log_scan_config(config)

    # Run the scan
    results = run_scan(args, config)

    # Process and save results
    process_results(results, args)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
