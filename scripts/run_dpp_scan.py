#!/usr/bin/env python3
"""Run a dpp scan by invoking the local create_a34 workflow and the energy optimiser,
collect results and plot input vs output similar to the provided figure.

This script is intentionally defensive: it imports the `create_a34` function from
the local `scripts/create_a34.py` using importlib so it can set the expected
module-level `args` used inside that function.

Usage: python scripts/run_dpp_scan.py
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

from aba_optimiser.config import (
    REL_K1_STD_DEV,
)  # Import here to ensure consistent value

logger = logging.getLogger("run_dpp_scan")


def plot_results(x: list[float], y: list[float], yerr: list[float], out_path: Path):
    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)

    # scale for display in 1e-4 units as in provided plot
    scale = 1e4
    # Increase font sizes globally for this figure
    font_size = 16
    with plt.rc_context(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": int(font_size * 0.9),
            "ytick.labelsize": int(font_size * 0.9),
            "legend.fontsize": int(font_size * 0.9),
        }
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.grid(which="both", linestyle="-", alpha=0.4)

        # Ideal line: y = x
        xlims = (float(np.nanmin(x)), float(np.nanmax(x)))
        xs = np.linspace(xlims[0], xlims[1], 100)
        ax.plot(
            xs * scale,
            xs * scale,
            linestyle="--",
            color="#ff8c00",
            label="Ideal (Input dpp = Calculated dpp)",
        )

        # Plot measured points with error bars
        ax.errorbar(
            x * scale,
            y * scale,
            yerr=yerr * scale,
            fmt="s",
            color="#1f77b4",
            label="Calculated dpp from DLMN",
            elinewidth=2.0,
            capsize=5,
            markersize=6,
        )

        ax.set_ylim(-3.0, 3.0)

        ax.set_xlabel(r"$\mathrm{Input}\;\Delta p/p\; (\times 10^{-4})$")
        ax.set_ylabel(r"$\mathrm{Calculated}\;\Delta p/p\; (\times 10^{-4})$")
        ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved dpp scan plot to {out_path}")


MAGNET_RANGES = [
    "BPM.11R1.B1/BPM.11L2.B1",
    "BPM.11R2.B1/BPM.11L3.B1",
    "BPM.11R3.B1/BPM.11L4.B1",
    "BPM.11R4.B1/BPM.11L5.B1",
    "BPM.11R5.B1/BPM.11L6.B1",
    "BPM.11R6.B1/BPM.11L7.B1",
    "BPM.11R7.B1/BPM.11L8.B1",
    "BPM.11R8.B1/BPM.11L1.B1",
]

BPM_STARTS = [
    "BPM.11R1.B1",
    "BPM.11R2.B1",
    "BPM.10R3.B1",
    "BPM.11R4.B1",
    "BPM.10R5.B1",
    "BPM.11R6.B1",
    "BPM.10R7.B1",
    "BPM.11R8.B1",
]


def main():
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
        "--pickle",
        default="runs/dpp_scan_results.pickle",
        help="Path to pickle file to read when --from-pickle is used",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # User-specified scan: n points between -2e-4 and 2e-4 (inclusive)
    inputs = np.linspace(-2e-4, 2e-4, 11).tolist()

    # Defaults requested by user
    flattop_turns = 6_600
    kick_both_planes = True
    num_tracks = 5
    track_batch_size = 1
    use_xsuite = True

    results_vals: list[float] = []
    results_errs: list[float] = []

    # If user requested to load from pickle, do that and plot and exit
    if args.from_pickle:
        p = Path(args.pickle)
        if not p.exists():
            logger.error("Pickle file %s not found", p)
            return

        # Load and normalize records into lists of (input, mean, std)
        try:
            with p.open("rb") as f:
                raw = pickle.load(f)
        except Exception:
            logger.exception("Failed to read pickle file %s", p)
            return

        # raw is expected to be a list of records; handle single-record fallback
        records = [raw] if isinstance(raw, dict) else list(raw)

        points = []
        for rec in records:
            try:
                inp = float(rec.get("input", rec.get("dpp", float("nan"))))
            except (TypeError, ValueError, AttributeError):
                inp = float("nan")

            # Prefer stored mean/std if present
            # mean = rec.get("extracted")[0]
            # std = rec.get("extracted")[1]
            mean = rec.get("mean")
            std = rec.get("std")
            print(mean, std, rec.get("error"))

            # Fallback ensure numeric
            try:
                mean = float(mean)
            except (TypeError, ValueError):
                mean = float("nan")
            try:
                std = float(std)
            except (TypeError, ValueError):
                std = float("nan")

            points.append((inp, mean, std))

        # Sort by input
        points.sort(key=lambda t: t[0])
        xs = [p[0] for p in points]
        means = [p[1] for p in points]
        stds = [p[2] for p in points]

        plot_results(xs, means, stds, Path(args.out))
        # print out the mean of ALL the results and the stddev
        return

    # Import optimiser/controller for actual run
    try:
        from create_a34 import create_a34

        from aba_optimiser.config import DPP_OPT_SETTINGS
        from aba_optimiser.training.controller import Controller
    except Exception:
        logger.exception(
            "Failed to import aba_optimiser modules; ensure the environment has the package installed"
        )
        return

    # Number of repeats per input to compute mean and std
    # Prepare fresh results container and output path — overwrite when complete
    out_raw = Path("runs") / "dpp_scan_results.pickle"
    out_raw.parent.mkdir(exist_ok=True, parents=True)
    raw_results: list[dict] = []

    for idx, machine_deltap in enumerate(inputs):
        logger.info(
            f"Running scan {idx + 1}/{len(inputs)}: machine_deltap={machine_deltap}"
        )

        per_repeat_vals: list[float] = []
        per_repeat_uncs: list[float] = []
        per_repeat_raw: list[dict] = []

        # Call create_a34 for this input to generate the machine with errors
        max_retries = 3
        create_a34_success = False

        for attempt in range(max_retries):
            try:
                create_a34(
                    flattop_turns,
                    kick_both_planes,
                    machine_deltap,
                    num_tracks,
                    REL_K1_STD_DEV,
                    track_batch_size,
                    use_xsuite,
                )
                create_a34_success = True
                break
            except (RuntimeError, OSError):
                logger.warning(
                    "create_a34 failed for machine_deltap=%s (attempt %d/%d)",
                    machine_deltap,
                    attempt + 1,
                    max_retries,
                )
                if attempt == max_retries - 1:
                    logger.exception(
                        "create_a34 failed after %d attempts for machine_deltap=%s",
                        max_retries,
                        machine_deltap,
                    )

        if not create_a34_success:
            # record a failed run as NaNs
            per_repeat_vals.append(float("nan"))
            per_repeat_uncs.append(float("nan"))
            per_repeat_raw.append({"error": "create_a34_failed_after_retries"})
            raise RuntimeError("create_a34 failed, aborting scan")
        if create_a34_success:
            for rep, mrange in enumerate(MAGNET_RANGES):
                logger.info(
                    f"  Repeat {rep + 1}/{len(MAGNET_RANGES)} for input dpp={machine_deltap}"
                )
                # Run the energy optimiser to extract resultant energy knob for this repeat
                print(rep, mrange)
                try:
                    energy_controller = Controller(
                        DPP_OPT_SETTINGS,
                        show_plots=False,
                        machine_deltap=machine_deltap,
                        magnet_range=mrange,
                        bpm_start_points=[BPM_STARTS[rep]],
                    )
                    energy_res, uncertainties = energy_controller.run()
                except AssertionError:
                    logger.exception(
                        "Energy optimiser failed for machine_deltap=%s (repeat %s)",
                        machine_deltap,
                        rep + 1,
                    )
                    per_repeat_vals.append(float("nan"))
                    per_repeat_uncs.append(float("nan"))
                    per_repeat_raw.append({"error": "energy_opt_failed"})
                    continue

                # Extract deltap and its reported uncertainty
                try:
                    val = float(energy_res["deltap"])
                except (KeyError, TypeError, ValueError):
                    val = float("nan")
                try:
                    unc = float(uncertainties.get("deltap", float("nan")))
                except (TypeError, ValueError):
                    unc = float("nan")

                per_repeat_vals.append(val)
                per_repeat_uncs.append(unc)
                per_repeat_raw.append(
                    {"result": energy_res, "uncertainties": uncertainties}
                )

            # Compute mean and std across repeats, ignoring NaNs
            arr = np.asarray(per_repeat_vals, dtype=float)
            valid_mask = ~np.isnan(arr)
            if valid_mask.sum() >= 1:
                mean_val = float(np.nanmean(arr))
                # use sample std when more than 1 valid sample, else 0.0
                std_val = float(np.nanstd(arr, ddof=1)) if valid_mask.sum() > 1 else 0.0
            else:
                mean_val = float("nan")
                std_val = float("nan")

            # Accumulate raw results in-memory; we'll overwrite the pickle after the full run
            raw_results.append(
                {
                    "input": machine_deltap,
                    "per_repeat": per_repeat_raw,
                    "per_repeat_extracted": per_repeat_vals,
                    "per_repeat_unc": per_repeat_uncs,
                    "mean": mean_val,
                    "std": std_val,
                }
            )
            results_vals.append(mean_val)
            results_errs.append(std_val)

    # Plot collected results
    # Write complete pickle (overwrite)
    try:
        with out_raw.open("wb") as f:
            pickle.dump(raw_results, f)
        logger.info("Wrote full results pickle to %s", out_raw)
    except Exception:
        logger.exception("Failed to write results pickle to %s", out_raw)

    plot_results(inputs, results_vals, results_errs, Path(args.out))


if __name__ == "__main__":
    main()
