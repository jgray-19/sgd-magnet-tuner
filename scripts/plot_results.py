#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_deltap_file(filepath):
    """Parse a results file into arrays and summary statistics."""
    arcs: list[int] = []
    arc_deltaps: list[float] = []
    mean_arcs: float | None = None
    std_arcs: float | None = None
    stderr_arcs: float | None = None

    if "0.txt" in filepath.name:
        print(f"Reading base file: {filepath}")

    with Path(filepath).open() as f:
        for line in f.readlines()[1:]:  # Skip header
            key, *rest = line.strip().split("\t")
            if len(rest) != 1:
                continue
            value = rest[0]
            if key.startswith("arc"):
                arc_num = int(key[3:])
                arcs.append(arc_num)
                arc_deltaps.append(float(value))
            elif key == "MeanArcs":
                mean_arcs = float(value)
            elif key == "StdDevArcs":
                std_arcs = float(value)
            elif key == "StdErrArcs":
                stderr_arcs = float(value)

    return {
        "arcs": np.array(arcs),
        "arc_deltaps": np.array(arc_deltaps),
        "mean_arcs": mean_arcs,
        "std_arcs": std_arcs,
        "stderr_arcs": stderr_arcs,
    }


EXPECTED_MAP = {
    "0": 0.0,
    "0p1": 0.1e-3,  # 0.1 per mil = 1e-4
    "0p2": 0.2e-3,
    "m0p1": -0.1e-3,
    "m0p2": -0.2e-3,
}


@dataclass
class BeamData:
    beam: int
    data: dict  # key: file, value: dict with arcs and summary stats


co = "co"  # Set to "" to use non closed orbit optimisation


def get_beam_data(beam: int) -> BeamData:
    folder = Path(f"b{beam}{co}_results")
    data = {}
    for file_key in EXPECTED_MAP:
        filepath = folder / f"{file_key}.txt"
        parsed = read_deltap_file(filepath)
        parsed["expected"] = EXPECTED_MAP[file_key]
        data[file_key] = parsed
    return BeamData(beam=beam, data=data)


def plot_all_deltap_vs_range(beam_data: BeamData):
    """Plot deltap vs arc range across all result files."""
    plt.figure(figsize=(10, 6))
    for file_key, d in beam_data.data.items():
        # Plot arcs
        if len(d["arcs"]) > 0:
            plt.plot(
                d["arcs"],
                d["arc_deltaps"],
                "o-",
                label=f"{file_key} arcs (exp: {d['expected'] * 1e5:.1f})",
            )
    plt.xlabel("Range")
    plt.ylabel(r"Deltap ($\times 10^{-5}$)")
    plt.title(f"Deltap vs Range for Beam {beam_data.beam} - All Files")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(f"b{beam_data.beam}{co}_results/deltap_all_beam{beam_data.beam}.png")
    plt.show()


def plot_difference_vs_arc(beam_data: BeamData):
    """Plot differences (file - 0) vs arc for all files in subplots."""
    files = [k for k in beam_data.data if k != "0"]
    n_files = len(files)
    if n_files == 0:
        return
    fig, axes = plt.subplots(n_files, 1, figsize=(10, 6 * n_files), sharex=True)
    if n_files == 1:
        axes = [axes]  # Make it a list for consistency
    base_deltaps = beam_data.data["0"]["arc_deltaps"]
    if len(base_deltaps) == 0:
        return
    for i, file_key in enumerate(files):
        ax = axes[i]
        diff_deltaps = beam_data.data[file_key]["arc_deltaps"]
        diffs = diff_deltaps - base_deltaps
        expected_diff = (
            beam_data.data[file_key]["expected"] - beam_data.data["0"]["expected"]
        )
        n = len(diffs)
        if n == 0:
            continue
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        ci = 1.96 * std_diff / np.sqrt(n) if n > 1 else 0.0
        ax.plot(
            beam_data.data["0"]["arcs"],
            diffs,
            "o-",
            color="blue",
            label=f"{file_key} Diff",
        )
        ax.axhline(
            y=mean_diff,
            linestyle="--",
            color="orange",
            alpha=0.7,
            label=f"Mean: {mean_diff * 1e5:.1f}",
        )
        ax.fill_between(
            beam_data.data["0"]["arcs"],
            mean_diff - ci,
            mean_diff + ci,
            alpha=0.2,
            color="orange",
            label="95% CI",
        )
        ax.axhline(
            y=expected_diff,
            linestyle=":",
            color="purple",
            alpha=0.7,
            label=f"Expected: {expected_diff * 1e5:.1f}",
        )
        # Highlight the average offset across arcs when available
        if None not in (
            beam_data.data[file_key]["mean_arcs"],
            beam_data.data["0"]["mean_arcs"],
        ):
            mean_arcs_diff = (
                beam_data.data[file_key]["mean_arcs"] - beam_data.data["0"]["mean_arcs"]
            )
            ax.axhline(
                y=mean_arcs_diff,
                linestyle="-",
                color="green",
                alpha=0.7,
                label=f"Mean Arcs Diff: {mean_arcs_diff * 1e5:.1f}",
            )
        ax.set_ylabel(r"Deltap Difference ($\times 10^{-5}$)")
        ax.set_title(
            f"Deltap Difference ({file_key} - 0) vs Arc for Beam {beam_data.beam}\nStd Dev: {std_diff:.2e}"
        )
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}")
        )
        ax.legend()
        ax.grid(visible=True)
    axes[-1].set_xlabel("Arc")
    plt.tight_layout()
    plt.savefig(
        f"b{beam_data.beam}{co}_results/deltap_diffs_all_beam{beam_data.beam}.png"
    )
    plt.show()


def plot_expected_vs_measured_mean(beam_data: BeamData):
    """Scatter plot of expected vs measured weighted mean deltap."""
    plt.figure(figsize=(8, 6))

    expecteds = []
    means = []
    stderr = []
    for parsed in beam_data.data.values():
        if parsed["mean_arcs"] is None:
            continue
        expecteds.append(parsed["expected"])
        means.append(parsed["mean_arcs"])
        stderr.append(parsed["stderr_arcs"] or 0.0)

    if not means:
        return

    plt.errorbar(
        expecteds,
        means,
        yerr=stderr,
        fmt="o",
        capsize=5,
        markersize=8,
        color="blue",
        label="Measured",
    )
    if len(means) > 1:
        fit = np.polyfit(expecteds, means, 1)
        fit_line = np.poly1d(fit)
        x_fit = np.linspace(min(expecteds), max(expecteds), 100)
        plt.plot(
            x_fit,
            fit_line(x_fit),
            "b-",
            label=f"Fit: y = {fit[0]:.2e}x + {fit[1]:.2e}",
        )

    plt.xlabel(r"Expected Deltap ($\times 10^{-5}$)")
    plt.ylabel(r"Measured Mean Deltap ($\times 10^{-5}$)")
    plt.title(f"Expected vs Measured Mean Deltap - Beam {beam_data.beam}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(
        f"b{beam_data.beam}{co}_results/expected_vs_measured_beam{beam_data.beam}.png"
    )
    plt.show()


def plot_difference_vs_measured_difference(beam_data: BeamData):
    """Plot expected difference vs measured difference for all pairwise file differences."""
    plt.figure(figsize=(8, 6))
    expected_diffs = []
    measured_diffs = []
    for file_i in beam_data.data:
        for file_j in beam_data.data:
            if file_i != file_j:
                exp_diff = (
                    beam_data.data[file_i]["expected"]
                    - beam_data.data[file_j]["expected"]
                )
                meas_diff = (beam_data.data[file_i]["mean_arcs"] or 0.0) - (
                    beam_data.data[file_j]["mean_arcs"] or 0.0
                )
                expected_diffs.append(exp_diff)
                measured_diffs.append(meas_diff)
    if len(expected_diffs) < 2:
        return
    # Linear fit
    fit = np.polyfit(expected_diffs, measured_diffs, 1)
    fit_line = np.poly1d(fit)
    x_fit = np.linspace(min(expected_diffs), max(expected_diffs), 100)
    y_fit = fit_line(x_fit)
    # Calculate R-squared
    y_pred = fit_line(np.array(expected_diffs))
    ss_res = np.sum((np.array(measured_diffs) - y_pred) ** 2)
    ss_tot = np.sum((np.array(measured_diffs) - np.mean(measured_diffs)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    plt.scatter(
        expected_diffs, measured_diffs, marker="x", s=30, label="Pairwise differences"
    )
    plt.plot(
        x_fit,
        y_fit,
        "r-",
        label=f"Fit: y = {fit[0]:.2e}x + {fit[1]:.2e}, R² - 1 = {r_squared - 1:.2e}",
    )
    plt.xlabel(r"Expected Difference ($\times 10^{-5}$)")
    plt.ylabel(r"Measured Difference ($\times 10^{-5}$)")
    plt.title(f"All Pairwise Differences: Expected vs Measured - Beam {beam_data.beam}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(
        f"b{beam_data.beam}{co}_results/all_pairwise_diffs_beam{beam_data.beam}.png"
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot deltap results for beam 1 or 2.")
    parser.add_argument(
        "beam", type=int, choices=[1, 2], nargs="?", help="Beam number (1 or 2)"
    )
    parser.add_argument("--both", action="store_true", help="Plot for both beams")
    args = parser.parse_args()

    if args.both:
        for beam in [1, 2]:
            beam_data = get_beam_data(beam)
            plot_all_deltap_vs_range(beam_data)
            plot_difference_vs_arc(beam_data)
            plot_expected_vs_measured_mean(beam_data)
            plot_difference_vs_measured_difference(beam_data)
        return

    if args.beam is None:
        parser.error("Beam number required unless --both is used")

    beam_data = get_beam_data(args.beam)
    print(beam_data)
    plot_all_deltap_vs_range(beam_data)
    plot_difference_vs_arc(beam_data)
    plot_expected_vs_measured_mean(beam_data)
    plot_difference_vs_measured_difference(beam_data)


if __name__ == "__main__":
    main()
