#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_deltap_file(filepath):
    """Read deltap results file and return arcs and deltap values."""
    arcs = []
    deltaps = []
    with Path(filepath).open() as f:
        lines = f.readlines()
        for line in lines[1:-2]:  # Skip header and Mean/StdDev lines
            parts = line.strip().split("\t")
            if len(parts) == 2:
                arc = int(parts[0])
                deltap = float(parts[1])
                arcs.append(arc)
                deltaps.append(deltap)
    return np.array(arcs), np.array(deltaps)


BEAM_CONFIG = {
    1: {
        "normal": "analysis/deltap_results.txt",
        "trim": "analysis_trim/deltap_results.txt",
        "expected": -3e-5,
        "color": "blue",
        "label": "Beam 1",
        "expected_color": "purple",
        "mean_color": "orange",
    },
    2: {
        "normal": "analysis_b2/deltap_results.txt",
        "trim": "analysis_trim_b2/deltap_results.txt",
        "expected": 4e-5,
        "color": "red",
        "label": "Beam 2",
        "expected_color": "green",
        "mean_color": "red",
    },
}


@dataclass
class BeamData:
    arcs: np.ndarray
    normal: np.ndarray
    trim: np.ndarray
    diffs: np.ndarray
    mean_diff: float
    ci: float
    expected: float
    color: str
    label: str
    mean_color: str
    expected_color: str


def get_beam_data(beam: int) -> BeamData:
    config = BEAM_CONFIG[beam]
    arcs1, deltaps1 = read_deltap_file(config["normal"])
    arcs2, deltaps2 = read_deltap_file(config["trim"])
    min_len = min(len(arcs1), len(arcs2))
    arcs = arcs1[:min_len]
    normal = deltaps1[:min_len]
    trim = deltaps2[:min_len]
    diffs = trim - normal
    std_diff = np.std(diffs)
    mean_diff = np.mean(diffs)
    n = len(diffs)
    ci = 1.96 * std_diff / np.sqrt(n)
    return BeamData(
        arcs=arcs,
        normal=normal,
        trim=trim,
        diffs=diffs,
        mean_diff=mean_diff,
        ci=ci,
        expected=config["expected"],
        color=config["color"],
        label=config["label"],
        mean_color=config["mean_color"],
        expected_color=config["expected_color"],
    )


def plot_differences(beam_data: BeamData, ax=None):
    """Plot differences with mean, CI, and expected lines."""
    if ax is None:
        ax = plt.gca()
    ax.plot(
        beam_data.arcs,
        beam_data.diffs,
        "o-",
        color=beam_data.color,
        label=f"{beam_data.label} Diff",
    )
    ax.axhline(
        y=beam_data.mean_diff,
        linestyle="--",
        color=beam_data.mean_color,
        alpha=0.7,
        label=f"{beam_data.label} Mean: {beam_data.mean_diff * 1e5:.1f}",
    )
    ax.fill_between(
        beam_data.arcs,
        beam_data.mean_diff - beam_data.ci,
        beam_data.mean_diff + beam_data.ci,
        alpha=0.2,
        color=beam_data.mean_color,
        label=f"{beam_data.label} 95% CI",
    )
    ax.axhline(
        y=beam_data.expected,
        linestyle=":",
        color=beam_data.expected_color,
        alpha=0.7,
        label=f"{beam_data.label} Expected: {beam_data.expected * 1e5:.1f}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot deltap differences for beam 1 or 2."
    )
    parser.add_argument(
        "beam", type=int, choices=[1, 2], nargs="?", help="Beam number (1 or 2)"
    )
    parser.add_argument(
        "--both", action="store_true", help="Plot both beams on the same plot"
    )
    args = parser.parse_args()

    if args.both:
        plt.figure(figsize=(10, 6))
        for beam in [1, 2]:
            beam_data = get_beam_data(beam)

            plot_differences(beam_data)

        plt.xlabel("Arc")
        plt.ylabel(r"Deltap Difference ($\times 10^{-5}$)")
        plt.title("Deltap Difference vs Arc for Both Beams")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}")
        )
        plt.legend()
        plt.grid(visible=True)
        plt.savefig("deltap_difference_both.png")
        plt.show()
        return

    if args.beam is None:
        parser.error("Beam number required unless --both is used")

    beam_data = get_beam_data(args.beam)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(beam_data.arcs, beam_data.normal, "o-", label="Normal", color="blue")
    plt.plot(beam_data.arcs, beam_data.trim, "s-", label="Trim", color="red")
    plt.xlabel("Arc")
    plt.ylabel(r"Deltap ($\times 10^{-5}$)")
    plt.title(f"Deltap vs Arc for {beam_data.label}")
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.grid(visible=True)
    plt.savefig(f"deltap_comparison_beam{args.beam}.png")

    # Plot differences
    plt.figure(figsize=(10, 6))
    plot_differences(beam_data)
    plt.xlabel("Arc")
    plt.ylabel(r"Deltap Difference ($\times 10^{-5}$)")
    plt.title(
        f"Deltap Difference vs Arc for Beam {args.beam}\nStd Dev: {np.std(beam_data.diffs):.2e}"
    )
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(f"deltap_difference_beam{args.beam}.png")
    plt.show()


if __name__ == "__main__":
    main()
