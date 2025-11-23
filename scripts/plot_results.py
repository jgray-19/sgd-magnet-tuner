#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_deltap_file(filepath):
    """Parse a results file into arrays and summary statistics."""
    arcs: list[int] = []
    arc_deltaps: list[float] = []
    arc_uncertainties: list[float] = []
    mean_arcs: float | None = None
    std_arcs: float | None = None
    stderr_arcs: float | None = None

    if "0.txt" in filepath.name:
        print(f"Reading base file: {filepath}")

    with Path(filepath).open() as f:
        for line in f.readlines()[1:]:  # Skip header
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            # Handle both arc/ir entries (with optional uncertainty column)
            if key.startswith("arc") or key.startswith("ir"):
                range_num = int(key[2:] if key.startswith("ir") else key[3:])
                arcs.append(range_num)
                arc_deltaps.append(float(parts[1]))
                # Read uncertainty if present
                if len(parts) >= 3 and parts[2]:
                    arc_uncertainties.append(float(parts[2]))
            elif key == "MeanArcs" or key == "MeanIrs":
                mean_arcs = float(parts[1])
            elif key == "StdDevArcs" or key == "StdDevIrs":
                std_arcs = float(parts[1])
            elif key == "StdErrArcs" or key == "StdErrIrs":
                stderr_arcs = float(parts[1])

    return {
        "arcs": np.array(arcs),
        "arc_deltaps": np.array(arc_deltaps),
        "arc_uncertainties": np.array(arc_uncertainties) if arc_uncertainties else None,
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


BASE_COLORS = {
    "0": "blue",
    "0p1": "green",
    "0p2": "red",
    "m0p1": "orange",
    "m0p2": "purple",
}


READABLE_NAMES = {
    "0": "No shift",
    "0p1": "+0.1",
    "0p2": "+0.2",
    "m0p1": "-0.1",
    "m0p2": "-0.2",
}


@dataclass
class BeamData:
    beam: int
    data: dict  # key: file, value: dict with arcs and summary stats
    estimated_dispersion: dict  # key: file_key, value: estimated dpp from estimated dispersion
    estimated_dispersion_err: dict  # key: file_key, value: uncertainty for estimated dispersion
    model_dispersion: dict  # key: file_key, value: estimated dpp from model dispersion
    model_dispersion_err: dict  # key: file_key, value: uncertainty for model dispersion


def format_fit_label(
    beam: int,
    slope: float,
    intercept: float,
    extra: str = "",
    is_estimated: bool = False,
) -> str:
    sign = "+ " if intercept >= 0 else "- "
    est = " Corrector" if is_estimated else " DLMN"
    return f"Beam {beam}{est} Fit: y = {slope:.2e}x {sign}{abs(intercept):.2e}{extra}"


def collect_expected_measured_points(beam_data, dispersion_type="estimated"):
    """Collect expected and measured points for DLMN and corrector data.

    Args:
        beam_data: BeamData object
        dispersion_type: "estimated" or "model" - which dispersion calculation to use
    """
    dlmn_points = []
    corrector_points = []
    for parsed in beam_data.data.values():
        if parsed["mean_arcs"] is not None:
            dlmn_points.append(
                {
                    "expected": parsed["expected"],
                    "measured": parsed["mean_arcs"],
                    "stderr": parsed["stderr_arcs"] or 0.0,
                }
            )

    if dispersion_type == "estimated":
        estimated = beam_data.estimated_dispersion
        estimated_err = beam_data.estimated_dispersion_err
    elif dispersion_type == "model":
        estimated = beam_data.model_dispersion
        estimated_err = beam_data.model_dispersion_err
    else:
        raise ValueError(f"Invalid dispersion_type: {dispersion_type}")

    for file_key in estimated:
        corrector_points.append(
            {
                "expected": beam_data.data[file_key]["expected"],
                "measured": estimated[file_key],
                "stderr": estimated_err.get(file_key, 0.0),
            }
        )
    return dlmn_points, corrector_points


co = "co"  # Set to "" to use non closed orbit optimisation


def get_beam_data(beam: int) -> BeamData:
    """Load beam data from results files and pre-computed estimates from JSON.

    Args:
        beam: Beam number (1 or 2)

    Returns:
        BeamData object containing measurement results and estimated dpp values

    Raises:
        FileNotFoundError: If estimates JSON file is not found
    """
    folder = Path(f"b{beam}{co}_results")
    data = {}
    for file_key in EXPECTED_MAP:
        filepath = folder / f"{file_key}.txt"
        parsed = read_deltap_file(filepath)
        parsed["expected"] = EXPECTED_MAP[file_key]
        data[file_key] = parsed

    # Load estimates from JSON file (generated by estimate_dpp_from_correctors.py)
    estimates_file = folder / f"estimates_b{beam}.json"
    if not estimates_file.exists():
        raise FileNotFoundError(
            f"Estimates file {estimates_file} not found. "
            f"Run scripts/estimate_dpp_from_correctors.py first to generate it."
        )

    with estimates_file.open() as f:
        results = json.load(f)
    results = {float(k): v for k, v in results.items()}

    estimated_dispersion = {}
    estimated_dispersion_err = {}
    model_dispersion = {}
    model_dispersion_err = {}
    for file_key, parsed in data.items():
        per_mil = parsed["expected"] * 1000
        if per_mil in results:
            result = results[per_mil]
            # Handle new format with estimated_dispersion and model_dispersion
            if isinstance(result, dict) and "estimated_dispersion" in result:
                estimated_dispersion[file_key] = result["estimated_dispersion"]["value"]
                estimated_dispersion_err[file_key] = result["estimated_dispersion"]["uncertainty"]
                model_dispersion[file_key] = result["model_dispersion"]["value"]
                model_dispersion_err[file_key] = result["model_dispersion"]["uncertainty"]
            # Handle old format (backward compatibility)
            elif isinstance(result, dict) and "value" in result:
                estimated_dispersion[file_key] = result["value"]
                estimated_dispersion_err[file_key] = result["uncertainty"]
                model_dispersion[file_key] = result["value"]  # fallback
                model_dispersion_err[file_key] = result["uncertainty"]  # fallback
            else:
                estimated_dispersion[file_key] = result
                estimated_dispersion_err[file_key] = 0.0
                model_dispersion[file_key] = result  # fallback
                model_dispersion_err[file_key] = 0.0  # fallback

    return BeamData(
        beam=beam,
        data=data,
        estimated_dispersion=estimated_dispersion,
        estimated_dispersion_err=estimated_dispersion_err,
        model_dispersion=model_dispersion,
        model_dispersion_err=model_dispersion_err,
    )


def plot_all_deltap_vs_range(beam_data_list):
    """Plot deltap vs arc range across all result files."""
    if isinstance(beam_data_list, BeamData):
        beam_data_list = [beam_data_list]
    plt.figure(figsize=(10, 6))
    for beam_data in beam_data_list:
        for file_key, d in beam_data.data.items():
            # Plot arcs
            if len(d["arcs"]) > 0:
                base_color = BASE_COLORS[file_key]
                if beam_data.beam == 1:
                    color = base_color
                    linestyle = "-"
                    marker = "o"
                else:
                    rgb = mcolors.to_rgb(base_color)
                    color = tuple(min(1, c + 0.2) for c in rgb)
                    linestyle = "--"
                    marker = "s"
                plt.plot(
                    d["arcs"],
                    d["arc_deltaps"],
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                )
    plt.xlabel("Range")
    plt.ylabel(r"Deltap ($\times 10^{-5}$)")
    beam_str = f"Beam {beam_data_list[0].beam}" if len(beam_data_list) == 1 else "Beams 1 and 2"
    plt.title(f"Deltap vs Range for {beam_str} - All Files")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    # Custom legend
    handles = []
    labels = []
    for file_key in EXPECTED_MAP:
        color = BASE_COLORS[file_key]
        readable = READABLE_NAMES[file_key]
        handle = plt.Line2D([0], [0], color=color, linestyle="-", linewidth=2)
        handles.append(handle)
        labels.append(readable)
    # Add beam styles
    handle_b1 = plt.Line2D([0], [0], color="black", linestyle="-", marker="o", markersize=6)
    handles.append(handle_b1)
    labels.append("Beam 1")
    handle_b2 = plt.Line2D([0], [0], color="black", linestyle="--", marker="s", markersize=6)
    handles.append(handle_b2)
    labels.append("Beam 2")
    plt.legend(handles, labels, loc="best")
    plt.grid(visible=True)
    if len(beam_data_list) == 1:
        plt.savefig(
            f"b{beam_data_list[0].beam}{co}_results/deltap_all_beam{beam_data_list[0].beam}.png"
        )
    else:
        Path("combined_results").mkdir(parents=True, exist_ok=True)
        plt.savefig("combined_results/deltap_all_beams.png")
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
            label=f"{READABLE_NAMES[file_key]} Diff",
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
            f"Deltap Difference ({READABLE_NAMES[file_key]} - 0) vs Arc for Beam {beam_data.beam}\nStd Dev: {std_diff:.2e}"
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


def plot_expected_vs_measured_mean_common(
    beam_data_list, dispersion_type="estimated", include_fits=True
):
    """Common plotting logic for expected vs measured mean deltap plots."""
    if isinstance(beam_data_list, BeamData):
        beam_data_list = [beam_data_list]
    plt.figure(figsize=(8, 6))

    for beam_data in beam_data_list:
        dlmn_points, corrector_points = collect_expected_measured_points(beam_data, dispersion_type)

        if not dlmn_points:
            continue

        expecteds = [p["expected"] for p in dlmn_points]
        means = [p["measured"] for p in dlmn_points]
        stderr = [p["stderr"] for p in dlmn_points]

        color = "blue" if beam_data.beam == 1 else "red"
        plt.errorbar(
            expecteds,
            means,
            yerr=stderr,
            fmt="o",
            capsize=8,
            markersize=6,
            color=color,
            label=f"Beam {beam_data.beam} DLMN result",
        )

        if include_fits and len(means) > 1:
            fit = np.polyfit(expecteds, means, 1)
            fit_line = np.poly1d(fit)
            x_fit = np.linspace(min(expecteds), max(expecteds), 100)
            plt.plot(
                x_fit,
                fit_line(x_fit),
                color=color,
                linestyle="-",
                label=format_fit_label(beam_data.beam, fit[0], fit[1]),
            )

        if corrector_points:
            expecteds_est = [p["expected"] for p in corrector_points]
            means_est = [p["measured"] for p in corrector_points]
            stderr_est = [p["stderr"] for p in corrector_points]

            color_est = "blue" if beam_data.beam == 1 else "red"
            marker = "s"
            dispersion_label = "Estimated" if dispersion_type == "estimated" else "Model"
            plt.errorbar(
                expecteds_est,
                means_est,
                yerr=stderr_est,
                fmt=marker,
                capsize=8,
                markersize=6,
                color=color_est,
                label=f"Beam {beam_data.beam} Corrector calc ({dispersion_label})",
            )
            if include_fits and len(means_est) > 1:
                fit_est = np.polyfit(expecteds_est, means_est, 1)
                fit_line_est = np.poly1d(fit_est)
                x_fit_est = np.linspace(min(expecteds_est), max(expecteds_est), 100)
                plt.plot(
                    x_fit_est,
                    fit_line_est(x_fit_est),
                    color=color_est,
                    linestyle="--",
                    label=format_fit_label(
                        beam_data.beam, fit_est[0], fit_est[1], is_estimated=True
                    ),
                )

    if not include_fits:
        # Add reference line y = x
        all_expecteds = []
        for beam_data in beam_data_list:
            dlmn_points, corrector_points = collect_expected_measured_points(
                beam_data, dispersion_type
            )
            all_expecteds.extend([p["expected"] for p in dlmn_points + corrector_points])
        if all_expecteds:
            min_exp = min(all_expecteds)
            max_exp = max(all_expecteds)
            plt.plot([min_exp, max_exp], [min_exp, max_exp], "k--", label="Ideal: y = x")

    plt.xlabel(r"Expected Deltap ($\times 10^{-5}$)")
    plt.ylabel(r"Measured Mean Deltap ($\times 10^{-5}$)")
    beam_str = f"Beam {beam_data_list[0].beam}" if len(beam_data_list) == 1 else "Beams 1 and 2"
    dispersion_title = "Estimated" if dispersion_type == "estimated" else "Model"
    fit_suffix = "" if include_fits else ", No Fit"
    plt.title(
        f"Expected vs Measured Mean Deltap ({dispersion_title} Disp{fit_suffix}) - {beam_str}"
    )
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)

    if len(beam_data_list) == 1:
        dispersion_suffix = "_estimated" if dispersion_type == "estimated" else "_model"
        fit_suffix = "" if include_fits else "_no_fit"
        plt.savefig(
            f"b{beam_data_list[0].beam}{co}_results/expected_vs_measured{fit_suffix}{dispersion_suffix}_beam{beam_data_list[0].beam}.png"
        )
    else:
        Path("combined_results").mkdir(parents=True, exist_ok=True)
        dispersion_suffix = "_estimated" if dispersion_type == "estimated" else "_model"
        fit_suffix = "" if include_fits else "_no_fit"
        plt.savefig(
            f"combined_results/expected_vs_measured{fit_suffix}{dispersion_suffix}_beams.png"
        )
    plt.show()


def plot_expected_vs_measured_mean(beam_data_list, dispersion_type="estimated"):
    """Scatter plot of expected vs measured weighted mean deltap with fits."""
    plot_expected_vs_measured_mean_common(beam_data_list, dispersion_type, include_fits=True)


def plot_expected_vs_measured_mean_no_fit(beam_data_list, dispersion_type="estimated"):
    """Scatter plot of expected vs measured weighted mean deltap without fits."""
    plot_expected_vs_measured_mean_common(beam_data_list, dispersion_type, include_fits=False)


def plot_difference_vs_measured_difference(beam_data_list):
    """Plot expected difference vs measured difference for all pairwise file differences."""
    if isinstance(beam_data_list, BeamData):
        beam_data_list = [beam_data_list]
    plt.figure(figsize=(8, 6))
    for beam_data in beam_data_list:
        expected_diffs = []
        measured_diffs = []
        for file_i in beam_data.data:
            for file_j in beam_data.data:
                if file_i != file_j:
                    exp_diff = (
                        beam_data.data[file_i]["expected"] - beam_data.data[file_j]["expected"]
                    )
                    meas_diff = (beam_data.data[file_i]["mean_arcs"] or 0.0) - (
                        beam_data.data[file_j]["mean_arcs"] or 0.0
                    )
                    expected_diffs.append(exp_diff)
                    measured_diffs.append(meas_diff)
        if len(expected_diffs) < 2:
            continue
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
        color = "blue" if beam_data.beam == 1 else "red"
        plt.scatter(
            expected_diffs,
            measured_diffs,
            marker="x",
            s=30,
            color=color,
            label=f"Beam {beam_data.beam} Pairwise differences",
        )
        plt.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle="-",
            label=format_fit_label(
                beam_data.beam, fit[0], fit[1], extra=f", 1 - R² = {1 - r_squared:.2e}"
            ),
        )
    plt.xlabel(r"Expected Difference ($\times 10^{-5}$)")
    plt.ylabel(r"Measured Difference ($\times 10^{-5}$)")
    beam_str = f"Beam {beam_data_list[0].beam}" if len(beam_data_list) == 1 else "Beams 1 and 2"
    plt.title(f"All Pairwise Differences: Expected vs Measured - {beam_str}")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 1e5:.1f}"))
    plt.legend()
    plt.grid(visible=True)
    if len(beam_data_list) == 1:
        plt.savefig(
            f"b{beam_data_list[0].beam}{co}_results/all_pairwise_diffs_beam{beam_data_list[0].beam}.png"
        )
    else:
        Path("combined_results").mkdir(parents=True, exist_ok=True)
        plt.savefig("combined_results/all_pairwise_diffs_beams.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot deltap results for beam 1 or 2.")
    parser.add_argument(
        "beam", type=int, choices=[1, 2], nargs="?", help="Beam number (1 or 2)"
    )
    parser.add_argument("--both", action="store_true", help="Plot for both beams")
    parser.add_argument(
        "--dispersion",
        type=str,
        choices=["estimated", "model"],
        default="estimated",
        help="Which dispersion to use for corrector calculations (default: estimated)",
    )

    args = parser.parse_args()

    if args.both:
        beam_data_list = [
            get_beam_data(1),
            get_beam_data(2),
        ]
        plot_all_deltap_vs_range(beam_data_list)
        plot_expected_vs_measured_mean(beam_data_list, args.dispersion)
        plot_difference_vs_measured_difference(beam_data_list)
        return

    if args.beam is None:
        parser.error("Beam number required unless --both is used")

    beam_data = get_beam_data(args.beam)
    plot_all_deltap_vs_range(beam_data)
    plot_difference_vs_arc(beam_data)
    plot_expected_vs_measured_mean(beam_data, args.dispersion)
    plot_difference_vs_measured_difference(beam_data)


if __name__ == "__main__":
    main()
