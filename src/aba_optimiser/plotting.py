"""
Plotting utilities for the ABA optimiser results.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def _setup_scientific_formatting():
    """Setup scientific notation formatting for both axes."""
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def plot_strengths_comparison(
    magnet_names: list[str],
    final_vals: np.ndarray,
    true_vals: np.ndarray,
    uncertainties: np.ndarray,
    initial_vals: np.ndarray | None = None,
    show_errorbars: bool = False,
    plot_real: bool = False,
    save_path: str = "plots/relative_difference_comparison.png",
    *,
    # title: Optional[str] = None,
    style: str | None = "seaborn-v0_8-colorblind",
    unit: str | None = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths comparison (either raw strengths or relative differences).

    Args:
        magnet_names: List of knob names
        final_vals: Final knob values
        true_vals: True knob values
        uncertainties: Uncertainty values
        initial_vals: Initial knob values (optional)
        show_errorbars: Whether to show error bars
        save_path: Path to save the plot
    """

    if style:
        plt.style.use(style)

    # Basic input validation
    n = len(magnet_names)
    assert len(final_vals) == n == len(true_vals) == len(uncertainties), (
        "Input arrays must all have the same length as magnet_names"
    )
    if initial_vals is not None:
        assert len(initial_vals) == n, "initial_vals must have same length as others"

    # Determine baseline: use initial_vals when provided so diffs are w.r.t initial
    baseline = initial_vals if initial_vals is not None else true_vals

    # Calculate differences (relative) OR prepare raw values for plotting
    baseline_abs = np.abs(baseline)
    zero_mask = baseline_abs == 0

    if plot_real:
        final_plot = np.asarray(final_vals)
        true_plot = np.asarray(true_vals)
        uncertainties_on_plot = np.abs(np.asarray(uncertainties))
    else:
        # relative: fraction of baseline (guard divide-by-zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            final_plot = np.abs(np.asarray(final_vals) - baseline) / baseline_abs
            uncertainties_on_plot = np.abs(np.asarray(uncertainties)) / baseline_abs

        if np.any(zero_mask):
            warnings.warn(
                "Some baseline elements are zero: relative difference undefined for those indices. "
                "Those points will be NaN in the plot."
            )
            final_plot = np.where(zero_mask, np.nan, final_plot)
            uncertainties_on_plot = np.where(zero_mask, np.nan, uncertainties_on_plot)

    x = np.arange(n)

    if initial_vals is not None:
        # Side-by-side comparison plot comparing TRUE vs FINAL relative to INITIAL
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot true (either raw true values or true differences)
        if plot_real:
            ax.bar(
                x - width / 2,
                true_plot,
                width,
                color="lightcoral",
                label="True Model Strengths",
            )
        else:
            true_diff = np.abs(np.asarray(true_vals) - baseline) / baseline_abs
            true_diff = np.where(zero_mask, np.nan, true_diff)
            ax.bar(
                x - width / 2,
                true_diff,
                width,
                color="lightcoral",
                label="True vs Initial Model Strengths",
            )

        # Plot final (either raw or differences) with optional error bars
        error_bars = uncertainties_on_plot if show_errorbars else None
        if error_bars is not None:
            ax.bar(
                x + width / 2,
                final_plot,
                width,
                color="mediumpurple",
                label=(
                    "Final Model Strengths"
                    if plot_real
                    else "Final vs Initial Model Strengths"
                ),
                yerr=error_bars,
                capsize=5,
            )
        else:
            ax.bar(
                x + width / 2,
                final_plot,
                width,
                color="mediumpurple",
                label=(
                    "Final Model Strengths"
                    if plot_real
                    else "Final vs Initial Model Strengths"
                ),
            )

        # ax.set_title(title or "Initial vs Final Relative Difference")
    else:
        # Single plot for final (relative or raw) only
        fig, ax = plt.subplots(figsize=(12, 6))

        error_bars = uncertainties_on_plot if show_errorbars else None
        if plot_real:
            # show true and final side-by-side for reference
            ax.bar(
                x - 0.25,
                true_plot,
                0.5,
                color="lightcoral",
                label="True Model Strengths",
            )
            if error_bars is not None:
                ax.bar(
                    x + 0.25,
                    final_plot,
                    0.5,
                    color="mediumpurple",
                    label="Final Model Strengths",
                    yerr=error_bars,
                    capsize=5,
                )
            else:
                ax.bar(
                    x + 0.25,
                    final_plot,
                    0.5,
                    color="mediumpurple",
                    label="Final Model Strengths",
                )
        else:
            if error_bars is not None:
                ax.bar(
                    x,
                    final_plot,
                    0.5,
                    color="mediumpurple",
                    label="Final vs True",
                    yerr=error_bars,
                    capsize=5,
                )
            else:
                ax.bar(x, final_plot, 0.5, color="mediumpurple", label="Final vs True")

        # ax.set_title(title or "Relative Difference vs True")

    ax.set_xlabel("Magnet Name")
    if plot_real:
        ax.set_ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference (fraction)")
    ax.grid(axis="y", alpha=0.3)

    _setup_scientific_formatting()

    ax.set_xticks(x)
    ax.set_xticklabels(magnet_names, rotation=45, ha="right")

    if initial_vals is not None:
        ax.legend()

    # add acceptance band for relative plots if requested
    ax.margins(x=0.01)
    plt.xticks(fontsize=9)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_strengths_vs_position(
    elem_spos: np.ndarray,
    final_vals: np.ndarray,
    true_vals: np.ndarray,
    uncertainties: np.ndarray,
    initial_vals: np.ndarray | None = None,
    show_errorbars: bool = False,
    plot_real: bool = False,
    save_path: str = "plots/relative_difference_vs_position.png",
    *,
    style: str | None = "seaborn-v0_8-colorblind",
    unit: str | None = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths vs element position (either raw strengths or relative differences).

    Args:
        elem_spos: Element positions
        final_vals: Final knob values
        true_vals: True knob values
        uncertainties: Uncertainty values
        initial_vals: Initial knob values (optional)
        show_errorbars: Whether to show error bars
        save_path: Path to save the plot
    """
    if style:
        plt.style.use(style)

    # Validation
    n = len(elem_spos)
    assert len(final_vals) == n == len(true_vals) == len(uncertainties), (
        "All inputs must have same length"
    )
    if initial_vals is not None:
        assert len(initial_vals) == n

    baseline = initial_vals if initial_vals is not None else true_vals

    baseline_abs = np.abs(baseline)
    zero_mask = baseline_abs == 0

    if plot_real:
        final_plot = np.asarray(final_vals)
        true_plot = np.asarray(true_vals)
        uncertainties_on_plot = np.abs(np.asarray(uncertainties))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            final_plot = np.abs(np.asarray(final_vals) - baseline) / baseline_abs
            uncertainties_on_plot = np.abs(np.asarray(uncertainties)) / baseline_abs
        if np.any(zero_mask):
            warnings.warn(
                "Some baseline elements are zero: relative difference undefined for those indices and will be NaN in the plot."
            )
            final_plot = np.where(zero_mask, np.nan, final_plot)
            uncertainties_on_plot = np.where(zero_mask, np.nan, uncertainties_on_plot)

    fig, ax = plt.subplots(figsize=(12, 6))

    if initial_vals is not None:
        # Plot TRUE and FINAL (either raw or differences) relative to INITIAL baseline
        if plot_real:
            ax.plot(
                elem_spos,
                true_plot,
                "o",
                label="True Model Strengths",
                markersize=7,
                alpha=0.85,
            )
            if show_errorbars:
                ax.errorbar(
                    elem_spos,
                    final_plot,
                    yerr=uncertainties_on_plot,
                    fmt="o",
                    label="Final Model Strengths",
                    capsize=5,
                    markersize=7,
                    alpha=0.85,
                )
            else:
                ax.plot(
                    elem_spos,
                    final_plot,
                    "o",
                    label="Final Model Strengths",
                    markersize=7,
                    alpha=0.85,
                )
        else:
            true_diff = np.abs(true_vals - baseline) / np.abs(baseline)
            true_diff = np.where(zero_mask, np.nan, true_diff)
            ax.plot(
                elem_spos,
                true_diff,
                "o",
                label="True vs Initial Model Strengths",
                markersize=7,
                alpha=0.85,
            )

            if show_errorbars:
                ax.errorbar(
                    elem_spos,
                    final_plot,
                    yerr=uncertainties_on_plot,
                    fmt="o",
                    label="Final vs Initial Model Strengths",
                    capsize=5,
                    markersize=7,
                    alpha=0.85,
                )
            else:
                ax.plot(
                    elem_spos,
                    final_plot,
                    "o",
                    label="Final vs Initial Model Strengths",
                    markersize=7,
                    alpha=0.85,
                )
    else:
        # Plot final only (or raw values with true as reference)
        if plot_real:
            ax.plot(
                elem_spos,
                true_plot,
                "o",
                label="True Model Strengths",
                markersize=7,
                alpha=0.85,
            )
            if show_errorbars:
                ax.errorbar(
                    elem_spos,
                    final_plot,
                    yerr=uncertainties_on_plot,
                    fmt="o",
                    label="Final Model Strengths",
                    capsize=5,
                    markersize=7,
                    alpha=0.85,
                )
            else:
                ax.plot(
                    elem_spos,
                    final_plot,
                    "o",
                    label="Final Model Strengths",
                    markersize=7,
                    alpha=0.85,
                )
        else:
            label = "Relative Difference"
            if show_errorbars:
                plt.errorbar(
                    elem_spos,
                    final_plot,
                    yerr=uncertainties_on_plot,
                    fmt="o",
                    label=label,
                    capsize=5,
                    markersize=7,
                    alpha=0.85,
                )
            else:
                plt.plot(
                    elem_spos, final_plot, "o", label=label, markersize=7, alpha=0.85
                )

        # plt.title("Relative Difference vs Element Position")

    plt.xlabel("Element Position [m]")
    if plot_real:
        plt.ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference (fraction)")

    plt.legend()
    plt.grid(alpha=0.3)

    _setup_scientific_formatting()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def show_plots():
    plt.show()
