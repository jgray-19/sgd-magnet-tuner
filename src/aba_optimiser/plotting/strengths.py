"""
Plotting utilities for the ABA optimiser results.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aba_optimiser.plotting.utils import setup_scientific_formatting

LOGGER = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 16})

# Color palette for magnet families (color encodes MQ/MS; style encodes True/Final)
MQ_COLOR = "#1f77b4"  # tab:blue (fallback)
MS_COLOR = "#ff7f0e"  # tab:orange (fallback)
# Distinct colors per (series × family)
MQ_TRUE_COLOR = "#1f77b4"  # tab:blue
MQ_FINAL_COLOR = "#17becf"  # tab:cyan
MS_TRUE_COLOR = "#ff7f0e"  # tab:orange
MS_FINAL_COLOR = "#d62728"  # tab:red


def _family_labels_and_colors(
    names: list[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (families, colors, palette) for magnet names."""
    families = ["MQ" if str(n).upper().startswith("MQ") else "MS" for n in names]
    palette = {"MQ": MQ_COLOR, "MS": MS_COLOR}
    colors = [palette[f] for f in families]
    return families, colors, palette


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
    style: str | None = "seaborn-v0_8-colorblind",
    unit: str | None = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths comparison (either raw strengths or relative differences).

    Colors encode magnet family (MQ/MS). Style (hatch) encodes True vs Final.
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

    # Family colors
    families, family_colors, palette = _family_labels_and_colors(magnet_names)
    true_colors = [MQ_TRUE_COLOR if f == "MQ" else MS_TRUE_COLOR for f in families]
    final_colors = [MQ_FINAL_COLOR if f == "MQ" else MS_FINAL_COLOR for f in families]

    # Determine baseline: use initial_vals when provided so diffs are w.r.t initial
    baseline = initial_vals if initial_vals is not None else true_vals

    # Calculate differences (relative) OR prepare raw values for plotting
    baseline_abs = np.abs(baseline)
    zero_mask = baseline_abs == 0
    if any(zero_mask) and not plot_real:
        warnings.warn(
            "Some baseline elements are zero: relative difference undefined for those indices."
        )

    if plot_real:
        final_plot = np.asarray(final_vals)
        true_plot = np.asarray(true_vals)
        uncertainties_on_plot = np.abs(np.asarray(uncertainties))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            final_plot = np.abs(np.asarray(final_vals) - baseline) / baseline_abs
            uncertainties_on_plot = np.abs(np.asarray(uncertainties)) / baseline_abs
            final_plot *= 1e4
            uncertainties_on_plot *= 1e4

    x = np.arange(n)

    if initial_vals is not None:
        # Side-by-side bars: color=MQ/MS, hatch distinguishes True vs Final
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 8))

        # TRUE bars
        if plot_real:
            y_true = true_plot
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                y_true = np.abs(np.asarray(true_vals) - baseline) / baseline_abs
                y_true *= 1e4
        ax.bar(
            x - width / 2,
            y_true,
            width,
            color=true_colors,
            edgecolor="black",
            label="True",
        )

        # FINAL bars (with optional error bars)
        err = uncertainties_on_plot if show_errorbars else None
        ax.bar(
            x + width / 2,
            final_plot,
            width,
            color=final_colors,
            edgecolor="black",
            label="Final",
            yerr=err,
            capsize=5 if show_errorbars else 0,
        )
    else:
        # Single series
        fig, ax = plt.subplots(figsize=(12, 6))
        err = uncertainties_on_plot if show_errorbars else None
        if plot_real:
            # show true and final side-by-side for reference
            ax.bar(
                x - 0.25,
                true_plot,
                0.5,
                color=true_colors,
                edgecolor="black",
                label="True",
            )
            ax.bar(
                x + 0.25,
                final_plot,
                0.5,
                color=final_colors,
                edgecolor="black",
                label="Final",
                yerr=err,
                capsize=5 if show_errorbars else 0,
            )
        else:
            ax.bar(
                x,
                final_plot,
                0.5,
                color=final_colors,
                edgecolor="black",
                label="Final vs True",
                yerr=err,
                capsize=5 if show_errorbars else 0,
            )

    # Labels, ticks, grid
    if plot_real:
        ax.set_ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference ($\\times10^{-4}$)")
    ax.grid(axis="y", alpha=0.3)
    setup_scientific_formatting(plane="y")

    ax.set_xticks(x)
    ax.set_xticklabels(magnet_names, rotation=45, ha="right")

    if initial_vals is not None:
        # Y limit based on initial or true diff (keeps your original behavior)
        if plot_real:
            top = np.ceil(np.max(initial_vals)) + 0.5
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                true_diff = np.abs(np.asarray(true_vals) - baseline) / baseline_abs
                true_diff *= 1e4
            top = np.ceil(np.nanmax(true_diff)) + 0.5
        ax.set_ylim(top=top, bottom=0)

    # Legend: four distinct colors (series × family)
    legend_handles = [
        Patch(facecolor=MQ_TRUE_COLOR, edgecolor="black", label="MQ • True"),
        Patch(facecolor=MQ_FINAL_COLOR, edgecolor="black", label="MQ • Final"),
        Patch(facecolor=MS_TRUE_COLOR, edgecolor="black", label="MS • True"),
        Patch(facecolor=MS_FINAL_COLOR, edgecolor="black", label="MS • Final"),
    ]
    ax.legend(handles=legend_handles, title="Series × Family", loc="upper right")

    ax.margins(x=0.01)
    ax.tick_params(axis="x", labelsize=12)
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
    dpi: int = 300,
    magnet_names: list[str] | None = None,  # <— NEW: used to color MQ/MS
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths vs element position (either raw strengths or relative differences).

    Colors encode magnet family (MQ/MS) if `magnet_names` is provided.
    Series (True/Final) uses marker shape to differentiate.
    """
    if style:
        plt.style.use(style)

    # Validation
    n = len(elem_spos)
    assert len(final_vals) == n == len(true_vals) == len(uncertainties), (
        f"All inputs must have same length, final_vals: {len(final_vals)}, true_vals: {len(true_vals)}, uncertainties: {len(uncertainties)}"
    )
    if initial_vals is not None:
        assert len(initial_vals) == n
    # Ensure numpy arrays for boolean mask indexing (elem_spos may be a list)
    elem_spos = np.asarray(elem_spos)
    if magnet_names is not None:
        assert len(magnet_names) == n, "magnet_names must match elem_spos length"

    # Family colors (optional)
    if magnet_names is not None:
        families, family_colors, palette = _family_labels_and_colors(magnet_names)
        is_mq = np.array([f == "MQ" for f in families], dtype=bool)
        is_ms = ~is_mq
    else:
        families = None
        is_mq = is_ms = None

    baseline = initial_vals if initial_vals is not None else true_vals
    baseline_abs = np.abs(baseline)
    zero_mask = baseline_abs == 0

    if plot_real:
        final_plot = np.asarray(final_vals)
        true_plot = np.asarray(true_vals)
        uncertainties_on_plot = np.abs(np.asarray(uncertainties))
        # (Note: Keeping your original scaling behavior as-is)
        final_plot *= 1e4
        true_plot *= 1e4
        uncertainties_on_plot *= 1e4
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            final_plot = np.abs(np.asarray(final_vals) - baseline) / baseline_abs
            uncertainties_on_plot = np.abs(np.asarray(uncertainties)) / baseline_abs
            final_plot *= 1e4
            uncertainties_on_plot *= 1e4

        if np.any(zero_mask):
            warnings.warn(
                "Some baseline elements are zero: relative difference undefined for those indices and will be NaN in the plot."
            )
            final_plot = np.where(zero_mask, np.nan, final_plot)
            uncertainties_on_plot = np.where(zero_mask, np.nan, uncertainties_on_plot)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Helper to plot a subset with consistent style
    def _plot_subset(mask: np.ndarray, color: str, label: str, series: str):
        if not np.any(mask):
            return
        marker = "s" if series == "True" else "o"
        if series == "True":
            y = (
                true_plot
                if plot_real
                else (np.abs(true_vals - baseline) / np.abs(baseline)) * 1e4
            )
            if not plot_real:
                y = np.where(zero_mask, np.nan, y)
            ax.plot(
                elem_spos[mask],
                y[mask],
                marker,
                label=label,
                markersize=7,
                alpha=0.9,
                linestyle="None",
                color=color,
            )
        else:  # Final
            if show_errorbars:
                ax.errorbar(
                    elem_spos[mask],
                    final_plot[mask],
                    yerr=uncertainties_on_plot[mask],
                    fmt=marker,
                    label=label,
                    capsize=5,
                    markersize=7,
                    alpha=0.9,
                    linestyle="None",
                    color=color,
                )
            else:
                ax.plot(
                    elem_spos[mask],
                    final_plot[mask],
                    marker,
                    label=label,
                    markersize=7,
                    alpha=0.9,
                    linestyle="None",
                    color=color,
                )

    if initial_vals is not None:
        # TRUE and FINAL relative to INITIAL (or raw if plot_real)
        if magnet_names is not None:
            # Plot MQ, then MS to get per-family colors + series markers
            _plot_subset(is_mq, MQ_TRUE_COLOR, "MQ • True", "True")
            _plot_subset(is_ms, MS_TRUE_COLOR, "MS • True", "True")
            _plot_subset(is_mq, MQ_FINAL_COLOR, "MQ • Final", "Final")
            _plot_subset(is_ms, MS_FINAL_COLOR, "MS • Final", "Final")
        else:
            # Single color fallback
            _plot_subset(np.ones(n, dtype=bool), MQ_TRUE_COLOR, "True", "True")
            _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOR, "Final", "Final")
    else:
        # Only FINAL (or TRUE+FINAL in raw-mode section above)
        if plot_real:
            # Show both True and Final also when no initial (as in your original)
            if magnet_names is not None:
                _plot_subset(is_mq, MQ_TRUE_COLOR, "MQ • True", "True")
                _plot_subset(is_ms, MS_TRUE_COLOR, "MS • True", "True")
                _plot_subset(is_mq, MQ_FINAL_COLOR, "MQ • Final", "Final")
                _plot_subset(is_ms, MS_FINAL_COLOR, "MS • Final", "Final")
            else:
                _plot_subset(np.ones(n, dtype=bool), MQ_TRUE_COLOR, "True", "True")
                _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOR, "Final", "Final")
        else:
            # Final only
            if magnet_names is not None:
                _plot_subset(is_mq, MQ_FINAL_COLOR, "MQ • Final", "Final")
                _plot_subset(is_ms, MS_FINAL_COLOR, "MS • Final", "Final")
            else:
                _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOR, "Final", "Final")

    # Y limit when initial provided (keeps original behavior)
    if initial_vals is not None:
        if plot_real:
            top = np.ceil(np.max(initial_vals)) + 0.5
        else:
            true_diff = np.abs(true_vals - baseline) / np.abs(baseline)
            true_diff = np.where(zero_mask, np.nan, true_diff)
            true_diff *= 1e4
            top = np.ceil(np.nanmax(true_diff)) + 0.5
        ax.set_ylim(top=top, bottom=0)

    ax.set_xlabel("Element Position [m]")
    if plot_real:
        ax.set_ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference ($\\times10^{-4}$)")

    if magnet_names is not None:
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color=MQ_TRUE_COLOR,
                label="MQ • True",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=MQ_FINAL_COLOR,
                label="MQ • Final",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color=MS_TRUE_COLOR,
                label="MS • True",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=MS_FINAL_COLOR,
                label="MS • Final",
            ),
        ]
        ax.legend(handles=handles, title="Series × Family", loc="upper right")
    else:
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color=MQ_TRUE_COLOR,
                label="True",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=MQ_FINAL_COLOR,
                label="Final",
            ),
        ]
        ax.legend(handles=handles, title="Series", loc="upper right")

    ax.grid(alpha=0.3)
    setup_scientific_formatting(plane="y")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def plot_deltap_comparison(
    actual_deltap: float,
    estimated_deltap: float,
    uncertainty: float,
    save_path: str = "plots/deltap_comparison.png",
    *,
    style: str | None = "seaborn-v0_8-colorblind",
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot deltap comparison before and after optimisation as a single bar chart.

    Axis in units of 10^-4.
    """
    if style:
        plt.style.use(style)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = [0, 1]
    labels = ["Actual", "Estimated"]
    values = [actual_deltap * 1e4, estimated_deltap * 1e4]
    colors = [MQ_TRUE_COLOR, MQ_FINAL_COLOR]

    ax.bar(x, values, color=colors, edgecolor="black", width=0.6)

    # Error bar on final
    ax.errorbar(
        1,
        values[1],
        yerr=uncertainty * 1e4,
        capsize=5,
        color="black",
        fmt="none",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta p / p$ ($\times 10^{-4}$)")
    ax.grid(axis="y", alpha=0.3)
    setup_scientific_formatting(plane="y")

    # Legend
    legend_handles = [
        Patch(facecolor=MQ_TRUE_COLOR, edgecolor="black", label="Initial"),
        Patch(facecolor=MQ_FINAL_COLOR, edgecolor="black", label="Final"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax
