"""
Plotting utilities for the ABA optimiser results.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aba_optimiser.plotting.utils import setup_scientific_formatting

if TYPE_CHECKING:
    from aba_optimiser.accelerators import Accelerator

LOGGER = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 16})

# Color palette for magnet families (color encodes MQ/MS; style encodes True/Final)
MB_COLOUR = "#2ca02c"  # tab:green (fallback)
MQ_COLOUR = "#1f77b4"  # tab:blue (fallback)
MS_COLOUR = "#ff7f0e"  # tab:orange (fallback)

# Distinct colors per (series × family)
MB_TRUE_COLOUR = "#2ca02c"  # tab:green
MB_FINAL_COLOUR = "#98df8a"  # tab:lightgreen
MQ_TRUE_COLOUR = "#1f77b4"  # tab:blue
MQ_FINAL_COLOUR = "#17becf"  # tab:cyan
MS_TRUE_COLOUR = "#ff7f0e"  # tab:orange
MS_FINAL_COLOUR = "#d62728"  # tab:red

TRUE_COLOURS = {
    "MB": MB_TRUE_COLOUR,
    "MQ": MQ_TRUE_COLOUR,
    "MS": MS_TRUE_COLOUR,
}
FINAL_COLOURS = {
    "MB": MB_FINAL_COLOUR,
    "MQ": MQ_FINAL_COLOUR,
    "MS": MS_FINAL_COLOUR,
}


def _generate_family_colors_from_accelerator(
    accelerator: Accelerator,
) -> tuple[dict[str, str], dict[str, str]]:
    """Generate true/final colors for magnet families from accelerator knob specs.

    Returns:
        Tuple of (true_colors, final_colors) dictionaries mapping family names to colors.
    """
    knob_specs = accelerator.get_supported_knob_specs()

    # Extract unique families from patterns (e.g., "MB.*" -> "MB")
    families = set()
    for _, _, pattern, _ in knob_specs:
        # Extract family from pattern (first 2 chars before any special chars)
        family = pattern.split(".")[0][:2]  # e.g., "MB.*" -> "MB"
        families.add(family)

    families = sorted(families)
    n_families = len(families)

    # Generate colors using matplotlib's color cycle
    import matplotlib.cm as cm

    tab10 = cm.get_cmap("tab10")
    base_colors = [tab10(i / max(1, n_families - 1)) for i in range(n_families)]

    true_colors = {}
    final_colors = {}

    for i, family in enumerate(families):
        base_color = base_colors[i]

        # Create true color (darker/saturated)
        true_colors[family] = base_color

        # Create final color (lighter version)
        # Convert to HSV and lighten
        h, s, v = mcolors.rgb_to_hsv(base_color[:3])  # Take RGB components
        final_colors[family] = mcolors.hsv_to_rgb((h, s * 0.6, min(1.0, v * 1.3)))

    return true_colors, final_colors


def _family_labels_and_colors(
    names: list[str],
    accelerator: Accelerator | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return (families, colors, palette) for magnet names."""
    families = []
    for n in names:
        families.append(n[:2])  # e.g., "MQ", "MS", "MB"

    if accelerator is not None:
        true_colors, final_colors = _generate_family_colors_from_accelerator(accelerator)
        # Use final colors as the main palette for backward compatibility
        palette = final_colors
    else:
        # Fallback to hardcoded colors
        palette = {"MQ": MQ_COLOUR, "MS": MS_COLOUR, "MB": MB_COLOUR}

    colors = [palette.get(f, "#808080") for f in families]  # Default to grey for unknown families
    return families, colors, palette


def _prepare_plot_data(final_vals, true_vals, uncertainties, initial_vals, plot_real):
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
        final_plot *= 1e4
        uncertainties_on_plot *= 1e4
        if any(zero_mask):
            warnings.warn(
                "Some baseline elements are zero: relative difference undefined for those indices."
            )
            final_plot = np.where(zero_mask, np.nan, final_plot)
            uncertainties_on_plot = np.where(zero_mask, np.nan, uncertainties_on_plot)
        with np.errstate(divide="ignore", invalid="ignore"):
            true_plot = np.abs(np.asarray(true_vals) - baseline) / baseline_abs
        true_plot *= 1e4
        true_plot = np.where(zero_mask, np.nan, true_plot)

    return final_plot, true_plot, uncertainties_on_plot, baseline, zero_mask


def _compute_y_top(final_vals, baseline, zero_mask, plot_real):
    if plot_real:
        top = np.ceil(np.max(final_vals)) * 1.1
    else:
        final_diff = np.abs(final_vals - baseline) / np.abs(baseline)
        final_diff = np.where(zero_mask, np.nan, final_diff)
        final_diff *= 1e4
        top = np.ceil(np.nanmax(final_diff)) * 1.1
    return top


def _save_plot(fig, save_path, dpi):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def _validate_inputs(
    length_base, final_vals, true_vals, uncertainties, initial_vals=None, magnet_names=None
):
    n = len(length_base)
    assert len(final_vals) == n == len(true_vals) == len(uncertainties), (
        f"Input arrays must have length {n}"
    )
    if initial_vals is not None:
        assert len(initial_vals) == n
    if magnet_names is not None:
        assert len(magnet_names) == n
    return n


def _get_family_colors_and_masks(magnet_names, accelerator: Accelerator | None = None):
    families, _, _ = _family_labels_and_colors(magnet_names, accelerator)

    if accelerator is not None:
        true_colours_dict, final_colours_dict = _generate_family_colors_from_accelerator(
            accelerator
        )
    else:
        # Fallback to hardcoded colors
        true_colours_dict = TRUE_COLOURS
        final_colours_dict = FINAL_COLOURS

    true_colours = [true_colours_dict.get(f, "#808080") for f in families]
    final_colours = [final_colours_dict.get(f, "#808080") for f in families]
    is_mq = np.array([f == "MQ" for f in families], dtype=bool)
    is_ms = np.array([f == "MS" for f in families], dtype=bool)
    is_mb = np.array([f == "MB" for f in families], dtype=bool)
    return families, true_colours, final_colours, is_mq, is_ms, is_mb


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
    accelerator: Accelerator | None = None,
    style: str | None = "seaborn-v0_8-colorblind",
    unit: str | None = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths comparison (either raw strengths or relative differences).

    Colors encode magnet family (MQ/MS). Style (hatch) encodes True vs Final.
    If accelerator is provided, colors are generated dynamically from supported knob specs.
    """
    if style:
        plt.style.use(style)

    n = _validate_inputs(magnet_names, final_vals, true_vals, uncertainties, initial_vals)

    families, true_colours, final_colours, _, _, _ = _get_family_colors_and_masks(
        magnet_names, accelerator
    )

    final_plot, true_plot, uncertainties_on_plot, baseline, zero_mask = _prepare_plot_data(
        final_vals, true_vals, uncertainties, initial_vals, plot_real
    )

    x = np.arange(n)

    if initial_vals is not None:
        # Side-by-side bars: colour=MQ/MS, hatch distinguishes True vs Final
        width = 0.35
        fig, ax = plt.subplots(figsize=(14, 8))

        # TRUE as red dotted line
        ax.plot(
            x,
            true_plot,
            color="red",
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=5,
            label="True",
        )

        # FINAL bars (with optional error bars)
        err = uncertainties_on_plot if show_errorbars else None
        ax.bar(
            x,
            final_plot,
            width,
            color=final_colours,
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
                color=true_colours,
                edgecolor="black",
                label="True",
            )
            ax.bar(
                x + 0.25,
                final_plot,
                0.5,
                color=final_colours,
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
                color=final_colours,
                edgecolor="black",
                label="Final vs True",
                yerr=err,
                capsize=5 if show_errorbars else 0,
            )

    # Labels, ticks, grid
    if plot_real:
        ax.set_ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference (units)")
    ax.grid(axis="y", alpha=0.3)
    setup_scientific_formatting(plane="y")

    ax.set_xticks(x)
    ax.set_xticklabels(magnet_names, rotation=45, ha="right")

    # Y limit based on initial or true diff (keeps your original behavior)
    top = _compute_y_top(final_vals, baseline, zero_mask, plot_real)
    ax.set_ylim(top=top, bottom=0)

    # Legend: True as line, Finals as bars
    unique_families = set(families)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            marker="o",
            markersize=5,
            label="True",
        ),
    ]

    if accelerator is not None:
        true_colors_dict, final_colors_dict = _generate_family_colors_from_accelerator(accelerator)
        for fam in sorted(unique_families):
            if fam in final_colors_dict:
                legend_handles.append(
                    Patch(
                        facecolor=final_colors_dict[fam],
                        edgecolor="black",
                        label=f"{fam} • Estimate",
                    )
                )
    else:
        # Fallback to hardcoded families
        for fam in ["MQ", "MS", "MB"]:
            if fam in unique_families:
                colour = {"MQ": MQ_FINAL_COLOUR, "MS": MS_FINAL_COLOUR, "MB": MB_FINAL_COLOUR}[fam]
                legend_handles.append(
                    Patch(facecolor=colour, edgecolor="black", label=f"{fam} • Estimate")
                )

    ax.margins(x=0.01)
    ax.tick_params(axis="x", labelsize=12)
    plt.tight_layout()

    _save_plot(fig, save_path, dpi)
    return fig, ax


def plot_strengths_vs_position(
    elem_spos: list[float],
    final_vals: np.ndarray,
    true_vals: np.ndarray,
    uncertainties: np.ndarray,
    initial_vals: np.ndarray | None = None,
    show_errorbars: bool = False,
    plot_real: bool = False,
    save_path: str = "plots/relative_difference_vs_position.png",
    *,
    accelerator: Accelerator | None = None,
    style: str | None = "seaborn-v0_8-colorblind",
    unit: str | None = None,
    dpi: int = 300,
    magnet_names: list[str] | None = None,  # <— NEW: used to color MQ/MS
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot strengths vs element position (either raw strengths or relative differences).

    Colors encode magnet family (MQ/MS) if `magnet_names` is provided.
    Series (True/Final) uses marker shape to differentiate.
    If accelerator is provided, colors are generated dynamically from supported knob specs.
    """
    if style:
        plt.style.use(style)

    n = _validate_inputs(
        elem_spos, final_vals, true_vals, uncertainties, initial_vals, magnet_names
    )

    # Ensure numpy arrays for boolean mask indexing (elem_spos may be a list)
    spos_arr = np.asarray(elem_spos)

    if magnet_names is not None:
        families, _, _, is_mq, is_ms, is_mb = _get_family_colors_and_masks(
            magnet_names, accelerator
        )
    else:
        families = None

    final_plot, true_plot, uncertainties_on_plot, baseline, zero_mask = _prepare_plot_data(
        final_vals, true_vals, uncertainties, initial_vals, plot_real
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Helper to plot a subset with consistent style
    def _plot_subset(mask: np.ndarray, color: str, label: str, series: str):
        if not np.any(mask):
            return
        marker = "s" if series == "True" else "o"
        if series == "True":
            ax.plot(
                spos_arr[mask],
                true_plot[mask],
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
                    spos_arr[mask],
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
                    spos_arr[mask],
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
            # Plot MQ, then MS, then MB to get per-family colors + series markers
            _plot_subset(is_mq, MQ_TRUE_COLOUR, "MQ • True", "True")
            _plot_subset(is_ms, MS_TRUE_COLOUR, "MS • True", "True")
            _plot_subset(is_mb, MB_TRUE_COLOUR, "MB • True", "True")
            _plot_subset(is_mq, MQ_FINAL_COLOUR, "MQ • Final", "Final")
            _plot_subset(is_ms, MS_FINAL_COLOUR, "MS • Final", "Final")
            _plot_subset(is_mb, MB_FINAL_COLOUR, "MB • Final", "Final")
        else:
            # Single color fallback
            _plot_subset(np.ones(n, dtype=bool), MQ_TRUE_COLOUR, "True", "True")
            _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOUR, "Final", "Final")
    else:
        # Only FINAL (or TRUE+FINAL in raw-mode section above)
        if plot_real:
            # Show both True and Final also when no initial (as in your original)
            if magnet_names is not None:
                _plot_subset(is_mq, MQ_TRUE_COLOUR, "MQ • True", "True")
                _plot_subset(is_ms, MS_TRUE_COLOUR, "MS • True", "True")
                _plot_subset(is_mb, MB_TRUE_COLOUR, "MB • True", "True")
                _plot_subset(is_mq, MQ_FINAL_COLOUR, "MQ • Final", "Final")
                _plot_subset(is_ms, MS_FINAL_COLOUR, "MS • Final", "Final")
                _plot_subset(is_mb, MB_FINAL_COLOUR, "MB • Final", "Final")
            else:
                _plot_subset(np.ones(n, dtype=bool), MQ_TRUE_COLOUR, "True", "True")
                _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOUR, "Final", "Final")
        else:
            # Final only
            if magnet_names is not None:
                _plot_subset(is_mq, MQ_FINAL_COLOUR, "MQ • Final", "Final")
                _plot_subset(is_ms, MS_FINAL_COLOUR, "MS • Final", "Final")
                _plot_subset(is_mb, MB_FINAL_COLOUR, "MB • Final", "Final")
            else:
                _plot_subset(np.ones(n, dtype=bool), MQ_FINAL_COLOUR, "Final", "Final")

    # Y limit when initial provided (keeps original behavior)
    top = _compute_y_top(final_vals, baseline, zero_mask, plot_real)
    ax.set_ylim(top=top, bottom=0)

    ax.set_xlabel("Element Position [m]")
    if plot_real:
        ax.set_ylabel("Strength" + (f" [{unit}]" if unit else ""))
    else:
        ax.set_ylabel("Relative difference ($\\times10^{-4}$)")

    if magnet_names is not None:
        assert families is not None
        unique_families = set(families)
        handles = []

        if accelerator is not None:
            true_colors_dict, final_colors_dict = _generate_family_colors_from_accelerator(
                accelerator
            )
            for fam in sorted(unique_families):
                if fam in true_colors_dict and fam in final_colors_dict:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            linestyle="None",
                            color=true_colors_dict[fam],
                            label=f"{fam} • True",
                        )
                    )
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="None",
                            color=final_colors_dict[fam],
                            label=f"{fam} • Final",
                        )
                    )
        else:
            # Fallback to hardcoded families
            for fam in ["MQ", "MS", "MB"]:
                if fam in unique_families:
                    true_color = TRUE_COLOURS[fam]
                    final_color = FINAL_COLOURS[fam]
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="s",
                            linestyle="None",
                            color=true_color,
                            label=f"{fam} • True",
                        )
                    )
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="None",
                            color=final_color,
                            label=f"{fam} • Final",
                        )
                    )
        ax.legend(handles=handles, title="Legend", loc="upper right")
    else:
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                color=MQ_TRUE_COLOUR,
                label="True",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color=MQ_FINAL_COLOUR,
                label="Final",
            ),
        ]
        ax.legend(handles=handles, title="Series", loc="upper right")

    ax.grid(alpha=0.3)
    setup_scientific_formatting(plane="y")

    plt.tight_layout()
    _save_plot(fig, save_path, dpi)
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
    # Make a wider figure so legend can sit above the bar without overlapping
    fig, ax = plt.subplots(figsize=(5.5, 6))

    x = [0]
    labels = ["Estimated"]
    values = [estimated_deltap * 1e4]
    colors = [MQ_FINAL_COLOUR]

    # Bar (estimated) with edge and errorbar
    ax.bar(x, values, color=colors, edgecolor="black", width=0.55)
    ax.errorbar(
        0,
        values[0],
        yerr=uncertainty * 1e4,
        capsize=6,
        color="black",
        fmt="none",
        zorder=5,
    )

    # Red dotted line for actual value
    actual_y = actual_deltap * 1e4
    ax.axhline(
        actual_y,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Actual",
        zorder=6,
    )

    # Annotate numeric values above the bar and near the line
    # Bar annotation
    bar_top = values[0]

    # X ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(r"$\Delta p / p$ ($\times 10^{-4}$)")
    ax.grid(axis="y", alpha=0.3)
    setup_scientific_formatting(plane="y")

    # Y-limits: include both estimated and actual with a small margin
    top = max(bar_top, actual_y)
    margin = max(0.05 * abs(top), 0.002)
    bottom = min(0, min(bar_top, actual_y) - margin)
    top = max(0, top + margin)
    ax.set_ylim(bottom=bottom, top=top)

    # Legend above the plot, centered, with transparent box to avoid hiding bar
    legend_handles = [
        Patch(facecolor=MQ_FINAL_COLOUR, edgecolor="black", label="Estimated"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Actual"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=False,
    )
    leg.get_frame().set_alpha(0.95)

    # Improve layout and save
    plt.tight_layout()
    _save_plot(fig, save_path, dpi)
    return fig, ax
