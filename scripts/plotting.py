from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_error_bars_bpm_range(
    s_sub: NDArray,
    baseline_x: NDArray,
    std_x: NDArray,
    baseline_y: NDArray,
    std_y: NDArray,
    bpm_range: str = "BPM.13R3.B1/BPM.12L4.B1",
    y_lim: tuple[float, float] | None = None,
):
    """Plot baseline positions with error bars for a specific BPM range."""
    start_bpm, end_bpm = bpm_range.split("/")
    avg_x = np.mean(std_x)
    avg_y = np.mean(std_y)
    """Plot baseline positions with error bars for a specific BPM range."""
    fig_range, (ax1_r, ax2_r) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax1_r.errorbar(
        s_sub,
        baseline_x,
        yerr=std_x,
        fmt="x",
        color="k",
        markersize=4,
        capsize=3,
        elinewidth=1,
        markeredgecolor="k",
        label=f"X rms error (avg={avg_x:.2e})",
    )
    ax1_r.set_ylabel("X [m]")
    ax1_r.set_title(f"Baseline X positions with σ error-bars ({start_bpm} → {end_bpm})")
    ax1_r.grid(visible=True, alpha=0.3)
    ax1_r.legend()
    if y_lim:
        ax1_r.set_ylim(y_lim)

    ax2_r.errorbar(
        s_sub,
        baseline_y,
        yerr=std_y,
        fmt="x",
        color="k",
        markersize=4,
        capsize=3,
        elinewidth=1,
        markeredgecolor="k",
        label=f"Y rms error (avg={avg_y:.2e})",
    )
    ax2_r.set_xlabel("s [m]")
    ax2_r.set_ylabel("Y [m]")
    ax2_r.set_title(f"Baseline Y positions with σ error-bars ({start_bpm} → {end_bpm})")
    ax2_r.grid(visible=True, alpha=0.3)
    ax2_r.legend()
    if y_lim:
        ax2_r.set_ylim(y_lim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_range


def plot_std_log_comparison(
    s_sorted: NDArray,
    std_x_quad: NDArray,
    std_y_quad: NDArray,
    std_x_ic: NDArray,
    std_y_ic: NDArray,
    std_x_combined: NDArray,
    std_y_combined: NDArray,
    bpm_range: str = "BPM.13R3.B1/BPM.12L4.B1",
    nturns: int = 100,
):
    """Plot standard deviations on a logarithmic scale."""
    fig_log, (ax1_l, ax2_l) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig_log.suptitle(
        f"Logarithmic Plot of Standard Deviations ({bpm_range})", fontsize=14, y=0.98
    )

    # X-component (only these lines are given labels so we can create a single
    # shared legend for the whole figure)
    (h_quad_x,) = ax1_l.plot(
        s_sorted,
        std_x_quad,
        marker="x",
        color="red",
        label="Quadrupole error",
    )
    (h_instr_x,) = ax1_l.plot(
        s_sorted,
        std_x_ic,
        marker="x",
        color="blue",
        label="Initial Condition error",
    )
    (h_comb_x,) = ax1_l.plot(
        s_sorted,
        std_x_combined,
        marker="x",
        color="orange",
        label="Combined error",
    )
    ax1_l.set_yscale("log")
    h_noise_x = ax1_l.axhline(
        1e-4, color="green", linestyle="--", label="Measurement noise"
    )
    ax1_l.set_ylabel("Std X [m]")
    ax1_l.grid(visible=True, alpha=0.3)
    # Add panel label '1' in the top-left of the X subplot
    ax1_l.text(
        0.02,
        0.95,
        "X Plane",
        transform=ax1_l.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        zorder=10,
    )
    ax1_l.set_ylim(1e-7, 2e-3)

    # Y-component (no labels here to avoid duplicate legend entries)
    ax2_l.plot(s_sorted, std_y_quad, marker="x", color="red")
    ax2_l.plot(s_sorted, std_y_ic, marker="x", color="blue")
    ax2_l.plot(s_sorted, std_y_combined, marker="x", color="orange")
    ax2_l.set_yscale("log")
    ax2_l.axhline(1e-4, color="green", linestyle="--")
    ax2_l.set_xlabel("s [m]")
    ax2_l.set_ylabel("Std Y [m]")
    ax2_l.grid(visible=True, alpha=0.3)
    # Add panel label '2' in the top-left of the Y subplot
    ax2_l.text(
        0.02,
        0.95,
        "Y Plane",
        transform=ax2_l.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        zorder=10,
    )
    ax2_l.set_ylim(1e-7, 2e-3)

    # Create a single legend for the whole figure, placed at the top and
    # spanning both subplots. Use a few columns so it sits in one row.
    handles = [h_quad_x, h_instr_x, h_comb_x, h_noise_x]
    labels = [h.get_label() for h in handles]
    fig_log.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
    )

    # Leave space at the top for suptitle and legend, then save.
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig_log.savefig(
        f"plots/std_log_comparison_{nturns}t.png", dpi=300, bbox_inches="tight"
    )
    return fig_log


def show_plots():
    """Show all generated plots."""
    plt.show()
