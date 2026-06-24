"""Full-ring optics and closed-orbit plotting CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from tmom_recon import build_twiss_from_measurements

from aba_optimiser.measurements.b2_errors import read_b2_error_table
from aba_optimiser.measurements.plotting.core import (
    BEST_KNOWLEDGE_LABEL,
    BETTER_KNOWLEDGE_LABEL,
    DESIGN_OPTICS_LABEL,
    MEASUREMENT_LABEL,
    PLOT_COLORS,
    add_ip_positions_to_plot,
    get_element_positions,
    get_fullring_twiss,
    get_ip_positions,
    prepare_plot_context,
)

if TYPE_CHECKING:
    import pandas as pd


def plot_fullring_comparison(
    accelerator,
    design_accelerator,
    all_estimates: dict[str, float] | None,
    analysis_dir: Path,
    squeeze_step: str,
    results_dir: Path,
    tune_knobs_file: Path,
    corrector_file: Path | None,
    beam: int,
    include_best_knowledge_model: bool = True,
    deltap: float = 0.0,
) -> None:
    """Plot full-ring optics differences relative to the measurement."""

    def _header_float(df, *keys: str) -> float | None:
        headers = getattr(df, "headers", None)
        if headers is None:
            return None
        for key in keys:
            value = headers.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    meas_twiss, _ = build_twiss_from_measurements(
        analysis_dir, include_errors=True, reverse_bpm_order=beam == 2
    )
    meas_twiss.columns = [col.lower() for col in meas_twiss.columns]

    twiss_basic = get_fullring_twiss(design_accelerator, deltap=deltap)
    twiss_online = get_fullring_twiss(
        accelerator,
        tune_knobs_file=tune_knobs_file,
        corrector_file=corrector_file,
        deltap=deltap,
    )
    twiss_eff_online = None
    if all_estimates is not None:
        twiss_eff_online = get_fullring_twiss(
            accelerator,
            estimated_magnets=all_estimates,
            tune_knobs_file=tune_knobs_file,
            corrector_file=corrector_file,
            deltap=deltap,
        )
    ip_positions = get_ip_positions(accelerator)

    common_bpms = meas_twiss.index.intersection(twiss_basic.index).intersection(twiss_online.index)
    if twiss_eff_online is not None:
        common_bpms = common_bpms.intersection(twiss_eff_online.index)
    if len(common_bpms) == 0:
        raise ValueError("No common BPMs found for full-ring comparison.")

    meas_full = meas_twiss.loc[common_bpms].copy()
    base_full = twiss_basic.loc[common_bpms].copy()
    online_full = twiss_online.loc[common_bpms].copy()
    est_full = twiss_eff_online.loc[common_bpms].copy() if twiss_eff_online is not None else None

    start_bpm = common_bpms[0]
    meas_full = _normalize_phase(meas_full, ("mux", "muy"), start_bpm)
    base_full = _normalize_phase(base_full, ("mux", "muy"), start_bpm)
    online_full = _normalize_phase(online_full, ("mux", "muy"), start_bpm)
    if est_full is not None:
        est_full = _normalize_phase(est_full, ("mux", "muy"), start_bpm)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    xvals = meas_full["s"]

    def _diff(model, model_err, measurement, measurement_err):
        delta = model - measurement
        if model_err is None and measurement_err is None:
            return delta, None
        model_err_vals = model_err if model_err is not None else 0.0
        meas_err_vals = measurement_err if measurement_err is not None else 0.0
        return delta, np.sqrt(model_err_vals**2 + meas_err_vals**2)

    def _rel_diff(model, model_err, measurement, measurement_err):
        rel_delta = (model - measurement) / measurement * 100.0
        if model_err is None and measurement_err is None:
            return rel_delta, None
        model_err_vals = model_err if model_err is not None else 0.0
        meas_err_vals = measurement_err if measurement_err is not None else 0.0
        rel_err = np.sqrt(model_err_vals**2 + meas_err_vals**2) / np.abs(measurement) * 100.0
        return rel_delta, rel_err

    def _rms_label(label: str, values, unit: str) -> str:
        arr = np.asarray(values, dtype=float)
        rms = float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")
        return f"{label} (RMS {rms:.3f} {unit})"

    series_to_plot = [(DESIGN_OPTICS_LABEL, base_full, "s-", 1.0)]
    if include_best_knowledge_model:
        series_to_plot.append((BEST_KNOWLEDGE_LABEL, online_full, "^-", 1.0))
    if est_full is not None:
        series_to_plot.append((BETTER_KNOWLEDGE_LABEL, est_full, "d-", 1.0))
    plot_specs = [
        (axes[0, 0], "mux", "phase", r"$\Delta \mu_x$ [2$\pi$turns]"),
        (axes[0, 1], "muy", "phase", r"$\Delta \mu_y$ [2$\pi$turns]"),
        (axes[1, 0], "betx", "beta", r"$\Delta \beta_x / \beta_{x,\mathrm{meas}}$ [%]"),
        (axes[1, 1], "bety", "beta", r"$\Delta \beta_y / \beta_{y,\mathrm{meas}}$ [%]"),
    ]

    def _plot(ax, xdata, ydata, yerr, color_key, legend_label, fmt, linewidth):
        ax.errorbar(
            xdata,
            ydata,
            yerr=yerr,
            fmt=fmt,
            markersize=2.5,
            linewidth=linewidth,
            label=legend_label,
            color=PLOT_COLORS[color_key],
            alpha=0.9,
            capsize=2,
        )

    for ax, column, plot_type, ylabel in plot_specs:
        meas_err_col = {"mux": "errmux", "muy": "errmuy", "betx": "errbetx", "bety": "errbety"}.get(column)
        meas_err = meas_full.get(meas_err_col) if meas_err_col is not None else None
        for label, df, fmt, linewidth in series_to_plot:
            if plot_type == "phase":
                yvals, yerr = _diff(df[column], df.get(f"{column}_err"), meas_full[column], meas_err)
                plot_label = label
            else:
                yvals, yerr = _rel_diff(df[column], df.get(f"{column}_err"), meas_full[column], meas_err)
                plot_label = _rms_label(label, yvals, "%")
            _plot(ax, xvals, yvals, yerr, label, plot_label, fmt, linewidth)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(visible=True, alpha=0.3)
        ax.legend(fontsize=8)
        add_ip_positions_to_plot(ax, ip_positions)

    axes[1, 0].set_xlabel("S (m)")
    axes[1, 1].set_xlabel("S (m)")

    plt.tight_layout()
    _path = results_dir / f"phase_advance_{squeeze_step}_fullring.png"
    plt.savefig(_path, dpi=150)
    print(f"Saved plot: {_path.resolve()}")

    def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        return next((col for col in candidates if col in df.columns), None)

    x_col = _pick_col(meas_full, ("x", "orbitx", "cox"))
    y_col = _pick_col(meas_full, ("y", "orbity", "coy"))
    x_err_col = _pick_col(meas_full, ("errx", "errorbitx", "errcox"))
    y_err_col = _pick_col(meas_full, ("erry", "errorbity", "errcoy"))

    if x_col is None and y_col is None:
        logging.getLogger(__name__).warning(
            "No closed-orbit columns found in measurement twiss. Skipping full-ring closed-orbit plot."
        )
    else:
        fig_orbit, orbit_axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        orbit_specs = [
            (orbit_axes[0], x_col, r"$x$ [mm]"),
            (orbit_axes[1], y_col, r"$y$ [mm]"),
        ]

        for ax, col, ylabel in orbit_specs:
            if col is None:
                ax.text(0.5, 0.5, "Orbit column unavailable in measurement", transform=ax.transAxes, ha="center", va="center")
                ax.grid(visible=True, alpha=0.3)
                add_ip_positions_to_plot(ax, ip_positions)
                continue

            meas_yerr = None
            if col == x_col and x_err_col is not None:
                meas_yerr = 1.0e3 * meas_full[x_err_col]
            elif col == y_col and y_err_col is not None:
                meas_yerr = 1.0e3 * meas_full[y_err_col]

            ax.errorbar(
                xvals, 1.0e3 * meas_full[col], yerr=meas_yerr,
                fmt="o-", markersize=2.5, linewidth=1.0,
                label=MEASUREMENT_LABEL, color=PLOT_COLORS[MEASUREMENT_LABEL], alpha=0.9, capsize=2,
            )
            ax.plot(
                xvals, 1.0e3 * base_full[col], "s-", markersize=2.5, linewidth=1.0,
                label=_rms_label(DESIGN_OPTICS_LABEL, 1.0e3 * (base_full[col] - meas_full[col]), "mm"),
                color=PLOT_COLORS[DESIGN_OPTICS_LABEL], alpha=0.9,
            )
            if include_best_knowledge_model:
                ax.plot(
                    xvals, 1.0e3 * online_full[col], "^-", markersize=2.5, linewidth=1.0,
                    label=_rms_label(BEST_KNOWLEDGE_LABEL, 1.0e3 * (online_full[col] - meas_full[col]), "mm"),
                    color=PLOT_COLORS[BEST_KNOWLEDGE_LABEL], alpha=0.9,
                )
            if est_full is not None:
                ax.plot(
                    xvals, 1.0e3 * est_full[col], "d-", markersize=2.5, linewidth=1.0,
                    label=_rms_label(BETTER_KNOWLEDGE_LABEL, 1.0e3 * (est_full[col] - meas_full[col]), "mm"),
                    color=PLOT_COLORS[BETTER_KNOWLEDGE_LABEL], alpha=0.9,
                )
            ax.set_ylabel(ylabel)
            ax.grid(visible=True, alpha=0.3)
            ax.legend(fontsize=8)
            add_ip_positions_to_plot(ax, ip_positions)

        orbit_axes[1].set_xlabel("S (m)")
        plt.tight_layout()
        _path = results_dir / f"closed_orbit_{squeeze_step}_fullring.png"
        plt.savefig(_path, dpi=150)
        print(f"Saved plot: {_path.resolve()}")

    assumed_meas_chrom = 10.0
    chrom_labels = [MEASUREMENT_LABEL, DESIGN_OPTICS_LABEL]
    chrom_dq1 = [assumed_meas_chrom, _header_float(twiss_basic, "dq1", "DQ1")]
    chrom_dq2 = [assumed_meas_chrom, _header_float(twiss_basic, "dq2", "DQ2")]
    if include_best_knowledge_model:
        chrom_labels.append(BEST_KNOWLEDGE_LABEL)
        chrom_dq1.append(_header_float(twiss_online, "dq1", "DQ1"))
        chrom_dq2.append(_header_float(twiss_online, "dq2", "DQ2"))
    if twiss_eff_online is not None:
        chrom_labels.append(BETTER_KNOWLEDGE_LABEL)
        chrom_dq1.append(_header_float(twiss_eff_online, "dq1", "DQ1"))
        chrom_dq2.append(_header_float(twiss_eff_online, "dq2", "DQ2"))

    if any(v is not None for v in chrom_dq1 + chrom_dq2):
        fig_chrom, ax_chrom = plt.subplots(figsize=(10, 5))
        xpos = np.array([0.0, 1.0], dtype=float)
        width = 0.65
        for label, dq1, dq2 in zip(chrom_labels, chrom_dq1, chrom_dq2, strict=False):
            ax_chrom.bar(
                xpos,
                [np.nan if dq1 is None else dq1, np.nan if dq2 is None else dq2],
                width=width, label=label,
                color=PLOT_COLORS.get(label, "tab:gray"), alpha=0.45,
            )
        ax_chrom.set_xticks(xpos)
        ax_chrom.set_xticklabels([r"$dq_1$", r"$dq_2$"])
        ax_chrom.set_ylabel("Chromaticity")
        ax_chrom.set_title("Full-ring chromaticity from Twiss headers")
        ax_chrom.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
        ax_chrom.grid(axis="y", alpha=0.3)
        ax_chrom.legend()
        ax_chrom.text(
            0.01, 0.98,
            "Measurement chromaticity assumed for now: dq1 = dq2 = 10",
            transform=ax_chrom.transAxes, va="top", ha="left", fontsize=9,
        )
        plt.tight_layout()
        _path = results_dir / f"chromaticity_{squeeze_step}_fullring.png"
        plt.savefig(_path, dpi=150)
        print(f"Saved plot: {_path.resolve()}")
    else:
        logging.getLogger(__name__).warning(
            "Could not find dq1/dq2 in twiss headers for any full-ring model. Skipping chromaticity plot."
        )

    if accelerator.b2_errors is not None:
        b2_table = read_b2_error_table(accelerator.b2_errors)
        elem_names = list(b2_table.keys())
        elem_pos = get_element_positions(accelerator, elem_names)
        b2_with_pos = sorted(
            ((name, b2_table[name], pos) for name, pos in elem_pos.items() if name in b2_table),
            key=lambda t: t[2],
        )
        if b2_with_pos:
            fig_b2, ax_b2 = plt.subplots(figsize=(16, 4))
            ax_b2.bar(
                [pos for _, _, pos in b2_with_pos],
                [k1l for _, k1l, _ in b2_with_pos],
                width=20.0,
                color="tab:purple",
                alpha=0.7,
            )
            ax_b2.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax_b2.set_xlabel("S (m)")
            ax_b2.set_ylabel(r"$K_1 L$ [m$^{-1}$]")
            ax_b2.set_title("Dipole b2 errors")
            ax_b2.grid(visible=True, alpha=0.3)
            add_ip_positions_to_plot(ax_b2, ip_positions)
            plt.tight_layout()
            _path = results_dir / f"b2_errors_{squeeze_step}_fullring.png"
            plt.savefig(_path, dpi=150)
            print(f"Saved plot: {_path.resolve()}")

    plt.show()


def _normalize_phase(df: pd.DataFrame, columns: tuple[str, str], start_bpm: str) -> pd.DataFrame:
    from aba_optimiser.measurements.plotting.core import _normalize_phase as impl

    return impl(df, columns, start_bpm)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True)
    parser.add_argument("--squeeze-step", type=str, required=True)
    parser.add_argument("--optics", action="store_true")
    parser.add_argument("--frequency", type=str, default="0Hz")
    parser.add_argument(
        "--estimate-source",
        type=str,
        choices=["none", "estimates", "quad-checkpoint", "bend-checkpoint"],
        default="estimates",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-uncertainty", type=float, default=None)
    parser.add_argument("--fullring", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--without-best-knowledge", action="store_true")
    args = parser.parse_args()

    context = prepare_plot_context(
        beam=args.beam,
        squeeze_step=args.squeeze_step,
        use_optics=args.optics,
        frequency=args.frequency,
        estimate_source=args.estimate_source,
        checkpoint_dir=args.checkpoint_dir,
        max_uncertainty=args.max_uncertainty,
        fullring_knob_diffs=True,
    )
    plot_fullring_comparison(
        context.accelerator,
        context.design_accelerator,
        context.all_estimates,
        context.analysis_dir,
        context.squeeze_step,
        context.results_dir,
        context.tune_knobs_file,
        context.corrector_file,
        context.beam,
        include_best_knowledge_model=not args.without_best_knowledge,
        deltap=context.deltap,
    )


if __name__ == "__main__":
    main()
