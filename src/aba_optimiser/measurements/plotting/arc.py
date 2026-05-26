"""Arc-by-arc phase advance and beta plotting CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tmom_recon import build_twiss_from_measurements

from aba_optimiser.measurements.plotting.core import (
    BEST_KNOWLEDGE_LABEL,
    BETTER_KNOWLEDGE_LABEL,
    DESIGN_OPTICS_LABEL,
    PLOT_COLORS,
    EstimateSource,
    get_arc_ranges,
    get_twiss_through_arc,
    parse_arc_spec,
    prepare_plot_context,
)


def plot_phase_advances(
    accelerator,
    design_accelerator,
    all_estimates: dict[str, float] | None,
    analysis_dir: Path,
    squeeze_step: str,
    results_dir: Path,
    tune_knobs_file: Path,
    corrector_file: Path | None,
    beam: int,
    arcs: list[int] | None = None,
    deltap: float = 0.0,
) -> None:
    """Plot phase advance comparison through each arc."""
    meas_twiss, _ = build_twiss_from_measurements(
        analysis_dir, include_errors=True, reverse_bpm_order=beam == 2
    )
    meas_twiss.columns = [col.lower() for col in meas_twiss.columns]

    arc_ranges = get_arc_ranges(beam)
    arc_list = arcs if arcs is not None else list(range(1, 9))

    for arc_num in arc_list:
        arc_start, arc_end = arc_ranges[arc_num]
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        meas_phase_raw, candidate_bpms = get_measurement_phase_through_arc(meas_twiss, arc_start, arc_end)

        def choose_best_start_bpm(candidates: list[str]) -> str:
            best_bpm = candidates[0]
            min_error = float("inf")
            for start_bpm in candidates:
                if start_bpm not in meas_twiss.index:
                    continue
                phase_initial = get_twiss_through_arc(
                    design_accelerator,
                    start_bpm,
                    arc_end,
                    meas_twiss,
                    estimated_magnets=None,
                    tune_knobs_file=None,
                    corrector_file=None,
                )
                var_x = np.sum((phase_initial["mux_err"]) ** 2)
                var_y = np.sum((phase_initial["muy_err"]) ** 2)
                total_var = var_x + var_y
                if total_var < min_error:
                    min_error = total_var
                    best_bpm = start_bpm
            return best_bpm

        start_bpm = choose_best_start_bpm(candidate_bpms)
        meas_phase = _normalize_phase(meas_phase_raw, ("mux", "muy"), start_bpm=start_bpm)

        twiss_basic = get_twiss_through_arc(
            design_accelerator,
            start_bpm,
            arc_end,
            meas_twiss,
            estimated_magnets=None,
            tune_knobs_file=None,
            corrector_file=None,
        )
        twiss_online = get_twiss_through_arc(
            accelerator,
            start_bpm,
            arc_end,
            meas_twiss,
            estimated_magnets=None,
            tune_knobs_file=tune_knobs_file,
            corrector_file=corrector_file,
        )

        twiss_eff_online = None
        common_bpms = meas_phase.index.intersection(twiss_basic.index).intersection(twiss_online.index)
        if all_estimates is not None:
            twiss_eff_online = get_twiss_through_arc(
                accelerator,
                start_bpm,
                arc_end,
                meas_twiss,
                estimated_magnets=all_estimates,
                tune_knobs_file=tune_knobs_file,
                corrector_file=corrector_file,
                deltap=deltap,
            )
            common_bpms = common_bpms.intersection(twiss_eff_online.index)
        if len(common_bpms) == 0:
            logging.warning("No common BPMs found in Arc %s, skipping plot.", arc_num)
            continue

        def _diff(
            series: pd.Series,
            series_err: pd.Series | None,
            meas: pd.Series,
            meas_err: pd.Series | None,
        ) -> tuple[pd.Series, pd.Series | None]:
            delta = series.loc[common_bpms] - meas.loc[common_bpms]
            if series_err is None and meas_err is None:
                return delta, None
            series_err_vals = series_err.loc[common_bpms] if series_err is not None else 0.0
            meas_err_vals = meas_err.loc[common_bpms] if meas_err is not None else 0.0
            return delta, np.sqrt(series_err_vals**2 + meas_err_vals**2)

        def _rel_diff(
            series: pd.Series,
            series_err: pd.Series | None,
            meas: pd.Series,
            meas_err: pd.Series | None,
        ) -> tuple[pd.Series, pd.Series | None]:
            rel_delta = (series.loc[common_bpms] - meas.loc[common_bpms]) / meas.loc[common_bpms] * 100
            if series_err is None and meas_err is None:
                return rel_delta, None
            series_err_vals = series_err.loc[common_bpms] if series_err is not None else 0.0
            meas_err_vals = meas_err.loc[common_bpms] if meas_err is not None else 0.0
            rel_err = np.sqrt(series_err_vals**2 + meas_err_vals**2) / np.abs(meas.loc[common_bpms]) * 100
            return rel_delta, rel_err

        def _plot(ax, xvals, series, yerr, label, fmt):
            ax.errorbar(
                xvals,
                series,
                yerr=yerr,
                fmt=fmt,
                markersize=3,
                label=label,
                color=PLOT_COLORS[label],
                alpha=0.8,
                capsize=2,
            )

        ax_phase_x = axes[0, 0]
        meas_mux = meas_phase["mux"]
        meas_mux_err = meas_phase.get("errmux")
        base_dx, base_dx_err = _diff(twiss_basic["mux"], twiss_basic.get("mux_err"), meas_mux, meas_mux_err)
        _plot(ax_phase_x, meas_phase.loc[common_bpms, "s"], base_dx, base_dx_err, DESIGN_OPTICS_LABEL, "s-")
        online_dx, online_dx_err = _diff(
            twiss_online["mux"], twiss_online.get("mux_err"), meas_mux, meas_mux_err
        )
        _plot(ax_phase_x, meas_phase.loc[common_bpms, "s"], online_dx, online_dx_err, BEST_KNOWLEDGE_LABEL, "^-")
        if twiss_eff_online is not None:
            eff_dx, eff_dx_err = _diff(
                twiss_eff_online["mux"], twiss_eff_online.get("mux_err"), meas_mux, meas_mux_err
            )
            _plot(ax_phase_x, meas_phase.loc[common_bpms, "s"], eff_dx, eff_dx_err, BETTER_KNOWLEDGE_LABEL, "d-")
        ax_phase_x.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_phase_x.set_ylabel(r"$\Delta \mu_x$ vs meas [turns, $1 = 2\pi$]")
        ax_phase_x.set_title(rf"Arc {arc_num} - Horizontal phase advance $\Delta \mu_x$")
        ax_phase_x.legend(fontsize=8)
        ax_phase_x.grid(visible=True, alpha=0.3)
        ax_phase_x.set_xlabel("S (m)")

        ax_phase_y = axes[0, 1]
        meas_muy = meas_phase["muy"]
        meas_muy_err = meas_phase.get("errmuy")
        base_dy, base_dy_err = _diff(twiss_basic["muy"], twiss_basic.get("muy_err"), meas_muy, meas_muy_err)
        _plot(ax_phase_y, meas_phase.loc[common_bpms, "s"], base_dy, base_dy_err, DESIGN_OPTICS_LABEL, "s-")
        online_dy, online_dy_err = _diff(
            twiss_online["muy"], twiss_online.get("muy_err"), meas_muy, meas_muy_err
        )
        _plot(ax_phase_y, meas_phase.loc[common_bpms, "s"], online_dy, online_dy_err, BEST_KNOWLEDGE_LABEL, "^-")
        if twiss_eff_online is not None:
            eff_dy, eff_dy_err = _diff(
                twiss_eff_online["muy"], twiss_eff_online.get("muy_err"), meas_muy, meas_muy_err
            )
            _plot(ax_phase_y, meas_phase.loc[common_bpms, "s"], eff_dy, eff_dy_err, BETTER_KNOWLEDGE_LABEL, "d-")
        ax_phase_y.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_phase_y.set_ylabel(r"$\Delta \mu_y$ vs meas [2$\pi$turns]")
        ax_phase_y.set_title(rf"Arc {arc_num} - Vertical phase advance $\Delta \mu_y$")
        ax_phase_y.legend(fontsize=8)
        ax_phase_y.grid(visible=True, alpha=0.3)
        ax_phase_y.set_xlabel("S (m)")

        ax_beta_x = axes[1, 0]
        meas_betx = meas_twiss.loc[common_bpms, "betx"]
        meas_betx_err = meas_twiss.loc[common_bpms].get("errbetx")
        base_dbetx, base_dbetx_err = _rel_diff(
            twiss_basic["betx"], twiss_basic.get("betx_err"), meas_betx, meas_betx_err
        )
        _plot(ax_beta_x, meas_twiss.loc[common_bpms, "s"], base_dbetx, base_dbetx_err, DESIGN_OPTICS_LABEL, "s-")
        online_dbetx, online_dbetx_err = _rel_diff(
            twiss_online["betx"], twiss_online.get("betx_err"), meas_betx, meas_betx_err
        )
        _plot(ax_beta_x, meas_twiss.loc[common_bpms, "s"], online_dbetx, online_dbetx_err, BEST_KNOWLEDGE_LABEL, "^-")
        if twiss_eff_online is not None:
            eff_dbetx, eff_dbetx_err = _rel_diff(
                twiss_eff_online["betx"], twiss_eff_online.get("betx_err"), meas_betx, meas_betx_err
            )
            _plot(ax_beta_x, meas_twiss.loc[common_bpms, "s"], eff_dbetx, eff_dbetx_err, BETTER_KNOWLEDGE_LABEL, "d-")
        ax_beta_x.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_beta_x.set_ylabel("Δ β_x / β_x vs meas (%)")
        ax_beta_x.set_title(f"Arc {arc_num} - Horizontal β Δ")
        ax_beta_x.legend(fontsize=8)
        ax_beta_x.grid(visible=True, alpha=0.3)
        ax_beta_x.set_xlabel("S (m)")

        ax_beta_y = axes[1, 1]
        meas_bety = meas_twiss.loc[common_bpms, "bety"]
        meas_bety_err = meas_twiss.loc[common_bpms].get("errbety")
        base_dbety, base_dbety_err = _rel_diff(
            twiss_basic["bety"], twiss_basic.get("bety_err"), meas_bety, meas_bety_err
        )
        _plot(ax_beta_y, meas_twiss.loc[common_bpms, "s"], base_dbety, base_dbety_err, DESIGN_OPTICS_LABEL, "s-")
        online_dbety, online_dbety_err = _rel_diff(
            twiss_online["bety"], twiss_online.get("bety_err"), meas_bety, meas_bety_err
        )
        _plot(ax_beta_y, meas_twiss.loc[common_bpms, "s"], online_dbety, online_dbety_err, BEST_KNOWLEDGE_LABEL, "^-")
        if twiss_eff_online is not None:
            eff_dbety, eff_dbety_err = _rel_diff(
                twiss_eff_online["bety"], twiss_eff_online.get("bety_err"), meas_bety, meas_bety_err
            )
            _plot(ax_beta_y, meas_twiss.loc[common_bpms, "s"], eff_dbety, eff_dbety_err, BETTER_KNOWLEDGE_LABEL, "d-")
        ax_beta_y.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_beta_y.set_ylabel("Δ β_y / β_y vs meas (%)")
        ax_beta_y.set_title(f"Arc {arc_num} - Vertical β Δ")
        ax_beta_y.legend(fontsize=8)
        ax_beta_y.grid(visible=True, alpha=0.3)
        ax_beta_y.set_xlabel("S (m)")

        fig.suptitle(
            f"Phase advance & beta comparison - {squeeze_step} - Arc {arc_num} (start: {start_bpm})",
            fontsize=12,
        )
        plt.tight_layout()
        _path = results_dir / f"phase_advance_{squeeze_step}_arc{arc_num}.png"
        plt.savefig(_path, dpi=150)
        print(f"Saved plot: {_path.resolve()}")
    plt.show()


def _normalize_phase(df: pd.DataFrame, columns: tuple[str, ...], start_bpm: str) -> pd.DataFrame:
    from aba_optimiser.measurements.plotting.core import _normalize_phase as impl

    return impl(df, columns, start_bpm)


def get_measurement_phase_through_arc(meas_twiss: pd.DataFrame, arc_start: str, arc_end: str):
    from aba_optimiser.measurements.plotting.core import (
        get_measurement_phase_through_arc as impl,
    )

    return impl(meas_twiss, arc_start, arc_end)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True)
    parser.add_argument("--squeeze-step", type=str, required=True)
    parser.add_argument("--optics", action="store_true")
    parser.add_argument("--arcs", type=str, default=None)
    parser.add_argument("--frequency", type=str, default="0Hz")
    parser.add_argument(
        "--estimate-source",
        type=str,
        choices=["none", "estimates", "quad-checkpoint", "bend-checkpoint"],
        default="estimates",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-uncertainty", type=float, default=None)
    args = parser.parse_args()

    context = prepare_plot_context(
        beam=args.beam,
        squeeze_step=args.squeeze_step,
        use_optics=args.optics,
        frequency=args.frequency,
        estimate_source=args.estimate_source,
        checkpoint_dir=args.checkpoint_dir,
        max_uncertainty=args.max_uncertainty,
        fullring_knob_diffs=False,
    )
    plot_phase_advances(
        context.accelerator,
        context.design_accelerator,
        context.all_estimates,
        context.analysis_dir,
        context.squeeze_step,
        context.results_dir,
        context.tune_knobs_file,
        context.corrector_file,
        context.beam,
        arcs=parse_arc_spec(args.arcs),
        deltap=context.deltap,
    )


if __name__ == "__main__":
    main()
