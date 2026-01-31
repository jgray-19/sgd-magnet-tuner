import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.measurements.squeeze_helpers import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
    get_or_make_sequence,
    get_results_dir,
    load_estimates,
)
from aba_optimiser.measurements.twiss_from_measurement import build_twiss_from_measurements

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import argparse
import logging

import matplotlib.pyplot as plt
import tfs


def get_twiss_without_errors(
    sequence_file: Path,
    just_bpms: bool,
    beam: int,
    beam_energy: float,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs_file: Path | None = None,
    corrector_file: Path | None = None,
) -> tfs.TfsDataFrame:
    """Get twiss data from a model with optional tune knobs and estimated magnets."""
    mad = OptimisationMadInterface(
        sequence_file=str(sequence_file),
        seq_name=f"lhcb{beam}",
        beam_energy=beam_energy,
        bpm_pattern="BPM",
        corrector_strengths=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    if estimated_magnets is not None:
        mad.set_magnet_strengths(estimated_magnets)
    return mad.run_twiss(observe=int(just_bpms))


def find_true_values(
    seq_file: Path,
    beam: int,
    estimates: dict[str, dict[str, float]],
    tune_knobs_file: Path,
    beam_energy: float,
) -> dict[str, dict[str, float]]:
    """Find true quadrupole values from the sequence file with tune knobs applied."""
    mad = OptimisationMadInterface(
        sequence_file=str(seq_file),
        seq_name=f"lhcb{beam}",
        beam_energy=beam_energy,
        tune_knobs_file=tune_knobs_file,
        corrector_strengths=None,
    )
    actual = {}
    for arc, mags in estimates.items():
        actual[arc] = {}
        for mag in mags:
            elem_name, property_name = mag.rsplit(".", 1)
            true_k1 = mad.mad.MADX[elem_name][property_name]
            actual[arc][mag] = true_k1
    return actual


def plot_quad_diffs(estimates: dict, actual: dict, squeeze_step: str, results_dir: Path) -> None:
    """Plot quadrupole differences for all arcs."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for arc_num in range(1, 9):
        ax = axes[arc_num - 1]
        arc_key = f"Arc {arc_num}"
        if arc_key in estimates:
            mags = list(estimates[arc_key].keys())
            rel_diffs = [
                abs(estimates[arc_key][m] - actual[arc_key][m])
                / abs(actual[arc_key][m])
                * 100
                * np.sign(actual[arc_key][m])
                for m in mags
            ]
            ax.bar(range(len(mags)), rel_diffs)
            ax.set_xticks(range(len(mags)))
            ax.set_xticklabels([m.split(".")[1] for m in mags], rotation=90)
            ax.set_title(f"Arc {arc_num}")
            ax.set_ylabel("Estimated k1")
    plt.tight_layout()
    plt.savefig(results_dir / f"quad_diffs_{squeeze_step}.png")
    plt.show()


def get_arc_ranges(beam: int) -> dict[int, tuple[str, str]]:
    """Get arc ranges (start BPM -> end BPM) for each arc.

    Returns dictionary mapping arc number to (start_element, end_element) tuples.
    Arc goes from 13R -> 12L (e.g., Arc 1 goes from IP1 to IP2).
    """
    if beam == 1:
        return {
            1: ("BPM.13R1.B1", "BPM.12L2.B1"),
            2: ("BPM.13R2.B1", "BPM.12L3.B1"),
            3: ("BPM.13R3.B1", "BPM.12L4.B1"),
            4: ("BPM.13R4.B1", "BPM.12L5.B1"),
            5: ("BPM.13R5.B1", "BPM.12L6.B1"),
            6: ("BPM.13R6.B1", "BPM.12L7.B1"),
            7: ("BPM.13R7.B1", "BPM.12L8.B1"),
            8: ("BPM.13R8.B1", "BPM.12L1.B1"),
        }
    return {
        1: ("BPM.13L1.B2", "BPM.12R2.B2"),
        2: ("BPM.13L2.B2", "BPM.12R3.B2"),
        3: ("BPM.13L3.B2", "BPM.12R4.B2"),
        4: ("BPM.13L4.B2", "BPM.12R5.B2"),
        5: ("BPM.13L5.B2", "BPM.12R6.B2"),
        6: ("BPM.13L6.B2", "BPM.12R7.B2"),
        7: ("BPM.13L7.B2", "BPM.12R8.B2"),
        8: ("BPM.13L8.B2", "BPM.12R1.B2"),
    }


def _normalize_phase(df: pd.DataFrame, mu_cols: tuple[str, str], start_bpm: str) -> pd.DataFrame:
    """Shift phase columns so the chosen start BPM sits at zero."""
    if df.empty:
        return df

    df = df.copy()
    start_row = df.loc[start_bpm] if start_bpm is not None else df.iloc[0]
    df[mu_cols[0]] = df[mu_cols[0]] - start_row[mu_cols[0]]
    df[mu_cols[1]] = df[mu_cols[1]] - start_row[mu_cols[1]]
    return df


def get_phase_advance_through_arc(
    seq_file: Path,
    beam: int,
    beam_energy: float,
    arc_start: str,
    arc_end: str,
    meas_twiss: pd.DataFrame,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs_file: Path | None = None,
) -> pd.DataFrame:
    """Get phase advance through an arc using model with range and beta0 from measurement.

    Args:
        seq_file: Path to sequence file
        beam: Beam number
        beam_energy: Beam energy in GeV
        arc_start: Starting element name
        arc_end: Ending element name
        meas_twiss: Twiss from measurement (for beta0 initialization)
        estimated_magnets: Optional magnet strength estimates
        tune_knobs_file: Optional tune knobs file

    Returns:
        DataFrame with phase advances at BPM locations
    """
    mad = OptimisationMadInterface(
        sequence_file=str(seq_file),
        seq_name=f"lhcb{beam}",
        beam_energy=beam_energy,
        bpm_pattern="BPM",
        corrector_strengths=None,
        tune_knobs_file=tune_knobs_file,
    )

    if estimated_magnets is not None:
        mad.set_magnet_strengths(estimated_magnets)

    # Optics at arc start (measurement beta0)
    optics = {
        "betx": meas_twiss.loc[arc_start, "betx"],
        "bety": meas_twiss.loc[arc_start, "bety"],
        "alfx": meas_twiss.loc[arc_start, "alfx"],
        "alfy": meas_twiss.loc[arc_start, "alfy"],
        "dx": meas_twiss.loc[arc_start, "dx"],
        "dpx": meas_twiss.loc[arc_start, "dpx"],
        "dy": meas_twiss.loc[arc_start, "dy"],
        "dpy": meas_twiss.loc[arc_start, "dpy"],
    }
    optics_err = {
        "betx": meas_twiss.loc[arc_start, "errbetx"],
        "bety": meas_twiss.loc[arc_start, "errbety"],
        "alfx": meas_twiss.loc[arc_start, "erralfx"],
        "alfy": meas_twiss.loc[arc_start, "erralfy"],
        "dx": meas_twiss.loc[arc_start, "errdx"],
        "dpx": 0.0,
        "dy": meas_twiss.loc[arc_start, "errdy"],
        "dpy": 0.0,
    }

    run_twiss_string = f"""
    local B0 = MAD.beta0 {{
        beta11=py:recv(),
        beta22=py:recv(),
        alfa11=py:recv(),
        alfa22=py:recv(),
        dx=py:recv(),
        dpx=py:recv(),
        dy=py:recv(),
        dpy=py:recv(),
    }}
    twiss_result = twiss {{
        sequence = loaded_sequence,
        range ="{arc_start}/{arc_end}",
        X0 = B0,
        observe = 1,
    }}
    """

    def run_twiss(values: list[float]) -> pd.DataFrame:
        mad.mad.send(run_twiss_string)
        for val in values:
            mad.mad.send(val)
        df = mad.mad.twiss_result.to_df().set_index("name")
        return df.rename(columns={"mu1": "mux", "mu2": "muy"})

    base = run_twiss(list(optics.values()))
    plus = run_twiss([optics[k] + optics_err[k] for k in optics])
    minus = run_twiss([optics[k] - optics_err[k] for k in optics])

    base["mux_err"] = abs(plus["mux"] - minus["mux"]) / 2
    base["muy_err"] = abs(plus["muy"] - minus["muy"]) / 2

    return _normalize_phase(base, ("mux", "muy"), start_bpm=arc_start)


def get_measurement_phase_through_arc(
    meas_twiss: pd.DataFrame,
    arc_start: str,
    arc_end: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Get phase advance through arc from measurement data.

    Args:
        meas_twiss: Twiss from measurement
        arc_start: Starting element name
        arc_end: Ending element name

    Returns:
        DataFrame with relative phase advances
    """
    # Get all BPMs in the arc range
    start_s = meas_twiss.loc[arc_start, "s"]
    end_s = meas_twiss.loc[arc_end, "s"]

    if end_s < start_s:
        arc_bpms = meas_twiss[(meas_twiss["s"] >= start_s) | (meas_twiss["s"] <= end_s)]
    else:
        arc_bpms = meas_twiss[(meas_twiss["s"] >= start_s) & (meas_twiss["s"] <= end_s)]

    if arc_bpms.empty:
        return arc_bpms, []

    candidate_bpms = list(arc_bpms.head(2).index)
    print(candidate_bpms)
    return arc_bpms, candidate_bpms


def plot_phase_advances(
    seq_file: Path,
    all_estimates: dict,
    analysis_dir: Path,
    squeeze_step: str,
    results_dir: Path,
    tune_knobs_file: Path,
    beam_energy: float,
    beam: int,
) -> None:
    """Plot phase advance comparison through each arc with four models."""
    meas_twiss, _ = build_twiss_from_measurements(
        analysis_dir, include_errors=True, reverse_bpm_order=beam == 2
    )
    meas_twiss.columns = [col.lower() for col in meas_twiss.columns]

    # Get arc ranges
    arc_ranges = get_arc_ranges(beam)

    for arc_num in range(1, 9):
        arc_start, arc_end = arc_ranges[arc_num]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Get measurement arc (unnormalized) and candidate BPMs near arc start
        meas_phase_raw, candidate_bpms = get_measurement_phase_through_arc(meas_twiss, arc_start, arc_end)

        # Initial base model (no knobs/estimates) to pick best start BPM by minimal phase error
        def choose_best_start_bpm(candidates: list[str]) -> str:
            best_bpm = candidates[0]
            min_error = float("inf")
            for start_bpm in candidates:
                if start_bpm not in meas_twiss.index:
                    continue
                phase_initial = get_phase_advance_through_arc(
                    seq_file,
                    beam,
                    beam_energy,
                    start_bpm,
                    arc_end,
                    meas_twiss,
                    estimated_magnets=None,
                    tune_knobs_file=None,
                )
                var_x = np.sum((phase_initial["mux_err"]) ** 2)
                var_y = np.sum((phase_initial["muy_err"]) ** 2)
                total_var = var_x + var_y
                if total_var < min_error:
                    min_error = total_var
                    best_bpm = start_bpm
            return best_bpm

        start_bpm = choose_best_start_bpm(candidate_bpms)

        # Normalize measurement to chosen start BPM
        meas_phase = _normalize_phase(meas_phase_raw, ("mux", "muy"), start_bpm=start_bpm)

        # Recompute models using chosen start BPM (shared beta0 and range)
        phase_basic = get_phase_advance_through_arc(
            seq_file,
            beam,
            beam_energy,
            start_bpm,
            arc_end,
            meas_twiss,
            estimated_magnets=None,
            tune_knobs_file=None,
        )

        phase_online = get_phase_advance_through_arc(
            seq_file,
            beam,
            beam_energy,
            start_bpm,
            arc_end,
            meas_twiss,
            estimated_magnets=None,
            tune_knobs_file=tune_knobs_file,
        )

        phase_eff_online = get_phase_advance_through_arc(
            seq_file,
            beam,
            beam_energy,
            start_bpm,
            arc_end,
            meas_twiss,
            estimated_magnets=all_estimates,
            tune_knobs_file=tune_knobs_file,
        )

        # Find common BPMs
        common_bpms = (
            meas_phase.index.intersection(phase_basic.index)
            .intersection(phase_online.index)
            .intersection(phase_eff_online.index)
        )
        if len(common_bpms) == 0:
            logging.warning(f"No common BPMs found in Arc {arc_num}, skipping plot.")
            continue

        def _diff(series: pd.Series, series_err: pd.Series | None, meas: pd.Series, meas_err: pd.Series | None) -> tuple[pd.Series, pd.Series | None]:
            delta = series.loc[common_bpms] - meas.loc[common_bpms]
            if series_err is None and meas_err is None:
                return delta, None
            series_err_vals = series_err.loc[common_bpms] if series_err is not None else 0.0
            meas_err_vals = meas_err.loc[common_bpms] if meas_err is not None else 0.0
            return delta, np.sqrt(series_err_vals**2 + meas_err_vals**2)

        def _plot(ax, xvals, series, yerr, label, fmt):
            ax.errorbar(
                xvals,
                series,
                yerr=yerr,
                fmt=fmt,
                markersize=3,
                label=label,
                alpha=0.8,
                capsize=2,
            )

        # Horizontal
        ax_x = axes[0]
        meas_mux = meas_phase["mux"]
        meas_mux_err = meas_phase.get("errmux")
        base_dx, base_dx_err = _diff(phase_basic["mux"], phase_basic.get("mux_err"), meas_mux, meas_mux_err)
        _plot(ax_x, meas_phase.loc[common_bpms, "s"], base_dx, base_dx_err, "Base model - meas", "s-")
        online_dx, online_dx_err = _diff(phase_online["mux"], phase_online.get("mux_err"), meas_mux, meas_mux_err)
        _plot(ax_x, meas_phase.loc[common_bpms, "s"], online_dx, online_dx_err, "Model+tunes - meas", "^-")
        eff_dx, eff_dx_err = _diff(phase_eff_online["mux"], phase_eff_online.get("mux_err"), meas_mux, meas_mux_err)
        _plot(ax_x, meas_phase.loc[common_bpms, "s"], eff_dx, eff_dx_err, "Model+tunes+ests - meas", "d-")
        ax_x.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_x.set_ylabel("Δ Phase X vs meas (2π)")
        ax_x.set_title(f"Arc {arc_num} - Horizontal Δ")
        ax_x.legend(fontsize=8)
        ax_x.grid(visible=True, alpha=0.3)

        # Vertical
        ax_y = axes[1]
        meas_muy = meas_phase["muy"]
        meas_muy_err = meas_phase.get("errmuy")
        base_dy, base_dy_err = _diff(phase_basic["muy"], phase_basic.get("muy_err"), meas_muy, meas_muy_err)
        _plot(ax_y, meas_phase.loc[common_bpms, "s"], base_dy, base_dy_err, "Base model - meas", "s-")
        online_dy, online_dy_err = _diff(phase_online["muy"], phase_online.get("muy_err"), meas_muy, meas_muy_err)
        _plot(ax_y, meas_phase.loc[common_bpms, "s"], online_dy, online_dy_err, "Model+tunes - meas", "^-")
        eff_dy, eff_dy_err = _diff(phase_eff_online["muy"], phase_eff_online.get("muy_err"), meas_muy, meas_muy_err)
        _plot(ax_y, meas_phase.loc[common_bpms, "s"], eff_dy, eff_dy_err, "Model+tunes+ests - meas", "d-")
        ax_y.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax_y.set_ylabel("Δ Phase Y vs meas (2π)")
        ax_y.set_title(f"Arc {arc_num} - Vertical Δ")
        ax_y.legend(fontsize=8)
        ax_y.grid(visible=True, alpha=0.3)

        ax_x.set_xlabel("S (m)")
        ax_y.set_xlabel("S (m)")

        fig.suptitle(f"Phase advance comparison - {squeeze_step} - Arc {arc_num} (start: {start_bpm})", fontsize=12)
        plt.tight_layout()
        plt.savefig(results_dir / f"phase_advance_{squeeze_step}_arc{arc_num}.png", dpi=150)
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True, help="Beam number")
    parser.add_argument(
        "--squeeze-step", type=str, required=True, help="Squeeze step, e.g., '1.2m'"
    )
    parser.add_argument(
        "--optics", action="store_true", help="Use the optimisation from optics measurements"
    )
    args = parser.parse_args()
    beam = args.beam
    squeeze_step = args.squeeze_step
    use_optics = args.optics

    beam_path = BETABEAT_DIR / get_measurement_date(squeeze_step) / f"LHCB{beam}/"
    model_base_dir = beam_path / "Models/"
    analysis_base_dir = beam_path / "Results/"

    results_dir = get_results_dir(beam)

    temp_dir = PROJECT_ROOT / "temp_analysis_plots"
    temp_dir.mkdir(exist_ok=True)

    model_dir = model_base_dir / MODEL_DIRS[beam][squeeze_step]
    seq_file = get_or_make_sequence(beam, model_dir)

    fldr_name = "optics" if use_optics else "squeeze"
    estimates_file = PROJECT_ROOT / f"b{beam}_{fldr_name}_results" / f"quad_estimates_{squeeze_step}.txt"
    if not estimates_file.exists():
        print(f"Estimates file not found: {estimates_file}")
        return

    beam_energy = 6800.0
    print(f"Beam energy set to: {beam_energy} GeV")

    estimates = load_estimates(estimates_file)
    # Load tune knobs from 0Hz measurement
    tune_knobs_file = results_dir / f"tune_knobs_{squeeze_step}_0Hz.txt"
    if not tune_knobs_file.exists():
        raise FileNotFoundError(f"Tune knobs file not found: {tune_knobs_file}")
    print(f"Using tune knobs file: {tune_knobs_file}")

    actual = find_true_values(seq_file, beam, estimates, tune_knobs_file, beam_energy)

    plot_quad_diffs(estimates, actual, squeeze_step, results_dir)

    all_estimates = {}
    for arc in estimates.values():
        all_estimates.update(arc)

    analysis_dir = analysis_base_dir / ANALYSIS_DIRS[beam][squeeze_step]
    plot_phase_advances(
        seq_file,
        all_estimates,
        analysis_dir,
        squeeze_step,
        results_dir,
        tune_knobs_file,
        beam_energy,
        beam,
    )


if __name__ == "__main__":
    main()
