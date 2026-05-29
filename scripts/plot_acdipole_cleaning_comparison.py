#!/usr/bin/env python3
"""Plot raw, SVD-cleaned, and SVD+AC-dipole-cleaned phase spaces for one squeeze step.

The script reuses the same measurement bookkeeping as the squeeze optimisation flow,
but keeps all reconstruction stages in memory. It plots the BPM immediately upstream
and downstream of the AC dipole, with overlays for:

1. raw/noisy reconstruction,
2. reconstruction after SVD cleaning,
3. reconstruction after SVD cleaning plus AC-dipole cleaning.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from tmom_recon import (
    ACDipoleConfig,
    build_twiss_from_measurements,
    calculate_ac_dipole_momentum,
    calculate_pz_measurement,
    calculate_transverse_pz_nbpm,
)
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from tmom_recon.svd import svd_clean_measurements
from turn_by_turn import read_tbt

from aba_optimiser.accelerators import LHC
from aba_optimiser.measurements.create_datafile import (
    build_madng_twiss_table,
)
from aba_optimiser.measurements.loading import convert_tbt_to_dataframes
from aba_optimiser.measurements.optimise_squeeze_quads import (
    MEAS_TIMES,
    ZEROHZ,
    get_analysis_folders,
    get_knob_files,
    get_measurement_files,
    get_sequence_creation_time,
)
from aba_optimiser.measurements.squeeze_config import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
)
from aba_optimiser.measurements.squeeze_helpers import (
    extract_tunes_from_job_file,
    get_or_make_sequence,
    get_results_dir,
)
from aba_optimiser.measurements.utils import find_all_bad_bpms
from aba_optimiser.measurements.variances import assign_known_noise_variances
from aba_optimiser.plotting.utils import setup_scientific_formatting

LOGGER = logging.getLogger(__name__)

RAW_LABEL = "Raw reconstruction"
SVD_LABEL = "SVD cleaned"
ACD_LABEL = "SVD + ACD cleaned"
NBPM_LABEL = "SVD + N-BPM"


def _get_paths(beam: int, squeeze_step: str) -> tuple[Path, Path, Path]:
    meas_date = get_measurement_date(squeeze_step)
    beam_root = BETABEAT_DIR / meas_date / f"LHCB{beam}"
    meas_base_dir = beam_root / "Measurements"
    model_dir = beam_root / "Models" / MODEL_DIRS[beam][squeeze_step]
    analysis_dir_name = ANALYSIS_DIRS[beam][squeeze_step]
    analysis_dir = beam_root / "Results" / analysis_dir_name
    return meas_base_dir, model_dir, analysis_dir


def _load_model_twiss_data(
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    pc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    job_file = model_dir / "job.create_model_nominal.madx"
    nat_x, nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
    nattunes = [nat_x, nat_y, 0.0]
    tunes = [drv_x, drv_y, 0.0]
    accelerator = LHC(
        beam=beam,
        pc=pc,
        sequence_file=sequence_path,
    )
    madng_output_dir = Path(
        tempfile.mkdtemp(prefix=f"acd_clean_twiss_b{beam}_{model_dir.name}_", dir="/tmp")
    )
    tws = build_madng_twiss_table(
        model_dir=model_dir,
        accelerator=accelerator,
        output_dir=madng_output_dir,
        nattunes=nattunes,
        tunes=tunes,
    )
    twiss_elements = tfs.read(madng_output_dir / "twiss_elements.dat")
    for file in madng_output_dir.iterdir():
        file.unlink()
    madng_output_dir.rmdir()  # Clean up the temporary directory after reading the twiss table

    def _normalise_twiss_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [str(col).lower() for col in df.columns]
        df = df.rename(
            columns={
                "betx": "beta11",
                "bety": "beta22",
                "alfx": "alfa11",
                "alfy": "alfa22",
                "mux": "mu1",
                "muy": "mu2",
            }
        )
        df.headers = {key.lower(): value for key, value in df.headers.items()}
        df = df.set_index("name")
        df.index = df.index.astype(str)
        return df

    tws = _normalise_twiss_columns(tws)
    twiss_elements = _normalise_twiss_columns(twiss_elements)
    return tws, twiss_elements


def _load_model_twiss(
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    pc: float,
) -> pd.DataFrame:
    tws, _ = _load_model_twiss_data(
        model_dir=model_dir,
        sequence_path=sequence_path,
        beam=beam,
        pc=pc,
    )
    return tws


def _load_model_twiss_elements(
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    pc: float,
) -> pd.DataFrame:
    _, twiss_elements = _load_model_twiss_data(
        model_dir=model_dir,
        sequence_path=sequence_path,
        beam=beam,
        pc=pc,
    )
    return twiss_elements


def _load_combined_measurements(files: list[Path], bad_bpms: list[str]) -> pd.DataFrame:
    measurements = [read_tbt(file, datatype="lhc") for file in files]
    per_bunch = convert_tbt_to_dataframes(
        measurements,
        bad_bpms=bad_bpms,
        combine_measurements=True,
    )
    combined = pd.concat(per_bunch, ignore_index=True)
    combined["name"] = combined["name"].astype(str)
    combined["turn"] = combined["turn"].astype("int32", copy=False)
    return combined


def _prepare_reconstruction_input(
    combined: pd.DataFrame,
    model_tws: pd.DataFrame,
    bad_bpms: list[str],
    apply_svd: bool,
) -> pd.DataFrame:
    df = combined.copy(deep=True)
    if apply_svd:
        df = svd_clean_measurements(df)
    df = df[df["name"].isin(model_tws.index)].copy()
    df.set_index("name", inplace=True)
    df = assign_known_noise_variances(df, bad_bpms)
    df.reset_index(inplace=True)
    return df


def _load_measurement_twiss(analysis_dir: Path, beam: int) -> pd.DataFrame:
    tws, _ = build_twiss_from_measurements(analysis_dir, include_errors=True, reverse_bpm_order=beam == 2)
    tws.columns = [str(col).lower() for col in tws.columns]
    if "name" in tws.columns:
        tws = tws.set_index("name")
    else:
        tws.index = tws.index.astype(str)
    tws.index = tws.index.astype(str)
    return tws


def _reconstruct(
    input_df: pd.DataFrame,
    *,
    analysis_dir: Path,
    model_tws: pd.DataFrame,
    beam: int,
    ac_dipole_config: ACDipoleConfig | None = None,
) -> pd.DataFrame:
    result = calculate_pz_measurement(
        input_df,
        measurement_folder=analysis_dir,
        model_tws=model_tws,
        reverse_meas_tws=beam == 2,
        info=False,
        include_errors=True,
        include_optics_errors=True,
        dpp_override=0.0,
        ac_dipole_config=ac_dipole_config,
    )
    result["name"] = result["name"].astype(str)
    result["turn"] = result["turn"].astype("int32", copy=False)
    return result


def _apply_acd_details_to_reconstruction(
    base_recon: pd.DataFrame,
    acd_details: pd.DataFrame,
) -> pd.DataFrame:
    """Apply cleaned ACD BPM momenta to an existing reconstruction dataframe."""
    out = base_recon.copy(deep=True)

    bpm_upstream = str(acd_details.attrs.get("bpm_upstream", acd_details["bpm_upstream"].iloc[0]))
    bpm_downstream = str(
        acd_details.attrs.get("bpm_downstream", acd_details["bpm_downstream"].iloc[0])
    )

    side_specs = [
        (bpm_upstream, "px_bpm_upstream_cleaned", "py_bpm_upstream_cleaned"),
        (bpm_downstream, "px_bpm_downstream_cleaned", "py_bpm_downstream_cleaned"),
    ]
    for bpm_name, px_col, py_col in side_specs:
        side = acd_details[["turn", px_col, py_col]].rename(columns={px_col: "px", py_col: "py"})
        side = side.set_index("turn")
        mask = out["name"].astype(str) == bpm_name
        if not mask.any():
            continue
        turns = out.loc[mask, "turn"].to_numpy()
        out.loc[mask, "px"] = side.reindex(turns)["px"].to_numpy(dtype=float)
        out.loc[mask, "py"] = side.reindex(turns)["py"].to_numpy(dtype=float)

    # Keep resolved ACD metadata for debugging and labels.
    out.attrs["ac_dipole_bpm_upstream"] = bpm_upstream
    out.attrs["ac_dipole_bpm_downstream"] = bpm_downstream
    out.attrs["ac_dipole_marker"] = acd_details.attrs.get("acd_marker", acd_details.attrs.get("acd_element"))
    out.attrs["ac_dipole_smooth_lambda"] = float(acd_details.attrs.get("smooth_lambda", np.nan))
    return out


def _select_bpm(data: pd.DataFrame, bpm_name: str, max_turns: int | None) -> pd.DataFrame:
    out = data[data["name"] == bpm_name].copy()
    if max_turns is not None:
        out = out[out["turn"] <= max_turns].copy()
    return out.sort_values("turn")


def _resolve_bpm_name(bpm_name: str, candidates: pd.Index | list[str], beam: int) -> str:
    candidate_set = {str(name).upper() for name in candidates}
    requested = bpm_name.upper()
    if requested in candidate_set:
        return requested

    beam_qualified = f"{requested}.B{beam}"
    if beam_qualified in candidate_set:
        return beam_qualified

    matches = sorted(name for name in candidate_set if name.startswith(f"{requested}."))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous BPM name {bpm_name!r}. Matches: {matches}")
    raise ValueError(f"Could not resolve BPM name {bpm_name!r} for beam {beam}.")


def _plot_overlay(ax, datasets: list[tuple[str, pd.DataFrame, str]], coord: str, mom: str, title: str) -> None:
    for label, df, color in datasets:
        ax.scatter(
            df[coord] * 1e3,  # Convert to mm
            df[mom] * 1e6,  # Convert to urad
            label=label,
            alpha=0.45,
            s=1.5,
            color=color,
        )
    ax.set_xlabel(f"${coord}$ [mm]")
    ax.set_ylabel(f"$p_{coord[-1]}$ [$\\mu$rad]")
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    setup_scientific_formatting(ax, powerlimits=(-1, 1))


def _normalise_phase_space(
    data: pd.DataFrame,
    measurement_twiss: pd.DataFrame,
    coord: str,
    mom: str,
) -> pd.DataFrame:
    if coord not in ("x", "y"):
        raise ValueError(f"coord must be 'x' or 'y', got {coord!r}")

    beta_candidates = ("betx", "beta11") if coord == "x" else ("bety", "beta22")
    alpha_candidates = ("alfx", "alfa11") if coord == "x" else ("alfy", "alfa22")
    beta_col = next((col for col in beta_candidates if col in measurement_twiss.columns), None)
    alpha_col = next((col for col in alpha_candidates if col in measurement_twiss.columns), None)
    required_cols = [col for col in (beta_col, alpha_col) if col is not None]
    missing_cols = [col for col in required_cols if col not in measurement_twiss.columns]
    if beta_col is None or alpha_col is None or missing_cols:
        expected = [*beta_candidates, *alpha_candidates]
        raise KeyError(f"Missing measurement optics columns for {coord}-plane. Expected one of: {expected}")

    optics = measurement_twiss[required_cols].rename(
        columns={beta_col: "beta", alpha_col: "alpha"}
    )
    out = data.join(optics, on="name", how="inner")
    if out.empty:
        raise ValueError("No BPMs matched between reconstructed data and measurement optics.")

    beta = out["beta"].to_numpy(dtype=float)
    alpha = out["alpha"].to_numpy(dtype=float)
    coord_vals = out[coord].to_numpy(dtype=float)
    mom_vals = out[mom].to_numpy(dtype=float)
    sqrt_beta = np.sqrt(beta)

    out[f"{coord}_norm"] = coord_vals / sqrt_beta
    out[f"{mom}_norm"] = alpha * coord_vals / sqrt_beta + mom_vals * sqrt_beta
    return out


def _plot_phase_space_figure(
    datasets: list[tuple[str, pd.DataFrame, str]],
    bpm_rows: list[tuple[str, str]],
    *,
    max_turns: int | None,
    figsize: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(len(bpm_rows), 2, figsize=figsize, squeeze=False)
    for row_idx, (row_label, bpm_name) in enumerate(bpm_rows):
        for col_idx, (coord, mom) in enumerate((("x", "px"), ("y", "py"))):
            bpm_datasets = [
                (label, _select_bpm(df, bpm_name, max_turns), color)
                for label, df, color in datasets
            ]
            _plot_overlay(
                axes[row_idx, col_idx],
                bpm_datasets,
                coord=coord,
                mom=mom,
                title=f"{row_label} ({bpm_name}.{coord.upper()})",
            )
    return fig, axes


def _plot_normalised_overlay(
    ax,
    datasets: list[tuple[str, pd.DataFrame, str]],
    coord: str,
    mom: str,
    title: str,
) -> None:
    coord_label = f"{coord}_n"
    mom_label = f"p_{{{coord}}},n"
    for label, df, color in datasets:
        ax.scatter(
            df[f"{coord}_norm"] * 1e3,
            df[f"{mom}_norm"] * 1e3,
            label=label,
            alpha=0.45,
            s=1.5,
            color=color,
        )
    ax.set_xlabel(f"${coord_label}$ [$\\sqrt{{\\mathrm{{m}}}}$]")
    ax.set_ylabel(f"${mom_label}$ [$\\mathrm{{m}}^{{-1/2}}$]")
    ax.set_title(title)
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(visible=True, alpha=0.3)
    setup_scientific_formatting(ax, powerlimits=(-1, 1))


def _acd_jump_frame(data: pd.DataFrame, plane: str, max_turns: int | None) -> pd.DataFrame:
    if plane not in ("x", "y"):
        raise ValueError(f"plane must be 'x' or 'y', got {plane!r}")
    suffix = "px" if plane == "x" else "py"
    raw_col = f"d{suffix}_rad"
    fit_col = f"d{suffix}_fit_rad"
    out = data[["turn"]].copy()

    if raw_col not in data.columns or fit_col not in data.columns:
        raise KeyError(f"Expected AC-dipole jump columns {raw_col} and {fit_col} in dataframe.")

    out["raw"] = data[raw_col].to_numpy()
    out["fit"] = data[fit_col].to_numpy()

    if max_turns is not None:
        out = out[out["turn"] <= max_turns].copy()
    return out.sort_values("turn")


def _plot_jump_fit(
    ax,
    jump_datasets: list[tuple[str, pd.DataFrame, str]],
    plane: str,
) -> None:
    if not jump_datasets:
        raise ValueError("jump_datasets must not be empty")

    for label, jump_df, color in jump_datasets:
        ax.plot(
            jump_df["turn"],
            jump_df["raw"] * 1e6,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label=f"Raw jump ({label})",
        )
        ax.plot(
            jump_df["turn"],
            jump_df["fit"] * 1e6,
            color=color,
            linewidth=1.5,
            label=f"Fitted jump ({label})",
        )
    if plane == "y":
        ax.set_xlabel("Turn")
    ax.set_ylabel(f"$\\Delta p_{plane}$ [$\\mu$rad]")
    if plane == "x":
        ax.set_title("ACD Jump vs turn")
    ax.grid(visible=True, alpha=0.3)
    setup_scientific_formatting(ax, powerlimits=(-1, 1))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Compare raw, SVD-cleaned, and SVD+AC-dipole-cleaned phase-space reconstructions."
    )
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True, help="Beam number")
    parser.add_argument("--squeeze-step", type=str, required=True, help="Squeeze step, e.g. inj_rdt")
    parser.add_argument(
        "--freq",
        type=str,
        default=ZEROHZ,
        help="Measurement frequency bucket to plot. Default: 0Hz",
    )
    parser.add_argument(
        "--smooth-lambda",
        type=float,
        default=1.0,
        help="Smoothing strength used by the AC-dipole cleaning.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional turn cutoff applied before plotting.",
    )
    parser.add_argument(
        "--before-bpm",
        type=str,
        default=None,
        help="Override the upstream BPM to plot.",
    )
    parser.add_argument(
        "--after-bpm",
        type=str,
        default=None,
        help="Override the downstream BPM to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is only shown.",
    )
    args = parser.parse_args()

    if args.squeeze_step not in MEAS_TIMES[args.beam]:
        raise ValueError(f"Unknown squeeze step {args.squeeze_step!r} for beam {args.beam}.")

    meas_times = MEAS_TIMES[args.beam][args.squeeze_step]
    if args.freq not in meas_times or not meas_times[args.freq]:
        raise ValueError(
            f"No measurement times configured for frequency {args.freq!r} "
            f"at beam {args.beam}, squeeze {args.squeeze_step}."
        )

    meas_base_dir, model_dir, analysis_dir = _get_paths(args.beam, args.squeeze_step)
    sequence_time = get_sequence_creation_time(meas_times, args.squeeze_step)
    sequence_path = get_or_make_sequence(args.beam, model_dir, time=sequence_time)
    model_tws, twiss_elements = _load_model_twiss_data(
        model_dir,
        sequence_path=sequence_path,
        beam=args.beam,
        pc=6800.0,
    )
    accelerator = LHC(
        beam=args.beam,
        pc=6800.0,
        sequence_file=sequence_path,
    )
    measurement_twiss = _load_measurement_twiss(analysis_dir, beam=args.beam)

    times = meas_times[args.freq]
    analysed_folders = get_analysis_folders(times, args.beam, meas_base_dir, args.squeeze_step)
    files = get_measurement_files(times, analysed_folders, args.beam, args.squeeze_step)
    results_dir = get_results_dir(args.beam)
    tune_knobs_file, corrector_knobs_file = get_knob_files(results_dir, args.squeeze_step, args.freq)
    if not tune_knobs_file.exists():
        raise FileNotFoundError(f"Tune knobs file not found: {tune_knobs_file}")
    if not corrector_knobs_file.exists():
        raise FileNotFoundError(f"Corrector knobs file not found: {corrector_knobs_file}")
    bad_bpms = sorted({bpm for folder in analysed_folders for bpm in find_all_bad_bpms(folder)})

    selected_file = files[0]
    LOGGER.info("Loading 1 measurement file for %s: %s", args.freq, selected_file)
    combined = _load_combined_measurements([selected_file], bad_bpms)

    raw_input = _prepare_reconstruction_input(combined, model_tws, bad_bpms, apply_svd=False)
    svd_input = _prepare_reconstruction_input(combined, model_tws, bad_bpms, apply_svd=True)

    raw_recon = _reconstruct(
        raw_input,
        analysis_dir=analysis_dir,
        model_tws=model_tws,
        beam=args.beam,
    )
    svd_recon = _reconstruct(
        svd_input,
        analysis_dir=analysis_dir,
        model_tws=model_tws,
        beam=args.beam,
    )

    accelerator = LHC(
        beam=args.beam,
        pc=6800.0,
        sequence_file=sequence_path,
    )

    acd_model = ACDipoleMadDriver(
        accelerator=accelerator,
        deltap=0.0,
        observed_elements=accelerator.get_ac_dipole_marker(),
        # tune_knobs_file=tune_knobs_file,
        # corrector_knobs_file=corrector_knobs_file,
        discard_mad_output=True,
    )
    acd_details = calculate_ac_dipole_momentum(
        svd_input,
        model_tws,
        ac_dipole_marker=accelerator.get_ac_dipole_marker(),
        model=acd_model,
        smooth_lambda=args.smooth_lambda,
        inject_noise=False,
    )
    model_bpm_tws = model_tws.loc[model_tws.index.isin(svd_input["name"].astype(str))].copy()
    nbpm_recon = calculate_transverse_pz_nbpm(
        svd_input.copy(deep=True),
        tws=model_bpm_tws,
        twiss_elements=twiss_elements,
        inject_noise=False,
        info=False,
    )
    nbpm_recon["name"] = nbpm_recon["name"].astype(str)
    nbpm_recon["turn"] = nbpm_recon["turn"].astype("int32", copy=False)

    before_bpm = args.before_bpm or acd_details.attrs.get("bpm_upstream")
    if before_bpm is None and "bpm_upstream" in acd_details.columns:
        before_bpm = str(acd_details["bpm_upstream"].iloc[0])
    after_bpm = args.after_bpm or acd_details.attrs.get("bpm_downstream")
    if after_bpm is None and "bpm_downstream" in acd_details.columns:
        after_bpm = str(acd_details["bpm_downstream"].iloc[0])
    if before_bpm is None or after_bpm is None:
        raise ValueError(
            "Failed to resolve the AC-dipole BPM pair. Pass --before-bpm and --after-bpm explicitly."
        )
    available_bpms = raw_recon["name"].astype(str).unique().tolist()
    before_bpm = _resolve_bpm_name(before_bpm, available_bpms, args.beam)
    after_bpm = _resolve_bpm_name(after_bpm, available_bpms, args.beam)
    extra_bpm = _resolve_bpm_name("BPM.15R1", available_bpms, args.beam)

    if before_bpm != str(acd_details.attrs.get("bpm_upstream", acd_details["bpm_upstream"].iloc[0])) or (
        after_bpm != str(acd_details.attrs.get("bpm_downstream", acd_details["bpm_downstream"].iloc[0]))
    ):
        acd_details = calculate_ac_dipole_momentum(
            svd_input,
            model_tws,
            ac_dipole_marker=accelerator.get_ac_dipole_marker(),
            model=acd_model,
            bpm_upstream=before_bpm,
            bpm_downstream=after_bpm,
            smooth_lambda=args.smooth_lambda,
            inject_noise=False,
        )

    acd_recon = _apply_acd_details_to_reconstruction(svd_recon, acd_details)

    datasets = [
        (RAW_LABEL, raw_recon, "tab:blue"),
        (SVD_LABEL, svd_recon, "tab:orange"),
        (ACD_LABEL, acd_recon, "tab:green"),
        (NBPM_LABEL, nbpm_recon, "tab:red"),
    ]

    phase_fig, phase_axes = _plot_phase_space_figure(
        datasets,
        [("Before ACD", before_bpm), ("After ACD", after_bpm)],
        max_turns=args.max_turns,
        figsize=(12, 8),
    )

    phase_legend = phase_axes[0, 0].legend(loc="best", fontsize=8)
    for handle in phase_legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([24.0])
    phase_fig.tight_layout()

    extra_phase_fig, extra_phase_axes = _plot_phase_space_figure(
        datasets,
        [("BPM.15R1", extra_bpm)],
        max_turns=args.max_turns,
        figsize=(12, 4),
    )
    extra_phase_legend = extra_phase_axes[0, 0].legend(loc="best", fontsize=8)
    for handle in extra_phase_legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([24.0])
    extra_phase_fig.tight_layout()

    norm_fig, norm_axes = plt.subplots(2, 2, figsize=(12, 8))
    norm_plot_specs = [
        (norm_axes[0, 0], before_bpm, "x", "px", f"Before ACD ({before_bpm}.X)"),
        (norm_axes[0, 1], before_bpm, "y", "py", f"Before ACD ({before_bpm}.Y)"),
        (norm_axes[1, 0], after_bpm, "x", "px", f"After ACD ({after_bpm}.X)"),
        (norm_axes[1, 1], after_bpm, "y", "py", f"After ACD ({after_bpm}.Y)"),
    ]
    for ax, bpm_name, coord, mom, title in norm_plot_specs:
        bpm_datasets = []
        for label, df, color in datasets:
            bpm_df = _select_bpm(df, bpm_name, args.max_turns)
            bpm_datasets.append(
                (
                    label,
                    _normalise_phase_space(
                        bpm_df,
                        measurement_twiss=measurement_twiss,
                        coord=coord,
                        mom=mom,
                    ),
                    color,
                )
            )
        _plot_normalised_overlay(ax, bpm_datasets, coord=coord, mom=mom, title=title)

    norm_legend = norm_axes[0, 0].legend(loc="best", fontsize=8)
    for handle in norm_legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([24.0])
    norm_fig.tight_layout()

    dpx_jump = _acd_jump_frame(acd_details, plane="x", max_turns=args.max_turns)
    dpy_jump = _acd_jump_frame(acd_details, plane="y", max_turns=args.max_turns)
    jump_fig, jump_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _plot_jump_fit(
        jump_axes[0],
        [(ACD_LABEL, dpx_jump, "tab:green")],
        plane="x",
    )
    _plot_jump_fit(
        jump_axes[1],
        [(ACD_LABEL, dpy_jump, "tab:green")],
        plane="y",
    )
    jump_axes[0].legend(loc="best", fontsize=8)
    jump_fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        phase_output = args.output.with_name(f"{args.output.stem}_phase_space{args.output.suffix}")
        extra_phase_output = args.output.with_name(
            f"{args.output.stem}_bpm15r1_phase_space{args.output.suffix}"
        )
        norm_output = args.output.with_name(f"{args.output.stem}_normalised_phase_space{args.output.suffix}")
        jump_output = args.output.with_name(f"{args.output.stem}_deltas{args.output.suffix}")
        phase_fig.savefig(phase_output, dpi=200)
        extra_phase_fig.savefig(extra_phase_output, dpi=200)
        norm_fig.savefig(norm_output, dpi=200)
        jump_fig.savefig(jump_output, dpi=200)
        LOGGER.info("Saved phase-space plot to %s", phase_output)
        LOGGER.info("Saved BPM.15R1 phase-space plot to %s", extra_phase_output)
        LOGGER.info("Saved normalised phase-space plot to %s", norm_output)
        LOGGER.info("Saved delta plot to %s", jump_output)

    plt.show()
    acd_model.close()


if __name__ == "__main__":
    main()
