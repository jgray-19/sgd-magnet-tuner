#!/usr/bin/env python3
"""Plot raw, SVD-cleaned, weighted-SVD-cleaned, and SVD+AC-dipole-cleaned phase spaces.

The script reuses the same measurement bookkeeping as the squeeze optimisation flow,
but keeps all reconstruction stages in memory. It plots the BPM immediately upstream
and downstream of the AC dipole, with overlays for:

1. raw/noisy reconstruction,
2. reconstruction after SVD cleaning,
3. reconstruction after weighted SVD cleaning,
4. reconstruction after SVD cleaning plus AC-dipole cleaning.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tfs
from tmom_recon import (
    ACDipoleConfig,
    build_twiss_from_measurements,
    calculate_pz,
)
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements

from aba_optimiser.accelerators import LHC
from aba_optimiser.measurements.create_datafile import (
    build_madng_twiss_table,
)
from aba_optimiser.measurements.loading import (
    convert_tbt_to_dataframes,
    load_measurement_files,
)
from aba_optimiser.measurements.optimise_squeeze_quads import (
    MEAS_TIMES,
    ZEROHZ,
)
from aba_optimiser.measurements.sequence import (
    extract_tunes_from_job_file,
    get_or_make_sequence,
)
from aba_optimiser.measurements.squeeze.config import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
    get_results_dir,
)
from aba_optimiser.measurements.squeeze.io import (
    get_sequence_creation_time,
    prepare_frequency_metadata,
)
from aba_optimiser.measurements.uncompensated_analysis import (
    get_uncompensated_analysis_dir,
    rerun_optics_analysis_without_compensation,
)
from aba_optimiser.measurements.variances import assign_known_noise_variances

LOGGER = logging.getLogger(__name__)

RAW_LABEL = "Raw reconstruction"
SVD_LABEL = "SVD cleaned"
WEIGHTED_SVD_LABEL = "Weighted SVD cleaned"
ACD_LABEL = "Weighted SVD + ACD cleaned"
PHASE_PLOT_COLORS = {
    RAW_LABEL: "#2563eb",
    SVD_LABEL: "#dc2626",
    WEIGHTED_SVD_LABEL: "#7c3aed",
    ACD_LABEL: "#059669",
}
UNCOMP_PHASE_PLOT_COLORS = {
    RAW_LABEL: "#7c3aed",
    SVD_LABEL: "#d97706",
    WEIGHTED_SVD_LABEL: "#2563eb",
    ACD_LABEL: "#0f766e",
}
NORMALISED_PLOT_COLORS = {
    RAW_LABEL: "#3b82f6",
    SVD_LABEL: "#ef4444",
    WEIGHTED_SVD_LABEL: "#8b5cf6",
    ACD_LABEL: "#10b981",
}
UNCOMP_NORMALISED_PLOT_COLORS = {
    RAW_LABEL: "#8b5cf6",
    SVD_LABEL: "#f59e0b",
    WEIGHTED_SVD_LABEL: "#3b82f6",
    ACD_LABEL: "#14b8a6",
}


def setup_scientific_formatting(ax, powerlimits: tuple[int, int] = (-1, 1)) -> None:
    """Apply scientific tick formatting while avoiding noisy mathtext warnings."""
    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits(powerlimits)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


def _get_paths(beam: int, squeeze_step: str) -> tuple[Path, Path, Path]:
    meas_date = get_measurement_date(squeeze_step)
    beam_root = BETABEAT_DIR / meas_date / f"LHCB{beam}"
    meas_base_dir = beam_root / "Measurements"
    model_dir = beam_root / "Models" / MODEL_DIRS[beam][squeeze_step]
    analysis_dir_name = ANALYSIS_DIRS[beam][squeeze_step]
    analysis_dir = beam_root / "Results" / analysis_dir_name
    return meas_base_dir, model_dir, analysis_dir


def _prepare_writable_model_dir(model_dir: Path, beam: int, squeeze_step: str) -> Path:
    """Copy the reference model into a writable temp directory for sequence generation."""
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f"acd_clean_model_b{beam}_{squeeze_step.replace('.', '_')}_",
            dir="/tmp",
        )
    )
    writable_model_dir = temp_root / model_dir.name
    shutil.copytree(model_dir, writable_model_dir, symlinks=True)
    return writable_model_dir


def _prepare_runtime_dir(beam: int, squeeze_step: str) -> Path:
    """Create a writable temp directory for derived analysis artifacts."""
    return Path(
        tempfile.mkdtemp(
            prefix=f"acd_clean_runtime_b{beam}_{squeeze_step.replace('.', '_')}_",
            dir="/tmp",
        )
    )


def _load_model_twiss_data(
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    kinetic_energy: float,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    job_file = model_dir / "job.create_model_nominal.madx"
    nat_x, nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
    LOGGER.info(
        "Using job file tunes for %s: nat=(%.6f, %.6f), drv=(%.6f, %.6f)",
        job_file,
        nat_x,
        nat_y,
        drv_x,
        drv_y,
    )
    nattunes = [nat_x, nat_y, 0.0]
    tunes = [drv_x, drv_y, 0.0]
    accelerator = LHC(
        beam=beam,
        sequence_file=sequence_path,
        kinetic_energy=kinetic_energy,
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

    if "name" in tws.columns:
        tws["name"] = tws["name"].astype(str).str.upper()
        tws = tws.set_index("name")
    elif tws.index.name is not None:
        tws.index = tws.index.astype(str).str.upper()
    return tws, twiss_elements, (drv_x, drv_y)

def _load_combined_measurements(
    files: list[Path], bad_bpms: list[str], *, beam: int
) -> pd.DataFrame:
    measurements = load_measurement_files(files, beam=beam)
    per_bunch = convert_tbt_to_dataframes(
        measurements,
        bad_bpms=bad_bpms,
        combine_measurements=True,
    )
    combined = pd.concat(per_bunch, ignore_index=True)
    combined["name"] = combined["name"].astype(str).str.upper()
    combined["turn"] = combined["turn"].astype("int32")
    return combined


def _prepare_reconstruction_input(
    combined: pd.DataFrame,
    model_tws: pd.DataFrame,
    bad_bpms: list[str],
    cleaning: str,
) -> pd.DataFrame:
    df = combined.copy(deep=True)
    model_bpm_names = pd.Index(model_tws.index).astype(str).str.upper()
    measured_bpm_names = pd.Index(df["name"].astype(str).str.upper().unique())
    df["name"] = df["name"].astype(str).str.upper()
    if cleaning == "svd":
        df = svd_clean_measurements(df)
    elif cleaning == "weighted_svd":
        df = df[df["name"].isin(model_bpm_names)].copy()
        if df.empty:
            missing_bpms = measured_bpm_names.difference(model_bpm_names).tolist()[:10]
            LOGGER.warning(
                "Weighted SVD input is empty after model BPM filter. Example measured BPMs not in model: %s",
                missing_bpms,
            )
            raise ValueError("Weighted SVD input is empty after model BPM filtering")
        df = assign_known_noise_variances(df, bad_bpms)
        df = weighted_svd_clean_measurements(df)
        df["name"] = df["name"].astype(str)
        df["turn"] = df["turn"].astype("int32")
        return df
    elif cleaning != "raw":
        raise ValueError(f"Unknown cleaning mode: {cleaning!r}")
    df = df[df["name"].isin(model_bpm_names)].copy()
    if df.empty:
        raise ValueError(f"{cleaning} input is empty after model BPM filtering")
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
    tws.index = tws.index.astype(str).str.upper()
    return tws


def _build_reconstruction_datasets(
    *,
    analysis_dir: Path,
    raw_input: pd.DataFrame,
    svd_input: pd.DataFrame,
    weighted_svd_input: pd.DataFrame,
    model_tws: pd.DataFrame,
    beam: int,
    acd_recon: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame, str]]]:
    measurement_twiss = _load_measurement_twiss(analysis_dir, beam=beam)
    raw_recon = _reconstruct(
        raw_input,
        analysis_dir=analysis_dir,
        model_tws=model_tws,
        beam=beam,
    )
    svd_recon = _reconstruct(
        svd_input,
        analysis_dir=analysis_dir,
        model_tws=model_tws,
        beam=beam,
    )
    weighted_svd_recon = _reconstruct(
        weighted_svd_input,
        analysis_dir=analysis_dir,
        model_tws=model_tws,
        beam=beam,
    )
    datasets = [
        (RAW_LABEL, raw_recon, "tab:blue"),
        (SVD_LABEL, svd_recon, "tab:orange"),
        (WEIGHTED_SVD_LABEL, weighted_svd_recon, "tab:purple"),
        (ACD_LABEL, acd_recon, "tab:green"),
    ]
    return measurement_twiss, datasets


def _reconstruct(
    input_df: pd.DataFrame,
    *,
    analysis_dir: Path,
    model_tws: pd.DataFrame,
    beam: int,
    ac_dipole_config: ACDipoleConfig | None = None,
) -> pd.DataFrame:
    result = calculate_pz(
        input_df,
        measurement_dir=analysis_dir,
        model_tws=model_tws,
        reverse_meas_tws=beam == 2,
        info=False,
        pt_override=0.0,
        acd=ac_dipole_config,
    )
    result = _scale_position_variances_after_svd(result)
    result["name"] = result["name"].astype(str)
    result["turn"] = result["turn"].astype("int32")
    return result


def _scale_position_variances_after_svd(df: pd.DataFrame) -> pd.DataFrame:
    """Match the production measurement pipeline's post-SVD variance scaling."""
    result = df.copy()
    result["var_x"] = result["var_x"] / 100.0
    result["var_y"] = result["var_y"] / 100.0
    return result


def _build_ac_dipole_config(
    *,
    accelerator: LHC,
    pt: float,
    dpx_tune: float,
    dpy_tune: float,
    tune_knobs: Path | None = None,
    corrector_knobs: Path | None = None,
    smooth_lambda: float = 1.0,
) -> ACDipoleConfig:
    model = ACDipoleMadDriver(
        accelerator=accelerator,
        pt=pt,
        observed_elements=accelerator.get_ac_dipole_marker(),
        tune_knobs=tune_knobs,
        corrector_knobs=corrector_knobs,
        discard_mad_output=True,
    )
    return ACDipoleConfig(
        ac_dipole_marker=accelerator.get_ac_dipole_marker(),
        model=model,
        dpx_tune=dpx_tune,
        dpy_tune=dpy_tune,
        tune_knobs=tune_knobs,
        corrector_knobs=corrector_knobs,
        smooth_lambda=smooth_lambda,
    )


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


def _acd_corrected_bpms(data: pd.DataFrame) -> set[str]:
    corrected: set[str] = set()
    for attr_name in (
        "ac_dipole_bpm_upstream",
        "ac_dipole_bpm_downstream",
        "bpm_upstream",
        "bpm_downstream",
    ):
        bpm_name = data.attrs.get(attr_name)
        if bpm_name:
            corrected.add(str(bpm_name).upper())
    return corrected


def _datasets_for_bpm(
    datasets: list[tuple[str, pd.DataFrame, str]],
    bpm_name: str,
    max_turns: int | None,
) -> list[tuple[str, pd.DataFrame, str]]:
    """Return the plot-ready datasets for one BPM.

    The AC-dipole cleaning step only changes the two BPMs adjacent to the AC dipole,
    so we omit that series elsewhere to avoid plotting an unchanged duplicate of the
    SVD-cleaned reconstruction under a misleading label.
    """
    plot_datasets: list[tuple[str, pd.DataFrame, str]] = []
    bpm_name_upper = bpm_name.upper()
    for label, df, color in datasets:
        if label == ACD_LABEL:
            corrected_bpms = _acd_corrected_bpms(df)
            if corrected_bpms and bpm_name_upper not in corrected_bpms:
                continue
        plot_datasets.append((label, _select_bpm(df, bpm_name, max_turns), color))
    return plot_datasets


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
    optics_twiss: pd.DataFrame,
    coord: str,
    mom: str,
) -> pd.DataFrame:
    if coord not in ("x", "y"):
        raise ValueError(f"coord must be 'x' or 'y', got {coord!r}")

    beta_candidates = ("betx", "beta11") if coord == "x" else ("bety", "beta22")
    alpha_candidates = ("alfx", "alfa11") if coord == "x" else ("alfy", "alfa22")
    beta_col = next((col for col in beta_candidates if col in optics_twiss.columns), None)
    alpha_col = next((col for col in alpha_candidates if col in optics_twiss.columns), None)
    required_cols = [col for col in (beta_col, alpha_col) if col is not None]
    missing_cols = [col for col in required_cols if col not in optics_twiss.columns]
    if beta_col is None or alpha_col is None or missing_cols:
        expected = [*beta_candidates, *alpha_candidates]
        raise KeyError(f"Missing optics columns for {coord}-plane. Expected one of: {expected}")

    optics = optics_twiss[required_cols].rename(columns={beta_col: "beta", alpha_col: "alpha"})
    out = data.join(optics, on="name", how="inner")
    if out.empty:
        raise ValueError("No BPMs matched between reconstructed data and the chosen optics table.")

    beta = out["beta"].to_numpy(dtype=float)
    alpha = out["alpha"].to_numpy(dtype=float)
    coord_vals = out[coord].to_numpy(dtype=float)
    mom_vals = out[mom].to_numpy(dtype=float)
    sqrt_beta = np.sqrt(beta)

    out[f"{coord}_norm"] = coord_vals / sqrt_beta
    out[f"{mom}_norm"] = alpha * coord_vals / sqrt_beta + mom_vals * sqrt_beta
    return out


def _circle_closeness_score(data: pd.DataFrame, coord: str, mom: str) -> float:
    """Return a 0..1 score for how tightly the normalised points follow a circle.

    A perfect circle has constant radius in normalised phase space, so we score the
    point cloud by the coefficient of variation of its radius and map that onto a
    bounded 0..1 scale where 1 is best.
    """
    radius = np.hypot(
        data[f"{coord}_norm"].to_numpy(dtype=float),
        data[f"{mom}_norm"].to_numpy(dtype=float),
    )
    finite_radius = radius[np.isfinite(radius)]
    if finite_radius.size == 0:
        return float("nan")
    mean_radius = float(np.mean(finite_radius))
    if np.isclose(mean_radius, 0.0):
        return 1.0 if np.allclose(finite_radius, 0.0) else 0.0
    radius_cv = float(np.std(finite_radius) / mean_radius)
    return float(np.clip(1.0 - radius_cv, 0.0, 1.0))


def _format_circle_closeness_summary(
    datasets: list[tuple[str, pd.DataFrame, str]],
    coord: str,
    mom: str,
) -> str:
    lines = ["circle closeness"]
    for label, df, _color in datasets:
        score = _circle_closeness_score(df, coord=coord, mom=mom)
        score_text = "n/a" if not np.isfinite(score) else f"{score:.3f}"
        lines.append(f"{label}: {score_text}")
    return "\n".join(lines)


def _normalisation_optics_for_label(
    label: str,
    *,
    measurement_twiss: pd.DataFrame,
    model_tws: pd.DataFrame,
) -> pd.DataFrame:
    """Choose the optics basis consistent with each reconstruction path.

    Raw/SVD reconstructions are produced by `calculate_pz`, which uses
    the measurement optics from the analysis folder. The ACD-cleaned momenta come
    from `calculate_pz(..., acd_only=True)` against the supplied model twiss, so we
    normalise that series with the same model optics to avoid introducing a
    spurious phase-space offset from mixing conventions.
    """
    if label == ACD_LABEL:
        return model_tws
    return measurement_twiss


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
            bpm_datasets = _datasets_for_bpm(datasets, bpm_name, max_turns)
            _plot_overlay(
                axes[row_idx, col_idx],
                bpm_datasets,
                coord=coord,
                mom=mom,
                title=f"{row_label} ({bpm_name}.{coord.upper()})",
            )
    return fig, axes


def _plot_phase_space_on_axes(
    axes: np.ndarray,
    datasets: list[tuple[str, pd.DataFrame, str]],
    bpm_rows: list[tuple[str, str]],
    *,
    max_turns: int | None,
    title_prefix: str = "",
) -> None:
    for row_idx, (row_label, bpm_name) in enumerate(bpm_rows):
        for col_idx, (coord, mom) in enumerate((("x", "px"), ("y", "py"))):
            bpm_datasets = _datasets_for_bpm(datasets, bpm_name, max_turns)
            title = f"{row_label} ({bpm_name}.{coord.upper()})"
            if title_prefix:
                title = f"{title_prefix} - {title}"
            _plot_overlay(
                axes[row_idx, col_idx],
                bpm_datasets,
                coord=coord,
                mom=mom,
                title=title,
            )


def _apply_palette(
    datasets: list[tuple[str, pd.DataFrame, str]],
    palette: dict[str, str],
) -> list[tuple[str, pd.DataFrame, str]]:
    return [(label, df, palette.get(label, color)) for label, df, color in datasets]


def _style_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(loc="best", fontsize=8)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([24.0])


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
            df[f"{coord}_norm"],
            df[f"{mom}_norm"],
            label=label,
            alpha=0.45,
            s=1.5,
            color=color,
        )
    ax.set_xlabel(f"${coord_label}$ [$\\sqrt{{\\mathrm{{m}}}}$]")
    ax.set_ylabel(f"${mom_label}$ [$\\mathrm{{m}}^{{-1/2}}$]")
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    setup_scientific_formatting(ax, powerlimits=(-1, 1))
    ax.text(
        0.02,
        0.98,
        _format_circle_closeness_summary(datasets, coord=coord, mom=mom),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )


def _plot_normalised_grid(
    axes: np.ndarray,
    layout_specs: list[tuple[int, int, str, str, str, str]],
    datasets: list[tuple[str, pd.DataFrame, str]],
    *,
    max_turns: int | None,
    measurement_twiss: pd.DataFrame,
    model_tws: pd.DataFrame,
    palette: dict[str, str],
    title_prefix: str = "",
) -> None:
    for row_idx, col_idx, bpm_name, coord, mom, title in layout_specs:
        ax = axes[row_idx, col_idx]
        bpm_datasets = []
        for label, bpm_df, color in _datasets_for_bpm(datasets, bpm_name, max_turns):
            bpm_datasets.append(
                (
                    label,
                    _normalise_phase_space(
                        bpm_df,
                        optics_twiss=_normalisation_optics_for_label(
                            label,
                            measurement_twiss=measurement_twiss,
                            model_tws=model_tws,
                        ),
                        coord=coord,
                        mom=mom,
                    ),
                    palette.get(label, color),
                )
            )
        plot_title = title if not title_prefix else f"{title_prefix} - {title}"
        _plot_normalised_overlay(ax, bpm_datasets, coord=coord, mom=mom, title=plot_title)


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
        description=(
            "Compare raw, SVD-cleaned, weighted-SVD-cleaned, and "
            "SVD+AC-dipole-cleaned phase-space reconstructions."
        )
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
        default=1,
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
    results_dir = get_results_dir(args.beam)
    runtime_dir = _prepare_runtime_dir(args.beam, args.squeeze_step)
    writable_model_dir = _prepare_writable_model_dir(model_dir, args.beam, args.squeeze_step)
    sequence_time = get_sequence_creation_time(meas_times, args.squeeze_step)
    sequence_path = get_or_make_sequence(args.beam, writable_model_dir, time=sequence_time)
    times = meas_times[args.freq]
    (
        files,
        tune_knobs,
        corrector_knobs,
        bad_bpms,
        kinetic_energy,
    ) = prepare_frequency_metadata(
        args.freq,
        times,
        args.beam,
        meas_base_dir,
        results_dir,
        args.squeeze_step,
    )
    model_tws, twiss_elements, acd_driven_tunes = _load_model_twiss_data(
        writable_model_dir,
        sequence_path=sequence_path,
        beam=args.beam,
        kinetic_energy=kinetic_energy,
    )
    selected_file = files[0]
    LOGGER.info("Loading 1 measurement file for %s: %s", args.freq, selected_file)
    bad_bpms = list(bad_bpms)  # Convert to a list for typing - could change the typing
    combined = _load_combined_measurements([selected_file], bad_bpms, beam=args.beam)

    raw_input = _prepare_reconstruction_input(combined, model_tws, bad_bpms, cleaning="raw")
    svd_input = _prepare_reconstruction_input(combined, model_tws, bad_bpms, cleaning="svd")
    weighted_svd_input = _prepare_reconstruction_input(
        combined,
        model_tws,
        bad_bpms,
        cleaning="weighted_svd",
    )

    accelerator = LHC(
        beam=args.beam,
        sequence_file=sequence_path,
        kinetic_energy=kinetic_energy,
    )
    LOGGER.info(
        "ACD model inputs: tune_knobs=%s corrector_knobs=%s",
        tune_knobs,
        corrector_knobs,
    )

    acd_config = _build_ac_dipole_config(
        accelerator=accelerator,
        pt=0.0,
        dpx_tune=acd_driven_tunes[0],
        dpy_tune=acd_driven_tunes[1],
        tune_knobs=tune_knobs,
        corrector_knobs=corrector_knobs,
        smooth_lambda=args.smooth_lambda,
    )
    acd_model = acd_config.model
    acd_details = calculate_pz(
        weighted_svd_input,
        model_tws=model_tws,
        acd=acd_config,
        acd_only=True,
        info=False,
    )
    before_bpm = args.before_bpm or acd_details.attrs.get("bpm_upstream")
    if before_bpm is None:
        before_bpm = acd_details.attrs.get("bpm_upstream")
    if before_bpm is None and "bpm_upstream" in acd_details.columns:
        before_bpm = str(acd_details["bpm_upstream"].iloc[0])
    after_bpm = args.after_bpm or acd_details.attrs.get("bpm_downstream")
    if after_bpm is None:
        after_bpm = acd_details.attrs.get("bpm_downstream")
    if after_bpm is None and "bpm_downstream" in acd_details.columns:
        after_bpm = str(acd_details["bpm_downstream"].iloc[0])
    if before_bpm is None or after_bpm is None:
        raise ValueError(
            "Failed to resolve the AC-dipole BPM pair. Pass --before-bpm and --after-bpm explicitly."
        )
    available_bpms = raw_input["name"].astype(str).unique().tolist()
    before_bpm = _resolve_bpm_name(before_bpm, available_bpms, args.beam)
    after_bpm = _resolve_bpm_name(after_bpm, available_bpms, args.beam)
    extra_bpm = _resolve_bpm_name("BPM.15R1", available_bpms, args.beam)

    if before_bpm != str(acd_details.attrs.get("bpm_upstream", acd_details["bpm_upstream"].iloc[0])) or (
        after_bpm != str(acd_details.attrs.get("bpm_downstream", acd_details["bpm_downstream"].iloc[0]))
    ):
        acd_config = ACDipoleConfig(
            ac_dipole_marker=acd_config.ac_dipole_marker,
            model=acd_model,
            dpx_tune=acd_config.dpx_tune,
            dpy_tune=acd_config.dpy_tune,
            bpm_upstream=before_bpm,
            bpm_downstream=after_bpm,
            smooth_lambda=acd_config.smooth_lambda,
        )
        acd_details = calculate_pz(
            weighted_svd_input,
            model_tws=model_tws,
            acd=acd_config,
            acd_only=True,
            info=False,
        )

    acd_recon = _apply_acd_details_to_reconstruction(
        _reconstruct(
            weighted_svd_input,
            analysis_dir=analysis_dir,
            model_tws=model_tws,
            beam=args.beam,
        ),
        acd_details,
    )

    measurement_twiss, datasets = _build_reconstruction_datasets(
        analysis_dir=analysis_dir,
        raw_input=raw_input,
        svd_input=svd_input,
        weighted_svd_input=weighted_svd_input,
        model_tws=model_tws,
        beam=args.beam,
        acd_recon=acd_recon,
    )

    uncompensated_analysis_dir = runtime_dir / get_uncompensated_analysis_dir(analysis_dir).name
    uncompensated_analysis_dir, _ = rerun_optics_analysis_without_compensation(
        analysis_dir,
        target_analysis_dir=uncompensated_analysis_dir,
    )
    uncomp_acd_recon = _apply_acd_details_to_reconstruction(
        _reconstruct(
            weighted_svd_input,
            analysis_dir=uncompensated_analysis_dir,
            model_tws=model_tws,
            beam=args.beam,
        ),
        acd_details,
    )
    uncomp_measurement_twiss, uncomp_datasets = _build_reconstruction_datasets(
        analysis_dir=uncompensated_analysis_dir,
        raw_input=raw_input,
        svd_input=svd_input,
        weighted_svd_input=weighted_svd_input,
        model_tws=model_tws,
        beam=args.beam,
        acd_recon=uncomp_acd_recon,
    )

    bpm_rows = [("Before ACD", before_bpm), ("After ACD", after_bpm)]
    phase_fig, phase_axes = plt.subplots(4, 2, figsize=(12, 16), squeeze=False)
    _plot_phase_space_on_axes(
        phase_axes[:2, :],
        _apply_palette(datasets, PHASE_PLOT_COLORS),
        bpm_rows,
        max_turns=args.max_turns,
        title_prefix="Compensated",
    )
    _plot_phase_space_on_axes(
        phase_axes[2:, :],
        _apply_palette(uncomp_datasets, UNCOMP_PHASE_PLOT_COLORS),
        bpm_rows,
        max_turns=args.max_turns,
        title_prefix="Uncompensated",
    )
    _style_legend(phase_axes[0, 0])
    _style_legend(phase_axes[2, 0])
    phase_fig.tight_layout()

    extra_phase_fig, extra_phase_axes = _plot_phase_space_figure(
        _apply_palette(datasets, PHASE_PLOT_COLORS),
        [("BPM.15R1", extra_bpm)],
        max_turns=args.max_turns,
        figsize=(12, 4),
    )
    _style_legend(extra_phase_axes[0, 0])
    extra_phase_fig.tight_layout()

    norm_layout_specs = [
        (0, 0, before_bpm, "x", "px", f"Before ACD ({before_bpm}.X)"),
        (0, 1, before_bpm, "y", "py", f"Before ACD ({before_bpm}.Y)"),
        (1, 0, after_bpm, "x", "px", f"After ACD ({after_bpm}.X)"),
        (1, 1, after_bpm, "y", "py", f"After ACD ({after_bpm}.Y)"),
    ]

    norm_fig, norm_axes = plt.subplots(4, 2, figsize=(12, 16), squeeze=False)
    _plot_normalised_grid(
        norm_axes[:2, :],
        norm_layout_specs,
        datasets,
        max_turns=args.max_turns,
        measurement_twiss=measurement_twiss,
        model_tws=model_tws,
        palette=NORMALISED_PLOT_COLORS,
        title_prefix="Compensated",
    )
    _plot_normalised_grid(
        norm_axes[2:, :],
        norm_layout_specs,
        uncomp_datasets,
        max_turns=args.max_turns,
        measurement_twiss=uncomp_measurement_twiss,
        model_tws=model_tws,
        palette=UNCOMP_NORMALISED_PLOT_COLORS,
        title_prefix="Uncompensated",
    )
    _style_legend(norm_axes[0, 0])
    _style_legend(norm_axes[2, 0])
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
    _style_legend(jump_axes[0])
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
        LOGGER.info("Saved combined phase-space plot to %s", phase_output)
        LOGGER.info("Saved BPM.15R1 phase-space plot to %s", extra_phase_output)
        LOGGER.info("Saved combined normalised phase-space plot to %s", norm_output)
        LOGGER.info("Saved delta plot to %s", jump_output)

    plt.show()
    acd_model.close()


if __name__ == "__main__":
    main()
