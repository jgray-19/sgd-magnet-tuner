"""Diagnostic plotting utilities for quadrupole estimates and phase advances."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import MEASUREMENTS_ARTIFACTS_ROOT, PROJECT_ROOT
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.measurements.sequence import get_or_make_sequence
from aba_optimiser.measurements.squeeze.config import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
    get_results_dir,
)
from aba_optimiser.measurements.squeeze.io import load_estimates_and_uncertainties

LOGGER = logging.getLogger(__name__)

EstimateSource = Literal["none", "estimates", "quad-checkpoint", "bend-checkpoint"]

MEASUREMENT_LABEL = "Measurement"
DESIGN_OPTICS_LABEL = "Design Optics"
BEST_KNOWLEDGE_LABEL = "Best Knowledge Model"
BETTER_KNOWLEDGE_LABEL = "Better Knowledge Model"

PLOT_COLORS = {
    MEASUREMENT_LABEL: "black",
    DESIGN_OPTICS_LABEL: "tab:blue",
    BEST_KNOWLEDGE_LABEL: "tab:orange",
    BETTER_KNOWLEDGE_LABEL: "tab:red",
}

ARC56_OFFSETS_FILE = PROJECT_ROOT / "data" / "LHC_BBA_arc56_bpm_to_quad_offsets.csv"


@dataclass(frozen=True)
class PlotContext:
    accelerator: LHC
    design_accelerator: LHC
    all_estimates: dict[str, float] | None
    analysis_dir: Path
    squeeze_step: str
    results_dir: Path
    tune_knobs: Path
    corrector_file: Path | None
    beam: int
    deltap: float
    b2_errors: Path | None


def parse_arc_spec(arcs: str | None) -> list[int] | None:
    """Parse arc selection CLI input into a list of arc numbers."""
    if not arcs:
        return None
    if "-" in arcs:
        start_str, end_str = arcs.split("-", 1)
        start, end = int(start_str), int(end_str)
        return list(range(start, end + 1))
    return [int(x.strip()) for x in arcs.split(",")]


def prepare_plot_context(
    *,
    beam: int,
    squeeze_step: str,
    use_optics: bool,
    frequency: str,
    estimate_source: EstimateSource,
    checkpoint_dir: Path | None,
    max_uncertainty: float | None,
    fullring_knob_diffs: bool,
) -> PlotContext:
    """Build the common plotting context shared by the arc and full-ring CLIs."""
    beam_path = BETABEAT_DIR / get_measurement_date(squeeze_step) / f"LHCB{beam}/"
    model_base_dir = beam_path / "Models/"
    analysis_base_dir = beam_path / "Results/"

    results_dir = get_results_dir(beam)

    squeeze_step_id = squeeze_step.replace(".", "_")
    temp_analysis_dir = (
        MEASUREMENTS_ARTIFACTS_ROOT / "temp" / f"temp_analysis_squeeze_b{beam}_{squeeze_step_id}"
    )
    madng_model_dir = temp_analysis_dir / "madng_model"
    if not madng_model_dir.exists():
        model_dir = model_base_dir / MODEL_DIRS[beam][squeeze_step]
        seq_file = get_or_make_sequence(beam, model_dir)
    else:
        seq_file = get_or_make_sequence(beam, madng_model_dir)

    estimates_file: Path | None = None
    if estimate_source == "estimates":
        fldr_name = "optics" if use_optics else "squeeze"
        estimates_file = (
            MEASUREMENTS_ARTIFACTS_ROOT
            / "results"
            / f"b{beam}_{fldr_name}_results"
            / f"quad_estimates_{squeeze_step}.json"
        )
        if not estimates_file.exists():
            legacy_estimates_file = estimates_file.with_suffix(".txt")
            if legacy_estimates_file.exists():
                estimates_file = legacy_estimates_file
            else:
                raise FileNotFoundError(f"Estimates file not found: {estimates_file}")

    kinetic_energy, deltap, b2_errors = load_model_metadata(beam, squeeze_step)
    print(f"Beam energy set to: {kinetic_energy} GeV")

    design_accelerator = LHC(
        beam=beam,
        kinetic_energy=kinetic_energy,
        sequence_file=seq_file,
        optimise_bends=True,
        optimise_quadrupoles=True,
        optimise_other_quadrupoles=True,
        optimise_sextupoles=True,
        optimise_correctors=False,
        optimise_quad_dx=True,
        optimise_quad_dy=True,
    )
    accelerator = LHC(
        beam=beam,
        kinetic_energy=kinetic_energy,
        sequence_file=seq_file,
        optimise_bends=True,
        optimise_quadrupoles=True,
        optimise_other_quadrupoles=True,
        optimise_sextupoles=True,
        optimise_correctors=False,
        optimise_quad_dx=True,
        optimise_quad_dy=True,
    )

    if estimate_source == "none":
        estimates = {}
        uncertainties = {}
    elif estimate_source == "estimates":
        assert estimates_file is not None
        estimates, uncertainties = load_estimates_and_uncertainties(estimates_file)
    else:
        estimates = load_estimates_from_checkpoints(
            beam=beam,
            squeeze_step=squeeze_step,
            checkpoint_kind=estimate_source,
            checkpoint_dir=checkpoint_dir,
        )
        uncertainties = {
            arc: dict.fromkeys(arc_estimates, 0.0) for arc, arc_estimates in estimates.items()
        }

    tune_knobs = results_dir / f"tune_knobs_{squeeze_step}_0Hz.txt"
    if not tune_knobs.exists():
        raise FileNotFoundError(f"Tune knobs file not found: {tune_knobs}")
    print(f"Using tune knobs file: {tune_knobs}")

    corrector_file = results_dir / f"corrector_strengths_{squeeze_step}_{frequency}.txt"
    if not corrector_file.exists():
        raise FileNotFoundError(f"Corrector file not found: {corrector_file}")
    print(f"Using corrector file: {corrector_file}")

    if estimate_source == "none":
        LOGGER.info(
            "Skipping estimate loading and conversion; plotting only design and best-knowledge models"
        )
        all_estimates = None
    else:
        actual = find_true_values(accelerator, estimates, tune_knobs)
        estimates, uncertainties, actual = filter_estimates_by_max_uncertainty(
            estimates,
            uncertainties,
            actual,
            max_uncertainty,
        )

        plot_quad_diffs(
            estimates,
            uncertainties,
            actual,
            squeeze_step,
            results_dir,
            fullring=fullring_knob_diffs,
            accelerator=accelerator if fullring_knob_diffs else None,
            overlay_arc56_offsets=fullring_knob_diffs,
        )

        all_estimates = {}
        for arc in estimates.values():
            all_estimates.update(arc)

    analysis_dir = analysis_base_dir / ANALYSIS_DIRS[beam][squeeze_step]
    return PlotContext(
        accelerator=accelerator,
        design_accelerator=design_accelerator,
        all_estimates=all_estimates,
        analysis_dir=analysis_dir,
        squeeze_step=squeeze_step,
        results_dir=results_dir,
        tune_knobs=tune_knobs,
        corrector_file=corrector_file,
        beam=beam,
        deltap=deltap,
        b2_errors=b2_errors,
    )


def load_model_metadata(beam: int, squeeze_step: str) -> tuple[float, float, Path | None]:
    """Load beam energy, machine deltap, and optional b2 error table from optimisation metadata."""
    metadata_file = (
        MEASUREMENTS_ARTIFACTS_ROOT
        / "temp"
        / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
        / "metadata.json"
    )
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_file}. Run optimisation first or regenerate without --skip-reload."
        )
    with metadata_file.open("r") as f:
        metadata = json.load(f)
    energy = float(metadata.get("energy", 450.0))
    deltap = float(metadata.get("machine_deltap", 0.0))
    b2_errors_raw = metadata.get("b2_errors")
    b2_errors = Path(b2_errors_raw) if b2_errors_raw else None
    return energy, deltap, b2_errors


def load_estimates_from_checkpoints(
    beam: int,
    squeeze_step: str,
    checkpoint_kind: Literal["quad-checkpoint", "bend-checkpoint"],
    checkpoint_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load per-arc estimate knobs from optimisation checkpoints."""
    squeeze_step_id = squeeze_step.replace(".", "_")
    default_dir = (
        MEASUREMENTS_ARTIFACTS_ROOT
        / "temp"
        / f"temp_analysis_squeeze_b{beam}_{squeeze_step_id}"
        / "checkpoints"
    )
    ckpt_dir = checkpoint_dir if checkpoint_dir is not None else default_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    stage_suffix = "quads" if checkpoint_kind == "quad-checkpoint" else "bends"
    pattern = f"checkpoint_b{beam}_{squeeze_step_id}_arc*_{stage_suffix}.json"
    checkpoint_files = sorted(ckpt_dir.glob(pattern))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No {stage_suffix} checkpoints found in {ckpt_dir} matching {pattern}"
        )

    estimates: dict[str, dict[str, float]] = {}
    for checkpoint_file in checkpoint_files:
        payload = json.loads(checkpoint_file.read_text())
        stem = checkpoint_file.stem
        try:
            arc_token = stem.split("_arc", 1)[1].split("_", 1)[0]
            arc_num = int(arc_token)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse arc number from checkpoint name: {checkpoint_file}"
            ) from exc

        arc_key = f"Arc {arc_num}"
        knob_map = payload.get("current_knobs") or {}
        # knob_map = payload.get("best_knobs") or {}
        estimates[arc_key] = {str(k): float(v) for k, v in knob_map.items()}

    LOGGER.info(
        "Loaded %d arcs of estimates from %s checkpoints in %s",
        len(estimates),
        stage_suffix,
        ckpt_dir,
    )
    return estimates


def get_twiss_without_errors(
    accelerator: LHC,
    just_bpms: bool,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs: Path | None = None,
    corrector_file: Path | None = None,
    deltap: float = 0.0,
    b2_errors: Path | None = None,
) -> pd.DataFrame:
    """Get twiss data from a model with optional tune knobs and estimated magnets."""
    mad = GradientDescentMadInterface(
        accelerator,
        corrector_knobs=corrector_file,
        tune_knobs=tune_knobs,
        b2_errors=b2_errors,
    )
    if estimated_magnets is not None:
        # Saved estimates can reference knobs absent from this optics (e.g. a dy offset
        # on a quad whose k1 is zero here, so no misalignment knob was created). Such a
        # knob has no optical effect, so drop it rather than fail the whole plot.
        known = {n: v for n, v in estimated_magnets.items() if n in mad.knob_name_set}
        dropped = sorted(set(estimated_magnets) - set(known))
        if dropped:
            LOGGER.warning(
                "Ignoring %d estimate knob(s) not present in the loaded model: %s",
                len(dropped),
                ", ".join(dropped[:10]) + ("..." if len(dropped) > 10 else ""),
            )
        mad.update_knob_values(known)
    return mad.run_twiss(deltap=deltap, observe=int(just_bpms), chrom=True)


def get_fullring_twiss(
    accelerator: LHC,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs: Path | None = None,
    corrector_file: Path | None = None,
    deltap: float = 0.0,
    b2_errors: Path | None = None,
) -> pd.DataFrame:
    """Get full-ring twiss data at BPMs with column names aligned to measurement data."""
    twiss = get_twiss_without_errors(
        accelerator,
        just_bpms=True,
        estimated_magnets=estimated_magnets,
        tune_knobs=tune_knobs,
        corrector_file=corrector_file,
        deltap=deltap,
        b2_errors=b2_errors,
    )
    return twiss.rename(columns={"mu1": "mux", "mu2": "muy", "beta11": "betx", "beta22": "bety"})


def find_true_values(
    accelerator: LHC,
    estimates: dict[str, dict[str, float]],
    tune_knobs: Path,
    corrector_file: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Return zero-reference values for optimisation-space plots."""
    del accelerator, tune_knobs, corrector_file
    return {arc: dict.fromkeys(mags, 0.0) for arc, mags in estimates.items()}


def convert_estimates_to_optimisation_space(
    accelerator: LHC,
    estimates: dict[str, dict[str, float]],
    tune_knobs: Path,
    corrector_file: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Return saved result values unchanged because outputs are already in optimisation space."""
    del accelerator, tune_knobs, corrector_file
    return {arc: mags.copy() for arc, mags in estimates.items()}


def convert_uncertainties_to_optimisation_space(
    accelerator: LHC,
    uncertainties: dict[str, dict[str, float]],
    tune_knobs: Path,
    corrector_file: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Return saved uncertainties unchanged because outputs are already in optimisation space."""
    del accelerator, tune_knobs, corrector_file
    return {arc: mags.copy() for arc, mags in uncertainties.items()}


def filter_estimates_by_max_uncertainty(
    estimates: dict[str, dict[str, float]],
    uncertainties: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
    max_uncertainty: float | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Discard estimates whose relative uncertainty exceeds the requested threshold."""
    if max_uncertainty is None:
        return estimates, uncertainties, actual

    filtered_estimates: dict[str, dict[str, float]] = {}
    filtered_uncertainties: dict[str, dict[str, float]] = {}
    filtered_actual: dict[str, dict[str, float]] = {}
    dropped = 0

    for arc, arc_estimates in estimates.items():
        arc_uncertainties = uncertainties.get(arc, {})
        arc_actual = actual.get(arc, {})

        kept_estimates: dict[str, float] = {}
        kept_uncertainties: dict[str, float] = {}
        kept_actual: dict[str, float] = {}

        for knob, value in arc_estimates.items():
            uncertainty = float(arc_uncertainties.get(knob, 0.0))
            scale = abs(float(value))
            relative_uncertainty = (
                0.0 if uncertainty == 0.0 else float("inf") if scale == 0.0 else uncertainty / scale
            )
            if not np.isfinite(relative_uncertainty) or relative_uncertainty > max_uncertainty:
                dropped += 1
                continue
            kept_estimates[knob] = value
            kept_uncertainties[knob] = uncertainty
            if knob in arc_actual:
                kept_actual[knob] = arc_actual[knob]

        if kept_estimates:
            filtered_estimates[arc] = kept_estimates
            filtered_uncertainties[arc] = kept_uncertainties
            filtered_actual[arc] = kept_actual

    LOGGER.info(
        "Applied max-relative-uncertainty filter %.3e: kept %d estimates, dropped %d",
        max_uncertainty,
        sum(len(v) for v in filtered_estimates.values()),
        dropped,
    )
    return filtered_estimates, filtered_uncertainties, filtered_actual


def _knob_diff(estimate: float, truth: float) -> float:
    """Return the signed difference between an estimate and its true value."""
    return estimate - truth


def _is_displacement_knob(knob_name: str) -> bool:
    """Return True when a knob name denotes a transverse displacement (dx/dy)."""
    knob_suffix = knob_name.rsplit(".", 1)[-1].lower()
    return knob_suffix.endswith(("dx", "dy"))


def _load_arc56_vertical_offsets(beam: int, data_file: Path) -> dict[str, pd.DataFrame]:
    """Load measured arc56 vertical-offset values (per reconstruction method) from a CSV."""
    if not data_file.exists():
        return {}

    data = pd.read_csv(data_file)
    if beam == 1:
        name_col = "name"
        methods = {
            "pQMS 1": ("y offset beam 1 pQMS 1", "y offset unc beam 1 pQMS 1"),
            "pQMS 2": ("y offset beam 1 pQMS 2", "y offset unc beam 1 pQMS 2"),
            "k-mod": ("y offset beam 1 k-mod", "y offset unc beam 1 k-mod"),
        }
    else:
        name_col = "name 2"
        methods = {
            "pQMS 1": ("y offset beam 2 pQMS 1", "y offset unc beam 2 pQMS 1"),
            "pQMS 2": ("y offset beam 2 pQMS 2", "y offset unc beam 2 pQMS 2"),
            "k-mod": ("y offset beam 2 k-mod", "y offset unc beam 2 k-mod"),
        }

    if name_col not in data.columns:
        return {}

    results: dict[str, pd.DataFrame] = {}
    names = data[name_col].astype(str).str.strip().str.upper()
    for method_label, (value_col, err_col) in methods.items():
        if value_col not in data.columns or err_col not in data.columns:
            continue
        method_df = pd.DataFrame(
            {
                "name": names,
                "value_mm": pd.to_numeric(data[value_col], errors="coerce") * 1.0e-3,
                "err_mm": pd.to_numeric(data[err_col], errors="coerce") * 1.0e-3,
            }
        ).dropna(subset=["name", "value_mm", "err_mm"])
        if not method_df.empty:
            results[method_label] = method_df
    return results


def _overlay_arc56_offsets_on_displacement_axis(
    ax: plt.Axes,
    accelerator: LHC | None,
    data_file: Path,
) -> None:
    """Overlay measured arc56 vertical offsets onto the displacement axis."""
    if accelerator is None:
        return

    beam = int(getattr(accelerator, "beam", 1))
    offsets_by_method = _load_arc56_vertical_offsets(beam, data_file)
    if not offsets_by_method:
        LOGGER.warning("No arc56 vertical offsets loaded from %s", data_file)
        return

    method_styles = {
        "pQMS 1": ("P", "tab:green"),
        "pQMS 2": ("X", "tab:cyan"),
        "k-mod": ("*", "tab:brown"),
    }

    for method_label, method_df in offsets_by_method.items():
        quad_elements = [f"MQ{name}.B{beam}" for name in method_df["name"]]
        quad_pos = get_element_positions(accelerator, quad_elements)

        xcoords: list[float] = []
        ycoords: list[float] = []
        yerrs: list[float] = []
        missing_names: list[str] = []
        for _, row in method_df.iterrows():
            quad_element = f"MQ{row['name']}.B{beam}"
            s_pos = quad_pos.get(quad_element)
            if s_pos is None:
                missing_names.append(str(row["name"]))
                continue
            xcoords.append(float(s_pos))
            ycoords.append(float(row["value_mm"]))
            yerrs.append(float(row["err_mm"]))

        if not xcoords:
            continue

        marker_style, color = method_styles.get(method_label, ("D", "tab:gray"))
        ax.errorbar(
            xcoords,
            ycoords,
            yerr=yerrs,
            fmt=marker_style,
            linestyle="None",
            markersize=6,
            markeredgewidth=0.8,
            color=color,
            alpha=0.95,
            capsize=2,
            label=f"Arc56 dy offset ({method_label})",
            zorder=5,
        )

        if missing_names:
            LOGGER.warning(
                "Could not map %d arc56 quadrupole names for beam %d (%s): %s",
                len(missing_names),
                beam,
                method_label,
                ", ".join(sorted(set(missing_names))),
            )


def _plot_knobs_on_axis(
    ax: plt.Axes,
    knob_names: list[str],
    knob_values: list[float],
    knob_uncertainties: list[float],
    colors: list[str] | str,
    accelerator: LHC | None,
    ip_positions: dict[str, float] | None,
    width: float = 20.0,
) -> None:
    """Bar-plot knob values on one axis, using element s-positions when an accelerator is given."""
    if accelerator is None:
        ax.bar(
            range(len(knob_names)),
            knob_values,
            yerr=knob_uncertainties,
            color=colors,
            capsize=2,
        )
        ax.set_xticks(range(len(knob_names)))
        ax.set_xticklabels([m.split(".")[1] for m in knob_names], rotation=90)
        return

    elem_names = [m.rsplit(".", 1)[0] for m in knob_names]
    elem_pos = get_element_positions(accelerator, elem_names)
    knobs_with_pos = [
        (knob_name, knob_value, elem_pos[knob_name.rsplit(".", 1)[0]])
        for knob_name, knob_value in zip(knob_names, knob_values, strict=False)
        if knob_name.rsplit(".", 1)[0] in elem_pos
    ]

    if len(knobs_with_pos) == len(knob_names):
        x_positions = [pos for _, _, pos in knobs_with_pos]
        y_values = [value for _, value, _ in knobs_with_pos]
        y_uncertainties = [
            knob_uncertainties[knob_names.index(name)] for name, _, _ in knobs_with_pos
        ]
        if isinstance(colors, list):
            plot_colors = [colors[knob_names.index(name)] for name, _, _ in knobs_with_pos]
        else:
            plot_colors = colors
        ax.bar(
            x_positions,
            y_values,
            yerr=y_uncertainties,
            color=plot_colors,
            width=width,
            capsize=2,
        )
        ax.set_xlabel("S (m)")
        if ip_positions is not None:
            add_ip_positions_to_plot(ax, ip_positions)
        return

    ax.bar(
        range(len(knob_names)), knob_values, yerr=knob_uncertainties, color=colors, capsize=2
    )
    ax.set_xticks(range(len(knob_names)))
    ax.set_xticklabels([m.split(".")[1] for m in knob_names], rotation=90)


def _plot_fullring_quad_diffs(
    estimates: dict,
    uncertainties: dict,
    actual: dict,
    squeeze_step: str,
    results_dir: Path,
    accelerator: LHC | None,
    overlay_arc56_offsets: bool,
) -> None:
    """Plot full-ring knob differences as quadrupole, displacement and bend panels."""
    arc_key = "Arc 1"
    if arc_key not in estimates or arc_key not in actual:
        raise KeyError(
            "Full-ring mode expects all strengths under 'Arc 1' in both estimates and actual values."
        )

    mags = list(estimates[arc_key].keys())
    strength_mags = [m for m in mags if not _is_displacement_knob(m)]
    displacement_mags = [m for m in mags if _is_displacement_knob(m)]

    quads = [m for m in strength_mags if "mq" in m.lower()]
    bends = [m for m in strength_mags if "mq" not in m.lower() and "mb" in m.lower()]
    others = [m for m in strength_mags if "mq" not in m.lower() and "mb" not in m.lower()]
    bends_diffs = [_knob_diff(estimates[arc_key][m], actual[arc_key][m]) for m in bends]
    bends_unc = [uncertainties[arc_key].get(m, 0.0) for m in bends]
    others_diffs = [_knob_diff(estimates[arc_key][m], actual[arc_key][m]) for m in others]
    others_unc = [uncertainties[arc_key].get(m, 0.0) for m in others]
    displacement_diffs_mm = [
        1e3 * _knob_diff(estimates[arc_key][m], actual[arc_key][m])
        for m in displacement_mags
    ]
    displacement_unc_mm = [1e3 * uncertainties[arc_key].get(m, 0.0) for m in displacement_mags]

    displacement_colors = [
        "tab:orange" if m.rsplit(".", 1)[-1].lower().endswith("dx") else "tab:purple"
        for m in displacement_mags
    ]

    ip_positions = get_ip_positions(accelerator) if accelerator is not None else None
    has_others_panel = len(others) > 0

    import matplotlib.patches as mpatches

    fig_main, main_axes = plt.subplots(2, 1, figsize=(22, 10))
    ax1, displacement_ax = main_axes

    if accelerator is not None:
        quad_elems = [m.rsplit(".", 1)[0] for m in quads]
        quad_pos = get_element_positions(accelerator, quad_elems)
        quads_with_pos = [
            (quad_name, quad_pos[quad_name.rsplit(".", 1)[0]])
            for quad_name in quads
            if quad_name.rsplit(".", 1)[0] in quad_pos
        ]
        quads_without_pos = [
            quad_name for quad_name in quads if quad_name.rsplit(".", 1)[0] not in quad_pos
        ]

        if quads_without_pos:
            others.extend(quads_without_pos)
            others_diffs.extend(
                [
                    _knob_diff(estimates[arc_key][q], actual[arc_key][q])
                    for q in quads_without_pos
                ]
            )
            others_unc.extend([uncertainties[arc_key].get(q, 0.0) for q in quads_without_pos])
            LOGGER.info(
                "Moved %d quadrupoles without s-position to 'Others' panel.",
                len(quads_without_pos),
            )

        if quads_with_pos:
            quad_x = [pos for _, pos in quads_with_pos]
            quad_y = [
                _knob_diff(estimates[arc_key][q], actual[arc_key][q])
                for q, _ in quads_with_pos
            ]
            quad_unc = [uncertainties[arc_key].get(q, 0.0) for q, _ in quads_with_pos]
            ax1.bar(quad_x, quad_y, yerr=quad_unc, color="red", width=20.0, capsize=2)
            ax1.set_xlabel("S (m)")
            add_ip_positions_to_plot(ax1, ip_positions)
        else:
            ax1.text(
                0.5,
                0.5,
                "No quadrupoles with available s-position",
                transform=ax1.transAxes,
                ha="center",
                va="center",
            )
            ax1.set_xticks([])
    else:
        quad_diffs = [_knob_diff(estimates[arc_key][m], actual[arc_key][m]) for m in quads]
        quad_unc = [uncertainties[arc_key].get(m, 0.0) for m in quads]
        ax1.bar(range(len(quads)), quad_diffs, yerr=quad_unc, color="red", capsize=2)
        ax1.set_xticks(range(len(quads)))
        ax1.set_xticklabels([m.split(".")[1] for m in quads], rotation=90)
    ax1.set_title("Quadrupoles")
    ax1.set_ylabel("Optimisation knob value")

    _plot_knobs_on_axis(
        displacement_ax,
        displacement_mags,
        displacement_diffs_mm,
        displacement_unc_mm,
        displacement_colors,
        accelerator,
        ip_positions,
    )
    if overlay_arc56_offsets:
        _overlay_arc56_offsets_on_displacement_axis(
            displacement_ax,
            accelerator,
            ARC56_OFFSETS_FILE,
        )
    displacement_ax.set_title("Quadrupole Displacements")
    displacement_ax.set_ylabel("Displacement difference (mm)")

    displacement_handles = [
        mpatches.Patch(color="tab:orange", label="dx"),
        mpatches.Patch(color="tab:purple", label="dy"),
    ]
    displacement_ax.legend(handles=displacement_handles, loc="upper right")

    plt.figure(fig_main.number)
    plt.tight_layout()
    _path = results_dir / f"quad_diffs_{squeeze_step}_fullring.png"
    fig_main.savefig(_path)
    print(f"Saved plot: {_path.resolve()}")

    bend_nrows = 2 if has_others_panel else 1
    fig_bends, bend_axes = plt.subplots(bend_nrows, 1, figsize=(22, 5 * bend_nrows))
    if bend_nrows == 1:
        bend_axes = [bend_axes]

    bend_elem_names = [m.rsplit(".", 1)[0] for m in bends]
    bend_kinds = get_element_kinds(accelerator, bend_elem_names) if accelerator is not None else {}
    bend_colors = [
        "tab:orange" if bend_kinds.get(m.rsplit(".", 1)[0], "sbend") == "rbend" else "tab:blue"
        for m in bends
    ]
    _plot_knobs_on_axis(
        bend_axes[0],
        bends,
        bends_diffs,
        bends_unc,
        bend_colors,
        accelerator,
        ip_positions,
    )
    bend_axes[0].set_title("Bending Magnets")
    bend_axes[0].set_ylabel("Optimisation knob value")
    bend_legend_handles = [
        mpatches.Patch(color="tab:blue", label="sbend (MB)"),
        mpatches.Patch(color="tab:orange", label="rbend (MBR)"),
    ]
    bend_axes[0].legend(handles=bend_legend_handles, loc="upper right")

    if has_others_panel:
        _plot_knobs_on_axis(
            bend_axes[1],
            others,
            others_diffs,
            others_unc,
            "green",
            accelerator,
            ip_positions,
        )
        bend_axes[1].set_title("Others")
        bend_axes[1].set_ylabel("Optimisation knob value")

    plt.figure(fig_bends.number)
    plt.tight_layout()
    _path = results_dir / f"quad_diffs_{squeeze_step}_fullring_bends.png"
    fig_bends.savefig(_path)
    print(f"Saved plot: {_path.resolve()}")



def _plot_per_arc_quad_diffs(
    estimates: dict,
    uncertainties: dict,
    actual: dict,
    squeeze_step: str,
    results_dir: Path,
) -> None:
    """Plot one knob-difference panel per arc on a shared 2x4 grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for arc_num in range(1, 9):
        ax = axes[arc_num - 1]
        arc_key = f"Arc {arc_num}"
        if arc_key in estimates:
            mags = list(estimates[arc_key].keys())
            rel_diffs = [_knob_diff(estimates[arc_key][m], actual[arc_key][m]) for m in mags]
            rel_unc = [uncertainties[arc_key].get(m, 0.0) for m in mags]
            ax.bar(range(len(mags)), rel_diffs, yerr=rel_unc, capsize=2)
            ax.set_xticks(range(len(mags)))
            ax.set_xticklabels([m.split(".")[1] for m in mags], rotation=90)
            ax.set_title(f"Arc {arc_num}")
            ax.set_ylabel("Optimisation knob value")
    plt.tight_layout()
    _path = results_dir / f"quad_diffs_{squeeze_step}.png"
    plt.savefig(_path)
    print(f"Saved plot: {_path.resolve()}")



def plot_quad_diffs(
    estimates: dict,
    uncertainties: dict,
    actual: dict,
    squeeze_step: str,
    results_dir: Path,
    fullring: bool = False,
    accelerator: LHC | None = None,
    overlay_arc56_offsets: bool = False,
) -> None:
    """Plot optimisation-space knob values."""
    if fullring:
        _plot_fullring_quad_diffs(
            estimates,
            uncertainties,
            actual,
            squeeze_step,
            results_dir,
            accelerator,
            overlay_arc56_offsets,
        )
        return
    _plot_per_arc_quad_diffs(estimates, uncertainties, actual, squeeze_step, results_dir)
def get_arc_ranges(beam: int) -> dict[int, tuple[str, str]]:
    """Get arc ranges (start BPM -> end BPM) for each arc."""
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


def get_ip_positions(accelerator: LHC) -> dict[str, float]:
    """Get longitudinal positions for IP1..IP8 from a model twiss."""
    mad_iface = GradientDescentMadInterface(accelerator)
    mad_iface.observe(pattern="IP[1-8]$")
    tws = mad_iface.run_twiss()
    return {f"IP{ip}": float(tws.loc[f"IP{ip}", "s"]) for ip in range(1, 9)}


def add_ip_positions_to_plot(ax: plt.Axes, ip_positions: dict[str, float]) -> None:
    """Add dashed vertical lines and labels for interaction points."""
    for ip, pos in ip_positions.items():
        ax.axvline(pos, color="grey", linestyle="--", alpha=0.5)
        ax.text(
            pos,
            ax.get_ylim()[1] * 0.9,
            ip,
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            color="grey",
        )


def get_element_positions(accelerator: LHC, element_names: list[str]) -> dict[str, float]:
    """Get model s-positions for selected lattice elements."""
    mad_iface = GradientDescentMadInterface(accelerator)
    tws = mad_iface.run_twiss(observe=0)
    s_by_name = {str(name).upper(): float(s) for name, s in tws["s"].items()}
    return {name: s_by_name[name.upper()] for name in element_names if name.upper() in s_by_name}


def get_element_kinds(accelerator: LHC, element_names: list[str]) -> dict[str, str]:
    """Get the MAD-NG kind attribute for selected lattice elements."""
    mad_iface = GradientDescentMadInterface(accelerator)
    tws = mad_iface.run_twiss(observe=0)
    if "kind" in tws.columns:
        kind_by_name = {str(name).upper(): str(kind) for name, kind in tws["kind"].items()}
        return {name: kind_by_name[name.upper()] for name in element_names if name.upper() in kind_by_name}
    kinds: dict[str, str] = {}
    for name in element_names:
        with contextlib.suppress(KeyError, AttributeError):
            kinds[name] = mad_iface.mad.loaded_sequence[name].kind
    return kinds


def _normalize_phase(df: pd.DataFrame, mu_cols: tuple[str, str], start_bpm: str) -> pd.DataFrame:
    """Shift phase columns so the chosen start BPM sits at zero."""
    if df.empty:
        return df

    df = df.copy()
    start_row = df.loc[start_bpm] if start_bpm is not None else df.iloc[0]
    df[mu_cols[0]] = df[mu_cols[0]] - start_row[mu_cols[0]]
    df[mu_cols[1]] = df[mu_cols[1]] - start_row[mu_cols[1]]
    return df


def get_twiss_through_arc(
    accelerator: LHC,
    arc_start: str,
    arc_end: str,
    meas_twiss: pd.DataFrame,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs: Path | None = None,
    corrector_file: Path | None = None,
    deltap: float = 0.0,
) -> pd.DataFrame:
    """Get twiss data (phase and beta) through an arc using measurement beta0 as initial conditions."""
    mad = GradientDescentMadInterface(
        accelerator=accelerator,
        corrector_knobs=corrector_file,
        tune_knobs=tune_knobs,
    )

    if estimated_magnets is not None:
        mad.update_knob_values(estimated_magnets)

    if arc_start not in meas_twiss.index:
        raise KeyError(
            f"Start BPM {arc_start} not found in measurement twiss; bet/alf are required."
        )
    row = meas_twiss.loc[arc_start]

    required_cols = ("betx", "bety", "alfx", "alfy", "errbetx", "errbety", "erralfx", "erralfy")
    missing = [c for c in required_cols if c not in row.index or pd.isna(row[c])]
    if missing:
        raise KeyError(
            f"Missing required optics columns at {arc_start}: {missing}. betx/bety/alfx/alfy must be present."
        )

    optics = {
        "betx": row["betx"],
        "bety": row["bety"],
        "alfx": row["alfx"],
        "alfy": row["alfy"],
        "dx": row.get("dx", 0.0),
        "dpx": row.get("dpx", 0.0),
        "dy": row.get("dy", 0.0),
        "dpy": row.get("dpy", 0.0),
    }

    optics_err = {
        "betx": row["errbetx"],
        "bety": row["errbety"],
        "alfx": row["erralfx"],
        "alfy": row["erralfy"],
        "dx": row.get("errdx", 0.0),
        "dpx": row.get("errdpx", 0.0),
        "dy": row.get("errdy", 0.0),
        "dpy": row.get("errdpy", 0.0),
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
        deltap = {deltap:.6e},
    }}
    """

    def run_twiss(values: list[float]) -> pd.DataFrame:
        mad.mad.send(run_twiss_string)
        for val in values:
            mad.mad.send(val)
        df = mad.mad.twiss_result.to_df().set_index("name")
        return df.rename(columns={"mu1": "mux", "mu2": "muy", "beta11": "betx", "beta22": "bety"})

    base = run_twiss(list(optics.values()))
    plus = run_twiss([optics[k] + optics_err[k] for k in optics])
    minus = run_twiss([optics[k] - optics_err[k] for k in optics])

    base["mux_err"] = abs(plus["mux"] - minus["mux"]) / 2
    base["muy_err"] = abs(plus["muy"] - minus["muy"]) / 2
    base["betx_err"] = abs(plus["betx"] - minus["betx"]) / 2
    base["bety_err"] = abs(plus["bety"] - minus["bety"]) / 2

    return _normalize_phase(base, ("mux", "muy"), start_bpm=arc_start)


def get_measurement_phase_through_arc(
    meas_twiss: pd.DataFrame,
    arc_start: str,
    arc_end: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Get phase advance through arc from measurement data."""
    start_s = meas_twiss.loc[arc_start, "s"]
    end_s = meas_twiss.loc[arc_end, "s"]

    if end_s < start_s:
        arc_bpms = meas_twiss[(meas_twiss["s"] >= start_s) | (meas_twiss["s"] <= end_s)]
    else:
        arc_bpms = meas_twiss[(meas_twiss["s"] >= start_s) & (meas_twiss["s"] <= end_s)]

    if arc_bpms.empty:
        return arc_bpms, []

    candidate_bpms = list(arc_bpms.head(2).index)
    return arc_bpms, candidate_bpms


def plot_phase_advances(*args, **kwargs) -> None:
    from aba_optimiser.measurements.plotting.arc import plot_phase_advances as impl

    return impl(*args, **kwargs)


def plot_fullring_comparison(*args, **kwargs) -> None:
    from aba_optimiser.measurements.plotting.fullring import plot_fullring_comparison as impl

    return impl(*args, **kwargs)


def main() -> None:
    """Compatibility dispatcher for the legacy combined entry point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fullring", action="store_true")
    args, _ = parser.parse_known_args()

    if args.fullring:
        from aba_optimiser.measurements.plotting.fullring import main as impl
    else:
        from aba_optimiser.measurements.plotting.arc import main as impl
    impl()


if __name__ == "__main__":
    main()
