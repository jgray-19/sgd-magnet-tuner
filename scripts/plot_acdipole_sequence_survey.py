#!/usr/bin/env python3
"""Plot the surveyed machine segment between the AC-dipole BPM pair.

Assumptions are intentionally aggressive:
- use the same squeeze/beam conventions as the measurement scripts,
- read the upstream/downstream BPMs from temp-analysis metadata if present,
- otherwise require them on the command line,
- use the standard generated sequence for the squeeze unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt

from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.measurements.optimise_squeeze_quads import MEAS_TIMES, get_sequence_creation_time
from aba_optimiser.measurements.squeeze_helpers import get_model_dir, get_or_make_sequence

LOGGER = logging.getLogger(__name__)

PLOT_ELEMENT_NAMES = [
    "bpmyb.6l4.b1",
    "mqy.6l4.b1",
    "mcbyv.6l4.b1",
    "bqkv.6l4.b1",
    "mkqa.6l4.b1",
    "bqkh.b6l4.b1",
    "bpmya.5l4.b1",
]


def _remove_drift_rows(df):
    return df.loc[~df["name"].astype(str).str.contains("drift")].copy()


def _load_metadata_bpm_pair(beam: int, squeeze_step: str) -> tuple[str | None, str | None]:
    temp_dir = PROJECT_ROOT / f"temp_analysis_squeeze_b{beam}_{squeeze_step.replace('.', '_')}"
    metadata_file = temp_dir / "metadata.json"
    if not metadata_file.exists():
        return None, None
    payload = json.loads(metadata_file.read_text())
    return payload.get("ac_dipole_bpm_upstream"), payload.get("ac_dipole_bpm_downstream")


def _build_line(sequence_file: Path, beam: int) -> xt.Line:
    seq_name = f"lhcb{beam}"
    env = xt.load_madx_lattice(file=sequence_file)
    if seq_name not in env.lines:
        raise ValueError(f"Sequence {seq_name} not found in {sequence_file}")
    return env.lines[seq_name].copy()


def _select_segment(line: xt.Line, start_name: str, end_name: str) -> xt.Line:
    segment = line.select(start=start_name, end=end_name, name=f"{start_name}_to_{end_name}")
    names = [str(name) for name in segment.element_names]
    if start_name not in names:
        raise ValueError(f"Start BPM {start_name} not found in selected segment.")
    if end_name not in names:
        raise ValueError(f"End BPM {end_name} not found in selected segment.")
    return segment


def _plotted_names(survey_df) -> list[str]:
    available_names = set(survey_df["name"].astype(str))
    return [name for name in PLOT_ELEMENT_NAMES if name in available_names]


def _label_config(label_names: list[str]) -> dict[str, dict[str, object]]:
    config: dict[str, dict[str, object]] = {}
    for ii, name in enumerate(label_names):
        is_upper = ii % 2 == 0
        is_bpm = name.startswith("bpm")
        x_offset = -5 if (is_bpm and ii == 0) else (5 if is_bpm else 0)
        more_offset = -10 if "bqkv" in name else 0
        config[f"^{re.escape(name)}$"] = {
            "text": name.upper(),
            "fontsize": 10,
            "color": "0.15",
            "xytext": (x_offset, (12 if is_upper else -12) + more_offset),
            "textcoords": "offset points",
            "ha": "right" if (is_bpm and is_upper) else ("left" if is_bpm else "center"),
            "va": "bottom" if is_upper else "top",
            "arrowprops": {"arrowstyle": "-", "color": "0.7", "shrinkB": 4},
        }
    return config


def _box_config() -> dict[str, dict[str, object] | bool]:
    return {
        "^bpmyb\\.6l4\\.b1$": {"color": "tab:blue", "length": 0.15, "alpha": 0.7},
        "^bpmya\\.5l4\\.b1$": {"color": "tab:blue", "length": 0.15, "alpha": 0.7},
        "^mqy\\.6l4\\.b1$": {"color": "tab:orange", "alpha": 0.7},
        "^mcbyv\\.6l4\\.b1$": {"color": "tab:purple", "length": 0.2, "alpha": 0.75},
        "^bqkv\\.6l4\\.b1$": {"color": "tab:red", "length": 0.2, "alpha": 0.75},
        "^mkqa\\.6l4\\.b1$": {"color": "tab:green", "length": 0.35, "alpha": 0.85},
        "^bqkh\\.b6l4\\.b1$": {"color": "tab:brown", "length": 0.2, "alpha": 0.75},
    }


def _thin_plot_artists(ax, linewidth: float = 0.8) -> None:
    for line in ax.lines:
        line.set_linewidth(linewidth)
    for patch in ax.patches:
        patch.set_linewidth(linewidth)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Plot the Xtrack survey segment between the inferred AC-dipole BPMs."
    )
    parser.add_argument("--beam", type=int, choices=[1, 2], required=True)
    parser.add_argument("--squeeze-step", type=str, required=True)
    parser.add_argument("--before-bpm", type=str, default=None)
    parser.add_argument("--after-bpm", type=str, default=None)
    parser.add_argument("--sequence-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.squeeze_step not in MEAS_TIMES[args.beam]:
        raise ValueError(f"Unknown squeeze step {args.squeeze_step!r} for beam {args.beam}.")

    meta_before, meta_after = _load_metadata_bpm_pair(args.beam, args.squeeze_step)
    before_bpm = args.before_bpm or meta_before
    after_bpm = args.after_bpm or meta_after
    if before_bpm is None or after_bpm is None:
        raise ValueError(
            "Could not infer the AC-dipole BPM pair from metadata. "
            "Pass --before-bpm and --after-bpm explicitly."
        )

    if args.sequence_file is not None:
        sequence_file = args.sequence_file
    else:
        model_dir = get_model_dir(args.beam, args.squeeze_step)
        sequence_time = get_sequence_creation_time(MEAS_TIMES[args.beam][args.squeeze_step], args.squeeze_step)
        sequence_file = get_or_make_sequence(args.beam, model_dir, time=sequence_time)

    # xsuite uses lowercase!
    before_bpm = before_bpm.lower()
    after_bpm = after_bpm.lower()
    LOGGER.info("Using sequence %s", sequence_file)
    LOGGER.info("Plotting segment from %s to %s", before_bpm, after_bpm)

    line = _build_line(sequence_file, args.beam)
    segment_line = _select_segment(line, before_bpm, after_bpm)
    survey = segment_line.survey()
    survey_df = survey.to_pandas()
    survey_df = _remove_drift_rows(survey_df)
    plotted_names = _plotted_names(survey_df)
    labels = _label_config(plotted_names)
    boxes = _box_config()

    print("Survey rows to be plotted:")
    print(survey_df.loc[survey_df["name"].astype(str).isin(plotted_names)].to_string(index=False))

    survey.plot(boxes=boxes, labels=labels, legend=False)
    ax = plt.gca()
    fig = ax.figure
    fig.set_size_inches(10, 2.5)
    _thin_plot_artists(ax)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.margins(0)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
        LOGGER.info("Saved plot to %s", args.output)

    plt.show()


if __name__ == "__main__":
    main()
