"""Presentation study for SPS separate vs diagonal kicks with single-plane BPMs.

The study has two layers:

- tracking-file physics plots that show what the BPM system actually records
- controller runs that show how the optimiser behaves on the two datasets

Outputs include:

- horizontal/vertical beta-beating comparison
- true and reconstructed magnet errors
- horizontal/vertical initial-condition grids at the selected start BPMs
- horizontal/vertical position variation along the ring, with start-BPM markers
- horizontal/vertical initial and final worker loss profiles along the ring
- CSV/JSON/Markdown summaries for slides or notes

Example:

```bash
../accpy/bin/python examples/study_sps_diagonal_vs_separate_kicks.py \
  --output-dir analysis/sps_diagonal_vs_separate_kicks_study
```
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, ScalarFormatter
from pymadng_utils.io.utils import save_knobs
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import line_to_dataframes
from xtrack_tools.tracking import run_tracking_without_ac_dipole

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad.aba_mad_interface import AbaMadInterface
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, OutputConfig, SequenceConfig
from aba_optimiser.training.data_manager import DataManager

if TYPE_CHECKING:
    import xtrack as xt


TRACK_COLUMNS = (
    "turn",
    "name",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
)
TARGET_QX = 0.13
TARGET_QY = 0.18
LOGGER = logging.getLogger(__name__)
CBF = {
    "reference": "#000000",
    "separate": "#0072B2",
    "diagonal": "#D55E00",
    "perturbed": "#009E73",
}
CBF_BPM = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#F0E442", "#999999"]


def configure_plot_style() -> None:
    """Apply one consistent readable style to all study figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.titlepad": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "savefig.bbox": "tight",
        }
    )


def format_axis(ax: plt.Axes, *, xbins: int = 7, ybins: int = 6) -> None:
    """Make plot axes easier to read with consistent ticks and formatting."""
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.grid(True, alpha=0.22, linewidth=0.8)


def format_log_y_axis(ax: plt.Axes, *, xbins: int = 7) -> None:
    """Format an axis with linear x and logarithmic y tick labels."""
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.grid(True, alpha=0.22, linewidth=0.8, which="both")


def coordinate_unit(column: str) -> tuple[float, str]:
    """Return a scale factor and display unit for the requested coordinate."""
    if column in {"x", "y"}:
        return 1e3, "mm"
    if column in {"px", "py"}:
        return 1e3, "mrad"
    return 1.0, ""


@dataclass(frozen=True)
class GeneratedTrackingCase:
    """Files and machine-state inputs for one generated study scenario."""

    name: str
    measurement_files: list[Path]
    corrector_file: Path | None
    tune_knobs_file: Path | None
    magnet_strengths: dict[str, float]


@dataclass(frozen=True)
class ScenarioResult:
    """Controller outputs and derived diagnostics for one scenario."""

    name: str
    estimate: dict[str, float]
    uncertainties: dict[str, float]
    initial_loss: pd.DataFrame
    final_loss: pd.DataFrame
    worker_count: int
    batch_count: int
    best_loss: float
    beta_rmse_x_pct: float
    beta_rmse_y_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "data" / "sequences" / "sps.seq",
        help="SPS sequence file used for the study.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "sps_diagonal_vs_separate_kicks_study",
        help="Directory where plots, tables, and summaries are written.",
    )
    parser.add_argument(
        "--flattop-turns",
        type=int,
        default=256,
        help="Number of turns tracked per generated file.",
    )
    parser.add_argument(
        "--num-start-bpms",
        type=int,
        default=4,
        help="Number of horizontal and vertical start BPMs used in the study.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Optimiser epochs used for each controller run.",
    )
    return parser.parse_args()


def build_sps_interface(sequence_file: Path) -> AbaMadInterface:
    return AbaMadInterface(accelerator=SPS(sequence_file=sequence_file, pc=450.0))


def get_bpm_positions(sequence_file: Path) -> pd.DataFrame:
    iface = build_sps_interface(sequence_file)
    iface.observe_bpms(iface.accelerator.bpm_pattern)
    twiss = iface.run_twiss().reset_index()
    return twiss[["name", "s"]]


def select_start_bpms(sequence_file: Path, num_start_bpms: int) -> list[str]:
    bpm_table = get_bpm_positions(sequence_file)
    h_bpms = bpm_table.loc[bpm_table["name"].str.startswith("BPH"), "name"].tolist()
    v_bpms = bpm_table.loc[bpm_table["name"].str.startswith("BPV"), "name"].tolist()
    h_spacing = max(1, len(h_bpms) // num_start_bpms)
    v_spacing = max(1, len(v_bpms) // num_start_bpms)
    return h_bpms[::h_spacing][:num_start_bpms] + v_bpms[::v_spacing][:num_start_bpms]


def split_start_bpms_by_plane(bpm_start_points: list[str]) -> dict[str, list[str]]:
    return {
        "H": [bpm for bpm in bpm_start_points if bpm.startswith("BPH")],
        "V": [bpm for bpm in bpm_start_points if bpm.startswith("BPV")],
    }


def build_sps_tracking_environment(
    sequence_file: Path,
) -> tuple[xt.Environment, dict[str, float], dict[str, float]]:
    """Build one perturbed and tune-matched SPS environment for both study cases."""
    interface = build_sps_interface(sequence_file)
    magnet_strengths, _ = interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="q",
    )
    matched_tunes = interface.match_tunes(target_qx=TARGET_QX, target_qy=TARGET_QY, deltap=0.0)
    perturbed_strengths = interface.get_magnet_strengths(list(magnet_strengths.keys()))

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=perturbed_strengths,
        corrector_table=tfs.TfsDataFrame(
            columns=["kind", "hkick", "hkick_old", "vkick", "vkick_old"]
        ),
        sequence_file=sequence_file,
        seq_name=interface.accelerator.seq_name,
        pc=interface.accelerator.pc,
        strict_set=False,
    )
    return env, perturbed_strengths, matched_tunes


def generate_sps_tracking_cases(
    *,
    sequence_file: Path,
    output_dir: Path,
    flattop_turns: int,
) -> tuple[GeneratedTrackingCase, GeneratedTrackingCase]:
    """Generate the separate-kick and diagonal-kick tracking inputs for the study."""
    env, magnet_strengths, matched_tunes = build_sps_tracking_environment(sequence_file)
    tune_knobs_file = output_dir / "tune_knobs_sps.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    separate_destination = output_dir / "separate_hv_kicks" / "track_off_magnet_sps.parquet"
    diagonal_destination = output_dir / "diagonal_kicks" / "track_off_magnet_sps.parquet"

    separate_files = sorted(
        write_tracking_measurements(
            env=env,
            destination=separate_destination,
            flattop_turns=flattop_turns,
            use_diagonal_kicks=False,
        ),
        key=lambda path: infer_file_plane(path),
    )
    diagonal_files = write_tracking_measurements(
        env=env,
        destination=diagonal_destination,
        flattop_turns=flattop_turns,
        use_diagonal_kicks=True,
    )
    if len(diagonal_files) != 1:
        raise RuntimeError("Expected a single diagonal tracking file")

    separate_case = GeneratedTrackingCase(
        name="separate_hv_kicks",
        measurement_files=separate_files,
        corrector_file=None,
        tune_knobs_file=tune_knobs_file,
        magnet_strengths=magnet_strengths,
    )
    diagonal_case = GeneratedTrackingCase(
        name="diagonal_kicks",
        measurement_files=diagonal_files,
        corrector_file=None,
        tune_knobs_file=tune_knobs_file,
        magnet_strengths=magnet_strengths,
    )
    return separate_case, diagonal_case


def write_tracking_measurements(
    *,
    env: xt.Environment,
    destination: Path,
    flattop_turns: int,
    use_diagonal_kicks: bool,
) -> list[Path]:
    """Run noiseless SPS tracking and write each particle to parquet."""
    line = env["sps"]
    monitored_line = run_tracking_without_ac_dipole(
        line=line,
        tws=line.twiss4d(),
        flattop_turns=flattop_turns,
        bpm_pattern="bp[hv].*",
        action_list=[4e-7],
        angle_list=[0.0],
        use_diagonal_kicks=use_diagonal_kicks,
        deltas=0.0,
        start_marker=None,
    )
    tracked_frames = list(line_to_dataframes(monitored_line))
    output_files: list[Path] = []

    for idx, true_df in enumerate(tracked_frames):
        df = prepare_track_dataframe(true_df, 0, flattop_turns)
        df = df.loc[:, TRACK_COLUMNS].copy()
        df["name"] = df["name"].astype(str)
        if len(tracked_frames) > 1:
            output_path = (
                destination.parent / f"{destination.stem}_particle_{idx}{destination.suffix}"
            )
        else:
            output_path = destination
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        output_files.append(output_path)

    return output_files


def infer_file_plane(path: Path) -> str:
    return DataManager.infer_kick_plane(pd.read_parquet(path))


def load_separate_measurements_by_plane(separate_files: list[Path]) -> dict[str, pd.DataFrame]:
    separate_by_plane = {infer_file_plane(path): pd.read_parquet(path) for path in separate_files}
    if {"x", "y"} - set(separate_by_plane):
        raise ValueError("Expected one x-only and one y-only file for the separate-kick case")
    return separate_by_plane


def rms(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float, copy=False)
    return float(np.sqrt(np.mean(array**2)))


def span(values: pd.Series) -> float:
    """Return max-min for one BPM signal across turns."""
    array = values.to_numpy(dtype=float, copy=False)
    return float(array.max() - array.min())


def build_bpm_signal_table(
    diagonal_file: Path,
    separate_files: list[Path],
    bpm_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-BPM measurable and orthogonal signal summaries."""
    diagonal_df = pd.read_parquet(diagonal_file)
    separate_by_plane = load_separate_measurements_by_plane(separate_files)
    rows: list[dict[str, float | str]] = []

    for row in bpm_positions.itertuples(index=False):
        bpm = str(row.name)
        diagonal_bpm = diagonal_df.loc[diagonal_df["name"] == bpm]
        if diagonal_bpm.empty:
            continue

        if bpm.startswith("BPH"):
            bpm_type = "H"
            coord, mom, orth_coord, orth_mom = "x", "px", "y", "py"
            separate_bpm = separate_by_plane["x"].loc[separate_by_plane["x"]["name"] == bpm]
        else:
            bpm_type = "V"
            coord, mom, orth_coord, orth_mom = "y", "py", "x", "px"
            separate_bpm = separate_by_plane["y"].loc[separate_by_plane["y"]["name"] == bpm]

        merged = diagonal_bpm[["turn", "name", coord]].merge(
            separate_bpm[["turn", "name", coord]],
            on=["turn", "name"],
            how="inner",
            suffixes=("_diag", "_sep"),
        )
        coord_diff = merged[f"{coord}_diag"] - merged[f"{coord}_sep"]

        rows.append(
            {
                "name": bpm,
                "s": float(row.s),
                "bpm_type": bpm_type,
                "measured_coord": coord,
                "orthogonal_coord": orth_coord,
                "position_rms_separate": rms(separate_bpm[coord]),
                "position_rms_diagonal": rms(diagonal_bpm[coord]),
                "position_rms_diagonal_orthogonal": rms(diagonal_bpm[orth_coord]),
                "position_span_separate": span(separate_bpm[coord]),
                "position_span_diagonal": span(diagonal_bpm[coord]),
                "position_span_diagonal_orthogonal": span(diagonal_bpm[orth_coord]),
                "momentum_rms_separate": rms(separate_bpm[mom]),
                "momentum_rms_diagonal": rms(diagonal_bpm[mom]),
                "momentum_rms_diagonal_orthogonal": rms(diagonal_bpm[orth_mom]),
                "mean_abs_position_diff_vs_separate": float(coord_diff.abs().mean()),
                "rms_position_diff_vs_separate": float(
                    np.sqrt(np.mean(coord_diff.to_numpy(dtype=float) ** 2))
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("s").reset_index(drop=True)


def start_bpm_positions(
    bpm_start_points: list[str],
    bpm_positions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    position_map = bpm_positions.set_index("name")
    grouped = split_start_bpms_by_plane(bpm_start_points)
    return {
        bpm_type: pd.DataFrame(
            {
                "name": names,
                "s": [float(position_map.loc[name, "s"]) for name in names],
            }
        )
        for bpm_type, names in grouped.items()
    }


def add_start_bpm_lines(ax: plt.Axes, start_table: pd.DataFrame) -> None:
    """Draw thin vertical markers for the selected start BPMs."""
    if start_table.empty:
        return
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.04 * (y_max - y_min if y_max != y_min else 1.0)
    for row in start_table.itertuples(index=False):
        ax.axvline(row.s, color="0.4", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(
            row.s,
            y_text,
            row.name,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="0.35",
        )


def plot_position_along_ring(
    bpm_signal_table: pd.DataFrame,
    start_positions_by_plane: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot measurable position span along the ring for the two study cases."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    specs = (
        ("H", "Horizontal BPMs", "x max - min [mm]"),
        ("V", "Vertical BPMs", "y max - min [mm]"),
    )
    for ax, (bpm_type, title, ylabel) in zip(axes, specs, strict=True):
        table = bpm_signal_table.loc[bpm_signal_table["bpm_type"] == bpm_type]
        ax.plot(
            table["s"],
            table["position_span_separate"] * 1e3,
            label="Separate kicks",
            linewidth=2.0,
            color=CBF["separate"],
        )
        ax.plot(
            table["s"],
            table["position_span_diagonal"] * 1e3,
            label="Diagonal kicks",
            linewidth=2.0,
            color=CBF["diagonal"],
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        add_start_bpm_lines(ax, start_positions_by_plane[bpm_type])
        format_axis(ax)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("s [m]")
    fig.suptitle("Measured position variation along the ring")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_initial_conditions_by_plane(
    *,
    diagonal_file: Path,
    separate_files: list[Path],
    bpm_start_points: list[str],
    output_dir: Path,
) -> None:
    """Plot compact initial-condition summaries for horizontal and vertical starts."""
    diagonal_df = pd.read_parquet(diagonal_file)
    separate_by_plane = load_separate_measurements_by_plane(separate_files)
    grouped = split_start_bpms_by_plane(bpm_start_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "H": {
            "bpms": grouped["H"],
            "coord": "x",
            "mom": "px",
            "orth_coord": "y",
            "orth_mom": "py",
            "separate_df": separate_by_plane["x"],
            "title": "Horizontal start BPM initial conditions",
            "filename": "initial_conditions_horizontal.png",
        },
        "V": {
            "bpms": grouped["V"],
            "coord": "y",
            "mom": "py",
            "orth_coord": "x",
            "orth_mom": "px",
            "separate_df": separate_by_plane["y"],
            "title": "Vertical start BPM initial conditions",
            "filename": "initial_conditions_vertical.png",
        },
    }

    for spec in specs.values():
        bpms: list[str] = spec["bpms"]
        if not bpms:
            continue

        coord = str(spec["coord"])
        mom = str(spec["mom"])
        orth_coord = str(spec["orth_coord"])
        orth_mom = str(spec["orth_mom"])
        separate_df = spec["separate_df"]
        coord_scale, coord_unit = coordinate_unit(coord)
        mom_scale, mom_unit = coordinate_unit(mom)
        orth_coord_scale, orth_coord_unit = coordinate_unit(orth_coord)
        orth_mom_scale, orth_mom_unit = coordinate_unit(orth_mom)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        colors = [CBF_BPM[idx % len(CBF_BPM)] for idx in range(len(bpms))]
        bpm_handles: list[Line2D] = []

        for bpm, color in zip(bpms, colors, strict=True):
            separate_bpm = separate_df.loc[separate_df["name"] == bpm]
            diagonal_bpm = diagonal_df.loc[diagonal_df["name"] == bpm]
            if separate_bpm.empty or diagonal_bpm.empty:
                continue

            axes[0].scatter(
                separate_bpm[coord] * coord_scale,
                separate_bpm[mom] * mom_scale,
                color=color,
                s=22,
                alpha=0.65,
                marker="o",
            )
            axes[0].scatter(
                diagonal_bpm[coord] * coord_scale,
                diagonal_bpm[mom] * mom_scale,
                color=color,
                s=26,
                alpha=0.75,
                marker="x",
            )
            axes[1].scatter(
                diagonal_bpm[orth_coord] * orth_coord_scale,
                diagonal_bpm[orth_mom] * orth_mom_scale,
                color=color,
                s=20,
                alpha=0.65,
                marker="^",
            )
            bpm_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=bpm))

        axes[0].set_title("Measured plane", fontsize=12)
        axes[0].set_xlabel(f"{coord} [{coord_unit}]")
        axes[0].set_ylabel(f"{mom} [{mom_unit}]")
        axes[1].set_title("Diagonal orthogonal plane", fontsize=12)
        axes[1].set_xlabel(f"{orth_coord} [{orth_coord_unit}]")
        axes[1].set_ylabel(f"{orth_mom} [{orth_mom_unit}]")
        for ax in axes:
            format_axis(ax, xbins=6, ybins=5)

        style_handles = [
            Line2D(
                [0],
                [0],
                color=CBF["reference"],
                marker="o",
                linestyle="None",
                label="Separate measurable",
            ),
            Line2D(
                [0],
                [0],
                color=CBF["reference"],
                marker="x",
                linestyle="None",
                label="Diagonal measurable",
            ),
            Line2D(
                [0],
                [0],
                color=CBF["reference"],
                marker="^",
                linestyle="None",
                label="Diagonal orthogonal",
            ),
        ]
        axes[0].legend(
            handles=style_handles,
            loc="upper left",
            frameon=True,
            borderpad=0.3,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        if bpm_handles:
            fig.legend(
                handles=bpm_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.005),
                ncol=min(4, len(bpm_handles)),
                frameon=False,
                columnspacing=1.2,
                handletextpad=0.5,
            )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), pad=0.7, w_pad=1.0)
        fig.savefig(output_dir / str(spec["filename"]), dpi=180)
        plt.close(fig)


def make_simulation_config() -> SimulationConfig:
    """Return the SPS multi-turn controller settings used for the study."""
    return SimulationConfig(
        tracks_per_worker=2,
        num_workers=8,
        num_batches=1,
        optimise_momenta=True,
        run_arc_by_arc=False,
        n_run_turns=1,
        bpm_loss_outlier_sigma=10.0,
        worker_loss_outlier_sigma=10.0,
        enable_preloop_outlier_screening=False,
    )


def make_optimiser_config(max_epochs: int, accelerator: SPS) -> OptimiserConfig:
    """Return the SPS quadrupole optimiser settings used for the study."""
    return OptimiserConfig(
        max_epochs=max_epochs,
        warmup_epochs=30,
        warmup_lr_start=5e-9,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=5e-16,
        expected_rel_error=accelerator.get_perturbation_families()["q"]["default_rel_std"],
        optimiser_type="lbfgs",
    )


def build_controller(
    *,
    sequence_file: Path,
    output_dir: Path,
    case: GeneratedTrackingCase,
    bpm_start_points: list[str],
    flattop_turns: int,
    max_epochs: int,
    initial_knob_strengths: dict[str, float] | None = None,
    run_label: str | None = None,
) -> Controller:
    """Build one controller configured exactly for the SPS comparison study."""
    accelerator = SPS(
        pc=450.0,
        sequence_file=sequence_file,
        optimise_quadrupoles=True,
    )
    label = case.name if run_label is None else run_label
    controller_dir = output_dir / label
    controller_dir.mkdir(parents=True, exist_ok=True)
    return Controller(
        accelerator,
        make_optimiser_config(max_epochs, accelerator),
        make_simulation_config(),
        SequenceConfig("$start/$end"),
        MeasurementConfig(
            measurement_files=case.measurement_files,
            corrector_files=case.corrector_file,
            tune_knobs_files=case.tune_knobs_file,
            flattop_turns=flattop_turns,
            bunches_per_file=1,
        ),
        bpm_start_points=bpm_start_points,
        bpm_end_points=[],
        show_plots=False,
        initial_knob_strengths=initial_knob_strengths,
        true_strengths=case.magnet_strengths,
        debug=False,
        mad_logfile=controller_dir / "controller.log",
        plots_dir=controller_dir / "controller_plots",
        output_config=OutputConfig(write_tensorboard_logs=False),
    )


def aggregate_loss_diagnostics(
    ctrl: Controller,
    diagnostics: list[dict[str, object]],
    bpm_positions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate worker diagnostic losses to one mean profile per BPM."""
    contributions: dict[str, list[float]] = {}
    payload_builder = ctrl.worker_manager.payload_builder
    for meta, diag in zip(ctrl.worker_manager.worker_metadata, diagnostics, strict=True):
        raw_loss = np.asarray(diag["loss_per_bpm"], dtype=np.float64)
        worker_loss = payload_builder.diagnostic_loss_per_bpm(
            loss_per_point=raw_loss,
            bpm_names=meta.bpm_names,
            n_run_turns=meta.n_run_turns,
            worker_id=meta.worker_id,
        )
        for bpm, loss_value in zip(meta.bpm_names, worker_loss, strict=True):
            contributions.setdefault(bpm, []).append(float(loss_value))

    rows: list[dict[str, float | int | str]] = []
    for row in bpm_positions.itertuples(index=False):
        bpm = str(row.name)
        if bpm not in contributions:
            continue
        values = np.asarray(contributions[bpm], dtype=np.float64)
        rows.append(
            {
                "name": bpm,
                "s": float(row.s),
                "bpm_type": "H" if bpm.startswith("BPH") else "V",
                "n_worker_contributions": int(values.size),
                "mean_loss": float(values.mean()),
                "sum_loss": float(values.sum()),
                "max_loss": float(values.max()),
            }
        )

    return pd.DataFrame(rows).sort_values("s").reset_index(drop=True)


def run_controller_with_diagnostics(
    ctrl: Controller,
    bpm_positions: pd.DataFrame,
) -> tuple[
    dict[str, float],
    dict[str, float],
    pd.DataFrame,
    int,
    int,
    float,
    list[list[int]],
    dict[int, int],
    int,
    int,
]:
    """Run one controller while capturing the initial loss profile."""
    total_turns = ctrl.data_manager.get_total_turns()

    try:
        ctrl.worker_manager.start_workers(
            ctrl.data_manager.track_data,
            ctrl.data_manager.turn_batches,
            ctrl.data_manager.file_map,
            ctrl.config_manager.start_bpms,
            ctrl.config_manager.end_bpms,
            ctrl.simulation_config,
            ctrl.machine_deltaps,
            ctrl.initial_knobs,
        )
        worker_count = len(ctrl.worker_manager.worker_metadata)
        batch_count = len(ctrl.data_manager.turn_batches)
        turn_batches = [list(batch) for batch in ctrl.data_manager.turn_batches]
        file_map = dict(ctrl.data_manager.file_map)
        sampling_num_workers = ctrl.data_manager.num_workers
        sampling_tracks_per_worker = ctrl.data_manager.tracks_per_worker

        initial_diagnostics = ctrl.worker_manager._request_worker_diagnostics(ctrl.initial_knobs)
        initial_loss = aggregate_loss_diagnostics(ctrl, initial_diagnostics, bpm_positions)

        ctrl._cleanup_memory()
        channels = ctrl.worker_manager.channels
        if channels is None:
            raise RuntimeError("Worker channels are not initialised")

        final_knobs = ctrl.optimisation_loop.run_optimisation(
            ctrl.initial_knobs,
            channels,
            writer=None,
            run_start=time.time(),
            total_turns=total_turns,
        )
        total_hessian = ctrl.worker_manager.termination_and_hessian(len(final_knobs))
        covariance = np.linalg.inv(total_hessian + 1e-8 * np.eye(total_hessian.shape[0]))
        uncertainties = dict(zip(final_knobs.keys(), np.sqrt(np.diag(covariance)), strict=True))
        return (
            final_knobs,
            uncertainties,
            initial_loss,
            worker_count,
            batch_count,
            float(ctrl.optimisation_loop.best_loss),
            turn_batches,
            file_map,
            sampling_num_workers,
            sampling_tracks_per_worker,
        )
    except Exception:
        ctrl.worker_manager.terminate_workers()
        raise


def probe_loss_profile(ctrl: Controller, bpm_positions: pd.DataFrame) -> pd.DataFrame:
    """Measure the worker loss profile at the controller's initial knob state."""
    workers_started = False
    try:
        ctrl.worker_manager.start_workers(
            ctrl.data_manager.track_data,
            ctrl.data_manager.turn_batches,
            ctrl.data_manager.file_map,
            ctrl.config_manager.start_bpms,
            ctrl.config_manager.end_bpms,
            ctrl.simulation_config,
            ctrl.machine_deltaps,
            ctrl.initial_knobs,
        )
        workers_started = True
        diagnostics = ctrl.worker_manager._request_worker_diagnostics(ctrl.initial_knobs)
        return aggregate_loss_diagnostics(ctrl, diagnostics, bpm_positions)
    finally:
        if workers_started:
            ctrl.worker_manager.termination_and_hessian(len(ctrl.initial_knobs))
        else:
            ctrl.worker_manager.terminate_workers()


def copy_sampling_state(source: Controller, target: Controller) -> None:
    """Force a probe controller to reuse the original controller's sampled turns."""
    target.data_manager.turn_batches = [list(batch) for batch in source.data_manager.turn_batches]
    target.data_manager.file_map = dict(source.data_manager.file_map)
    target.data_manager.num_workers = source.data_manager.num_workers
    target.data_manager.tracks_per_worker = source.data_manager.tracks_per_worker


def apply_sampling_state(
    target: Controller,
    *,
    turn_batches: list[list[int]],
    file_map: dict[int, int],
    num_workers: int,
    tracks_per_worker: int,
) -> None:
    """Force a probe controller to reuse a previously captured sampling state."""
    target.data_manager.turn_batches = [list(batch) for batch in turn_batches]
    target.data_manager.file_map = dict(file_map)
    target.data_manager.num_workers = num_workers
    target.data_manager.tracks_per_worker = tracks_per_worker


def compute_observed_twiss(
    *,
    sequence_file: Path,
    magnet_strengths: dict[str, float] | None = None,
    rematch_tunes: bool = False,
) -> pd.DataFrame:
    """Compute BPM-observed Twiss data for the requested magnet state."""
    iface = build_sps_interface(sequence_file)
    if magnet_strengths is not None:
        iface.set_magnet_strengths(magnet_strengths)
        if rematch_tunes:
            iface.match_tunes(target_qx=TARGET_QX, target_qy=TARGET_QY, deltap=0.0)
    iface.observe_bpms(iface.accelerator.bpm_pattern)
    return iface.run_twiss().reset_index()


def beta_rmse_pct(
    estimate_twiss: pd.DataFrame,
    true_twiss: pd.DataFrame,
    clean_twiss: pd.DataFrame,
    beta_column: str,
) -> float:
    """Return the RMSE of beta beating, in percent, relative to the true perturbed optics."""
    true_beating = (true_twiss[beta_column] - clean_twiss[beta_column]) / clean_twiss[beta_column]
    estimate_beating = (
        (estimate_twiss[beta_column] - clean_twiss[beta_column]) / clean_twiss[beta_column]
    )
    return float(np.sqrt(np.mean(((estimate_beating - true_beating) * 100.0) ** 2)))


def build_scenario_result(
    *,
    ctrl: Controller,
    sequence_file: Path,
    bpm_positions: pd.DataFrame,
    case: GeneratedTrackingCase,
    output_dir: Path,
    bpm_start_points: list[str],
    flattop_turns: int,
) -> tuple[ScenarioResult, pd.DataFrame]:
    """Run one controller case and build its summary container."""
    (
        estimate,
        uncertainties,
        initial_loss,
        worker_count,
        batch_count,
        best_loss,
        turn_batches,
        file_map,
        sampling_num_workers,
        sampling_tracks_per_worker,
    ) = run_controller_with_diagnostics(ctrl, bpm_positions)
    probe_ctrl = build_controller(
        sequence_file=sequence_file,
        output_dir=output_dir,
        case=case,
        bpm_start_points=bpm_start_points,
        flattop_turns=flattop_turns,
        max_epochs=1,
        initial_knob_strengths=estimate,
        run_label=f"{case.name}_final_probe",
    )
    apply_sampling_state(
        probe_ctrl,
        turn_batches=turn_batches,
        file_map=file_map,
        num_workers=sampling_num_workers,
        tracks_per_worker=sampling_tracks_per_worker,
    )
    final_loss = probe_loss_profile(probe_ctrl, bpm_positions)
    estimate_twiss = compute_observed_twiss(
        sequence_file=sequence_file,
        magnet_strengths=estimate,
        rematch_tunes=True,
    )
    clean_twiss = compute_observed_twiss(sequence_file=sequence_file)
    true_twiss = compute_observed_twiss(
        sequence_file=sequence_file,
        magnet_strengths=case.magnet_strengths,
        rematch_tunes=True,
    )
    result = ScenarioResult(
        name=case.name,
        estimate=estimate,
        uncertainties=uncertainties,
        initial_loss=initial_loss,
        final_loss=final_loss,
        worker_count=worker_count,
        batch_count=batch_count,
        best_loss=best_loss,
        beta_rmse_x_pct=beta_rmse_pct(estimate_twiss, true_twiss, clean_twiss, "beta11"),
        beta_rmse_y_pct=beta_rmse_pct(estimate_twiss, true_twiss, clean_twiss, "beta22"),
    )
    return result, estimate_twiss


def build_magnet_reconstruction_table(
    reference_ctrl: Controller,
    true_strengths: dict[str, float],
    separate_result: ScenarioResult,
    diagonal_result: ScenarioResult,
) -> pd.DataFrame:
    """Build a per-magnet comparison table for true and reconstructed errors."""
    knob_names = reference_ctrl.config_manager.knob_names
    nominal_strengths = reference_ctrl.config_manager.initial_strengths
    elem_positions = reference_ctrl.config_manager.elem_spos
    rows: list[dict[str, float | str]] = []

    for idx, knob in enumerate(knob_names):
        nominal = float(nominal_strengths[idx])
        true_value = float(true_strengths[knob])
        separate_value = float(separate_result.estimate[knob])
        diagonal_value = float(diagonal_result.estimate[knob])
        if nominal == 0.0:
            true_rel_error_pct = float("nan")
            separate_rel_error_pct = float("nan")
            diagonal_rel_error_pct = float("nan")
        else:
            denom = abs(nominal)
            true_rel_error_pct = 100.0 * (true_value - nominal) / denom
            separate_rel_error_pct = 100.0 * (separate_value - nominal) / denom
            diagonal_rel_error_pct = 100.0 * (diagonal_value - nominal) / denom

        if true_value == 0.0:
            separate_abs_rel_to_true_pct = abs(separate_value - true_value)
            diagonal_abs_rel_to_true_pct = abs(diagonal_value - true_value)
        else:
            denom = abs(true_value)
            separate_abs_rel_to_true_pct = 100.0 * abs(separate_value - true_value) / denom
            diagonal_abs_rel_to_true_pct = 100.0 * abs(diagonal_value - true_value) / denom

        rows.append(
            {
                "name": knob,
                "s": float(elem_positions[idx]),
                "nominal_strength": nominal,
                "true_strength": true_value,
                "separate_strength": separate_value,
                "diagonal_strength": diagonal_value,
                "true_rel_error_pct": true_rel_error_pct,
                "separate_rel_error_pct": separate_rel_error_pct,
                "diagonal_rel_error_pct": diagonal_rel_error_pct,
                "separate_abs_rel_to_true_pct": separate_abs_rel_to_true_pct,
                "diagonal_abs_rel_to_true_pct": diagonal_abs_rel_to_true_pct,
            }
        )

    return pd.DataFrame(rows).sort_values("s").reset_index(drop=True)


def plot_beta_beating(
    *,
    clean_twiss: pd.DataFrame,
    true_twiss: pd.DataFrame,
    separate_twiss: pd.DataFrame,
    diagonal_twiss: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot horizontal and vertical beta beating for truth and both reconstructions."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    specs = (
        ("beta11", "Horizontal beta beating", r"$\Delta \beta_x / \beta_x$ [%]"),
        ("beta22", "Vertical beta beating", r"$\Delta \beta_y / \beta_y$ [%]"),
    )
    cases = (
        ("True perturbation", true_twiss, CBF["reference"], "-"),
        ("Estimate from separate kicks", separate_twiss, CBF["separate"], "--"),
        ("Estimate from diagonal kicks", diagonal_twiss, CBF["diagonal"], "--"),
    )
    for ax, (column, title, ylabel) in zip(axes, specs, strict=True):
        base = clean_twiss[column]
        for label, twiss, color, linestyle in cases:
            beating = (twiss[column] - base) / base * 100.0
            ax.plot(
                twiss["s"],
                beating,
                label=label,
                linewidth=1.8,
                color=color,
                linestyle=linestyle,
            )
        ax.axhline(0.0, color="0.3", linewidth=0.8, alpha=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_axis(ax)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("s [m]")
    fig.suptitle("Beta beating reconstructed from separate and diagonal kick data")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_actual_beta_functions(
    *,
    clean_twiss: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the reference horizontal and vertical beta functions along the ring."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    specs = (
        ("beta11", "Horizontal beta function", r"$\beta_x$ [m]"),
        ("beta22", "Vertical beta function", r"$\beta_y$ [m]"),
    )
    for ax, (column, title, ylabel) in zip(axes, specs, strict=True):
        ax.plot(clean_twiss["s"], clean_twiss[column], color=CBF["reference"], label="Reference model")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_axis(ax)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("s [m]")
    fig.suptitle("Reference beta functions along the ring")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_magnet_errors(magnet_table: pd.DataFrame, output_path: Path) -> None:
    """Plot true and reconstructed quadrupole errors plus reconstruction residuals."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    scale = 1e-2  # convert percent values to 1e-4 relative units
    floor = 1e-6

    axes[0].plot(
        magnet_table["s"],
        np.clip(np.abs(magnet_table["true_rel_error_pct"] / scale), floor, None),
        label="True quadrupole error",
        color=CBF["reference"],
        linewidth=1.8,
    )
    axes[0].plot(
        magnet_table["s"],
        np.clip(np.abs(magnet_table["separate_rel_error_pct"] / scale), floor, None),
        label="Estimate from separate kicks",
        color=CBF["separate"],
        linewidth=1.8,
    )
    axes[0].plot(
        magnet_table["s"],
        np.clip(np.abs(magnet_table["diagonal_rel_error_pct"] / scale), floor, None),
        label="Estimate from diagonal kicks",
        color=CBF["diagonal"],
        linewidth=1.8,
    )
    axes[0].set_ylabel("Relative error vs nominal [units]")
    axes[0].set_title("Absolute true and reconstructed quadrupole errors")
    format_axis(axes[0])
    axes[0].legend(loc="upper right")

    axes[1].plot(
        magnet_table["s"],
        np.clip(magnet_table["separate_abs_rel_to_true_pct"] / scale, floor, None),
        label="Separate kicks",
        color=CBF["separate"],
        linewidth=1.8,
    )
    axes[1].plot(
        magnet_table["s"],
        np.clip(magnet_table["diagonal_abs_rel_to_true_pct"] / scale, floor, None),
        label="Diagonal kicks",
        color=CBF["diagonal"],
        linewidth=1.8,
    )
    axes[1].set_ylabel("|Estimate - true| / |true| [units]")
    axes[1].set_xlabel("s [m]")
    axes[1].set_title("Per-magnet reconstruction residual")
    axes[1].set_yscale("log")
    format_log_y_axis(axes[1])
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_along_ring(
    *,
    separate_result: ScenarioResult,
    diagonal_result: ScenarioResult,
    output_path: Path,
) -> None:
    """Plot initial and final mean worker loss along the ring for both cases."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    series = (
        ("Separate initial", separate_result.initial_loss, CBF["separate"], ":"),
        ("Separate final", separate_result.final_loss, CBF["separate"], "-"),
        ("Diagonal initial", diagonal_result.initial_loss, CBF["diagonal"], ":"),
        ("Diagonal final", diagonal_result.final_loss, CBF["diagonal"], "-"),
    )
    specs = (
        ("H", "Horizontal BPMs"),
        ("V", "Vertical BPMs"),
    )
    for ax, (bpm_type, title) in zip(axes, specs, strict=True):
        for label, table, color, linestyle in series:
            subset = table.loc[table["bpm_type"] == bpm_type]
            y_values = subset["mean_loss"].clip(lower=1e-30)
            ax.plot(
                subset["s"],
                y_values,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
        ax.set_title(title)
        ax.set_ylabel("Mean worker loss per BPM")
        ax.set_yscale("log")
        format_log_y_axis(ax)
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("s [m]")
    fig.suptitle("Initial and final loss profiles along the ring")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_scenario_summary_table(
    separate_result: ScenarioResult,
    diagonal_result: ScenarioResult,
    magnet_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact scenario summary for the report and slide notes."""
    metrics = {
        "separate_hv_kicks": {
            "mean_abs_rel_magnet_error_pct": float(
                magnet_table["separate_abs_rel_to_true_pct"].mean()
            ),
            "max_abs_rel_magnet_error_pct": float(
                magnet_table["separate_abs_rel_to_true_pct"].max()
            ),
        },
        "diagonal_kicks": {
            "mean_abs_rel_magnet_error_pct": float(
                magnet_table["diagonal_abs_rel_to_true_pct"].mean()
            ),
            "max_abs_rel_magnet_error_pct": float(
                magnet_table["diagonal_abs_rel_to_true_pct"].max()
            ),
        },
    }
    rows = []
    for result in (separate_result, diagonal_result):
        rows.append(
            {
                "scenario": result.name,
                "worker_count": result.worker_count,
                "batch_count": result.batch_count,
                "best_loss": result.best_loss,
                "initial_mean_loss": float(result.initial_loss["mean_loss"].mean()),
                "final_mean_loss": float(result.final_loss["mean_loss"].mean()),
                "beta_rmse_x_pct": result.beta_rmse_x_pct,
                "beta_rmse_y_pct": result.beta_rmse_y_pct,
                **metrics[result.name],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    *,
    output_dir: Path,
    scenario_summary: pd.DataFrame,
    magnet_table: pd.DataFrame,
    bpm_signal_table: pd.DataFrame,
    bpm_start_points: list[str],
) -> None:
    """Write a concise presentation-oriented Markdown report."""
    top_bpm_diff = bpm_signal_table.nlargest(8, "rms_position_diff_vs_separate")[
        ["name", "rms_position_diff_vs_separate"]
    ]
    top_diag_errors = magnet_table.nlargest(8, "diagonal_abs_rel_to_true_pct")[
        ["name", "diagonal_abs_rel_to_true_pct"]
    ]
    report = f"""# SPS separate vs diagonal kick study

## What is compared

- Same SPS quadrupole perturbation seed for both datasets
- `separate_hv_kicks`: one horizontal file plus one vertical file
- `diagonal_kicks`: one file with a simultaneous horizontal and vertical kick
- Same selected start BPMs for both controller runs: `{", ".join(bpm_start_points)}`

## Scenario summary

{scenario_summary.to_markdown(index=False)}

## BPMs with the largest measurable position mismatch

{top_bpm_diff.to_markdown(index=False)}

## Magnets with the largest diagonal-case reconstruction residual

{top_diag_errors.to_markdown(index=False)}

## Plot guide

- `beta_beating_comparison.png`: true beta beating vs the two reconstructed optics
- `beta_functions_reference.png`: reference and perturbed beta functions along the ring
- `magnet_error_reconstruction.png`: true quadrupole errors and residual reconstruction error
- `initial_conditions_horizontal.png`: start-BPM phase spaces for horizontal BPMs
- `initial_conditions_vertical.png`: start-BPM phase spaces for vertical BPMs
- `position_along_ring.png`: measurable position span (`max - min`) along the ring, with start-BPM markers
- `loss_along_ring.png`: mean worker diagnostic loss per BPM before and after optimisation
"""
    (output_dir / "report.md").write_text(report)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    configure_plot_style()
    logging.getLogger("xdeps").setLevel(logging.WARNING)
    logging.getLogger("aba_optimiser.mad.optimising_mad_interface").setLevel(logging.WARNING)
    logging.getLogger("aba_optimiser.mad.aba_mad_interface").setLevel(logging.WARNING)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bpm_positions = get_bpm_positions(args.sequence_file)
    bpm_start_points = select_start_bpms(args.sequence_file, args.num_start_bpms)
    start_positions_by_plane = start_bpm_positions(bpm_start_points, bpm_positions)
    LOGGER.info("Using start BPMs: %s", bpm_start_points)

    separate_case, diagonal_case = generate_sps_tracking_cases(
        sequence_file=args.sequence_file,
        output_dir=output_dir,
        flattop_turns=args.flattop_turns,
    )

    bpm_signal_table = build_bpm_signal_table(
        diagonal_case.measurement_files[0],
        separate_case.measurement_files,
        bpm_positions,
    )
    bpm_signal_table.to_csv(output_dir / "bpm_signal_summary.csv", index=False)

    plot_position_along_ring(
        bpm_signal_table,
        start_positions_by_plane,
        output_dir / "position_along_ring.png",
    )
    plot_initial_conditions_by_plane(
        diagonal_file=diagonal_case.measurement_files[0],
        separate_files=separate_case.measurement_files,
        bpm_start_points=bpm_start_points,
        output_dir=output_dir,
    )

    separate_ctrl = build_controller(
        sequence_file=args.sequence_file,
        output_dir=output_dir,
        case=separate_case,
        bpm_start_points=bpm_start_points,
        flattop_turns=args.flattop_turns,
        max_epochs=args.epochs,
    )
    diagonal_ctrl = build_controller(
        sequence_file=args.sequence_file,
        output_dir=output_dir,
        case=diagonal_case,
        bpm_start_points=bpm_start_points,
        flattop_turns=args.flattop_turns,
        max_epochs=args.epochs,
    )

    separate_result, separate_twiss = build_scenario_result(
        ctrl=separate_ctrl,
        sequence_file=args.sequence_file,
        bpm_positions=bpm_positions,
        case=separate_case,
        output_dir=output_dir,
        bpm_start_points=bpm_start_points,
        flattop_turns=args.flattop_turns,
    )
    diagonal_result, diagonal_twiss = build_scenario_result(
        ctrl=diagonal_ctrl,
        sequence_file=args.sequence_file,
        bpm_positions=bpm_positions,
        case=diagonal_case,
        output_dir=output_dir,
        bpm_start_points=bpm_start_points,
        flattop_turns=args.flattop_turns,
    )

    clean_twiss = compute_observed_twiss(sequence_file=args.sequence_file)
    true_twiss = compute_observed_twiss(
        sequence_file=args.sequence_file,
        magnet_strengths=separate_case.magnet_strengths,
        rematch_tunes=True,
    )

    plot_beta_beating(
        clean_twiss=clean_twiss,
        true_twiss=true_twiss,
        separate_twiss=separate_twiss,
        diagonal_twiss=diagonal_twiss,
        output_path=output_dir / "beta_beating_comparison.png",
    )
    plot_actual_beta_functions(
        clean_twiss=clean_twiss,
        output_path=output_dir / "beta_functions_reference.png",
    )

    magnet_table = build_magnet_reconstruction_table(
        separate_ctrl,
        separate_case.magnet_strengths,
        separate_result,
        diagonal_result,
    )
    magnet_table.to_csv(output_dir / "magnet_reconstruction.csv", index=False)
    plot_magnet_errors(magnet_table, output_dir / "magnet_error_reconstruction.png")

    loss_profiles = pd.concat(
        [
            separate_result.initial_loss.assign(scenario=separate_result.name, state="initial"),
            separate_result.final_loss.assign(scenario=separate_result.name, state="final"),
            diagonal_result.initial_loss.assign(scenario=diagonal_result.name, state="initial"),
            diagonal_result.final_loss.assign(scenario=diagonal_result.name, state="final"),
        ],
        ignore_index=True,
    )
    loss_profiles.to_csv(output_dir / "loss_profiles.csv", index=False)
    plot_loss_along_ring(
        separate_result=separate_result,
        diagonal_result=diagonal_result,
        output_path=output_dir / "loss_along_ring.png",
    )

    scenario_summary = build_scenario_summary_table(
        separate_result,
        diagonal_result,
        magnet_table,
    )
    scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)

    summary = {
        "start_bpms": bpm_start_points,
        "scenario_summary": scenario_summary.to_dict(orient="records"),
        "files": {
            "separate_measurements": [str(path) for path in separate_case.measurement_files],
            "diagonal_measurement": str(diagonal_case.measurement_files[0]),
            "tune_knobs_file": str(separate_case.tune_knobs_file) if separate_case.tune_knobs_file else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(to_builtin(summary), indent=2))
    write_report(
        output_dir=output_dir,
        scenario_summary=scenario_summary,
        magnet_table=magnet_table,
        bpm_signal_table=bpm_signal_table,
        bpm_start_points=bpm_start_points,
    )
    LOGGER.info("Study written to %s", output_dir)


if __name__ == "__main__":
    main()
