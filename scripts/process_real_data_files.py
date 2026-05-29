from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymadng_utils.io import read_knobs, save_knobs

# import tfs
from tmom_recon import build_twiss_from_measurements, calculate_transverse_pz
from tmom_recon.svd import svd_clean_measurements
from turn_by_turn import read_tbt

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.measurements.create_datafile import build_madng_twiss_table, save_online_knobs
from aba_optimiser.measurements.utils import find_all_bad_bpms_from_analysis
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import (
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.controller_helpers import create_arc_measurement_config

logger = logging.getLogger(__name__)

DELTAP = -5.337073499186168e-05


def read_tbt_file(file: Path):
    logger.info(f"Reading {file}")
    tbt_data = read_tbt(file)
    return tbt_data.matrices[0].X, tbt_data.matrices[0].Y


def build_long_measurement_df(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    turn_offset: int,
    bpm_categories: list[str],
) -> pd.DataFrame:
    X = X.copy()
    Y = Y.copy()
    X.index.name = "name"
    Y.index.name = "name"

    turn_cols = X.columns.astype(np.int32) + np.int32(turn_offset)
    X.columns = turn_cols
    Y.columns = turn_cols

    df_file = X.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
    df_file["y"] = Y.reset_index().melt(id_vars="name", var_name="turn", value_name="y")["y"]
    df_file["name"] = pd.Categorical(df_file["name"], categories=bpm_categories)
    df_file["turn"] = df_file["turn"].astype(np.int32, copy=False)
    df_file["x"] = df_file["x"].astype(np.float32, copy=False) / 1000  # Convert from mm to m
    df_file["y"] = df_file["y"].astype(np.float32, copy=False) / 1000  # Convert from mm to m
    return df_file


def measurement_time_from_sdds_file(file: Path) -> datetime:
    stem_parts = file.stem.split("@")
    if len(stem_parts) < 3:
        raise ValueError(f"Could not parse measurement time from SDDS filename: {file}")

    date_part = stem_parts[-2]
    time_part = stem_parts[-1]
    timestamp = datetime.strptime(f"{date_part} {time_part[:8]}", "%Y_%m_%d %H_%M_%S")
    return timestamp.replace(tzinfo=ZoneInfo("UTC"))


@dataclasses.dataclass
class RangeConfig:
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


def optimise_ranges(
    range_config: RangeConfig,
    accelerator: LHC,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    corrector_knobs_file: Path,
    tune_knobs_file: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    flattop_turns: int,
    plots_dir: Path,
    run_arc_by_arc: bool = False,
    initial_knob_strengths: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], LHC]:
    num_ranges = len(range_config.magnet_ranges)
    knobs_results = []
    uncertainty_results = []
    for i in range(num_ranges):
        logger.info(f"Starting optimisation for {title} {i + 1}/{num_ranges}")
        if not run_arc_by_arc and i > 0:
            # Double run turns and reduce appropriately
            old_n_run_turns = simulation_config.n_run_turns
            new_max_epochs = int(
                optimiser_config.max_epochs / (old_n_run_turns - 1) * old_n_run_turns
            )
            simulation_config = dataclasses.replace(
                simulation_config, n_run_turns=old_n_run_turns - 1
            )
            optimiser_config = dataclasses.replace(optimiser_config, max_epochs=new_max_epochs)

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
            first_bpm="BPM.33L2.B1" if accelerator.beam == 1 else "BPM.34R8.B2",
        )

        measurement_config = create_arc_measurement_config(
            measurement_file,
            num_tracks=1,
            flattop_turns=flattop_turns,
            corrector_files=corrector_knobs_file,
            tune_knobs_files=tune_knobs_file,
        )

        logger.info(
            f"Found the start and end BPMs for this range: {range_config.bpm_starts[i]} to {range_config.bpm_end_points[i]}"
        )

        controller = Controller(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            measurement_config,
            range_config.bpm_starts[i],
            range_config.bpm_end_points[i],
            show_plots=False,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=None,
            output_config=OutputConfig(write_tensorboard_logs=False),
        )
        final_knobs, uncs = controller.run()
        knobs_results.append(final_knobs)
        uncertainty_results.append(uncs)

    all_final_knobs = {}
    all_uncs = {}
    all_knobs = set()
    for knob_dict in knobs_results:
        all_knobs.update(knob_dict.keys())

    # Take the average of the knob values across all ranges for any knobs that appear in multiple ranges, and take the maximum uncertainty across all ranges for those knobs
    for knob in all_knobs:
        knob_values = [knob_dict.get(knob) for knob_dict in knobs_results if knob in knob_dict]
        knob_uncs = [unc_dict.get(knob) for unc_dict in uncertainty_results if knob in unc_dict]
        all_final_knobs[knob] = np.mean(knob_values)
        all_uncs[knob] = np.max(knob_uncs)

    return all_final_knobs, all_uncs, accelerator


logger = logging.getLogger(__name__)


def generate_comparison_plots(
    results_dir: Path,
    accelerator: LHC,
    corrector_path: Path,
    tune_knobs_path: Path,
    results: dict[str, float],
    model_tws: pd.DataFrame,
    measured_tws: pd.DataFrame,
    beam: int,
) -> None:
    """Generate closed orbit and dispersion comparison plots."""
    logger.info("Generating closed orbit and dispersion comparison plots")

    # Run twiss with optimised knobs
    mad_iface = GradientDescentMadInterface(
        accelerator,
        corrector_strengths=corrector_path,
        tune_knobs_file=tune_knobs_path,
    )
    mad_iface.set_madx_variables(**results)

    mad_iface.observe_elements(pattern="IP[1-8]$")
    tws = mad_iface.run_twiss()
    ip_positions = {f"IP{ip}": tws.loc[f"IP{ip}", "s"] for ip in range(1, 9)}

    def add_ip_positions_to_plot(ax):
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
            )

    mad_iface.observe_elements()
    optimised_tws = mad_iface.run_twiss(deltap=DELTAP)

    # Ensure consistent indexing
    for df in [model_tws, measured_tws, optimised_tws]:
        df.index = df.index.str.upper()

    bpm_index = model_tws.index.intersection(measured_tws.index).intersection(optimised_tws.index)

    model_plot = model_tws.loc[bpm_index]
    measured_plot = measured_tws.loc[bpm_index]
    optim_plot = optimised_tws.loc[bpm_index]
    s_values = model_plot["s"]

    # Plot closed orbit comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    title = f"Closed Orbit Comparison Beam {beam} (Real Data)"
    for diff, label in [
        ((measured_plot["X"] - model_plot["x"]) * 1e3, "Measured - Model"),
        ((optim_plot["x"] - model_plot["x"]) * 1e3, "Optimised - Model"),
        ((measured_plot["X"] - optim_plot["x"]) * 1e3, "Measured - Optimised"),
    ]:
        ax.plot(s_values, diff, label=label)
    add_ip_positions_to_plot(ax)
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylabel("x difference (mm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / f"closed_orbit_comparison_b{beam}.png", dpi=200)

    # Plot dispersion comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    title = f"Dispersion Comparison Beam {beam} (Real Data)"
    ax1.set_title(title)

    for ax, plane, err_col in [(ax1, "dx", "errdx"), (ax2, "dy", "errdy")]:
        diff_meas_model = measured_plot[plane.upper()] - model_plot[plane]
        if err_col in measured_plot.columns and measured_plot[err_col.upper()].notna().any():
            ax.errorbar(
                s_values,
                diff_meas_model,
                yerr=measured_plot[err_col.upper()],
                fmt="x",
                alpha=0.6,
                label="Measured - Model (±err)",
            )
        for diff, label in [
            (diff_meas_model, "Measured - Model"),
            (optim_plot[plane] - model_plot[plane], "Optimised - Model"),
            (measured_plot[plane.upper()] - optim_plot[plane], "Measured - Optimised"),
        ]:
            ax.plot(s_values, diff, label=label)
        add_ip_positions_to_plot(ax)
        ax.set_ylabel(f"d{plane} (m)")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

    ax2.set_xlabel("s (m)")
    fig.tight_layout()
    fig.savefig(results_dir / f"dispersion_comparison_b{beam}.png", dpi=200)

    # Plot actual closed orbit
    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label in [
        (model_plot["x"] * 1e3, "Model"),
        (measured_plot["X"] * 1e3, "Measured"),
        (optim_plot["x"] * 1e3, "Optimised"),
    ]:
        ax.plot(s_values, data, label=label)
    add_ip_positions_to_plot(ax)
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylabel("x (mm)")
    ax.set_title(f"Closed Orbit Beam {beam} (Real Data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / f"closed_orbit_actual_b{beam}.png", dpi=200)

    # Plot actual dispersion
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.set_title(f"Dispersion Beam {beam} (Real Data)")
    for ax, plane in [(ax1, "dx"), (ax2, "dy")]:
        for data, label in [
            (model_plot[plane], "Model"),
            (measured_plot[plane.upper()], "Measured"),
            (optim_plot[plane], "Optimised"),
        ]:
            ax.plot(s_values, data, label=label)
        add_ip_positions_to_plot(ax)
        ax.set_ylabel(f"{plane} (m)")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)
    ax2.set_xlabel("s (m)")
    fig.tight_layout()
    fig.savefig(results_dir / f"dispersion_actual_b{beam}.png", dpi=200)

    logger.info("Saved comparison plots")


def generate_pre_optimisation_orbit_plots(
    results_dir: Path,
    madng_tws: pd.DataFrame,
    measured_df: pd.DataFrame,
    beam: int,
    bad_bpms: list[str],
    turn: int,
) -> None:
    """Generate pre-optimisation orbit/momentum comparison plots from one measured turn vs MAD-NG."""
    logger.info("Generating pre-optimisation orbit comparison plots")

    model_plot = madng_tws.copy()
    model_plot.index = model_plot.index.str.upper()

    measured_turn = measured_df.loc[
        measured_df["turn"] == turn, ["name", "x", "y", "px", "py"]
    ].copy()
    measured_turn["name"] = measured_turn["name"].astype(str).str.upper()
    measured_plot = measured_turn.groupby("name", observed=False, sort=False)[
        ["x", "y", "px", "py"]
    ].mean()

    bad_bpm_index = {bpm.upper() for bpm in bad_bpms}
    model_plot = model_plot.loc[~model_plot.index.isin(bad_bpm_index)]
    measured_plot = measured_plot.loc[~measured_plot.index.isin(bad_bpm_index)]

    bpm_index = model_plot.index.intersection(measured_plot.index)
    comparison = pd.DataFrame(index=bpm_index)
    comparison["s"] = model_plot.loc[bpm_index, "s"]
    for col in ["x", "y", "px", "py"]:
        comparison[f"model_{col}"] = model_plot.loc[bpm_index, col]
        comparison[f"meas_{col}"] = measured_plot.loc[bpm_index, col]

    comparison = comparison.dropna().sort_values("s")
    if comparison.empty:
        logger.warning("No BPMs available for pre-optimisation plotting after filtering")
        return

    s_values = comparison["s"].to_numpy()

    # Difference plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Pre-optimisation Differences Beam {beam} (Turn {turn}, Measured - MAD-NG)")

    plot_specs = [
        ("x", "x", "Δx (mm)", 1e3),
        ("y", "y", "Δy (mm)", 1e3),
        ("px", "px", "Δpx", 1.0),
        ("py", "py", "Δpy", 1.0),
    ]

    for ax, (meas_col, model_col, ylabel, scale) in zip(axes.flat, plot_specs, strict=False):
        diff = (comparison[f"meas_{meas_col}"] - comparison[f"model_{model_col}"]) * scale
        ax.plot(s_values, diff, label=f"{meas_col} diff")
        ax.set_ylabel(ylabel)
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

    axes[1, 0].set_xlabel("s (m)")
    axes[1, 1].set_xlabel("s (m)")
    fig.tight_layout()
    fig.savefig(results_dir / f"pre_optimisation_orbit_difference_b{beam}.png", dpi=200)

    # Absolute plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Pre-optimisation Orbit/Momentum Beam {beam} (Turn {turn})")

    absolute_specs = [
        ("x", "x", "x (mm)", 1e3),
        ("y", "y", "y (mm)", 1e3),
        ("px", "px", "px", 1.0),
        ("py", "py", "py", 1.0),
    ]
    for ax, (meas_col, model_col, ylabel, scale) in zip(axes.flat, absolute_specs, strict=False):
        ax.plot(s_values, comparison[f"model_{model_col}"] * scale, label=f"MAD-NG {model_col}")
        ax.plot(s_values, comparison[f"meas_{meas_col}"] * scale, label=f"Measured {meas_col}")
        ax.set_ylabel(ylabel)
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

    axes[1, 0].set_xlabel("s (m)")
    axes[1, 1].set_xlabel("s (m)")
    fig.tight_layout()
    fig.savefig(results_dir / f"pre_optimisation_orbit_actual_b{beam}.png", dpi=200)

    logger.info("Saved pre-optimisation orbit comparison plots")


def remove_ir_correctors(
    accelerator, corrector_knobs_file: Path, beam: int, arc_magnet_ranges: list[str]
) -> None:
    valid_correctors = set()
    correctors = read_knobs(corrector_knobs_file)
    import re
    def strength_to_corr_name(corrector):
        # Change to uppercase, change the first letter from a to M and change [1-9]+\.[lr][1-8] to \.[1-9]+[lr][1-8]\.
        name = corrector.upper()
        name = name.replace("A", "M", 1)
        name = re.sub(r"([1-9]+)\.([LR])([1-8])", r".\1\2\3.", name)
        return name

    for arc_range in arc_magnet_ranges:
        mad_iface = GradientDescentMadInterface(accelerator, magnet_range=arc_range)
        tws = mad_iface.run_twiss(observe=0)
        valid_correctors.update(corrector for corrector in correctors if strength_to_corr_name(corrector) in tws.index)

    print(valid_correctors, correctors.keys())

    correctors = {
        corrector: strength
        for corrector, strength in correctors.items()
        if corrector in valid_correctors
    }
    save_knobs(correctors, corrector_knobs_file)

# def get_estimated_errors(beam, sequence_file):
#     accel = LHC(beam, )


def main(run_arc_by_arc: bool = False):
    logging.basicConfig(level=logging.INFO)

    # Define the SDDS files
    # fmt: off
    files = [
        # Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@13_44_45_038.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_14_16_935.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_15_01_886.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_15_18_980.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_15_32_838.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_15_45_961.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_16_00_824.sdds"),
        Path("/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11412/BPM/BUNCHTURN.BEAM1@Concentrated@2026_02_27@14_16_21_883.sdds"),
    ]
    # fmt: on

    beam = 1
    # Model directory
    model_dir = Path(
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2026-02-27/LHCB1/Models/2026-02-26_B1_120cm_flattop"
    )
    analysis_dir = Path(
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2026-02-27/LHCB1/Results/2026-02-27_B1_ft_120cm_onoffmom"
    )

    temp_analysis_dir = Path(
        "/afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner/temp_analysis_real_data_b1"
    )
    temp_dir_exists = temp_analysis_dir.exists()
    if not temp_dir_exists:
        temp_analysis_dir.mkdir(parents=True, exist_ok=True)

    ng_model_dir = temp_analysis_dir / "madng_model"
    ng_model_dir.mkdir(parents=True, exist_ok=True)

    # Load the model twiss
    logging.info("Loading model twiss data")
    tws = build_madng_twiss_table(model_dir, beam, ng_model_dir)
    tws.columns = [col.lower() for col in tws.columns]
    tws = tws.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws.headers = {k.lower(): v for k, v in tws.headers.items()}
    tws = tws.set_index("name")

    measurement_file = temp_analysis_dir / "pz_data.parquet"
    bad_bpms = find_all_bad_bpms_from_analysis(analysis_dir)
    n_turns = 19  # 19 cause I messed up originally, it doesn't matter.
    if not measurement_file.exists():
        # Read all TBT data in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(read_tbt_file, files))

        bpm_categories = results[0][0].index.tolist()
        turn_counts = np.array([X.shape[1] for X, _ in results], dtype=np.int32)
        turn_offsets = np.concatenate(([0], np.cumsum(turn_counts[:-1], dtype=np.int32)))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_data = list(
                executor.map(
                    build_long_measurement_df,
                    (X for X, _ in results),
                    (Y for _, Y in results),
                    turn_offsets.tolist(),
                    [bpm_categories] * len(results),
                )
            )

        df = pd.concat(all_data, ignore_index=True)

        # SVD clean
        df = svd_clean_measurements(df)

        logging.info("Data has %d turns and %d BPMs", df["turn"].nunique(), df["name"].nunique())

        # Compute mean and variance per BPM
        mean_x = df.groupby("name")["x"].mean()
        var_x = df.groupby("name")["x"].var()
        mean_y = df.groupby("name")["y"].mean()
        var_y = df.groupby("name")["y"].var()

        bpm_names = df["name"].unique().tolist()
        bpm_names = [bpm for bpm in bpm_names if bpm not in bad_bpms]

        # Filter to BPMs in tws
        tws = tws.loc[bpm_names]

        # Compute averages and variances
        rows = []
        for name in bpm_names:
            rows.append(
                {
                    "name": name,
                    "x": mean_x[name],
                    "y": mean_y[name],
                    "var_x": var_x[name],
                    "var_y": var_y[name],
                }
            )

        averaged = pd.DataFrame(rows)
        print(
            averaged["var_x"].describe(),
            averaged["var_y"].describe(),
        )

        # Create new DataFrame with 5 turns, each with averaged values
        new_rows = []
        for turn in range(1, n_turns + 1):
            for _, row in averaged.iterrows():
                new_rows.append(
                    {
                        "name": row["name"],
                        "turn": turn,
                        "x": row["x"],
                        "y": row["y"],
                        "var_x": row["var_x"],
                        "var_y": row["var_y"],
                    }
                )
        new_df = pd.DataFrame(new_rows)
        new_df["name"] = new_df["name"].astype("category")
        new_df["turn"] = new_df["turn"].astype("int32")

        # Calculate transverse momenta
        logging.info("Calculating transverse momenta")
        new_df = calculate_transverse_pz(new_df, tws, inject_noise=False)

        # Save the measurement file
        new_df.to_parquet(measurement_file, index=False)
        logger.info(f"Saved measurement file to {measurement_file}")
    else:
        logger.info(f"Measurement file {measurement_file} already exists, loading it")
        new_df = pd.read_parquet(measurement_file)

    # Now run the bends optimisation
    sequence_path = ng_model_dir / f"lhcb{beam}_saved.seq"

    # Create arc config
    if run_arc_by_arc and beam == 1:

        def _next_s(s):
            return s % 8 + 1

        arc_magnet_ranges = [f"BPM.9R{s}.B1/BPM.9L{_next_s(s)}.B1" for s in range(1, 9)]
        arc_bpm_starts = [
            [f"BPM.{i}R{s}.B1" for i in range(9, 34, 5)]
            # + [f"BPM.{i}L{_next_s(s)}.B1" for i in range(9, 34, 5)]
            for s in range(1, 9)
        ]
        arc_bpm_end_points = [
            [f"BPM.{i}L{_next_s(s)}.B1" for i in range(9, 35, 5)] for s in range(1, 9)
        ]
    elif run_arc_by_arc and beam == 2:

        def _next_s(s):
            return (s - 2) % 8 + 1

        arc_magnet_ranges = [f"BPM.9L{s}.B2/BPM.9R{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
        arc_bpm_starts = [[f"BPM.{i}L{s}.B2" for i in range(9, 34, 3)] for s in range(8, 0, -1)]
        arc_bpm_end_points = [
            [f"BPM.{i}L{(s - 2) % 8 + 1}.B2" for i in range(9, 35, 3)] for s in range(8, 0, -1)
        ]
    else:
        arc_magnet_ranges = ["$start/$end"]
        arc_bpm_starts = [[f"BPM.9{lr}{s}.B{beam}" for s in range(1, 9) for lr in "L"]]  # * 3
        arc_bpm_end_points = [[]]  # * 3

    arc_config = RangeConfig(
        magnet_ranges=arc_magnet_ranges,
        bpm_starts=arc_bpm_starts,
        bpm_end_points=arc_bpm_end_points,
    )

    # Create accelerator
    accelerator = LHC(
        beam=beam,
        pc=6800,
        sequence_file=sequence_path,
        optimise_energy=False,
        optimise_bends=True,
        optimise_correctors=False,
        optimise_quad_dx=False,
        optimise_other_quadrupoles=False,
    )

    # Extend the bad bpms with any BPMs that are not in the measurements or the model
    measurement_bpms = set(new_df["name"].unique())
    model_bpms = set(tws.index)
    missing_bpms = measurement_bpms.symmetric_difference(model_bpms)

    if missing_bpms:
        logger.warning(
            f"The following BPMs are in the measurements but not in the model, or vice versa: {missing_bpms}. They will be added to the bad BPM list."
        )
        bad_bpms = bad_bpms.union(missing_bpms)
    bad_bpms = list(bad_bpms)
    bad_bpms.extend(["BPM.27R3.B1", "BPM.9R1.B1"])

    # Save bad_bpms if any
    bad_bpms_file = temp_analysis_dir / "bad_bpms.txt"
    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")
    logger.info(f"Saved bad BPMs to {bad_bpms_file}")

    # Optimiser config
    optimiser_config = OptimiserConfig(
        max_epochs=1000 if run_arc_by_arc else 300,
        warmup_epochs=100 if run_arc_by_arc else 70,
        warmup_lr_start=8e-10,
        # max_lr=4e-9,
        # min_lr=1e-8,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-10,
        optimiser_type="lbfgs",
        # optimiser_type="adam",
        expected_rel_error=0,#10e-4,
    )

    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=1,
        num_batches=1,
        optimise_momenta=False,
        use_fixed_bpm=False,  # When run_arc_by_arc=False, use_fixed_bpm does nothing.
        run_arc_by_arc=run_arc_by_arc,
        n_run_turns=1 if run_arc_by_arc else 3,
        bpm_loss_outlier_sigma=10,
        worker_loss_outlier_sigma=5,
    )

    # Results dir
    results_dir = Path(f"/afs/cern.ch/work/j/jmgray/private/sgd-magnet-tuner/b{beam}_bends_results")
    results_dir.mkdir(exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    generate_pre_optimisation_orbit_plots(results_dir, tws, new_df, beam, bad_bpms, turn=1)
    # plt.show()

    corrector_knobs_file = results_dir / "corrector_knobs.txt"
    tune_knobs_file = results_dir / "tune_knobs.txt"
    if not corrector_knobs_file.exists() or not tune_knobs_file.exists():
        meas_time = measurement_time_from_sdds_file(files[0])
        save_online_knobs(
            meas_time,
            beam=beam,
            tune_knobs_file=tune_knobs_file,
            corrector_knobs_file=corrector_knobs_file,
        )

    # remove_ir_correctors(accelerator, corrector_knobs_file, beam, arc_magnet_ranges)

    # Run optimisation
    results, uncs, accelerator = optimise_ranges(
        arc_config,
        accelerator,
        optimiser_config,
        simulation_config,
        corrector_knobs_file,
        tune_knobs_file,
        measurement_file,
        list(bad_bpms),
        "bends",
        n_turns,  # flattop_turns
        plots_dir,
        run_arc_by_arc,
    )

    # Filter out knobs where half uncertainty > strength
    filtered_results = {}
    for knob in results:
        # if 0.5 * uncs[knob] > abs(results[knob]):
        #     logger.warning(
        #         f"Removing knob {knob}: strength {results[knob]:.6f}, uncertainty {uncs[knob]:.6f}, half unc {0.5 * uncs[knob]:.6f}"
        #     )
        # else:
            filtered_results[knob] = results[knob]

    # print out every knob that ends in dx and has an uncertainty greater than 0.5 cm
    for knob in results:
        if knob.endswith("dx") and uncs[knob] > 0.01/2:
            logger.warning(
                f"Knob {knob} has uncertainty greater than 1cm: strength {results[knob]:.6f}, uncertainty {uncs[knob]:.6f}"
            )

    # Save results
    bend_knobs_file = results_dir / "bend_knobs_bends_optimised.txt"
    with bend_knobs_file.open("w") as f:
        for k, v in filtered_results.items():
            f.write(f"{k} {v}\n")
    logger.info(f"Saved bend knobs to {bend_knobs_file}")

    lhc_displacement = LHC(
        beam=beam,
        pc=6800,
        sequence_file=sequence_path,
        optimise_energy=False,
        optimise_bends=True,
        optimise_correctors=False,
        optimise_quad_dx=True,
    )
    filtered_results["deltap"] = DELTAP
    results, uncs, accelerator = optimise_ranges(
        arc_config,
        lhc_displacement,
        optimiser_config,
        simulation_config,
        corrector_knobs_file,
        tune_knobs_file,
        measurement_file,
        list(bad_bpms),
        "bends",
        n_turns,  # flattop_turns
        plots_dir,
        run_arc_by_arc,
        initial_knob_strengths=filtered_results,
    )

    # Generate comparison plots
    measured_tws, disp_found = build_twiss_from_measurements(analysis_dir, include_errors=True)
    assert disp_found, (
        "Dispersion data not found in analysis results, cannot generate comparison plots"
    )
    plt.close("all")
    generate_comparison_plots(
        results_dir,
        accelerator,
        corrector_knobs_file,
        tune_knobs_file,
        results,
        tws,
        measured_tws,
        beam,
    )


if __name__ == "__main__":
    main(run_arc_by_arc=False)

    show_plots = True
    if show_plots:
        plt.show()
