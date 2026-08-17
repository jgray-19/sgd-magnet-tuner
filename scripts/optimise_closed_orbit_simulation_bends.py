from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from matplotlib import pyplot as plt
from pymadng_utils.io.utils import save_knobs
from tmom_recon import calculate_pz

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig, SimulationConfig
from aba_optimiser.mad import AbaMadInterface, GradientDescentMadInterface
from aba_optimiser.noise import assign_bpm_variances
from aba_optimiser.training.tracking_fitter import ArcByArcFitter
from aba_optimiser.training.controller_config import SequenceConfig
from aba_optimiser.training.controller_helpers import (
    create_arc_measurement_config,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

PC_GEV = 6800.0
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
DEFAULT_DPP_VALUES = [-1e-4, 0.0, 1e-4, 2e-4, -2e-4]


@dataclass
class RangeConfig:
    magnet_ranges: list[str]
    bpm_starts: list[list[str]]
    bpm_end_points: list[list[str]]


def create_tracking_dataframe(
    noisy_df: pd.DataFrame, turns: int = 3
) -> pd.DataFrame:
    """Convert noisy DataFrame to tracking DataFrame format."""
    expanded_rows = []
    for turn in range(1, turns + 1):
        for _, row in noisy_df.iterrows():
            expanded_rows.append(
                {
                    "name": row.name,
                    "turn": turn,
                    "x": row["x"],
                    "y": row["y"],
                    "var_x": row["var_x"],
                    "var_y": row["var_y"],
                }
            )
    result = pd.DataFrame(expanded_rows)
    result["name"] = result["name"].astype("category")
    result["turn"] = result["turn"].astype("int32")
    return result


def average_tracking_dataframe(tracking_df: pd.DataFrame, turns: int = 3) -> pd.DataFrame:
    """Average tracking over turns per BPM and replicate a fixed number of turns.

    Mirrors the measurement pipeline where per-BPM data are averaged and the
    variance of the mean is propagated, then expanded to a small number of
    turns for downstream consumers.
    """
    rows = []
    grouped = tracking_df.groupby("name")
    for name, sub in grouped:
        xs = sub["x"].to_numpy()
        ys = sub["y"].to_numpy()
        pxs = sub["px"].to_numpy()
        pys = sub["py"].to_numpy()

        rows.append(
            {
                "name": name,
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "px": float(np.mean(pxs)),
                "py": float(np.mean(pys)),
                "var_px": 3e-6**2,
                "var_py": 3e-6**2,
            }
        )

    averaged = pd.DataFrame(rows)
    # Assign BPM-specific variances using the noise module
    averaged = assign_bpm_variances(averaged, "lhc")
    expanded_rows = []
    for turn in range(1, turns + 1):
        for _, row in averaged.iterrows():
            expanded_rows.append(
                {
                    "name": row["name"],
                    "turn": turn,
                    "x": row["x"],
                    "y": row["y"],
                    "px": row["px"],
                    "py": row["py"],
                    "var_x": row["var_x"],
                    "var_y": row["var_y"],
                    "var_px": row["var_px"],
                    "var_py": row["var_py"],
                }
            )

    result = pd.DataFrame(expanded_rows)
    result["name"] = result["name"].astype("category")
    result["turn"] = result["turn"].astype("int32")
    return result


def format_dpp_label(dpp: float) -> str:
    if dpp == 0:
        return "0"
    sign = "" if dpp > 0 else "m"
    dpp = dpp * 1000  # Scale to match typical notation
    dpp_str = f"{dpp}".replace(".", "p").strip("-")
    return f"{sign}{dpp_str}"


def get_sequence_file(beam: int) -> Path:
    """Get the default sequence file path for a given beam."""
    return PROJECT_ROOT / f"tests/data/sequences/lhcb{beam}.seq"


def get_results_dir(beam: int) -> Path:
    """Get the results directory for a given beam."""
    return PROJECT_ROOT / f"b{beam}_sim_bends_results"


def get_temp_dir(beam: int) -> Path:
    """Get the temporary analysis directory for a given beam."""
    return PROJECT_ROOT / f"temp_analysis_sim_bends_b{beam}"


def create_full_ring_config(beam: int) -> RangeConfig:
    if beam == 1:
        magnet_ranges = ["BPM.9R1.B1/BPM.8R1.B1"]
        bpm_starts = [[f"BPM.9R{s}.B1" for s in range(1, 9)]]
        bpm_end_points = [[f"BPM.8R{s}.B1" for s in range(1, 9)]]
    elif beam == 2:
        magnet_ranges = ["BPM.8R1.B2/BPM.9R1.B2"]
        bpm_starts = [[f"BPM.8R{s}.B2" for s in range(1, 9)]]
        bpm_end_points = [[f"BPM.9R{s}.B2" for s in range(1, 9)]]
    else:
        raise ValueError(f"Unsupported beam: {beam}. Must be 1 or 2.")
    return RangeConfig(
        magnet_ranges=magnet_ranges,
        bpm_starts=bpm_starts,
        bpm_end_points=bpm_end_points,
    )


def create_arc_config(beam: int) -> RangeConfig:
    """Create arc range configuration for the given beam."""
    if beam == 1:
        arc_magnet_ranges = [f"BPM.9L{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]
        arc_bpm_starts = [
            [
                f"BPM.{i}{lr}{s if lr == 'R' else s % 8 + 1}.B1"
                for i in range(9, 35, 2)
                for lr in ["L", "R"]
                if not (i < 15 and lr == "L")
            ]
            for s in range(1, 9)
        ]
        # print(arc_bpm_starts)
        arc_bpm_end_points = [[f"BPM.9L{s % 8 + 1}.B1"] for s in range(1, 9)]
    elif beam == 2:
        arc_magnet_ranges = [f"BPM.9L{s}.B2/BPM.9L{(s - 2) % 8 + 1}.B2" for s in range(8, 0, -1)]
        arc_bpm_starts = [[f"BPM.{i}L{s}.B2" for i in range(9, 34, 2)] for s in range(8, 0, -1)]
        arc_bpm_end_points = [
            [f"BPM.{i}L{(s - 2) % 8 + 1}.B2" for i in range(9, 35, 2)] for s in range(8, 0, -1)
        ]
    else:
        raise ValueError(f"Unsupported beam: {beam}. Must be 1 or 2.")
    # For now just take the first start and end points for all arcs
    # arc_bpm_starts = [[starts[0]] for starts in arc_bpm_starts]
    # arc_bpm_end_points = [[ends[0]] for ends in arc_bpm_end_points]
    return RangeConfig(
        magnet_ranges=arc_magnet_ranges,
        bpm_starts=arc_bpm_starts,
        bpm_end_points=arc_bpm_end_points,
    )


def generate_track_with_errors(
    sequence_path: Path,
    destination_dir: Path,
    dpp_value: float,
    corrector_path: Path,
    tune_knobs_path: Path,
    beam: int,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    """Generate tracking data with magnet errors and AC dipole excitation.

    Args:
        sequence_path: Path to the MAD-NG sequence file
        destination_dir: Directory to save the generated measurement file
        dpp_value: Momentum deviation for the tracking
        corrector_path: Path to save corrector knob strengths
        tune_knobs_path: Path to save tune knob strengths
        beam: Beam number (1 or 2)
    Returns:
        Tuple of (measurement_file_path, before_twiss_df, after_twiss_df)
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    measurement_file = destination_dir / "pz_data.parquet"

    iface = AbaMadInterface(
        accelerator=LHC(beam=beam, sequence_file=sequence_path, kinetic_energy=PC_GEV)
    )
    iface.mad["zero_twiss", "_"] = iface.mad.twiss(sequence="loaded_sequence")

    iface.observe_elements()
    before_tws = iface.run_twiss(deltap=dpp_value)
    iface.unobserve_elements(["BPM"])

    magnet_strengths, _ = iface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="qds",
    )

    matched_tunes = iface.perform_orbit_correction(
        machine_deltap=0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_path,
    )
    if dpp_value != 0.0:
        matched_tunes = iface.perform_orbit_correction(
            machine_deltap=dpp_value,
            target_qx=0.28,
            target_qy=0.31,
            corrector_file=corrector_path,
        )

    corrector_table = tfs.read(corrector_path)
    corrector_table = corrector_table.loc[corrector_table.loc[:, "kind"] != "monitor"]

    iface.observe_elements()
    after_tws = iface.run_twiss(deltap=dpp_value)
    iface.unobserve_elements(["BPM"])
    save_knobs(matched_tunes, tune_knobs_path)

    # Apply BPM-specific noise variances using the noise module
    variance_df = assign_bpm_variances(after_tws.copy(), "lhc")
    tracking_df = create_tracking_dataframe(variance_df, turns=3)
    tracking_df = calculate_pz(tracking_df, model_tws=before_tws, inject_noise=False)

    # Check for nans for any bpms, remove that bpm add a warning and add to bad_bpms
    nans = tracking_df.isna().any(axis=1)
    bad_bpms = []
    if nans.any():
        bad_bpms = tracking_df.loc[nans, "name"].unique()
        logger.warning("Removing %s BPMs with NaNs in tracking data", len(bad_bpms))
        tracking_df = tracking_df[~tracking_df["name"].isin(bad_bpms)]

    tracking_df.to_parquet(measurement_file, index=False)
    return measurement_file, before_tws, after_tws, magnet_strengths, bad_bpms


def optimise_ranges(
    range_config: RangeConfig,
    accelerator: LHC,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    corrector_knobs: Path,
    tune_knobs: Path,
    measurement_file: Path,
    actual_knobs: dict[str, float] | None,
    bad_bpms: list[str],
    title: str,
    flattop_turns: int,
    plots_dir: Path,
) -> tuple[dict[str, float], dict[str, float], LHC]:
    num_ranges = len(range_config.magnet_ranges)
    all_final_knobs = {}
    all_uncs = {}
    for i in range(num_ranges):
        logger.info("Starting optimisation for arc %s/%s (%s)", i + 1, num_ranges, title)

        measurement_config = create_arc_measurement_config(
            measurement_file,
            machine_deltap=0.0,
            corrector_knobs=corrector_knobs,
            tune_knobs=tune_knobs,
        )

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
        )

        controller = ArcByArcFitter(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=range_config.bpm_starts[i],
            bpm_end_points=range_config.bpm_end_points[i],
            show_plots=False,
            initial_knob_strengths=None,
            true_strengths=None,
            plots_dir=plots_dir,
            debug=True,
            mad_logfile=measurement_file.parent / "mad_log"
        )
        final_knobs, uncs = controller.run()
        all_final_knobs.update(final_knobs)
        all_uncs.update(uncs)
    return all_final_knobs, all_uncs, accelerator


def create_optimiser_config() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=2000,
        warmup_epochs=50,
        warmup_lr_start=5e-11,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-10,
        optimiser_type="lbfgs",
        expected_rel_error=0,
    )


def create_simulation_config(no_fixed_bpm: bool) -> SimulationConfig:
    return SimulationConfig(
        num_workers=1,
        num_batches=1,
        optimise_momenta=False,
        use_fixed_bpm=not no_fixed_bpm,
    )


def process_data(beam: int, temp_dir: Path, results_dir: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, list[str], dict[str, float] | None, Path, float]:
    logger.info("Generating measurement data with errors for bends optimisation")
    sequence_file = PROJECT_ROOT / "sequences_from_models" / f"OMC3_LHCB{beam}_2025_28m010_31p012.seq"
    corrector_path = results_dir / "corrector_knobs.txt"
    tune_knobs_path = results_dir / "tune_knobs.txt"
    measurement_file, before_tws, after_tws, actual_knobs, bad_bpms = generate_track_with_errors(
        sequence_path=sequence_file,
        destination_dir=temp_dir,
        dpp_value=0,
        corrector_path=corrector_path,
        tune_knobs_path=tune_knobs_path,
        beam=beam,
    )
    pc = 6800.0
    return measurement_file, before_tws, after_tws, bad_bpms, actual_knobs, sequence_file, pc


def load_results(results_dir: Path) -> dict[str, float]:
    bend_knobs_file = results_dir / "bend_knobs_bends_optimised.txt"
    if not bend_knobs_file.exists():
        raise FileNotFoundError(f"Results file not found: {bend_knobs_file}")

    results = {}
    with bend_knobs_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                results[parts[0]] = float(parts[1])
    return results


def run_optimisation(
    arc_config: RangeConfig,
    accelerator: LHC,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    corrector_knobs: Path,
    tune_knobs: Path,
    measurement_file: Path,
    actual_knobs: dict[str, float] | None,
    bad_bpms: list[str],
    plots_dir: Path,
) -> tuple[dict[str, float], dict[str, float], LHC]:
    logger.info("Starting arc-by-arc closed orbit optimisations for bends")
    results, uncs, accel = optimise_ranges(
        range_config=arc_config,
        accelerator=accelerator,
        optimiser_config=optimiser_config,
        simulation_config=simulation_config,
        corrector_knobs=corrector_knobs,
        tune_knobs=tune_knobs,
        measurement_file=measurement_file,
        actual_knobs=actual_knobs,
        bad_bpms=bad_bpms,
        title="bends",
        flattop_turns=2_000,
        plots_dir=plots_dir,
    )
    plt.close("all")

    logger.info("All arc optimisations complete for bends")
    if not results:
        raise RuntimeError("No arc results produced for bends")
    return results, uncs, accel


def generate_plots(
    results_dir: Path,
    accelerator: LHC,
    corrector_path: Path,
    tune_knobs_path: Path,
    results: dict[str, float],
    before_tws: pd.DataFrame,
    after_tws: pd.DataFrame,
    measurement_file: Path,
    beam: int,
) -> None:
    logger.info("Generating closed orbit comparison plot with bends optimisation")
    mad_iface = GradientDescentMadInterface(
        accelerator,
        corrector_knobs=corrector_path,
        tune_knobs=tune_knobs_path,
    )
    if accelerator.optimise_bends:
        mad_iface.set_madx_variables(**results)
    else:
        mad_iface.set_magnet_strengths(results)
    mad_iface.observe_elements()
    after_optimised_tws = mad_iface.run_twiss()
    tracking_df = pd.read_parquet(measurement_file)
    tracking_turn2 = tracking_df[tracking_df["turn"] == 2]

    # Ensure all BPM names are uppercase for consistent intersection
    before_tws.index = before_tws.index.str.upper()
    after_tws.index = after_tws.index.str.upper()
    after_optimised_tws.index = after_optimised_tws.index.str.upper()
    tracking_turn2["name"] = tracking_turn2["name"].str.upper()

    bpm_index = (
        before_tws.index.intersection(after_tws.index)
        .intersection(after_optimised_tws.index)
        .intersection(tracking_turn2["name"].unique())
    )

    before_plot = before_tws.loc[bpm_index]
    after_plot = after_tws.loc[bpm_index]
    optim_plot = after_optimised_tws.loc[bpm_index]
    track_plot = tracking_turn2.set_index("name").loc[bpm_index]

    s_values = before_tws.loc[bpm_index, "s"]

    # Plot the difference between the closed orbit after tws with errors and correction
    # and averaged closed orbit from the tracking without optimisation
    fig, ax = plt.subplots(figsize=(10, 5))

    data_label = "Tracking turn 2 - model"
    title = f"Closed Orbit Difference Beam {beam} (Simulation)"
    rms_diff_track_before = np.sqrt(np.mean((track_plot["x"].to_numpy() - before_plot["x"].to_numpy()) ** 2)) * 1e3
    rms_diff_optim_before = np.sqrt(np.mean((after_optimised_tws["x"].to_numpy() - before_plot["x"].to_numpy()) ** 2)) * 1e3
    rms_diff_track_optim = np.sqrt(np.mean((track_plot["x"].to_numpy() - after_optimised_tws["x"].to_numpy()) ** 2)) * 1e3
    logger.info(
        "RMS x difference before optimisation: %.3f mm, after optimisation: %.3f mm, measurement vs optimised: %.3f mm",
        rms_diff_track_before,
        rms_diff_optim_before,
        rms_diff_track_optim,
    )

    ax.plot(
        s_values,
        (track_plot["x"].to_numpy() - before_plot["x"].to_numpy()) * 1e3,
        label=data_label,
    )
    ax.plot(
        s_values,
        (after_optimised_tws["x"].to_numpy() - before_plot["x"].to_numpy()) * 1e3,
        label="After Optimisation - Model",
    )
    ax.plot(
        s_values,
        (track_plot["x"].to_numpy() - after_optimised_tws["x"].to_numpy()) * 1e3,
        label="Measurement - After Optimisation",
    )
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylabel("x difference (mm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        results_dir / f"closed_orbit_difference_bends_b{beam}.png",
        dpi=200,
    )

    # Plot the dispersion
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    vx = "dx"
    vy = "dy"

    dx_diff = after_plot[vx] - before_plot[vx]
    dy_diff = after_plot[vy] - before_plot[vy]

    label1 = "After Errors & Correction - Model"
    label2 = "After Optimisation - Model"
    title = f"Dispersion Comparison Beam {beam} (Simulation)"

    if "errdx" in after_plot.columns:
        ax1.errorbar(
            before_plot["s"],
            dx_diff,
            yerr=after_plot["errdx"],
            fmt="x",
            alpha=0.6,
            label=f"{label1} (±errdx)",
        )
    ax1.plot(
        before_plot["s"],
        dx_diff,
        label=label1,
    )
    ax1.plot(
        before_plot["s"],
        optim_plot[vx] - before_plot[vx],
        label=label2,
    )
    ax1.plot(
        before_plot["s"],
        optim_plot[vx] - after_plot[vx],
        label="Measurement - After Optimisation",
    )
    ax1.set_ylabel(f"{vx} (m)")
    ax1.legend()
    ax1.set_title(title)
    ax1.grid(visible=True, alpha=0.3)
    if "errdy" in after_plot.columns:
        ax2.errorbar(
            before_plot["s"],
            dy_diff,
            yerr=after_plot["errdy"],
            fmt="x",
            alpha=0.6,
            label=f"{label1} (±errdy)",
        )
    ax2.plot(
        before_plot["s"],
        dy_diff,
        label=label1,
    )
    ax2.plot(
        before_plot["s"],
        optim_plot[vy] - before_plot[vy],
        label=label2,
    )
    ax2.plot(
        before_plot["s"],
        optim_plot[vy] - after_plot[vy],
        label="Measurement - After Optimisation",
    )
    ax2.set_xlabel("s (m)")
    ax2.set_ylabel(f"{vy} (m)")
    ax2.legend()
    ax2.grid(visible=True, alpha=0.3)
    output_path = results_dir / f"closed_orbit_comparison_bends_b{beam}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    logger.info("Saved plot to %s", output_path)
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam",
        type=int,
        choices=[1, 2],
        default=1,
        help="Beam number (1 or 2)",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve temporary directory",
    )
    parser.add_argument(
        "--no-fixed-bpm",
        action="store_true",
        help="Use full Cartesian BPM pairing",
    )
    parser.add_argument(
        "--full-ring",
        action="store_true",
        help="Optimise over the full ring instead of arc-by-arc",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate closed orbit comparison plot after optimisation",
    )
    parser.add_argument(
        "--just-plot",
        action="store_true",
        help="Load saved results and generate plot without running optimisation",
    )
    args = parser.parse_args()

    # Derive paths from beam number
    results_dir = get_results_dir(args.beam)
    temp_dir = get_temp_dir(args.beam)
    temp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    optimiser_config = create_optimiser_config()
    simulation_config = create_simulation_config(args.no_fixed_bpm)

    measurement_file, before_tws, after_tws, bad_bpms, actual_knobs, sequence_file, pc = process_data(
        args.beam, temp_dir, results_dir
    )

    if args.just_plot:
        results = load_results(results_dir)
    else:
        if args.full_ring:
            arc_config = create_full_ring_config(args.beam)
        else:
            arc_config = create_arc_config(args.beam)
        accel = LHC(
            beam=args.beam,
            sequence_file=sequence_file,
            kinetic_energy=pc,
            optimise_bends=True,
            normalise_bends=True,
        )
        results, uncs, accel = run_optimisation(
            arc_config,
            accel,
            optimiser_config,
            simulation_config,
            results_dir / "corrector_knobs.txt",
            results_dir / "tune_knobs.txt",
            measurement_file,
            actual_knobs,
            bad_bpms,
            temp_dir,
        )
        save_knobs(results, results_dir / "bend_knobs_bends_optimised.txt")

    if args.plot or args.just_plot:
        if 'accel' not in locals():
            # For just_plot, we need to recreate accel
            accel = LHC(
                beam=args.beam,
                sequence_file=sequence_file,
                kinetic_energy=pc,
                optimise_bends=True,
                normalise_bends=True,
            )
        generate_plots(
            results_dir,
            accel,
            results_dir / "corrector_knobs.txt",
            results_dir / "tune_knobs.txt",
            results,
            before_tws,
            after_tws,
            measurement_file,
            args.beam,
        )

    if not args.keep_temp:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
