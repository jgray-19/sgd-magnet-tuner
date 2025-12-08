"""
Integration test for quadrupole convergence with errors using AC dipole excitation.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.momentum_recon import calculate_dispersive_pz
from aba_optimiser.momentum_recon import calculate_transverse_pz as calc_pz
from aba_optimiser.physics.lhc_bends import normalise_lhcbend_magnets
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig
from aba_optimiser.xsuite.xsuite_tools import (
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
    run_tracking,
)
from tests.training.helpers import (
    TRACK_COLUMNS,
    generate_model_with_errors,
    get_twiss_without_errors,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def _run_track_with_acd(
    env: dict,
    flattop_turns: int,
    acd_ramp: int,
    destination: Path,
    dpp_values: list[float],
    lags: list[float],
    return_dataframes: bool = False,
) -> list[Path] | list[pd.DataFrame]:
    """
    Run tracking with AC dipole excitation for different lag values.

    Args:
        env: xsuite environment dictionary
        flattop_turns: Number of flattop turns to track (excluding ramp)
        acd_ramp: Number of turns for AC dipole ramp up/down
        destination: Path to save the parquet file (or base path for multiple files)
        dpp_value: Momentum deviation
        lags: List of lag values for each track (one track per lag)
        return_dataframes: If True, return dataframes instead of writing to disk

    Returns:
        List of Path objects for each generated parquet file, or list of DataFrames if return_dataframes=True
    """
    # Compute twiss for ACD insertion

    # AC dipole parameters
    total_turns = flattop_turns + acd_ramp
    driven_tunes = [0.27, 0.322]

    processed_dfs = []
    output_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)

    for idx, lag in enumerate(lags):
        tws = env["lhcb1"].twiss(method="4d", delta0=0)
        # Insert AC dipole with this lag value
        line = insert_ac_dipole(
            env["lhcb1"],
            tws,
            beam=1,
            acd_ramp=acd_ramp,
            total_turns=total_turns,
            driven_tunes=driven_tunes,
            lag=lag,
        )

        # Insert monitors for flattop turns only (after ramp)
        insert_particle_monitors_at_pattern(
            line,
            pattern="bpm.*[^k]",
            num_turns=total_turns,  # Monitor all turns, we'll filter later
            num_particles=1,
            inplace=True,
        )

        # Build particle at small initial offset
        particles = line.build_particles(
            x=[0.0],
            px=[0.0],
            y=[0.0],
            py=[0.0],
            delta=[dpp_values[idx]],
        )

        # Run tracking for total turns (ramp + flattop)
        run_tracking(
            line=line,
            particles=particles,
            nturns=total_turns,
        )

        # Get data from line
        true_dfs = line_to_dataframes(line)

        # Process dataframe - only keep flattop turns (after ramp)
        for true_df in true_dfs:
            # Filter to only include turns after the ramp
            df = true_df[true_df["turn"] >= acd_ramp].copy()
            # Renumber turns to start from 0
            df["turn"] = df["turn"] - acd_ramp

            df = prepare_track_dataframe(df, 0, flattop_turns, kick_both_planes=True)
            df = df.loc[:, TRACK_COLUMNS].copy()
            df["name"] = df["name"].astype(str)
            df["kick_plane"] = df["kick_plane"].astype(str)

            if return_dataframes:
                processed_dfs.append(df)
            else:
                # Save to file
                output_path = (
                    destination.parent / f"{destination.stem}_lag_{idx}{destination.suffix}"
                )
                df.to_parquet(output_path, index=False)
                output_files.append(output_path)

    if return_dataframes:
        return processed_dfs
    return output_files


def _make_optimiser_config_bend() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=15,
        warmup_epochs=5,
        warmup_lr_start=5e-10,
        max_lr=4e-8,
        min_lr=4e-8,
        gradient_converged_value=4e-11,
        optimiser_type="adam",
    )


def _make_simulation_config_bend() -> SimulationConfig:
    return SimulationConfig(
        tracks_per_worker=799,
        num_batches=200,
        num_workers=60,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
        optimise_momenta=False,
    )


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_tracks_quad_conv")


@pytest.mark.slow
def test_controller_bend_opt_simple(tmp_dir: Path, seq_b1: Path, json_b1: Path) -> None:
    """Test bend optimisation using AC dipole excitation with different lag values."""
    flattop_turns = 4_000
    acd_ramp = 1_000  # Ramp turns for AC dipole
    off_magnet_path = tmp_dir / "track_off_magnet.parquet"
    corrector_file = tmp_dir / "corrector_track_off_magnet.tfs"

    # Generate model with errors for all arcs
    env, magnet_strengths, twiss_errs, tune_knobs_file = generate_model_with_errors(
        sequence_file=seq_b1,
        json_file=json_b1,
        dpp_value=0,
        magnet_range="$start/$end",
        corrector_file=corrector_file,
        perturb_quads=True,
        perturb_bends=True,
    )

    # Get clean twiss for pz calculation
    tws_no_err = get_twiss_without_errors(seq_b1, just_bpms=False)

    # Generate tracks with AC dipole using 3 different lag values
    dpp_values = [0.0, 3e-4, -3e-4, 5e-4, -5e-4]
    lags = np.linspace(0, 2 * np.pi, len(dpp_values), endpoint=False).tolist()
    track_dfs = _run_track_with_acd(
        env=env,
        flattop_turns=flattop_turns,
        acd_ramp=acd_ramp,
        destination=off_magnet_path,
        dpp_values=dpp_values,
        lags=lags,
        return_dataframes=True,
    )
    xsuite_tws = env["lhcb1"].twiss4d().to_pandas()
    xsuite_tws = xsuite_tws.set_index("name")
    # convert name to be upper case to match measurement files
    xsuite_tws.index = xsuite_tws.index.str.upper()
    co_x = xsuite_tws["x"].to_dict()
    co_y = xsuite_tws["y"].to_dict()
    co_px = xsuite_tws["px"].to_dict()
    co_py = xsuite_tws["py"].to_dict()

    # Process each track dataframe and save
    processed_files = []
    tws = get_twiss_without_errors(seq_b1, just_bpms=True)

    for idx, track_df in enumerate(track_dfs):
        no_mean_file = svd_clean_measurements(track_df.reset_index())
        if idx == 0:
            no_mean_file = calc_pz(
                orig_data=no_mean_file,
                inject_noise=False,
                tws=tws,
                info=True,
                subtract_mean=False,
            )
        else:
            # Subtract the closed orbit from the mean
            no_mean_file["x"] = no_mean_file["x"] - no_mean_file["name"].map(co_x)
            no_mean_file["px"] = no_mean_file["px"] - no_mean_file["name"].map(co_px)
            no_mean_file["y"] = no_mean_file["y"] - no_mean_file["name"].map(co_y)
            no_mean_file["py"] = no_mean_file["py"] - no_mean_file["name"].map(co_py)
            no_mean_file = calculate_dispersive_pz(
                orig_data=no_mean_file,
                inject_noise=False,
                tws=tws,
                info=True,
            )

        # Plot phase space at a specific BPM for investigation
        bpm_to_plot = "BPM.9R1.B1"  # Change this to investigate different BPMs
        bpm_data = no_mean_file[no_mean_file["name"] == bpm_to_plot]

        if not bpm_data.empty:
            x_vals = bpm_data["x"].values
            px_vals = bpm_data["px"].values

            x_mean = np.mean(x_vals)
            px_mean = np.mean(px_vals)

            plt.figure(figsize=(8, 8))
            plt.scatter(x_vals, px_vals, alpha=0.5, s=10, label="Phase space points")
            plt.scatter(
                [x_mean],
                [px_mean],
                color="red",
                s=100,
                marker="x",
                linewidths=3,
                label=f"Mean ({x_mean:.2e}, {px_mean:.2e})",
            )
            plt.scatter(
                [0], [0], color="green", s=100, marker="+", linewidths=3, label="Zero point"
            )
            plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            plt.xlabel("x [m]")
            plt.ylabel("px [rad]")
            plt.title(f"Phase space at {bpm_to_plot} (lag {idx})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            print(f"Lag {idx}, BPM {bpm_to_plot}: x_mean = {x_mean:.6e}, px_mean = {px_mean:.6e}")

        # Save processed file
        output_path = (
            off_magnet_path.parent / f"{off_magnet_path.stem}_lag_{idx}{off_magnet_path.suffix}"
        )
        no_mean_file.to_parquet(output_path, index=False)
        processed_files.append(output_path)
    plt.show()

    # Create empty corrector file
    corrector_file = tmp_dir / "corrector_file.txt"
    corrector_file.write_text("")

    measurement_files = processed_files
    corrector_files = [corrector_file] * len(processed_files)
    tune_knobs_files = [tune_knobs_file] * len(processed_files)

    optimiser_config = _make_optimiser_config_bend()
    simulation_config = _make_simulation_config_bend()
    all_estimates = {}

    for arc_num in range(1, 9):
        a1, a2 = int(arc_num), int(arc_num) % 8 + 1
        magnet_range = f"BPM.13R{a1}.B1/BPM.13L{a2}.B1"
        bpm_start_points = [
            f"BPM.13R{a1}.B1",
            f"BPM.14R{a1}.B1",
            f"BPM.19R{a1}.B1",
            f"BPM.20R{a1}.B1",
        ]
        bpm_end_points = [
            f"BPM.13L{a2}.B1",
            f"BPM.14L{a2}.B1",
            f"BPM.19L{a2}.B1",
            f"BPM.20L{a2}.B1",
        ]
        num_workers = simulation_config.num_workers // (len(bpm_start_points) + len(bpm_end_points))
        sim_config = replace(
            simulation_config,
            num_workers=num_workers,
            tracks_per_worker=flattop_turns // num_workers,
        )

        sequence_config = SequenceConfig(
            sequence_file_path=seq_b1,
            magnet_range=magnet_range,
            beam_energy=6800,
            seq_name="lhcb1",
        )

        measurement_config = MeasurementConfig(
            measurement_files=measurement_files,
            corrector_files=corrector_files,
            tune_knobs_files=tune_knobs_files,
            flattop_turns=flattop_turns,
            machine_deltaps=dpp_values,
            bunches_per_file=1,
        )

        bpm_config = BPMConfig(
            start_points=bpm_start_points,
            end_points=bpm_end_points,
        )

        ctrl = Controller(
            optimiser_config=optimiser_config,
            simulation_config=sim_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_config=bpm_config,
            show_plots=False,
            true_strengths=magnet_strengths,
        )
        estimate, unc = ctrl.run()
        plt.show()

        all_estimates.update(estimate)

        normed_strengths = normalise_lhcbend_magnets(magnet_strengths)

        for magnet, value in estimate.items():
            true_value = normed_strengths[magnet]
            rel_diff = abs(value - true_value) / abs(true_value) if true_value != 0 else abs(value)
            fail_or_pass = "PASS" if rel_diff < 1e-4 else "FAIL"
            print(
                f"Magnet {magnet}: {fail_or_pass}, estimated {value}, "
                f"true {true_value}, rel diff {rel_diff}"
            )

    # Unnormalise the estimates to individual element keys
    unnormalised_estimates = {}
    for combined_key, value in all_estimates.items():
        for ind_key in magnet_strengths:
            if (
                re.sub(r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.k0", r"\1\3.k0", ind_key)
                == combined_key
            ):
                unnormalised_estimates[ind_key] = value
    all_estimates = unnormalised_estimates

    # Plot beta function errors
    tws_errs_betax = (twiss_errs["beta11"] - tws_no_err["beta11"]) / tws_no_err["beta11"]
    tws_errs_betay = (twiss_errs["beta22"] - tws_no_err["beta22"]) / tws_no_err["beta22"]

    plt.figure()
    plt.plot(twiss_errs["s"], tws_errs_betax * 100, label="BetaX error (%)")
    plt.plot(twiss_errs["s"], tws_errs_betay * 100, label="BetaY error (%)")
    plt.xlabel("s (m)")
    plt.ylabel("Relative beta error (%)")
    plt.title("Beta function errors due to magnet perturbations")
    plt.legend()
    plt.grid()

    tws_est = get_twiss_without_errors(
        seq_b1,
        just_bpms=False,
        estimated_magnets=all_estimates,
        corrector_file=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    tws_est_betax = (twiss_errs["beta11"] - tws_est["beta11"]) / tws_est["beta11"]
    tws_est_betay = (twiss_errs["beta22"] - tws_est["beta22"]) / tws_est["beta22"]
    plt.figure()
    plt.plot(tws_est["s"], tws_est_betax * 100, label="Estimated BetaX error (%)")
    plt.plot(tws_est["s"], tws_est_betay * 100, label="Estimated BetaY error (%)")
    plt.xlabel("s (m)")
    plt.ylabel("Relative beta error (%)")
    plt.title("Estimated beta function errors after bend optimisation")
    plt.legend()
    plt.grid()
    plt.show()
