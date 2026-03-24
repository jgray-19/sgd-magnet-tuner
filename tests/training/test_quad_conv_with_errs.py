"""
Integration test for quadrupole convergence with errors using AC dipole excitation.
"""

from __future__ import annotations

import json
import logging

# import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")

import os

import tfs
from pymadng_utils.io.utils import save_knobs
from tmom_recon import (
    calculate_dispersive_pz,
    calculate_transverse_pz,
)
from tmom_recon.svd import svd_clean_measurements
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.monitors import line_to_dataframes

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.noise import apply_bpm_noise
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import (
    MeasurementConfig,
    OutputConfig,
    SequenceConfig,
)
from tests.training.helpers import (
    TRACK_COLUMNS,
    generate_xsuite_env_with_errors,
    get_twiss_without_errors,
)

if TYPE_CHECKING:
    from pathlib import Path

    from xtrack import xt

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

logger = logging.getLogger(__name__)


def _should_plot() -> bool:
    """Determine whether to generate plots based on environment variable."""
    return os.getenv("PLOT_TEST_OUTPUT", "0") == "1"


def _run_track_with_acd(
    env: xt.Environment,
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
    # AC dipole parameters
    driven_tunes = [0.27, 0.322]

    processed_dfs = []
    output_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    line: xt.Line = env["lhcb1"]
    tws = line.twiss(method="4d", delta0=0)

    for idx, lag in enumerate(lags):
        # Run tracking with AC dipole using the generalized function
        particle_coords = {
            "x": [0.0],
            "px": [0.0],
            "y": [0.0],
            "py": [0.0],
            "delta": [dpp_values[idx]],
        }

        monitored_line = run_ac_dipole_tracking_with_particles(
            line=line,
            tws=tws,
            beam=1,
            ramp_turns=acd_ramp,
            flattop_turns=flattop_turns,
            driven_tunes=driven_tunes,
            lag=lag,
            bpm_pattern="bpm.*[^k]",
            particle_coords=particle_coords,
        )

        # Get data from line
        true_dfs = line_to_dataframes(monitored_line)

        # Process dataframe - only keep flattop turns (after ramp)
        for true_df in true_dfs:
            # Filter to only include turns after the ramp
            df = true_df[true_df["turn"] > acd_ramp].copy()
            # Renumber turns to start from 0
            df["turn"] = df["turn"] - acd_ramp

            # Remove the bpmcs\. bpms which are not seen when measuring
            df = df[~df["name"].str.contains("bpmcs\\.", case=False, regex=True)].copy()

            df = prepare_track_dataframe(df, 0, flattop_turns)
            df = df.loc[:, TRACK_COLUMNS].copy()
            df["name"] = df["name"].astype(str)

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
        max_epochs=50,
        warmup_epochs=20,
        warmup_lr_start=4e-6,
        max_lr=4e-7,
        min_lr=4e-7,
        gradient_converged_value=1e-10,
        optimiser_type="adam",
        expected_rel_error=1e-3,
    )


# Skip if with can't have 30 processes which is needed for the parallelism in this test
@pytest.mark.skipif(
    os.cpu_count() is not None and (os.cpu_count() < 30),  # ty:ignore[unsupported-operator]
    reason="Requires at least 30 CPU cores for parallel processing",
)
@pytest.mark.slow
def test_controller_bend_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    estimated_strengths_file: Path,
    loaded_interface: AbaMadInterface,
) -> None:
    """Test bend optimisation using AC dipole excitation with different lag values."""
    flattop_turns = 400
    turns_per_batch = 50  # 134  # This number works quite well independently of flattop_turns
    acd_ramp = 2_000  # Ramp turns for AC dipole - should be long enough to avoid emittance growth

    off_magnet_path = tmp_path / "track_off_magnet.parquet"
    corrector_file = tmp_path / "corrector_track_off_magnet.tfs"
    tune_knobs_file = tmp_path / "tune_knobs_track_off_magnet.json"

    # Generate model with errors for all arcs
    env, magnet_strengths, matched_tunes, _ = generate_xsuite_env_with_errors(
        loaded_interface,
        dpp_value=0,
        corrector_file=corrector_file,
        perturb_quads=True,
        perturb_bends=True,
    )
    twiss_errs = loaded_interface.run_twiss(observe=0)  # Observe all elements
    save_knobs(matched_tunes, tune_knobs_file)

    # Get clean twiss for pz calculation
    tws_no_err = get_twiss_without_errors(seq_b1, just_bpms=False)

    # Generate tracks with AC dipole using 3 different lag values
    # dpp_values = [0.0, 4e-4, -4e-4, 8e-4, -8e-4]  # , 12e-4, -12e-4]
    dpp_values = [0.0, 0.0, 0.0]
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
    model_co_x = tws_no_err["x"].to_dict()
    model_co_y = tws_no_err["y"].to_dict()
    model_co_px = tws_no_err["px"].to_dict()
    model_co_py = tws_no_err["py"].to_dict()

    # Process each track dataframe and save
    processed_files = []
    tws = get_twiss_without_errors(seq_b1, just_bpms=True)

    for idx, track_df in enumerate(track_dfs):
        # Add noise to the tracking data and apply SVD cleaning
        track_df_noisy = track_df.copy(deep=True)
        apply_bpm_noise(
            track_df_noisy.set_index("name"),
            np.random.default_rng(42 + idx),
            accelerator_type="lhc",
        )
        cleaned = svd_clean_measurements(track_df_noisy.reset_index())
        cleaned["name"] = cleaned["name"].str.upper()
        # Convert BPM names to uppercase to match the closed orbit dictionary
        if idx == 0:
            cleaned = calculate_transverse_pz(
                orig_data=cleaned,
                inject_noise=False,
                tws=tws,
                info=True,
            )
            # Get the mean x, px, y, py for closed orbit subtraction
            co_x = (cleaned.groupby("name")["x"].mean()).to_dict()
            co_px = (cleaned.groupby("name")["px"].mean()).to_dict()
            co_y = (cleaned.groupby("name")["y"].mean()).to_dict()
            co_py = (cleaned.groupby("name")["py"].mean()).to_dict()
        else:
            # Subtract the closed orbit from the mean
            cleaned = calculate_dispersive_pz(
                orig_data=cleaned,
                inject_noise=False,
                tws=tws,
                info=True,
            )
        no_mean_file = cleaned.copy(deep=True)
        # Remove the closed orbit mean from the data, so that includes orbit bumps and different dipolar errors
        # Then I add back the model closed orbit to have the orbit bumps matched to the model
        no_mean_file["x"] = (
            cleaned["x"] - cleaned["name"].map(co_x) + cleaned["name"].map(model_co_x)
        )
        no_mean_file["px"] = (
            cleaned["px"] - cleaned["name"].map(co_px) + cleaned["name"].map(model_co_px)
        )
        no_mean_file["y"] = (
            cleaned["y"] - cleaned["name"].map(co_y) + cleaned["name"].map(model_co_y)
        )
        no_mean_file["py"] = (
            cleaned["py"] - cleaned["name"].map(co_py) + cleaned["name"].map(model_co_py)
        )
        assert no_mean_file["var_x"].notna().all()
        assert no_mean_file["var_y"].notna().all()
        del cleaned

        # Plot phase space at a specific BPM for investigation
        bpm_to_plot = "BPM.9R1.B1"  # Change this to investigate different BPMs
        bpm_data = no_mean_file[no_mean_file["name"] == bpm_to_plot]

        if not bpm_data.empty:
            x_vals = bpm_data["x"].values
            px_vals = bpm_data["px"].values

            x_mean = np.mean(x_vals)
            px_mean = np.mean(px_vals)

            logger.info(
                f"Lag {idx}, BPM {bpm_to_plot}: x_mean = {x_mean:.6e}, px_mean = {px_mean:.6e}"
            )

        # Save processed file
        output_path = (
            off_magnet_path.parent / f"{off_magnet_path.stem}_lag_{idx}{off_magnet_path.suffix}"
        )
        no_mean_file.to_parquet(output_path, index=False)
        processed_files.append(output_path)
        # Print the min and max turn numbers to verify the expected number of turns
        logger.info(
            f"Processed track with lag {lags[idx]:.2f}: saved to {output_path}, "
            f"turns range: {no_mean_file['turn'].min()} to {no_mean_file['turn'].max()} (expected {flattop_turns} turns)"
        )

    # Create empty corrector file
    corrector_file = tmp_path / "corrector_file.txt"
    corrector_file.write_text("")

    measurement_files = processed_files
    corrector_files = [corrector_file] * len(processed_files)
    tune_knobs_files = [tune_knobs_file] * len(processed_files)

    optimiser_config = _make_optimiser_config_bend()
    all_estimates = {}

    def _run_optimisation_for_range(
        magnet_range: str,
        start_points: list[str],
        end_points: list[str],
        optimise_quadrupoles: bool,
        optimise_other_quadrupoles: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        num_workers = 60 // (len(start_points) + len(end_points))
        tracks_per_worker = flattop_turns - 3
        num_batches = int(np.ceil(tracks_per_worker / turns_per_batch))

        lhc_accelerator = LHC(
            beam=1,
            sequence_file=seq_b1,
            optimise_correctors=False,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_other_quadrupoles=optimise_other_quadrupoles,
        )

        sim_config = SimulationConfig(
            tracks_per_worker=tracks_per_worker,
            num_batches=num_batches,
            num_workers=num_workers,
            optimise_momenta=False,
            use_fixed_bpm=True,
        )

        sequence_config = SequenceConfig(
            magnet_range=magnet_range,
            first_bpm="MSIA.EXIT.B1",
        )

        measurement_config = MeasurementConfig(
            measurement_files=measurement_files,
            corrector_files=corrector_files,
            tune_knobs_files=tune_knobs_files,
            flattop_turns=flattop_turns,
            machine_deltaps=dpp_values,
            bunches_per_file=1,
        )

        plots_dir_name = magnet_range.replace("/", "_").replace(".", "_")
        plots_dir = tmp_path / f"plots_{plots_dir_name}"

        ctrl = Controller(
            accelerator=lhc_accelerator,
            optimiser_config=optimiser_config,
            simulation_config=sim_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=start_points,
            bpm_end_points=end_points,
            output_config=OutputConfig(
                show_plots=False,
                plots_dir=plots_dir,
            ),
            true_strengths=magnet_strengths,
        )
        return ctrl.run()

    # for ip_num in range(1, 9):
    #     optimise_quadrupoles = ip_num in [3, 7]
    #     magnet_range = f"BPM.13L{ip_num}.B1/BPM.13R{ip_num}.B1"
    #     bpm_start_points = [f"BPM.{s}L{ip_num}.B1" for s in range(13, 8, -1)]
    #     bpm_end_points = [f"BPM.{s}R{ip_num}.B1" for s in range(13, 8, -1)]
    #     print(f"\nOptimising IP {ip_num} with magnets in range {magnet_range}")
    #     print(f"  BPM start points: {bpm_start_points}")
    #     print(f"  BPM end points: {bpm_end_points}")
    #     estimate, unc = _run_optimisation_for_range(
    #         magnet_range=magnet_range,
    #         start_points=bpm_start_points,
    #         end_points=bpm_end_points,
    #         optimise_quadrupoles=optimise_quadrupoles,
    #         optimise_other_quadrupoles=True,
    #     )

    #     all_estimates.update(estimate)
    # plt.close("all")

    for arc_num in range(1, 9):
        a1, a2 = int(arc_num), int(arc_num) % 8 + 1
        magnet_range = f"BPM.13R{a1}.B1/BPM.13L{a2}.B1"
        bpm_start_points = [f"BPM.{s}R{a1}.B1" for s in range(13, 30, 3)] + [
            f"BPM.{s}R{a1}.B1" for s in range(14, 30, 3)
        ]
        bpm_end_points = [f"BPM.{s}L{a2}.B1" for s in range(13, 30, 3)] + [
            f"BPM.{s}L{a2}.B1" for s in range(14, 30, 3)
        ]
        print(f"\nOptimising arc {arc_num} with magnets in range {magnet_range}")
        print(f"  BPM start points: {bpm_start_points}")
        print(f"  BPM end points: {bpm_end_points}")
        estimate, unc = _run_optimisation_for_range(
            magnet_range=magnet_range,
            start_points=bpm_start_points,
            end_points=bpm_end_points,
            optimise_quadrupoles=True,
            optimise_other_quadrupoles=False,
        )

        all_estimates.update(estimate)
    plt.close("all")

    # for magnet, value in all_estimates.items():
    #     true_value = magnet_strengths[magnet]
    #     rel_diff = abs(value - true_value) / abs(true_value) if true_value != 0 else abs(value)
    #     fail_or_pass = "PASS" if rel_diff < 1e-4 else "FAIL"
    #     print(
    #         f"Magnet {magnet}: {fail_or_pass}, estimated {value}, "
    #         f"true {true_value}, rel diff {rel_diff}"
    #     )

    # Save estimates to file
    with estimated_strengths_file.open("w") as f:
        json.dump(all_estimates, f)

    # Plot beta function errors
    # fmt: off
    tws_errs_betax = (twiss_errs.loc[:, "beta11"] - tws_no_err.loc[:, "beta11"]) / tws_no_err.loc[:, "beta11"]
    tws_errs_betay = (twiss_errs.loc[:, "beta22"] - tws_no_err.loc[:, "beta22"]) / tws_no_err.loc[:, "beta22"]

    # Save beta beating data before corrections
    beta_beating_before = pd.DataFrame(
        {
            "s": twiss_errs.loc[:, "s"],
            "name": twiss_errs.index,
            "betax_error_percent": tws_errs_betax * 100,
            "betay_error_percent": tws_errs_betay * 100,
        }
    )
    beta_beating_before_file = tmp_path / "beta_beating_before_correction.tfs"
    tfs.write(beta_beating_before_file, beta_beating_before)

    tws_est = get_twiss_without_errors(
        seq_b1,
        just_bpms=False,
        estimated_magnets=all_estimates,
        corrector_file=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    tws_est_betax = (twiss_errs.loc[:, "beta11"] - tws_est.loc[:, "beta11"]) / tws_est.loc[:, "beta11"]
    tws_est_betay = (twiss_errs.loc[:, "beta22"] - tws_est.loc[:, "beta22"]) / tws_est.loc[:, "beta22"]
    # fmt: on

    # Save beta beating data after corrections
    beta_beating_after = pd.DataFrame(
        {
            "s": tws_est["s"],
            "name": tws_est.index,
            "betax_error_percent": tws_est_betax * 100,
            "betay_error_percent": tws_est_betay * 100,
        }
    )
    beta_beating_after_file = tmp_path / "beta_beating_after_correction.tfs"
    tfs.write(beta_beating_after_file, beta_beating_after)

    if _should_plot():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(tws_errs_betax * 100, label="BetaX error before correction")
        plt.plot(tws_est_betax * 100, label="BetaX error after correction")
        plt.xlabel("Element index")
        plt.ylabel("BetaX error (%)")
        plt.title("BetaX beating before and after correction")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(tws_errs_betay * 100, label="BetaY error before correction")
        plt.plot(tws_est_betay * 100, label="BetaY error after correction")
        plt.xlabel("Element index")
        plt.ylabel("BetaY error (%)")
        plt.title("BetaY beating before and after correction")
        plt.legend()
        plt.grid()

        plot_file = tmp_path / "beta_beating_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.show()
        print(f"Saved beta beating comparison plot to {plot_file}")

    assert all(tws_est_betax.abs() < 0.0025), "BetaX errors exceed 0.25% after optimisation"
    assert all(tws_est_betay.abs() < 0.005), "BetaY errors exceed 0.5% after optimisation"
    # Check that the original beta beating was larger than 1%
    assert any(tws_errs_betax.abs() > 0.005), "Original BetaX errors were not larger than 0.5%"
    assert any(tws_errs_betay.abs() > 0.01), "Original BetaY errors were not larger than 1%"
