"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tfs

from aba_optimiser.config import BEND_ERROR_FILE, OptSettings
from aba_optimiser.io.utils import read_knobs, save_knobs
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.training.controller import Controller
from aba_optimiser.xsuite.xsuite_tools import (
    initialise_env,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
    run_tracking,
)

logger = logging.getLogger(__name__)

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
    "kick_plane",
)


def _generate_nonoise_track(
    tmp_dir: Path,
    sequence_file: Path,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    magnet_range: str,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    average_closed_orbit: int | bool = False,
    remove_mean: bool = False,
) -> tuple[Path, dict, Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    # Create unique corrector file path based on destination
    corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    magnet_strengths = {}
    if perturb_quads:
        mad.mad.send(f"""
local randseed, randn, abs in MAD.gmath
new_magnet_values = {{}}
for _, elm in loaded_sequence:iter('{magnet_range}') do
if elm.kind == 'quadrupole' and elm.k1 ~= 0.0 and elm.name:match("MQ%.") then
    elm.k1 = elm.k1 + 1e-4 * randn() * abs(elm.k1)
    new_magnet_values[elm.name .. ".k1"] = elm.k1
end
end
py:send(new_magnet_values, true)
        """)
        magnet_strengths = mad.mad.recv()
    if perturb_bends:
        bend_errors_table = tfs.read(BEND_ERROR_FILE)
        bend_errors_dict = bend_errors_table["K0L"].to_dict()
        for elm in mad.mad.loaded_sequence:
            # Dipoles
            if elm.kind == "sbend" and elm.k0 != 0 and elm.name[:3] == "MB.":
                if elm.name not in bend_errors_dict:
                    raise ValueError(
                        f"Bend error for {elm.name} not found in {BEND_ERROR_FILE}"
                    )
                k0l_error = bend_errors_dict[elm.name]
                elm.k0 += k0l_error / elm.l
                magnet_strengths[elm.name + ".k0"] = elm.k0

    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=dpp_value,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )
    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=tmp_dir / "env_config.json",
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    # save the tune knobs to file with unique name
    tune_knobs_file = destination.parent / f"tune_knobs_{destination.stem}.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    insert_particle_monitors_at_pattern(
        env["lhcb1"],
        pattern="bpm.*[^k]",
        num_turns=flattop_turns,
        num_particles=1,
        inplace=True,
    )
    particles = env["lhcb1"].build_particles(
        x=[1e-4],
        px=[-1e-6],
        y=[1e-4],
        py=[-1e-6],
        delta=[dpp_value],
    )
    run_tracking(
        line=env["lhcb1"],
        particles=particles,
        nturns=flattop_turns,
    )
    true_dfs = line_to_dataframes(env["lhcb1"])

    df = prepare_track_dataframe(true_dfs[0], 0, flattop_turns, kick_both_planes=True)
    df = df.loc[:, TRACK_COLUMNS].copy()
    df["name"] = df["name"].astype(str)
    df["kick_plane"] = df["kick_plane"].astype(str)

    if average_closed_orbit:
        # Compute averages per BPM
        averaged = (
            df.groupby("name")[
                ["x", "px", "y", "py", "var_x", "var_y", "var_px", "var_py"]
            ]
            .mean()
            .reset_index()
        )
        # Create new DataFrame with 3 turns, each with averaged values
        new_rows = []
        for turn in [1, 2, 3]:
            for _, row in averaged.iterrows():
                new_rows.append(
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
                        "kick_plane": "xy",
                    }
                )
        df = pd.DataFrame(new_rows)
        df["name"] = df["name"].astype("category")
        df["turn"] = df["turn"].astype("int32")

    if remove_mean:
        # Remove mean from coordinates for each BPM
        for coord in ["x", "px"]:
            df[coord] = df.groupby("name")[coord].transform(lambda x: x - x.mean())

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)
    return corrector_file, magnet_strengths, tune_knobs_file


@pytest.fixture(scope="module")
def dpp_value() -> float:
    return 1.25e-4


@pytest.fixture(scope="module")
def flattop_turns() -> int:
    return 256


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_tracks")


def _make_opt_settings_energy() -> OptSettings:
    return OptSettings(
        max_epochs=1000,
        tracks_per_worker=1,
        num_workers=3,
        num_batches=10,
        warmup_epochs=1,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-7,
        gradient_converged_value=5e-10,
        optimise_energy=True,
        optimise_quadrupoles=False,
        optimise_bends=False,
    )


def _make_opt_settings_quad() -> OptSettings:
    return OptSettings(
        max_epochs=300,
        tracks_per_worker=10,
        num_workers=8,
        num_batches=2,
        warmup_epochs=50,
        warmup_lr_start=1e-9,
        max_lr=1e-7,
        min_lr=1e-7,
        gradient_converged_value=5e-14,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
    )


def _make_opt_settings_bend() -> OptSettings:
    return OptSettings(
        max_epochs=3000,
        tracks_per_worker=100,
        num_batches=10,
        num_workers=1,
        warmup_epochs=3,
        warmup_lr_start=5e-10,
        max_lr=2e-8,
        min_lr=2e-8,
        gradient_converged_value=1e-6,
        optimiser_type="adam",
        optimise_energy=False,
        optimise_quadrupoles=False,
        optimise_bends=True,
    )


@pytest.mark.slow
def test_controller_energy_opt(
    tmp_dir: Path,
    flattop_turns: int,
    sequence_file: Path,
    dpp_value: float,
) -> None:
    """Test that the controller initializes correctly with custom num_tracks and flattop_turns."""
    opt_settings = _make_opt_settings_energy()

    off_dpp_path = tmp_dir / "track_off_dpp.parquet"
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"

    corrector_file, _, tune_knobs_file = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_dpp_path,
        dpp_value,
        magnet_range,
        perturb_quads=False,
    )

    # Constants for the test
    bpm_start_points = [
        "BPM.9R2.B1",
        "BPM.10R2.B1",
        "BPM.11R2.B1",
    ]
    bpm_end_points = [
        "BPM.9L3.B1",
        "BPM.10L3.B1",
        "BPM.11L3.B1",
    ]

    ctrl = Controller(
        opt_settings=opt_settings,
        sequence_file_path=sequence_file,
        show_plots=False,
        magnet_range=magnet_range,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        measurement_files=off_dpp_path,
        true_strengths=None,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        num_tracks=1,
    )

    estimate, unc = ctrl.run()  # Ensure that run works without errors

    assert np.allclose(estimate.pop("deltap"), dpp_value, rtol=1e-4, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert uncertainty < 1e-6 and uncertainty > 0

    # check that estimate and unc are now empty
    assert not estimate
    assert not unc


@pytest.mark.slow
def test_controller_quad_opt_simple(tmp_dir: Path, sequence_file: Path) -> None:
    """Test quadrupole optimisation using the simple opt script logic."""
    # Constants for the test
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"
    bpm_start_points = [
        "BPM.9R2.B1",
        # "BPM.10R2.B1",
    ]
    bpm_end_points = [
        "BPM.9L3.B1",
        # "BPM.10L3.B1",
    ]

    flattop_turns = 1000
    off_magnet_path = tmp_dir / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_magnet_path,
        0.0,
        magnet_range,
        perturb_quads=True,
    )

    opt_settings = _make_opt_settings_quad()
    true_values = magnet_strengths.copy()

    ctrl = Controller(
        opt_settings=opt_settings,
        sequence_file_path=sequence_file,
        show_plots=False,
        magnet_range=magnet_range,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        measurement_files=off_magnet_path,
        true_strengths=true_values,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        num_tracks=1,
    )
    estimate, unc = ctrl.run()
    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - true_values[magnet]) / abs(true_values[magnet])
            if true_values[magnet] != 0
            else abs(value)
        )
        assert rel_diff < 1e-8, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {true_values[magnet]}, rel diff {rel_diff}"
        )


@pytest.mark.slow
def test_controller_quad_opt_all_arcs_beta_beating(
    tmp_dir: Path, sequence_file: Path
) -> None:
    """
    Test quadrupole optimisation across all 8 arcs.

    1. Creates a perturbed MAD instance with errors in all arcs
    2. Measures beta beating: perturbed vs unperturbed (BEFORE)
    3. Optimizes quadrupoles in each arc sequentially
    4. Applies all optimized values to a new MAD instance
    5. Measures beta beating: optimized vs unperturbed (AFTER - should approach 0)

    Uses magnet ranges from create_datafile_loop.py for beam 1.
    """
    # Define magnet ranges for all 8 arcs (beam 1)
    magnet_ranges_b1 = [f"BPM.9R{s}.B1/BPM.9L{s % 8 + 1}.B1" for s in range(1, 9)]

    # Generate BPM start/end points for each arc
    bpm_starts_b1 = [[f"BPM.{i}R{s}.B1" for i in [9]] for s in range(1, 9)]
    bpm_end_points_b1 = [[f"BPM.{i}L{s % 8 + 1}.B1" for i in [9]] for s in range(1, 9)]

    flattop_turns = 1000
    opt_settings = OptSettings(
        max_epochs=3000,
        tracks_per_worker=1,
        num_workers=4,
        num_batches=1,
        warmup_epochs=50,
        warmup_lr_start=1e-9,
        max_lr=1e-6,
        min_lr=1e-6,
        gradient_converged_value=1e-11,
        optimise_energy=True,
        optimise_quadrupoles=False,
        optimise_bends=False,
    )

    # Get initial reference twiss (unperturbed)
    logger.info("Creating unperturbed reference MAD")
    mad_reference = BaseMadInterface()
    mad_reference.load_sequence(sequence_file, "lhcb1")
    mad_reference.setup_beam(beam_energy=6800)
    mad_reference.observe_elements()
    reference_tws = mad_reference.run_twiss()

    # Create perturbed MAD instance with errors across the entire ring
    logger.info("Creating perturbed MAD with quadrupole errors across entire ring")
    mad_perturbed = BaseMadInterface()
    mad_perturbed.load_sequence(sequence_file, "lhcb1")
    mad_perturbed.setup_beam(beam_energy=6800)
    mad_perturbed.observe_elements()

    # Generate tracking data with perturbations across the entire ring
    logger.info(
        "Generating tracking data with perturbed quadrupoles across entire ring"
    )
    off_magnet_path = tmp_dir / "track_full_ring_perturbed.parquet"
    deltap = 1e-4
    corrector_file, all_perturbed_strengths, tune_knobs_file = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_magnet_path,
        deltap,
        "$start/$end",  # Entire ring
        perturb_quads=True,
        perturb_bends=True,
        remove_mean=False,
    )
    logger.info(
        f"Perturbed {len(all_perturbed_strengths)} quadrupoles across entire ring"
    )
    empty_corrector_file = tmp_dir / "empty_correctors.txt"
    empty_corrector_file.touch()  # Create an empty corrector file

    # Calculate beta beating BEFORE optimization (perturbed vs reference)
    logger.info("Calculating beta beating BEFORE optimization")
    mad_perturbed.set_magnet_strengths(all_perturbed_strengths)
    perturbed_tws = mad_perturbed.run_twiss(deltap=deltap)
    beta11_beat_before = (
        perturbed_tws["beta11"] - reference_tws["beta11"]
    ) / reference_tws["beta11"]
    beta22_beat_before = (
        perturbed_tws["beta22"] - reference_tws["beta22"]
    ) / reference_tws["beta22"]
    rms_before = np.sqrt((beta11_beat_before**2 + beta22_beat_before**2).mean())

    logger.info(f"RMS beta beating BEFORE optimization: {rms_before * 100:.4f}%")
    logger.info(
        f"Beta11 beating: mean={beta11_beat_before.mean() * 100:.4f}%, std={beta11_beat_before.std() * 100:.4f}%"
    )
    logger.info(
        f"Beta22 beating: mean={beta22_beat_before.mean() * 100:.4f}%, std={beta22_beat_before.std() * 100:.4f}%"
    )

    # Now optimize each arc and collect optimized strengths
    logger.info("\n=== Starting optimization for all arcs ===")
    all_optimized_strengths = {}
    perturbed_energy = {"deltap": deltap, **all_perturbed_strengths}
    optimised_deltaps = []

    for arc_idx in range(8):
        logger.info(f"\nOptimizing arc {arc_idx + 1}/8")

        magnet_range = magnet_ranges_b1[arc_idx]
        bpm_start_points = bpm_starts_b1[arc_idx]
        bpm_end_points = bpm_end_points_b1[arc_idx]

        logger.info(f"Arc {arc_idx + 1}")

        # Run optimization using the shared tracking data
        ctrl = Controller(
            opt_settings=opt_settings,
            sequence_file_path=sequence_file,
            show_plots=False,
            magnet_range=magnet_range,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            measurement_files=off_magnet_path,
            true_strengths=perturbed_energy.copy(),
            # corrector_files=corrector_file,
            corrector_files=empty_corrector_file,
            tune_knobs_files=tune_knobs_file,
            flattop_turns=flattop_turns,
            num_tracks=1,
        )
        estimate_energy, unc = ctrl.run()
        optimised_deltaps.append(estimate_energy["deltap"])
        # del ctrl
        quad_opt = replace(
            opt_settings,
            optimise_energy=True,
            optimise_quadrupoles=True,
            max_epochs=2000,
            max_lr=1e-7,
        )
        ctrl = Controller(
            opt_settings=quad_opt,
            initial_knob_strengths=estimate_energy,
            sequence_file_path=sequence_file,
            show_plots=False,
            magnet_range=magnet_range,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            measurement_files=off_magnet_path,
            true_strengths=perturbed_energy.copy(),
            # corrector_files=corrector_file,
            corrector_files=empty_corrector_file,
            tune_knobs_files=tune_knobs_file,
            flattop_turns=flattop_turns,
            num_tracks=1,
        )
        estimate, unc = ctrl.run()
        plt.close("all")  # Close any plots to save memory

        # Store optimized strengths
        all_optimized_strengths.update(estimate)
        logger.info(f"Arc {arc_idx + 1} optimized: {len(estimate)} quadrupoles")

    # Create optimized MAD instance with all optimized values
    logger.info(
        "\n=== Creating optimized MAD instance with all optimized quadrupole values ==="
    )
    mad_optimized = BaseMadInterface()
    mad_optimized.load_sequence(sequence_file, "lhcb1")
    mad_optimized.setup_beam(beam_energy=6800)
    mad_optimized.observe_elements()

    # Apply all optimized strengths
    optimised_deltap = np.mean(optimised_deltaps)
    all_optimized_strengths.pop("deltap", None)  # Remove deltap if present

    mad_optimized.set_magnet_strengths(all_optimized_strengths)
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]
    mad_optimized.apply_corrector_strengths(corrector_table)
    tune_knobs = read_knobs(tune_knobs_file)
    mad_optimized.set_madx_variables(**tune_knobs)

    # Calculate beta beating AFTER optimization (optimized vs reference)
    logger.info("Calculating beta beating AFTER optimization")
    optimised_tws = mad_optimized.run_twiss(deltap=optimised_deltap)
    beta11_beat_after = (
        perturbed_tws["beta11"] - optimised_tws["beta11"]
    ) / optimised_tws["beta11"]
    beta22_beat_after = (
        perturbed_tws["beta22"] - optimised_tws["beta22"]
    ) / optimised_tws["beta22"]
    rms_after = np.sqrt((beta11_beat_after**2 + beta22_beat_after**2).mean())

    logger.info(f"RMS beta beating AFTER optimization: {rms_after * 100:.4f}%")
    logger.info(
        f"Beta11 beating: mean={beta11_beat_after.mean() * 100:.4f}%, std={beta11_beat_after.std() * 100:.4f}%"
    )
    logger.info(
        f"Beta22 beating: mean={beta22_beat_after.mean() * 100:.4f}%, std={beta22_beat_after.std() * 100:.4f}%"
    )

    # Summary
    logger.info("\n=== Beta Beating Summary ===")
    logger.info(f"RMS BEFORE: {rms_before * 100:.4f}%")
    logger.info(f"RMS AFTER:  {rms_after * 100:.4f}%")
    logger.info(f"Improvement: {(rms_before - rms_after) / rms_before * 100:.2f}%")
    logger.info(f"Reduction factor: {rms_before / rms_after:.2f}x")

    # Plot beta beating around the ring
    logger.info("\n=== Creating beta beating plots ===")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    s_position = reference_tws["s"]

    # Plot horizontal beta beating
    ax1.plot(
        s_position,
        beta11_beat_before * 100,
        "r-",
        label="Before optimization",
        alpha=0.7,
    )
    ax1.plot(
        s_position, beta11_beat_after * 100, "b-", label="After optimization", alpha=0.7
    )
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("s [m]")
    ax1.set_ylabel(r"$\Delta\beta_x/\beta_x$ [%]")
    ax1.set_title("Horizontal Beta Beating Around the Ring")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot vertical beta beating
    ax2.plot(
        s_position,
        beta22_beat_before * 100,
        "r-",
        label="Before optimization",
        alpha=0.7,
    )
    ax2.plot(
        s_position, beta22_beat_after * 100, "b-", label="After optimization", alpha=0.7
    )
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("s [m]")
    ax2.set_ylabel(r"$\Delta\beta_y/\beta_y$ [%]")
    ax2.set_title("Vertical Beta Beating Around the Ring")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = tmp_dir / "beta_beating_comparison.png"
    # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Beta beating plot saved to {plot_path}")
    plt.show()

    # Assertions
    assert rms_after < rms_before, (
        f"Beta beating did not improve! Before: {rms_before * 100:.4f}%, After: {rms_after * 100:.4f}%"
    )

    # Check that we're approaching zero (at least 50% improvement)
    improvement_ratio = (rms_before - rms_after) / rms_before
    assert improvement_ratio > 0.5, (
        f"Insufficient improvement: only {improvement_ratio * 100:.2f}% (expected > 50%)"
    )

    logger.info("Test passed: Beta beating improved significantly after optimization!")
