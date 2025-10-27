"""Integration tests for transverse momentum reconstruction using xtrack data."""

from __future__ import annotations

import numpy as np
import pytest
import tfs

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")
pytest.importorskip("matplotlib")

from xobjects import ContextCpu  # noqa: E402

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.momentum_recon.transverse import calculate_pz
from aba_optimiser.simulation.magnet_perturbations import (
    apply_magnet_perturbations,
)
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import (
    create_xsuite_environment,
    initialise_env,
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(data_dir, sequence_file):
    json_path = data_dir / "lhcb1.json"
    env = create_xsuite_environment(
        json_file=json_path,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    baseline_line = env["lhcb1"].copy()
    tws = baseline_line.twiss(method="4d")

    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, 0.28, atol=1e-6, rtol=1e-6)
    assert np.isclose(qy, 0.31, atol=1e-6, rtol=1e-6)
    ramp_turns = 1000
    flattop_turns = 100
    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=tws,
        beam=1,
        acd_ramp=ramp_turns,
        total_turns=flattop_turns + ramp_turns,
        driven_tunes=[0.27, 0.322],
    )

    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=r"(?i)bpm.*",
        num_turns=ramp_turns + flattop_turns,
        num_particles=1,
    )

    ctx = ContextCpu()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
    )

    monitored_line.track(
        particles, num_turns=flattop_turns + ramp_turns, with_progress=False
    )

    tracking_df = line_to_dataframes(monitored_line)[0]
    # Delete the first 100 ramp turns and then reset turn count
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    # Add weights and kick plane info
    tracking_df["x_weight"] = 1.0
    tracking_df["y_weight"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    no_noise_result = calculate_pz(
        tracking_df.copy(deep=True),
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(42)
    noisy_result = calculate_pz(
        tracking_df.copy(deep=True),
        inject_noise=True,
        info=True,
        rng=rng,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(tracking_df.copy(deep=True))
    cleaned_noise_result = calculate_pz(
        cleaned_df,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_clean = _rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_clean = _rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = _rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = _rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = _rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = _rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    assert px_rmse_clean < 2e-7
    assert py_rmse_clean < 2e-7
    # Check noisy gets worse but within expected bounds
    assert px_rmse_noisy < 3e-6 and px_rmse_noisy > px_rmse_clean
    assert py_rmse_noisy < 3e-6 and py_rmse_noisy > py_rmse_clean
    # Check cleaned is better than noisy
    assert px_rmse_cleaned < px_rmse_noisy / 10
    assert py_rmse_cleaned < py_rmse_noisy / 10


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta_with_orbit_correction(
    data_dir, sequence_file, tmp_path
):
    """Test transverse momentum reconstruction with orbit correction for off-momentum beam."""

    json_path = data_dir / "lhcb1_corrected.json"
    corrector_file = tmp_path / "correctors.tfs"

    # Create MAD interface and load sequence
    mad = BaseMadInterface()
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)
    delta_p = 2e-4

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=delta_p,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )
    mad.observe_elements()
    ng_tws = mad.run_twiss(deltap=delta_p)

    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths={},  # No magnet perturbations
        corrector_table=corrector_table,
        json_file=json_path,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    baseline_line = env["lhcb1"].copy()
    tws = baseline_line.twiss(method="4d", delta0=delta_p)  # Off-momentum twiss

    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, 0.28, atol=1e-4, rtol=1e-4)
    assert np.isclose(qy, 0.31, atol=1e-4, rtol=1e-4)

    ramp_turns = 1000
    flattop_turns = 100
    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=tws,
        beam=1,
        acd_ramp=ramp_turns,
        total_turns=flattop_turns + ramp_turns,
        driven_tunes=[0.27, 0.322],
    )

    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=r"(?i)bpm.*",
        num_turns=ramp_turns + flattop_turns,
        num_particles=1,
    )

    ctx = ContextCpu()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=delta_p,  # Off-momentum particle
    )

    monitored_line.track(
        particles, num_turns=flattop_turns + ramp_turns, with_progress=False
    )

    tracking_df = line_to_dataframes(monitored_line)[0]
    # Delete the first 100 ramp turns and then reset turn count
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    # Add weights and kick plane info
    tracking_df["x_weight"] = 1.0
    tracking_df["y_weight"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "x", "y", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    no_noise_result = calculate_pz(
        tracking_df.copy(deep=True),
        tws=ng_tws,
        inject_noise=False,
        info=True,
        subtract_mean=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(42)
    noisy_result = calculate_pz(
        tracking_df.copy(deep=True),
        tws=ng_tws,
        inject_noise=True,
        info=True,
        subtract_mean=True,
        rng=rng,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(tracking_df.copy(deep=True))
    cleaned_noise_result = calculate_pz(
        cleaned_df,
        tws=ng_tws,
        inject_noise=False,
        info=True,
        subtract_mean=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_clean = _rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_clean = _rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = _rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = _rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = _rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = _rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    assert px_rmse_clean < 3e-6
    assert py_rmse_clean < 6e-7
    # Check noisy gets worse but within expected bounds
    assert px_rmse_noisy < 3.5e-6 and px_rmse_noisy > 3e-6
    assert py_rmse_noisy < 3e-6 and py_rmse_noisy > 6e-7

    # Momentum reconstruction error is dominated by the non-zero closed orbit
    # Check cleaned is better than noisy
    assert px_rmse_cleaned < px_rmse_noisy / 1.5
    assert py_rmse_cleaned < py_rmse_noisy / 4.0


@pytest.mark.slow
def test_calculate_pz_with_magnet_perturbations(data_dir, sequence_file, tmp_path):
    """Ensure transverse momentum reconstruction remains accurate after magnet perturbations."""

    json_path = data_dir / "lhcb1_corrected.json"
    corrector_file = tmp_path / "correctors_magnet.tfs"

    mad = BaseMadInterface()
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    magnet_strengths, _ = apply_magnet_perturbations(
        mad.mad, rel_k1_std_dev=1e-4, seed=123
    )
    assert magnet_strengths, "Expected magnet perturbations to update strengths"

    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=0.0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )

    mad.observe_elements()
    perturbed_tws = mad.run_twiss()

    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=json_path,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    baseline_line = env["lhcb1"].copy()
    tws = baseline_line.twiss(method="4d")

    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, 0.28, atol=5e-4, rtol=5e-4)
    assert np.isclose(qy, 0.31, atol=5e-4, rtol=5e-4)

    ramp_turns = 1000
    flattop_turns = 100
    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=tws,
        beam=1,
        acd_ramp=ramp_turns,
        total_turns=flattop_turns + ramp_turns,
        driven_tunes=[0.27, 0.322],
    )

    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=r"(?i)bpm.*",
        num_turns=ramp_turns + flattop_turns,
        num_particles=1,
    )

    ctx = ContextCpu()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
    )

    monitored_line.track(
        particles, num_turns=flattop_turns + ramp_turns, with_progress=False
    )

    tracking_df = line_to_dataframes(monitored_line)[0]
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    tracking_df["x_weight"] = 1.0
    tracking_df["y_weight"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    no_noise_result = calculate_pz(
        tracking_df.copy(deep=True),
        tws=perturbed_tws,
        inject_noise=False,
        info=True,
        subtract_mean=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(123)
    noisy_result = calculate_pz(
        tracking_df.copy(deep=True),
        tws=perturbed_tws,
        inject_noise=True,
        info=True,
        subtract_mean=True,
        rng=rng,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(tracking_df.copy(deep=True))
    cleaned_noise_result = calculate_pz(
        cleaned_df,
        tws=perturbed_tws,
        inject_noise=False,
        info=True,
        subtract_mean=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_clean = _rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_clean = _rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = _rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = _rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = _rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = _rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    assert px_rmse_clean < 3e-6
    assert py_rmse_clean < 6e-7
    assert px_rmse_noisy < 4e-6 and px_rmse_noisy > px_rmse_clean
    assert py_rmse_noisy < 3e-6 and py_rmse_noisy > py_rmse_clean
    # Check cleaned is better than noisy - Noise is dominated by non-zero closed orbit
    assert px_rmse_cleaned < px_rmse_noisy / 1.5
    assert py_rmse_cleaned < py_rmse_noisy / 4.5
