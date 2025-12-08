"""Shared utilities for momentum reconstruction integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xobjects import ContextCpu

from aba_optimiser.filtering.svd import svd_clean_measurements
from aba_optimiser.momentum_recon import inject_noise_xy
from aba_optimiser.xsuite.xsuite_tools import (
    create_xsuite_environment,
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
)

if TYPE_CHECKING:
    import tfs

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.nanmean((predicted - actual) ** 2)))


def setup_ac_dipole_tracking(
    json_path,
    sequence_file,
    delta_p: float = 0.0,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    create_env: bool = True,
):
    """Set up xsuite tracking with AC dipole and return tracking data + twiss.

    Parameters
    ----------
    json_path : Path
        Path to the xsuite JSON file.
    sequence_file : Path
        Path to the MAD-X sequence file.
    delta_p : float
        Momentum offset (δp/p).
    ramp_turns : int
        Number of AC dipole ramp turns.
    flattop_turns : int
        Number of flattop turns to use for reconstruction.
    create_env : bool
        If True, create environment from JSON. If False, environment should be
        provided separately (for custom initialization).

    Returns
    -------
    tracking_df : pd.DataFrame
        DataFrame with tracking data (x, y, px, py, etc.).
    tws : tfs.TfsDataFrame
        Twiss parameters.
    baseline_line : xtrack.Line
        The baseline accelerator line.
    """
    if create_env:
        env = create_xsuite_environment(
            json_file=json_path,
            sequence_file=sequence_file,
            seq_name="lhcb1",
        )
        baseline_line = env["lhcb1"].copy()
        tws_input = baseline_line.twiss(method="4d", delta0=delta_p)
    else:
        # This branch is used when environment is already set up elsewhere
        # (e.g., in transverse_momentum tests with custom initialization)
        raise ValueError("Must provide baseline_line and tws when create_env=False")

    # Verify tunes are as expected
    qx = float(tws_input.qx % 1)
    qy = float(tws_input.qy % 1)
    assert np.isclose(qx, 0.28, atol=1e-3), f"Qx = {qx}, expected 0.28"
    assert np.isclose(qy, 0.31, atol=1e-3), f"Qy = {qy}, expected 0.31"

    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=tws_input,
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
        delta=delta_p,
    )

    monitored_line.track(
        particles, num_turns=flattop_turns + ramp_turns, with_progress=False
    )

    tracking_df = _process_tracking_data(
        monitored_line, ramp_turns, flattop_turns, delta_p
    )

    return tracking_df, tws_input, baseline_line


def process_tracking_data(
    monitored_line,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    delta_p: float = 0.0,
):
    """Process raw tracking data from xsuite monitoring.

    Parameters
    ----------
    monitored_line : xtrack.Line
        The monitored accelerator line after tracking.
    ramp_turns : int
        Number of AC dipole ramp turns to skip.
    flattop_turns : int
        Number of flattop turns (not used for processing, informational).
    delta_p : float
        Momentum offset (δp/p) - added to dataframe but not used here.

    Returns
    -------
    tracking_df : pd.DataFrame
        Processed tracking data with normalized turns and added columns.
    """
    return _process_tracking_data(monitored_line, ramp_turns, flattop_turns, delta_p)


def _process_tracking_data(
    monitored_line,
    ramp_turns: int,
    flattop_turns: int,
    delta_p: float,
):
    """Internal helper to process tracking data."""
    tracking_df = line_to_dataframes(monitored_line)[0]
    # Remove ramp turns and reset turn count
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    # Add required columns
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0
    tracking_df["var_px"] = 1.0
    tracking_df["var_py"] = 1.0
    tracking_df["kick_plane"] = "both"

    return tracking_df


def get_truth_and_twiss(
    baseline_line,
    tracking_df,
):
    """Extract truth momenta and prepare twiss from baseline line.

    Parameters
    ----------
    baseline_line : xtrack.Line
        The baseline accelerator line.
    tracking_df : pd.DataFrame
        The tracking DataFrame containing actual (true) momenta.

    Returns
    -------
    truth : pd.DataFrame
        DataFrame with true momenta (px_true, py_true).
    tws : tfs.TfsDataFrame
        Twiss parameters indexed by BPM names.
    """
    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    # Convert twiss to expected format
    ng = baseline_line.to_madng()
    ng["tws", "flw"] = ng.twiss(sequence=ng.seq)
    tws: tfs.TfsDataFrame = (
        ng.tws.to_df()
        .set_index("name")
        .rename(index=str.upper)
        .loc[lambda df: df.index.str.contains("BPM")]
    )

    return truth, tws


def verify_pz_reconstruction(
    tracking_df,
    truth,
    tws,
    calculate_pz_func,
    px_clean_max,
    py_clean_max,
    px_noisy_min,
    px_noisy_max,
    py_noisy_min,
    py_noisy_max,
    px_divisor,
    py_divisor,
    rng_seed=42,
    subtract_mean=True,
):
    """Verify momentum reconstruction with noise and SVD cleaning.

    Tests three scenarios: clean data, noisy data, and SVD-cleaned data.
    Verifies that: (1) clean reconstruction meets accuracy thresholds,
    (2) noisy reconstruction degrades in expected range, and
    (3) SVD cleaning significantly improves reconstruction.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        The tracking data containing measurements.
    truth : pd.DataFrame
        The true momentum values (px_true, py_true).
    tws : tfs.TfsDataFrame
        Twiss parameters.
    calculate_pz_func : callable
        Function to calculate momentum (e.g., calculate_pz or calculate_transverse_pz).
    px_clean_max : float
        Maximum acceptable RMSE for clean px reconstruction.
    py_clean_max : float
        Maximum acceptable RMSE for clean py reconstruction.
    px_noisy_min : float or str
        Minimum expected RMSE for noisy px (or "px_rmse_clean" to use clean value).
    px_noisy_max : float
        Maximum acceptable RMSE for noisy px.
    py_noisy_min : float or str
        Minimum expected RMSE for noisy py (or "py_rmse_clean" to use clean value).
    py_noisy_max : float
        Maximum acceptable RMSE for noisy py.
    px_divisor : float
        Divisor to verify SVD improvement for px.
    py_divisor : float
        Divisor to verify SVD improvement for py.
    rng_seed : int
        Random seed for noise generation.
    subtract_mean : bool
        Whether to subtract mean in the calculation function.
    """
    no_noise_result = calculate_pz_func(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(rng_seed)
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy(
        noisy_df,
        tracking_df,
        rng,
        low_noise_bpms=[],
        noise_std=1e-4,
    )
    noisy_result = calculate_pz_func(
        noisy_df,
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_noise_result = calculate_pz_func(
        cleaned_df,
        tws=tws,
        inject_noise=False,
        info=True,
        subtract_mean=subtract_mean,
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

    px_rmse_clean = rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_clean = rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    assert px_rmse_clean < px_clean_max
    assert py_rmse_clean < py_clean_max
    if isinstance(px_noisy_min, str) and px_noisy_min == "px_rmse_clean":
        px_noisy_min_val = px_rmse_clean
    else:
        px_noisy_min_val = px_noisy_min
    if isinstance(py_noisy_min, str) and py_noisy_min == "py_rmse_clean":
        py_noisy_min_val = py_rmse_clean
    else:
        py_noisy_min_val = py_noisy_min
    assert px_noisy_min_val < px_rmse_noisy < px_noisy_max
    assert py_noisy_min_val < py_rmse_noisy < py_noisy_max
    # Check cleaned is better than noisy
    assert px_rmse_cleaned < px_rmse_noisy / px_divisor
    assert py_rmse_cleaned < py_rmse_noisy / py_divisor
