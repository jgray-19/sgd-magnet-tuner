"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig, SequenceConfig
from aba_optimiser.xsuite.monitors import line_to_dataframes
from aba_optimiser.xsuite.tracking import run_tracking_without_ac_dipole
from tests.training.helpers import TRACK_COLUMNS, generate_xsuite_env_with_errors

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import xtrack as xt

    from aba_optimiser.mad.base_mad_interface import BaseMadInterface

logger = logging.getLogger(__name__)


def _run_track_with_model(
    env: xt.Environment,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    action_list: list[float],
    angle_list: list[float],
    start_marker: str | None = None,
    return_dataframes: bool = False,
) -> list[Path] | list[pd.DataFrame]:
    """
    Run tracking with the given model and initial conditions.

    Args:
        env: xsuite environment dictionary
        flattop_turns: Number of turns to track
        destination: Path to save the parquet file (or base path for multiple files)
        dpp_value: Momentum deviation
        action_list: List of action values for each particle
        angle_list: List of angle values for each particle
        return_dataframes: If True, return dataframes instead of writing to disk

    Returns:
        List of Path objects for each generated parquet file, or list of DataFrames if return_dataframes=True
    """
    num_particles = len(action_list)

    if len(angle_list) != num_particles:
        raise ValueError("action_list and angle_list must have the same length")

    line: xt.Line = env["lhcb1"]  # ty:ignore[not-subscriptable]

    monitored_line = run_tracking_without_ac_dipole(
        line=line,
        tws=line.twiss4d(),
        flattop_turns=flattop_turns,
        bpm_pattern="bpm.*[^k]",
        action_list=action_list,
        angle_list=angle_list,
        delta_values=[dpp_value] * num_particles,
        start_marker=start_marker,
    )

    true_dfs = line_to_dataframes(monitored_line)
    # Process dataframes
    processed_dfs = []
    for true_df in true_dfs:
        df = prepare_track_dataframe(true_df, 0, flattop_turns, kick_both_planes=True)
        df = df.loc[:, TRACK_COLUMNS].copy()
        df["name"] = df["name"].astype(str)
        df["kick_plane"] = df["kick_plane"].astype(str)
        processed_dfs.append(df)

    if return_dataframes:
        return processed_dfs

    # Save each particle track to a separate file
    output_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)

    for idx, df in enumerate(processed_dfs):
        if num_particles == 1:
            output_path = destination
        else:
            # Create unique filename for each particle
            output_path = (
                destination.parent / f"{destination.stem}_particle_{idx}{destination.suffix}"
            )

        df.to_parquet(output_path, index=False)
        output_files.append(output_path)

    return output_files


def _generate_nonoise_track(
    interface_with_beam: BaseMadInterface,
    sequence_file: Path,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    magnet_range: str,
    start_marker: str | None = None,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    num_particles: int = 1,
    phases: list[float] | None = None,
) -> tuple[Path, dict[str, float], Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create unique corrector file path based on destination
    corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"
    tune_knobs_file = destination.parent / f"tune_knobs_{destination.stem}.txt"

    # Generate model with errors
    env, magnet_strengths, matched_tunes, corrector_table = generate_xsuite_env_with_errors(
        interface_with_beam,
        sequence_file=sequence_file,
        dpp_value=dpp_value,
        magnet_range=magnet_range,
        corrector_file=corrector_file,
        beam=1,
        perturb_quads=perturb_quads,
        perturb_bends=perturb_bends,
    )

    save_knobs(matched_tunes, tune_knobs_file)

    # Create action and angle lists
    action = 4e-8  # action for larger kick
    angle = 0.0

    if num_particles == 1:
        action_list = [action]
        angle_list = [angle]
    else:
        # If explicit phases provided use them, otherwise spread phases evenly
        if phases is not None:
            if len(phases) != num_particles:
                raise ValueError("Length of phases must equal num_particles")
            angle_list = phases
        else:
            angle_list = (np.linspace(0.0, 2 * np.pi, num=num_particles, endpoint=False)).tolist()
        action_list = [action] * num_particles

    # Run tracking with the model
    _run_track_with_model(
        env=env,
        flattop_turns=flattop_turns,
        destination=destination,
        dpp_value=dpp_value,
        action_list=action_list,
        angle_list=angle_list,
        start_marker=start_marker,
    )

    return corrector_file, magnet_strengths, tune_knobs_file


@pytest.fixture(scope="module")
def dpp_value() -> float:
    return 1.25e-4


@pytest.fixture(scope="module")
def flattop_turns() -> int:
    return 256


def _make_optimiser_config_energy() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=1,
        warmup_lr_start=1e-8,
        max_lr=2e-6,
        min_lr=2e-7,
        gradient_converged_value=5e-10,
        expected_rel_error=0,
    )


def _make_simulation_config_energy() -> SimulationConfig:
    return SimulationConfig(
        tracks_per_worker=1,
        num_workers=3,
        num_batches=10,
        optimise_energy=True,
        optimise_quadrupoles=False,
        optimise_bends=False,
        optimise_momenta=True,
    )


def _make_optimiser_config_quad() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=300,
        warmup_epochs=50,
        warmup_lr_start=1e-8,
        max_lr=1e-6,
        min_lr=1e-8,
        gradient_converged_value=5e-14,
        expected_rel_error=0,
    )


def _make_simulation_config_quad() -> SimulationConfig:
    return SimulationConfig(
        tracks_per_worker=10,
        num_workers=8,
        num_batches=2,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
    )


@pytest.mark.slow
def test_controller_energy_opt(
    tmp_path: Path,
    flattop_turns: int,
    seq_b1: Path,
    dpp_value: float,
    loaded_interface_with_beam: BaseMadInterface,
) -> None:
    """Test that the controller initialises correctly with custom num_tracks and flattop_turns."""
    optimiser_config = _make_optimiser_config_energy()
    simulation_config = _make_simulation_config_energy()

    off_dpp_path = tmp_path / "track_off_dpp.parquet"
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"

    corrector_file, _, tune_knobs_file = _generate_nonoise_track(
        loaded_interface_with_beam,
        seq_b1,
        flattop_turns,
        off_dpp_path,
        dpp_value,
        magnet_range,
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

    sequence_config = SequenceConfig(
        sequence_file_path=seq_b1,
        magnet_range=magnet_range,
        beam_energy=6800,
    )

    measurement_config = MeasurementConfig(
        measurement_files=off_dpp_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )

    bpm_config = BPMConfig(
        start_points=bpm_start_points,
        end_points=bpm_end_points,
    )

    ctrl = Controller(
        optimiser_config=optimiser_config,
        simulation_config=simulation_config,
        sequence_config=sequence_config,
        measurement_config=measurement_config,
        bpm_config=bpm_config,
        show_plots=False,
        true_strengths=None,
        mad_logfile=tmp_path / "controller_energy_opt.log",
    )

    estimate, unc = ctrl.run()  # Ensure that run works without errors

    assert np.allclose(estimate.pop("deltap"), dpp_value, rtol=1e-4, atol=1e-10)
    uncertainty = unc.pop("deltap")
    assert uncertainty < 1e-6 and uncertainty > 0

    # check that estimate and unc are now empty
    assert not estimate
    assert not unc


@pytest.mark.slow
@pytest.mark.parametrize("start_marker", ["MSIA.EXIT.B1", "E.CELL.12.B1"])
def test_controller_quad_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    start_marker: str,
    loaded_interface_with_beam: BaseMadInterface,
) -> None:
    """Test quadrupole optimisation using the simple opt script logic."""
    # Constants for the test
    magnet_range = "BPM.9R1.B1/BPM.9L2.B1"
    bpm_start_points: list[str] = [
        "BPM.9R1.B1",
        "BPM.10R1.B1",
    ]
    bpm_end_points: list[str] = [
        "BPM.9L2.B1",
        "BPM.10L2.B1",
    ]

    flattop_turns = 1000
    off_magnet_path = tmp_path / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        loaded_interface_with_beam,
        seq_b1,
        flattop_turns,
        off_magnet_path,
        0.0,
        "$start/$end",
        start_marker=start_marker,
        perturb_quads=True,
    )

    optimiser_config = _make_optimiser_config_quad()
    simulation_config = _make_simulation_config_quad()
    true_values = magnet_strengths.copy()

    sequence_config = SequenceConfig(
        sequence_file_path=seq_b1,
        magnet_range=magnet_range,
        beam_energy=6800,
        first_bpm=start_marker,
    )

    measurement_config = MeasurementConfig(
        measurement_files=off_magnet_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=flattop_turns,
        bunches_per_file=1,
    )

    bpm_config = BPMConfig(
        start_points=bpm_start_points,
        end_points=bpm_end_points,
    )
    ctrl = Controller(
        optimiser_config=optimiser_config,
        simulation_config=simulation_config,
        sequence_config=sequence_config,
        measurement_config=measurement_config,
        bpm_config=bpm_config,
        show_plots=False,
        plots_dir=tmp_path / "plots",
        true_strengths=true_values,
        debug=False,
        mad_logfile=tmp_path / "mad_logfile.log",
    )
    logger.info(f"Starting controller with logfile at {tmp_path / 'mad_logfile.log'}")
    estimate, unc = ctrl.run()
    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - true_values[magnet]) / abs(true_values[magnet])
            if true_values[magnet] != 0
            else abs(value)
        )
        assert rel_diff < 1e-6, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {true_values[magnet]}, rel diff {rel_diff}"
        )
