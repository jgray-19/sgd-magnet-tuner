"""
Shared utilities for controller integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")

from pymadng_utils.io.utils import save_knobs
from xtrack_tools.monitors import line_to_dataframes
from xtrack_tools.tracking import run_tracking_without_ac_dipole

from aba_optimiser.accelerators import instantiate_accelerator_from
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, SequenceConfig
from tests.training.helpers import TRACK_COLUMNS, generate_xsuite_env_with_errors

if TYPE_CHECKING:
    from pathlib import Path

    import xtrack as xt

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


def _run_track_with_model(
    env: xt.Environment,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    action_list: list[float],
    angle_list: list[float],
    line_name: str,
    start_marker: str | None = None,
    bpm_pattern: str = "bpm.*[^k]",
    use_diagonal_kicks: bool = True,
    combine_particles_into_single_file: bool = False,
    return_dataframes: bool = False,
) -> list[Path] | list[pd.DataFrame]:
    """Run tracking with the given model and initial conditions."""
    input_particles = len(action_list)
    if len(angle_list) != input_particles:
        raise ValueError("action_list and angle_list must have the same length")

    line: xt.Line = env[line_name]
    monitored_line = run_tracking_without_ac_dipole(
        line=line,
        tws=line.twiss4d(),
        flattop_turns=flattop_turns,
        bpm_pattern=bpm_pattern,
        action_list=action_list,
        angle_list=angle_list,
        use_diagonal_kicks=use_diagonal_kicks,
        deltas=dpp_value,
        start_marker=start_marker,
    )

    true_dfs = line_to_dataframes(monitored_line)
    processed_dfs = []
    for true_df in true_dfs:
        df = prepare_track_dataframe(
            true_df,
            0,
            flattop_turns,
        )
        df = df.loc[:, TRACK_COLUMNS].copy()
        df["name"] = df["name"].astype(str)
        processed_dfs.append(df)

    # # Debugging, plot the phase space at the first BPM:
    # import matplotlib.pyplot as plt
    # first_bpm = processed_dfs[0].iloc[0]["name"]
    # for df in processed_dfs:
    #     bpm_df = df[df["name"] == first_bpm]
    #     plt.plot(bpm_df["x"], bpm_df["px"], "o", label="x-px")
    #     plt.plot(bpm_df["y"], bpm_df["py"], "o", label="y-py")
    # plt.xlabel("Position (m)")
    # plt.ylabel("Momentum (rad)")
    # plt.title("Phase space at first BPM")
    # plt.legend()
    # plt.show()

    if return_dataframes:
        return processed_dfs

    output_files = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    num_output_particles = len(processed_dfs)
    if combine_particles_into_single_file and num_output_particles > 1:
        offset_dfs = []
        for idx, df in enumerate(processed_dfs):
            particle_df = df.copy()
            particle_df["turn"] = particle_df["turn"] + idx * flattop_turns
            offset_dfs.append(particle_df)
        combined_df = pd.concat(offset_dfs, ignore_index=True)
        combined_df.to_parquet(destination, index=False)
        output_files.append(destination)
        return output_files

    for idx, df in enumerate(processed_dfs):
        if num_output_particles == 1:
            output_path = destination
        else:
            output_path = (
                destination.parent / f"{destination.stem}_particle_{idx}{destination.suffix}"
            )
        df.to_parquet(output_path, index=False)
        output_files.append(output_path)
    return output_files


def _generate_nonoise_track(
    interface_with_beam: AbaMadInterface,
    flattop_turns: int,
    destination: Path,
    dpp_value: float,
    start_marker: str | None = None,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    num_particles: int = 1,
    phases: list[float] | None = None,
    bpm_pattern: str = "bpm.*[^k]",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    use_diagonal_kicks: bool = True,
) -> tuple[Path | None, dict[str, float], Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    corrector_file: Path | None = None
    tune_knobs_file: Path | None = None
    if apply_orbit_correction:
        corrector_file = destination.parent / f"corrector_{destination.stem}.tfs"
    tune_knobs_file = destination.parent / f"tune_knobs_{destination.stem}.txt"

    env, magnet_strengths, matched_tunes, corrector_table = generate_xsuite_env_with_errors(
        interface_with_beam,
        dpp_value=dpp_value,
        corrector_file=corrector_file,
        perturb_quads=perturb_quads,
        perturb_bends=perturb_bends,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
    )
    del corrector_table
    save_knobs(matched_tunes, tune_knobs_file)

    action = 4e-7 if interface_with_beam.accelerator.seq_name.lower() == "sps" else 4e-8
    angle = 0.0
    if num_particles == 1:
        action_list = [action]
        angle_list = [angle]
    else:
        if phases is not None:
            if len(phases) != num_particles:
                raise ValueError("Length of phases must equal num_particles")
            angle_list = phases
        else:
            angle_list = (np.linspace(0.0, 2 * np.pi, num=num_particles, endpoint=False)).tolist()
        action_list = [action] * num_particles

    _run_track_with_model(
        env=env,
        flattop_turns=flattop_turns,
        destination=destination,
        dpp_value=dpp_value,
        action_list=action_list,
        angle_list=angle_list,
        start_marker=start_marker,
        line_name=interface_with_beam.accelerator.seq_name.lower(),
        bpm_pattern=bpm_pattern,
        use_diagonal_kicks=use_diagonal_kicks,
    )
    return corrector_file, magnet_strengths, tune_knobs_file


DPP_VALUE = 1.25e-4
FLATTOP_TURNS = 256
def _make_simulation_config_energy(optimise_momenta: bool = True) -> SimulationConfig:
    return SimulationConfig(
        tracks_per_worker=1,
        num_workers=3,
        num_batches=10,
        optimise_momenta=optimise_momenta,
    )


def _run_energy_optimisation_case(
    *,
    tmp_path: Path,
    loaded_interface: AbaMadInterface,
    simulation_config: SimulationConfig,
    optimiser_config: OptimiserConfig,
    bpm_start_points: list[str],
    bpm_end_points: list[str],
    magnet_range: str,
    mad_log_name: str,
    bpm_pattern: str = "bpm.*[^k]",
    apply_orbit_correction: bool = True,
    target_qx: float = 0.28,
    target_qy: float = 0.31,
    dpp_value: float = DPP_VALUE,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run one energy optimisation scenario and return estimate/uncertainty dictionaries."""
    off_dpp_path = tmp_path / "track_off_dpp.parquet"
    corrector_file, _, tune_knobs_file = _generate_nonoise_track(
        loaded_interface,
        FLATTOP_TURNS,
        off_dpp_path,
        dpp_value,
        bpm_pattern=bpm_pattern,
        apply_orbit_correction=apply_orbit_correction,
        target_qx=target_qx,
        target_qy=target_qy,
    )

    sequence_config = SequenceConfig(magnet_range=magnet_range)
    measurement_config = MeasurementConfig(
        measurement_files=off_dpp_path,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=FLATTOP_TURNS,
        bunches_per_file=1,
    )

    accel = instantiate_accelerator_from(loaded_interface.accelerator, optimise_energy=True)
    ctrl = Controller(
        accel,
        optimiser_config,
        simulation_config,
        sequence_config,
        measurement_config,
        bpm_start_points,
        bpm_end_points,
        show_plots=False,
        true_strengths=None,
        mad_logfile=tmp_path / mad_log_name,
        write_tensorboard_logs=False,
        optimise_knobs=None,
    )
    return ctrl.run()


def _make_optimiser_config_quad() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=500,
        warmup_epochs=100,
        warmup_lr_start=1e-6,
        max_lr=1e-5,
        min_lr=1e-5,
        gradient_converged_value=5e-14,
        expected_rel_error=0,
    )


def _make_simulation_config_quad() -> SimulationConfig:
    return SimulationConfig(
        tracks_per_worker=10,
        num_workers=8,
        num_batches=2,
        bpm_loss_outlier_sigma=4,
    )
