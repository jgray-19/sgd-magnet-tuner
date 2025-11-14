"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tfs

from aba_optimiser.config import BEND_ERROR_FILE, OptSettings
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.training.controller import Controller
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import (
    initialise_env,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
    run_tracking,
)
from aba_optimiser.io.utils import save_knobs

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
    average_closed_orbit: bool = False,
) -> tuple[Path, dict, Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)
    corrector_file = tmp_dir / "corrector_table.tfs"

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
    elif perturb_bends:
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

    # save the tune knobs to file
    tune_knobs_file = tmp_dir / "tune_knobs.txt"
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
        averaged = df.groupby("name")[
            ["x", "px", "y", "py", "var_x", "var_y", "var_px", "var_py"]
        ].mean().reset_index()
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
        max_lr=2e-5,
        min_lr=8e-7,
        gradient_converged_value=5e-14,
        optimise_energy=False,
        optimise_quadrupoles=True,
        optimise_bends=False,
    )


def _make_opt_settings_bend() -> OptSettings:
    return OptSettings(
        max_epochs=5000,
        tracks_per_worker=1,
        num_batches=1,
        num_workers=1,
        warmup_epochs=3,
        warmup_lr_start=5e-10,
        max_lr=1e-1,
        min_lr=1e-1,
        gradient_converged_value=1e-6,
        optimiser_type="lbfgs",
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

    print(estimate["deltap"], unc["deltap"])

    assert np.allclose(estimate.pop("deltap"), dpp_value, rtol=1e-4, atol=1e-10) 
    assert np.allclose(unc.pop("deltap"), 0, atol=5e-7)

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
        show_plots=True,
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
        assert rel_diff < 3e-9, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {true_values[magnet]}, rel diff {rel_diff}"
        )


@pytest.mark.slow
def test_controller_bend_opt_simple(tmp_dir: Path, sequence_file: Path) -> None:
    """Test bending magnet optimisation using the simple opt script logic."""
    # Constants for the test
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"
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

    flattop_turns = 1000
    off_magnet_path = tmp_dir / "track_off_magnet.parquet"

    corrector_file, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_magnet_path,
        0.0,
        magnet_range,
        perturb_quads=False,
        perturb_bends=True,
        average_closed_orbit=True,
    )

    opt_settings = _make_opt_settings_bend()
    true_values = magnet_strengths.copy()

    # Update bend keys to remove [ABCD] for consistency with knob_names
    import re
    pattern = r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.k0"
    new_true_values = {}
    for key, value in true_values.items():
        match = re.match(pattern, key)
        if match:
            new_key = match.group(1) + match.group(3) + ".k0"
            new_true_values[new_key] = value
        else:
            new_true_values[key] = value  # For non-bend keys

    ctrl = Controller(
        opt_settings=opt_settings,
        sequence_file_path=sequence_file,
        show_plots=True,
        magnet_range=magnet_range,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        measurement_files=off_magnet_path,
        true_strengths=true_values,
        corrector_files=corrector_file,
        tune_knobs_files=tune_knobs_file,
        flattop_turns=3,
        num_tracks=1,
    )
    estimate, unc = ctrl.run()
    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - new_true_values[magnet]) / abs(new_true_values[magnet])
            if new_true_values[magnet] != 0
            else abs(value)
        )
        assert rel_diff < 3e-9, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {new_true_values[magnet]}, rel diff {rel_diff}"
        )
