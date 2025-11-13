"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tfs

from aba_optimiser.config import OptSettings
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.data_processing import prepare_track_dataframe
from aba_optimiser.simulation.optics import match_tunes, perform_orbit_correction
from aba_optimiser.training.controller import Controller
from aba_optimiser.xsuite.xsuite_tools import (
    initialise_env,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
    run_tracking,
)

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
    save_knobs: bool = False,
) -> tuple[Path, dict, Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)
    corrector_file = tmp_dir / "corrector_table.tfs"

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    if dpp_value != 0:
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
        magnet_strengths = {}
        tune_knobs_file = None
    else:
        # Create an empty corrector table
        corrector_table = tfs.TfsDataFrame()
        if perturb_quads:
            mad.mad.send(f"""
local randseed, randn, abs in MAD.gmath
new_magnet_values = {{}}
for _, elm in loaded_sequence:iter('{magnet_range}') do
    if elm.kind == 'quadrupole' and elm.k1 ~= 0.0 then
        elm.k1 = elm.k1 + 1e-4 * randn() * abs(elm.k1)
        new_magnet_values[elm.name .. ".k1"] = elm.k1
    end
end
py:send(new_magnet_values, true)
            """)
            magnet_strengths = mad.mad.recv()
        else:
            magnet_strengths = {}
        matched_tunes = match_tunes(
            mad.mad, target_qx=0.28, target_qy=0.31, deltap=dpp_value
        )
        tune_knobs_file = None
        if save_knobs:
            tune_knobs_file = tmp_dir / "tune_knobs.txt"
            save_knobs(magnet_strengths, tune_knobs_file)
    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=tmp_dir / "env_config.json",
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

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
def track_files(
    tmp_path_factory: pytest.TempPathFactory,
    flattop_turns: int,
    sequence_file: Path,
    dpp_value: float,
) -> dict[str, Path | float]:
    tmp_dir = tmp_path_factory.mktemp("aba_controller_tracks")
    off_dpp_path = tmp_dir / "track_off_dpp.parquet"
    off_magnet_path = tmp_dir / "track_off_magnet.parquet"

    corrector_file, _, _ = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_dpp_path,
        dpp_value,
        "BPM.9R2.B1/BPM.9L3.B1",
        perturb_quads=False,
        save_knobs=False,
    )
    _, magnet_strengths, _ = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_magnet_path,
        0.0,
        "BPM.9R2.B1/BPM.9L3.B1",
        perturb_quads=True,
        save_knobs=False,
    )

    return {
        "corrector_file": corrector_file,
        "off_dpp": off_dpp_path,
        "off_magnet": off_magnet_path,
        "magnet_strengths": magnet_strengths,
    }


def _make_opt_settings(
    *,
    only_energy: bool = True,
    optimise_quadrupoles: bool = False,
    optimise_sextupoles: bool = False,
) -> OptSettings:
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
        only_energy=only_energy,
        optimise_quadrupoles=optimise_quadrupoles,
        optimise_sextupoles=optimise_sextupoles,
        use_off_energy_data=optimise_sextupoles,
    )


def _make_opt_settings_quad() -> OptSettings:
    return OptSettings(
        max_epochs=300,
        tracks_per_worker=124,
        num_workers=8,
        num_batches=2,
        warmup_epochs=100,
        warmup_lr_start=1e-9,
        max_lr=2e-7,
        min_lr=2e-8,
        gradient_converged_value=5e-12,
        only_energy=False,
        optimise_quadrupoles=True,
        optimise_sextupoles=False,
        use_off_energy_data=False,
    )


@pytest.mark.slow
def test_controller_energy_opt(
    track_files: dict[str, Path | float],
    flattop_turns: int,
    sequence_file: Path,
    dpp_value: float,
) -> None:
    """Test that the controller initializes correctly with custom num_tracks and flattop_turns."""
    opt_settings = _make_opt_settings()

    # Constants for the test
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"
    bpm_start_points = [
        "BPM.9R2.B1",
        "BPM.10R2.B1",
        "BPM.11R2.B1",
        "BPM.12R2.B1",
        "BPM.13R2.B1",
    ]
    bpm_end_points = [
        "BPM.9L3.B1",
        "BPM.10L3.B1",
        "BPM.11L3.B1",
        "BPM.12L3.B1",
        "BPM.13L3.B1",
    ]

    ctrl = Controller(
        opt_settings=opt_settings,
        sequence_file_path=sequence_file,
        show_plots=False,
        magnet_range=magnet_range,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        measurement_file=track_files["off_dpp"],
        true_strengths=None,
        corrector_file=track_files["corrector_file"],
        flattop_turns=flattop_turns,
        num_tracks=1,
    )

    estimate, unc = ctrl.run()  # Ensure that run works without errors

    assert np.allclose(estimate.pop("deltap"), dpp_value, atol=1e-7)
    assert np.allclose(unc.pop("deltap"), 0, atol=4e-7)

    # check that estimate and unc are now empty
    assert not estimate
    assert not unc


@pytest.mark.slow
def test_controller_quad_opt_simple(sequence_file: Path) -> None:
    """Test quadrupole optimisation using the simple opt script logic."""
    # Constants for the test
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"
    bpm_start_points = [
        "BPM.9R2.B1",
        "BPM.10R2.B1",
        "BPM.11R2.B1",
        "BPM.12R2.B1",
        "BPM.13R2.B1",
    ]
    bpm_end_points = [
        "BPM.9L3.B1",
        "BPM.10L3.B1",
        "BPM.11L3.B1",
        "BPM.12L3.B1",
        "BPM.13L3.B1",
    ]

    flattop_turns = 1000
    tmp_dir = Path(tempfile.mkdtemp())
    off_magnet_path = tmp_dir / "track_off_magnet.parquet"

    _, magnet_strengths, tune_knobs_file = _generate_nonoise_track(
        tmp_dir,
        sequence_file,
        flattop_turns,
        off_magnet_path,
        0.0,
        magnet_range,
        perturb_quads=True,
        save_knobs=True,
    )

    opt_settings = _make_opt_settings_quad()
    empty_file = tmp_dir / "empty_corrector_table.txt"
    empty_file.touch()
    true_values = magnet_strengths.copy()
    true_values["deltap"] = 0.0  # We tracked at on-momentum for this test

    ctrl = Controller(
        opt_settings=opt_settings,
        sequence_file_path=sequence_file,
        show_plots=False,
        magnet_range=magnet_range,
        bpm_start_points=bpm_start_points,
        bpm_end_points=bpm_end_points,
        measurement_file=off_magnet_path,
        true_strengths=true_values,
        corrector_file=empty_file,
        tune_knobs_file=tune_knobs_file,
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
        assert rel_diff < 1e-5, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {true_values[magnet]}, rel diff {rel_diff}"
        )
