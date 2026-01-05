"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tfs
from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn import convert_to_tbt, write_tbt

from aba_optimiser.config import BEND_ERROR_FILE, OptimiserConfig
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.training.controller_config import BPMConfig, SequenceConfig
from aba_optimiser.training_optics import OpticsController
from aba_optimiser.xsuite.xsuite_tools import (
    initialise_env,
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    run_tracking,
)

if TYPE_CHECKING:
    from pathlib import Path

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
    model_dir: Path,
    sequence_file: Path,
    flattop_turns: int,
    dpp_value: float,
    magnet_range: str,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
) -> tuple[Path, dict, Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    # Create unique corrector file path based on destination
    corrector_file = tmp_dir / "correctors.tfs"

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
                    raise ValueError(f"Bend error for {elm.name} not found in {BEND_ERROR_FILE}")
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
    tune_knobs_file = tmp_dir / "tune_knobs.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    # Compute twiss for ACD insertion
    tws = env["lhcb1"].twiss(method="4d", delta0=dpp_value)

    # Insert AC dipole
    acd_ramp = 1000
    total_turns = flattop_turns + acd_ramp
    driven_tunes = [0.27, 0.322]
    output_files = []
    for lag in [0, np.pi / 4, np.pi / 3]:
        lhcb1 = insert_ac_dipole(
            env["lhcb1"],
            tws,
            beam=1,
            acd_ramp=acd_ramp,
            total_turns=total_turns,
            driven_tunes=driven_tunes,
            lag=lag,
        )

        insert_particle_monitors_at_pattern(
            lhcb1,
            pattern="bpm.*[^k]",
            num_turns=flattop_turns,
            num_particles=1,
            inplace=True,
        )
        for dpp in [-2.5e-4, 0.0, 2.5e-4]:
            particles = lhcb1.build_particles(
                x=[1e-4],
                px=[-1e-6],
                y=[1e-4],
                py=[-1e-6],
                delta=[dpp + dpp_value],
            )
            run_tracking(
                line=lhcb1,
                particles=particles,
                nturns=flattop_turns,
            )
            sdds = convert_to_tbt(lhcb1)
            output_file = tmp_dir / f"track_result_{dpp}_{lag}.sdds"
            output_files.append(output_file)
            write_tbt(output_file, sdds, noise=1e-4)

    linfile_dir = tmp_dir / "linfiles"
    hole_in_one_entrypoint(
        harpy=True,
        files=output_files,
        tbt_datatype="lhc",
        outputdir=linfile_dir,
        to_write=["lin", "spectra"],
        opposite_direction=False,
        driven_excitation="acd",
        tunes=driven_tunes + [0.0],
        nattunes=[0.28, 0.31, 0.0],
        turn=[acd_ramp, 50e3],
        clean=True,
    )
    all_linfiles = list(linfile_dir.glob("*.linx"))
    all_files = [f.parent / f.name.strip(".linx") for f in all_linfiles]
    analysis_dir = tmp_dir / "analysis"

    hole_in_one_entrypoint(
        optics=True,
        files=all_files,
        outputdir=analysis_dir,
        accel="lhc",
        beam=1,
        model_dir=model_dir,
        year="2025",
        compensation="model",
    )
    return corrector_file, magnet_strengths, tune_knobs_file, analysis_dir


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_tracks")


@pytest.mark.slow
def test_controller_opt(
    tmp_dir: Path,
    sequence_file: Path,
    model_dir_b1: Path,
) -> None:
    """Test that the controller initializes correctly with custom num_tracks and flattop_turns."""
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"

    corrector_file, magnet_strengths, tune_knobs_file, analysis_dir = _generate_nonoise_track(
        tmp_dir,
        model_dir_b1,
        sequence_file,
        6600,
        0e-4,
        magnet_range,
        perturb_quads=True,
        # perturb_bends=True,
    )

    # Constants for the test
    bpm_start_points = [
        "BPM.9R2.B1",
        "BPM.10R2.B1",
        # "BPM.11R2.B1",
    ]
    bpm_end_points = [
        "BPM.9L3.B1",
        "BPM.10L3.B1",
        # "BPM.11L3.B1",
    ]

    # print all files in analysis_dir for debugging
    for f in analysis_dir.glob("*"):
        logger.info(f"Analysis dir file: {f}")

    optimiser_config = OptimiserConfig(
        max_epochs=2000,
        warmup_epochs=100,
        warmup_lr_start=1e-9,
        max_lr=2e-7,
        min_lr=1.5e-7,
        gradient_converged_value=1e-4,
    )

    sequence_config = SequenceConfig(
        sequence_file_path=sequence_file,
        magnet_range=magnet_range,
        seq_name="lhcb1",
    )

    bpm_config = BPMConfig(
        start_points=bpm_start_points,
        end_points=bpm_end_points,
    )

    ctrl = OpticsController(
        sequence_config=sequence_config,
        optics_folder=analysis_dir,
        bpm_config=bpm_config,
        optimiser_config=optimiser_config,
        show_plots=True,
        corrector_file=corrector_file,
        tune_knobs_file=tune_knobs_file,
        true_strengths=magnet_strengths,
        use_errors=True,
    )

    estimate, unc = ctrl.run()
    for magnet, value in estimate.items():
        rel_diff = (
            abs(value - magnet_strengths[magnet]) / abs(magnet_strengths[magnet])
            if magnet_strengths[magnet] != 0
            else abs(value)
        )
        assert rel_diff < 1e-8, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {magnet_strengths[magnet]}, rel diff {rel_diff}"
        )
