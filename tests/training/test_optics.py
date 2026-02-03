"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")

import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.simulation.magnet_perturbations import apply_magnet_perturbations
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.training.controller_config import SequenceConfig
from aba_optimiser.training_optics import OpticsController

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.base_mad_interface import BaseMadInterface

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


def _generate_fake_measurement(
    tmp_path: Path,
    model_dir: Path,
    interface: BaseMadInterface,
    flattop_turns: int,
    dpp_value: float,
    magnet_range: str,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
) -> tuple[Path, dict, Path | None, Path]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create unique corrector file path based on destination
    corrector_file = tmp_path / "correctors.tfs"
    interface.mad["zero_twiss", "_"] = interface.mad.twiss(sequence="loaded_sequence")

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    magnet_strengths = {}
    magnet_type = ("q" if perturb_quads else "") + ("d" if perturb_bends else "")
    if magnet_type:
        magnet_strengths, _ = apply_magnet_perturbations(
            interface.mad,
            rel_k1_std_dev=1e-4,
            seed=42,
            magnet_type=magnet_type,
        )

    matched_tunes = perform_orbit_correction(
        mad=interface.mad,
        machine_deltap=dpp_value,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
        beam=1,
    )
    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # save the tune knobs to file with unique name
    tune_knobs_file = tmp_path / "tune_knobs.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    analysis_dir = tmp_path / "analysis"

    interface.observe_elements()
    twiss = interface.run_twiss(coupling=True)

    # Convert all the columns to uppercase
    twiss.columns = [col.upper() for col in twiss.columns]
    twiss.rename(columns={"MU1": "MUX", "MU2": "MUY"}, inplace=True)

    # Rename mu1 and mu2 to mux and muy
    twiss.headers = {key.upper(): value for key, value in twiss.headers.items()}

    fake_measurement(
        twiss=twiss,
        outputdir=analysis_dir,
        # randomize=["values", "errors"],
        # relative_errors=[1e-2],
    )

    return corrector_file, magnet_strengths, tune_knobs_file, analysis_dir


@pytest.mark.slow
def test_controller_opt(
    tmp_path: Path,
    seq_b1: Path,
    loaded_interface_with_beam: BaseMadInterface,
    model_dir_b1: Path,
) -> None:
    """Test that the controller initializes correctly with custom num_tracks and flattop_turns."""
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"

    corrector_file, magnet_strengths, tune_knobs_file, analysis_dir = _generate_fake_measurement(
        tmp_path,
        model_dir_b1,
        loaded_interface_with_beam,
        6600,
        0e-4,
        magnet_range,
        perturb_quads=True,
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
        warmup_lr_start=1e-8,
        max_lr=6e-7,
        min_lr=1e-7,
        gradient_converged_value=1e-5,
    )

    sequence_config = SequenceConfig(
        magnet_range=magnet_range,
    )

    accel = LHC(beam=1, sequence_file=seq_b1, beam_energy=6800.0, optimise_quadrupoles=True)

    ctrl = OpticsController(
        accel,
        sequence_config,
        optimiser_config,
        analysis_dir,
        bpm_start_points,
        bpm_end_points,
        show_plots=False,
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
        assert rel_diff < 3e-6, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {magnet_strengths[magnet]}, rel diff {rel_diff}"
        )
