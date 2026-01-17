"""
Integration-style tests for the controller logic using lightweight tracking data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement

from aba_optimiser.config import BEND_ERROR_FILE, OptimiserConfig
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.training.controller_config import BPMConfig, SequenceConfig
from aba_optimiser.training_optics import OpticsController

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

    # save the tune knobs to file with unique name
    tune_knobs_file = tmp_dir / "tune_knobs.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    analysis_dir = tmp_dir / "analysis"

    mad.observe_elements()
    twiss = mad.run_twiss(coupling=True)

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


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_tracks")


@pytest.mark.slow
def test_controller_opt(
    tmp_dir: Path,
    seq_b1: Path,
    model_dir_b1: Path,
) -> None:
    """Test that the controller initializes correctly with custom num_tracks and flattop_turns."""
    magnet_range = "BPM.9R2.B1/BPM.9L3.B1"

    corrector_file, magnet_strengths, tune_knobs_file, analysis_dir = _generate_nonoise_track(
        tmp_dir,
        model_dir_b1,
        seq_b1,
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
        warmup_lr_start=1e-8,
        max_lr=3e-7,
        min_lr=1e-7,
        gradient_converged_value=1e-4,
    )

    sequence_config = SequenceConfig(
        sequence_file_path=seq_b1,
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
        assert rel_diff < 1e-7, (
            f"Magnet {magnet}: FAIL, estimated {value}, true {magnet_strengths[magnet]}, rel diff {rel_diff}"
        )
