"""
Shared helper functions for controller integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tfs

from aba_optimiser.config import BEND_ERROR_FILE
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad import BaseMadInterface, OptimisationMadInterface
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import initialise_env

if TYPE_CHECKING:
    from pathlib import Path

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


def convert_rbends_to_true_rbends(mad: BaseMadInterface) -> None:
    """Convert all rbends in the sequence to true rbends for correct tracking."""
    mad.mad.send("""
for _, elm in loaded_sequence:iter() do
    if elm.kind == 'rbend' then
        elm.true_rbend = true
    end
end
    """)


def generate_model_with_errors(
    sequence_file: Path,
    json_file: Path,
    dpp_value: float,
    magnet_range: str,
    corrector_file: Path,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
) -> tuple[dict, dict[str, float], tfs.TfsDataFrame, Path]:
    """
    Generate a MAD model with errors and return the xsuite environment.

    Args:
        sequence_file: Path to the sequence file
        json_file: Path to the xsuite JSON file
        dpp_value: Momentum deviation value
        magnet_range: Range of magnets to perturb
        corrector_file: Path to save corrector strengths
        perturb_quads: Whether to perturb quadrupoles
        perturb_bends: Whether to perturb bends

    Returns:
        Tuple of (env, magnet_strengths, twiss_data, tune_knobs_file)
    """
    # Create MAD interface and load sequence
    mad = BaseMadInterface()
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    # Perform orbit correction for off-momentum beam
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

    # Convert all rbends into true rbends to ensure correct tracking
    convert_rbends_to_true_rbends(mad)

    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=dpp_value,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )
    # Run twiss to get closed orbit and optics
    twiss_data = mad.run_twiss(observe=0)

    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=json_file,
        sequence_file=sequence_file,
        seq_name="lhcb1",
        beam_energy=6800,
    )

    # save the tune knobs to file with unique name
    tune_knobs_file = corrector_file.parent / f"tune_knobs_{corrector_file.stem}.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    return env, magnet_strengths, twiss_data, tune_knobs_file


def get_twiss_without_errors(
    sequence_file: Path,
    just_bpms: bool,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs_file: Path | None = None,
    corrector_file: Path | None = None,
) -> tfs.TfsDataFrame:
    """Get twiss data from a clean model without errors."""
    mad = OptimisationMadInterface(
        sequence_file=sequence_file,
        seq_name="lhcb1",
        beam_energy=6800,
        bpm_pattern="BPM" if just_bpms else ".*",
        corrector_strengths=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    convert_rbends_to_true_rbends(mad)
    if estimated_magnets is not None:
        mad.set_magnet_strengths(estimated_magnets)
    return mad.run_twiss(observe=0)
