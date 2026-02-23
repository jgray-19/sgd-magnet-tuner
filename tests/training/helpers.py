"""
Shared helper functions for controller integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import tfs
from xtrack_tools.env import initialise_env

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad import AbaMadInterface, GenericMadInterface
from aba_optimiser.simulation.magnet_perturbations import apply_magnet_perturbations
from aba_optimiser.simulation.optics import perform_orbit_correction

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import xtrack as xt

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


def convert_rbends_to_true_rbends(mad: AbaMadInterface) -> None:
    """Convert all rbends in the sequence to true rbends for correct tracking."""
    mad.mad.send("""
for _, elm in loaded_sequence:iter() do
    if elm.kind == 'rbend' then
        elm.true_rbend = true
    end
end
    """)


def generate_model_with_errors(
    loaded_interface_with_beam: AbaMadInterface,
    sequence_file: Path,
    dpp_value: float,
    magnet_range: str,
    corrector_file: Path,
    beam: Literal[1, 2] = 1,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
) -> tuple[dict[str, float], dict, tfs.TfsDataFrame]:
    """
    Generate a MAD model with errors and return the xsuite environment.

    Args:
        sequence_file: Path to the sequence file
        dpp_value: Momentum deviation value
        magnet_range: Range of magnets to perturb
        corrector_file: Path to save corrector strengths
        perturb_quads: Whether to perturb quadrupoles
        perturb_bends: Whether to perturb bends

    Returns:
        Tuple of (magnet_strengths, matched_tunes, corrector_table)
    """
    # Create MAD interface and load sequence
    interface = loaded_interface_with_beam
    interface.mad["zero_twiss", "_"] = interface.mad.twiss(sequence="loaded_sequence")

    # Perform orbit correction for off-momentum beam
    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    magnet_strengths = {}
    magnet_type = ("q" if perturb_quads else "") + ("d" if perturb_bends else "")
    if magnet_type:
        magnet_strengths, _ = apply_magnet_perturbations(
            interface.mad,
            rel_k1_std_dev=None,  # Use default from ERROR_TABLE
            seed=42,
            magnet_type=magnet_type,
        )

    # Convert all rbends into true rbends to ensure correct tracking
    # convert_rbends_to_true_rbends(mad)

    matched_tunes = perform_orbit_correction(
        mad=interface.mad,
        machine_deltap=dpp_value,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
        beam=beam,
    )

    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table.loc[corrector_table.loc[:, "kind"] != "monitor"]

    return magnet_strengths, matched_tunes, corrector_table


def generate_xsuite_env_with_errors(
    loaded_interface_with_beam: AbaMadInterface,
    sequence_file: Path,
    dpp_value: float,
    magnet_range: str,
    corrector_file: Path,
    beam: Literal[1, 2] = 1,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
) -> tuple[xt.Environment, dict[str, float], dict, tfs.TfsDataFrame]:
    """
    Generate a MAD model with errors and return the xsuite environment.

    Args:
        sequence_file: Path to the sequence file
        dpp_value: Momentum deviation value
        magnet_range: Range of magnets to perturb
        corrector_file: Path to save corrector strengths
        perturb_quads: Whether to perturb quadrupoles
        perturb_bends: Whether to perturb bends

    Returns:
        Tuple of (env, magnet_strengths, matched_tunes, corrector_table)
    """
    magnet_strengths, matched_tunes, corrector_table = generate_model_with_errors(
        loaded_interface_with_beam,
        sequence_file,
        dpp_value,
        magnet_range,
        corrector_file,
        beam,
        perturb_quads,
        perturb_bends,
    )

    seq_name = f"lhcb{beam}"
    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        beam=beam,
        sequence_file=sequence_file,
        seq_name=seq_name,
        beam_energy=6800,
    )

    return env, magnet_strengths, matched_tunes, corrector_table


def get_twiss_without_errors(
    sequence_file: Path,
    just_bpms: bool,
    beam: int = 1,
    estimated_magnets: dict[str, float] | None = None,
    tune_knobs_file: Path | None = None,
    corrector_file: Path | None = None,
) -> pd.DataFrame:
    """Get twiss data from a clean model without errors."""
    accelerator = LHC(
        beam=beam,
        sequence_file=sequence_file,
    )
    mad = GenericMadInterface(
        accelerator,
        bpm_pattern="BPM",
        corrector_strengths=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    convert_rbends_to_true_rbends(mad)
    if estimated_magnets is not None:
        mad.set_magnet_strengths(estimated_magnets)
    return mad.run_twiss(observe=int(just_bpms))
