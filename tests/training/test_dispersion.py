"""
Integration-style tests for dispersion estimation using tracking data.

This test validates that we can reproduce model dispersion at correctors
by tracking from nearby BPMs using measured optics parameters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tfs
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.model.constants import TWISS_ELEMENTS_DAT
from omc3.model.manager import get_accelerator
from turn_by_turn import convert_to_tbt, write_tbt

from aba_optimiser.config import BEND_ERROR_FILE
from aba_optimiser.dispersion.dispersion_estimation import estimate_corrector_dispersions
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import (
    initialise_env,
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    run_tracking,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _generate_nonoise_track(
    tmp_dir: Path,
    model_dir: Path,
    sequence_file: Path,
    flattop_turns: int,
    beam: int = 1,
    perturb_quads: bool = False,
    perturb_bends: bool = False,
    magnet_range: str = "$start/$end",
) -> Path:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs.

    Args:
        tmp_dir: Temporary directory for outputs
        model_dir: Directory containing model files
        sequence_file: Path to MAD-X sequence file
        flattop_turns: Number of turns for tracking
        beam: Beam number (1 or 2)
        perturb_quads: Whether to add quadrupole errors
        perturb_bends: Whether to add bending errors
        magnet_range: Range of magnets to perturb

    Returns:
        Path to analysis directory containing optics results
    """
    seq_name = f"lhcb{beam}"

    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, seq_name)
    mad.setup_beam(beam_energy=6800)

    # Apply magnet perturbations
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
                name_in_dict = elm.name.replace("B2", "B1")
                if name_in_dict not in bend_errors_dict:
                    raise ValueError(f"Bend error for {elm.name} not found in {BEND_ERROR_FILE}")
                k0l_error = bend_errors_dict[name_in_dict]
                elm.k0 += k0l_error / elm.l
                magnet_strengths[elm.name + ".k0"] = elm.k0

    # Create unique corrector file path based on destination
    corrector_file = tmp_dir / f"correctors_b{beam}.tfs"

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
        beam=beam,
    )
    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=tmp_dir / f"env_config_b{beam}.json",
        sequence_file=sequence_file,
        seq_name=seq_name,
    )

    # save the tune knobs to file with unique name
    tune_knobs_file = tmp_dir / f"tune_knobs_b{beam}.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    # Compute twiss for ACD insertion
    tws = env[seq_name].twiss(method="4d")

    # Insert AC dipole
    acd_ramp = 1000
    total_turns = flattop_turns + acd_ramp
    driven_tunes = [0.27, 0.322]
    output_files = []
    for lag in [np.pi / 3]:
        line = insert_ac_dipole(
            env[seq_name],
            tws,
            beam=beam,
            acd_ramp=acd_ramp,
            total_turns=total_turns,
            driven_tunes=driven_tunes,
            lag=lag,
        )

        insert_particle_monitors_at_pattern(
            line,
            pattern="bpm.*[^k]",
            num_turns=flattop_turns,
            num_particles=1,
            inplace=True,
        )
        for dpp in [-5e-4, 0.0, 5e-4]:
            particles = line.build_particles(
                x=[1e-4],
                px=[-1e-6],
                y=[1e-4],
                py=[-1e-6],
                delta=[dpp],
            )
            run_tracking(
                line=line,
                particles=particles,
                nturns=flattop_turns,
            )
            sdds = convert_to_tbt(line)
            output_file = tmp_dir / f"track_result_b{beam}_{dpp}_{lag}.sdds"
            output_files.append(output_file)
            write_tbt(output_file, sdds, noise=1e-4)

    linfile_dir = tmp_dir / f"linfiles_b{beam}"
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
    analysis_dir = tmp_dir / f"analysis_b{beam}"

    hole_in_one_entrypoint(
        optics=True,
        files=all_files,
        outputdir=analysis_dir,
        accel="lhc",
        beam=beam,
        model_dir=model_dir,
        year="2025",
        compensation="equation",
    )
    return analysis_dir


def _validate_dispersion_estimates(
    dispersion_df: tfs.TfsDataFrame,
    twiss_elements: tfs.TfsDataFrame,
    beam: int,
) -> None:
    """Validate dispersion estimates against model values.

    Args:
        dispersion_df: DataFrame with estimated dispersions
        twiss_elements: DataFrame with model twiss values
        beam: Beam number for logging

    Raises:
        pytest.fail: If validation fails for any correctors
    """
    num_validated = 0
    num_failed = 0
    failures = []

    for corrector in dispersion_df.index:
        estimated_disp = dispersion_df.loc[corrector, "DISPERSION"]
        std = dispersion_df.loc[corrector, "STD"]
        twiss_disp = twiss_elements.loc[corrector, "DX"]

        diff = abs(estimated_disp - twiss_disp)
        is_close_rtol = np.isclose(estimated_disp, twiss_disp, rtol=6e-2)
        is_within_std = diff <= std

        if is_close_rtol or is_within_std:
            num_validated += 1
        else:
            num_failed += 1
            failures.append(
                f"Dispersion mismatch for {corrector}:\n"
                f"  Estimated: {estimated_disp:.6e}\n"
                f"  Twiss:     {twiss_disp:.6e}\n"
                f"  Difference: {diff:.6e}\n"
                f"  Std:       {std:.6e}\n"
                f"  Within 6% rtol: {is_close_rtol}\n"
                f"  Within 1 std:   {is_within_std}"
            )

    # Report summary
    logger.info(f"Beam {beam}: Validated {num_validated}/{len(dispersion_df)} correctors")
    logger.info(f"Beam {beam}: Failed {num_failed}/{len(dispersion_df)} correctors")

    # Fail test if any correctors don't match
    if failures:
        failure_msg = "\n\n".join(failures[:5])  # Show first 5 failures
        if len(failures) > 5:
            failure_msg += f"\n\n... and {len(failures) - 5} more failures"
        pytest.fail(
            f"Beam {beam}: Dispersion estimation failed for {num_failed} correctors:\n\n{failure_msg}"
        )


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_disp")


@pytest.mark.slow
def test_dispersion_beam1(
    tmp_dir: Path,
    seq_b1: Path,
    model_dir_b1: Path,
) -> None:
    """Test that dispersion estimation reproduces model values at correctors for Beam 1.

    This test:
    1. Generates tracking data with off-momentum particles
    2. Analyzes the tracking to extract optics at BPMs
    3. Estimates dispersion at correctors by tracking from nearby BPMs
    4. Validates estimates match model within tolerance
    """
    beam = 1
    # Generate tracking data and analyze optics
    optics_dir = _generate_nonoise_track(
        tmp_dir,
        model_dir_b1,
        seq_b1,
        6600,
        beam=beam,
        perturb_quads=True,
        perturb_bends=True,
    )

    # Load model twiss for validation
    twiss_elements = tfs.read(model_dir_b1 / TWISS_ELEMENTS_DAT, index="NAME")

    # Estimate horizontal dispersion at all correctors
    dispersion_df, _statistics_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=seq_b1,
        model_dir=model_dir_b1,
        seq_name=f"lhcb{beam}",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=30,
        plane="x",
    )

    # Validate estimates against model
    _validate_dispersion_estimates(dispersion_df, twiss_elements, beam)


@pytest.mark.slow
def test_dispersion_beam2(
    tmp_dir: Path,
    seq_b2: Path,
    model_dir_b2: Path,
) -> None:
    """Test that dispersion estimation reproduces model values at correctors for Beam 2.

    This test:
    1. Generates tracking data with off-momentum particles
    2. Analyzes the tracking to extract optics at BPMs
    3. Estimates dispersion at correctors by tracking from nearby BPMs
    4. Validates estimates match model within tolerance
    """
    beam = 2
    # Generate tracking data and analyze optics
    optics_dir = _generate_nonoise_track(
        tmp_dir,
        model_dir_b2,
        seq_b2,
        6600,
        beam=beam,
        perturb_quads=True,
        perturb_bends=True,
    )

    # Load model twiss for validation
    twiss_elements = tfs.read(model_dir_b2 / TWISS_ELEMENTS_DAT, index="NAME")
    # Estimate horizontal dispersion at all correctors
    dispersion_df, _statistics_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=seq_b2,
        model_dir=model_dir_b2,
        seq_name=f"lhcb{beam}",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=30,
        plane="x",
    )

    # Validate estimates against model
    _validate_dispersion_estimates(dispersion_df, twiss_elements, beam)
