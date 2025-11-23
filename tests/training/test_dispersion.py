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
from turn_by_turn import convert_to_tbt, write_tbt

from aba_optimiser.dispersion.dispersion_estimation import estimate_corrector_dispersions
from aba_optimiser.io.utils import save_knobs
from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import (
    create_xsuite_environment,
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
) -> tuple[Path, dict, Path | None]:
    """Generate a parquet file containing noiseless tracking data for the requested BPMs."""
    # Create MAD interface and load sequence
    mad = BaseMadInterface()  # stdout="/dev/null", redirect_stderr=True
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    # Create unique corrector file path based on destination
    corrector_file = tmp_dir / "correctors.tfs"

    # Perform orbit correction for off-momentum beam (delta = 2e-4)
    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )
    # Read corrector table
    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    # Create xsuite environment with orbit correction applied
    env = create_xsuite_environment(
        beam=1,
        sequence_file=sequence_file,
        beam_energy=6800,
        seq_name="lhcb1",
        rerun_madx=False,
        json_file=tmp_dir / "env_config.json",
    )

    # save the tune knobs to file with unique name
    tune_knobs_file = tmp_dir / "tune_knobs.txt"
    save_knobs(matched_tunes, tune_knobs_file)

    # Compute twiss for ACD insertion
    tws = env["lhcb1"].twiss(method="4d")

    # Insert AC dipole
    acd_ramp = 1000
    total_turns = flattop_turns + acd_ramp
    driven_tunes = [0.27, 0.322]
    output_files = []
    for lag in [np.pi / 3]:
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
        for dpp in [-5e-4, 0.0, 5e-4]:
            particles = lhcb1.build_particles(
                x=[1e-4],
                px=[-1e-6],
                y=[1e-4],
                py=[-1e-6],
                delta=[dpp],
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
    return analysis_dir


@pytest.fixture(scope="module")
def tmp_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    return tmp_path_factory.mktemp("aba_controller_disp")


@pytest.mark.slow
def test_dispersion(
    tmp_dir: Path,
    sequence_file: Path,
    model_dir: Path,
) -> None:
    """Test that dispersion estimation reproduces model values at correctors.

    This test:
    1. Generates tracking data with off-momentum particles
    2. Analyzes the tracking to extract optics at BPMs
    3. Estimates dispersion at correctors by tracking from nearby BPMs
    4. Validates estimates match model within tolerance
    """
    # Generate tracking data and analyze optics
    optics_dir = _generate_nonoise_track(
        tmp_dir,
        model_dir,
        sequence_file,
        6600,
    )

    # Load model twiss for validation
    twiss_elements = tfs.read(model_dir / TWISS_ELEMENTS_DAT, index="NAME")

    # Estimate horizontal dispersion at all correctors
    dispersion_df, _statistics_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=sequence_file,
        model_dir=model_dir,
        seq_name="lhcb1",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=10,
        plane="x",
    )

    # Validate estimates against model
    num_validated = 0
    num_failed = 0
    failures = []

    for corrector in dispersion_df.index:
        estimated_disp = dispersion_df.loc[corrector, "DISPERSION"]
        std = dispersion_df.loc[corrector, "STD"]
        twiss_disp = twiss_elements.loc[corrector, "DX"]

        diff = abs(estimated_disp - twiss_disp)
        is_close_rtol = np.isclose(estimated_disp, twiss_disp, rtol=1e-2)
        is_within_std = diff <= std * 2

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
                f"  Within 1% rtol: {is_close_rtol}\n"
                f"  Within 2 std:   {is_within_std}"
            )

    # Report summary
    logger.info(f"Validated {num_validated}/{len(dispersion_df)} correctors")
    logger.info(f"Failed {num_failed}/{len(dispersion_df)} correctors")

    # Fail test if any correctors don't match
    if failures:
        failure_msg = "\n\n".join(failures[:5])  # Show first 5 failures
        if len(failures) > 5:
            failure_msg += f"\n\n... and {len(failures) - 5} more failures"
        pytest.fail(f"Dispersion estimation failed for {num_failed} correctors:\n\n{failure_msg}")
