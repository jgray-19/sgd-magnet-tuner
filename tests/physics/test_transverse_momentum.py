"""Integration tests for transverse momentum reconstruction using xtrack data."""

from __future__ import annotations

import numpy as np
import pytest
import tfs

pytest.importorskip("xtrack")
pytest.importorskip("xpart")
pytest.importorskip("xobjects")
pytest.importorskip("matplotlib")

from typing import TYPE_CHECKING

from xobjects import ContextCpu  # noqa: E402

from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.model_creator.config import (
    AC_MARKER_PATTERN,
    DRV_TUNES,
    NAT_TUNES,
)
from aba_optimiser.momentum_recon import calculate_transverse_pz as calculate_pz
from aba_optimiser.simulation.magnet_perturbations import (
    apply_magnet_perturbations,
)
from aba_optimiser.simulation.optics import perform_orbit_correction
from aba_optimiser.xsuite.xsuite_tools import (
    create_xsuite_environment,
    initialise_env,
    insert_ac_dipole,
    insert_particle_monitors_at_pattern,
    line_to_dataframes,
)

from .momentum_test_utils import (  # noqa: E402
    verify_pz_reconstruction,
)

if TYPE_CHECKING:
    from xtrack import Line


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _setup_xsuite_simulation(
    delta_p,
    do_apply_magnet_perturbations,
    magnet_seed,
    json_path,
    sequence_file,
    tmp_path,
    test_id,
):
    corrector_file = tmp_path / f"correctors_{test_id}.tfs"

    mad = BaseMadInterface()
    mad.load_sequence(sequence_file, "lhcb1")
    mad.setup_beam(beam_energy=6800)

    magnet_strengths = {}
    if do_apply_magnet_perturbations:
        magnet_strengths, _ = apply_magnet_perturbations(
            mad.mad, rel_k1_std_dev=1e-4, seed=magnet_seed
        )
        assert magnet_strengths, "Expected magnet perturbations to update strengths"

    # Perform orbit correction
    matched_tunes = perform_orbit_correction(
        mad=mad.mad,
        machine_deltap=delta_p,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=corrector_file,
    )

    mad.observe_elements()
    tws = mad.run_twiss(deltap=delta_p)
    tws = tws.loc[tws.index.str.upper().str.contains("BPM")]

    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    env = initialise_env(
        matched_tunes=matched_tunes,
        magnet_strengths=magnet_strengths,
        corrector_table=corrector_table,
        json_file=json_path,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    baseline_line = env["lhcb1"].copy()
    xsuite_tws = baseline_line.twiss(method="4d", delta0=delta_p)

    qx = float(xsuite_tws.qx % 1)
    qy = float(xsuite_tws.qy % 1)
    assert np.isclose(qx, 0.28, atol=5e-4, rtol=5e-4)
    assert np.isclose(qy, 0.31, atol=5e-4, rtol=5e-4)

    ramp_turns = 1000
    flattop_turns = 100
    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=xsuite_tws,
        beam=1,
        acd_ramp=ramp_turns,
        total_turns=flattop_turns + ramp_turns,
        driven_tunes=[0.27, 0.322],
    )

    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=r"(?i)bpm.*",
        num_turns=ramp_turns + flattop_turns,
        num_particles=1,
    )

    ctx = ContextCpu()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=delta_p,
    )

    monitored_line.track(particles, num_turns=flattop_turns + ramp_turns, with_progress=False)

    tracking_df = line_to_dataframes(monitored_line)[0]
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    return tracking_df, truth, tws


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(json_b1, sequence_file):
    env = create_xsuite_environment(
        json_file=json_b1,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )

    baseline_line: Line = env["lhcb1"].copy()
    ng = baseline_line.to_madng()
    tws = baseline_line.twiss(method="4d")

    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, NAT_TUNES[0], atol=1e-6, rtol=1e-6)
    assert np.isclose(qy, NAT_TUNES[1], atol=1e-6, rtol=1e-6)
    qxd = DRV_TUNES[0]
    qyd = DRV_TUNES[1]
    acd_marker = AC_MARKER_PATTERN.format(beam=1).lower()
    betxac = tws.rows[acd_marker]["betx"][0]
    betyac = tws.rows[acd_marker]["bety"][0]
    ac_marker_place = "6.7065629327563011e+03"

    ng.send(f"""
    -- Install AC Kicker (AC Quad) elements
local hackicker, vackicker in MAD.element
!MAD.option.debug = 2;
local a = seq:replace({{
    hackicker "hackicker" {{
        at = {ac_marker_place},

        -- quad part
        nat_q = {qx},
        drv_q = {qxd},
        ac_bet = {betxac},
    }},
    vackicker "vackicker" {{
        at = {ac_marker_place},

        -- quad part
        nat_q = {qy},
        drv_q = {qyd},
        ac_bet = {betyac},
    }}
}}, "{acd_marker}");""")

    ramp_turns = 1000
    flattop_turns = 100
    ac_line = insert_ac_dipole(
        line=baseline_line,
        tws=tws,
        beam=1,
        acd_ramp=ramp_turns,
        total_turns=flattop_turns + ramp_turns,
        driven_tunes=[qxd, qyd],
    )

    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=r"(?i)bpm.*",
        num_turns=ramp_turns + flattop_turns,
        num_particles=1,
    )

    ctx = ContextCpu()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
    )

    monitored_line.track(particles, num_turns=flattop_turns + ramp_turns, with_progress=False)

    tracking_df = line_to_dataframes(monitored_line)[0]
    # Delete the first 100 ramp turns and then reset turn count
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    # Add weights and kick plane info
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    ng["tws", "flw"] = ng.twiss(sequence=ng.seq)
    tws: tfs.TfsDataFrame = (
        ng.tws.to_df()
        .set_index("name")
        .rename(index=str.upper)
        .loc[lambda df: df.index.str.contains("BPM")]
    )

    _verify_pz_reconstruction(
        tracking_df,
        truth,
        tws,
        px_clean_max=2e-7,
        py_clean_max=2e-7,
        px_noisy_min="px_rmse_clean",
        px_noisy_max=3e-6,
        py_noisy_min="py_rmse_clean",
        py_noisy_max=3e-6,
        px_divisor=4.5,
        py_divisor=4.5,
        rng_seed=42,
        subtract_mean=False,
    )


def _verify_pz_reconstruction(
    tracking_df,
    truth,
    tws,
    px_clean_max,
    py_clean_max,
    px_noisy_min,
    px_noisy_max,
    py_noisy_min,
    py_noisy_max,
    px_divisor,
    py_divisor,
    rng_seed=42,
    subtract_mean=True,
):
    """Wrapper around shared verify_pz_reconstruction for backward compatibility."""
    verify_pz_reconstruction(
        tracking_df,
        truth,
        tws,
        calculate_pz,
        px_clean_max,
        py_clean_max,
        px_noisy_min,
        px_noisy_max,
        py_noisy_min,
        py_noisy_max,
        px_divisor,
        py_divisor,
        rng_seed,
        subtract_mean,
    )


@pytest.mark.parametrize(
    "delta_p, do_apply_magnet_perturbations, magnet_seed, px_clean_max, py_clean_max, px_noisy_min, px_noisy_max, py_noisy_min, py_noisy_max, px_divisor, py_divisor",
    [
        pytest.param(
            2e-4,
            False,
            None,
            3e-6,
            6e-7,
            3e-6,
            3.5e-6,
            6e-7,
            3e-6,
            1.4,
            3.0,
            id="orbit_correction_off_momentum",
        ),
        pytest.param(
            0.0,
            True,
            123,
            3e-6,
            6e-7,
            "px_rmse_clean",
            4e-6,
            "py_rmse_clean",
            3e-6,
            1.4,
            3.0,
            id="magnet_perturbations_on_momentum",
        ),
    ],
)
@pytest.mark.slow
def test_calculate_pz_with_corrections_and_perturbations(
    delta_p,
    do_apply_magnet_perturbations,
    magnet_seed,
    px_clean_max,
    py_clean_max,
    px_noisy_min,
    px_noisy_max,
    py_noisy_min,
    py_noisy_max,
    px_divisor,
    py_divisor,
    json_b1_corrected,
    sequence_file,
    tmp_path,
):
    """Test momentum reconstruction with orbit correction and/or magnet perturbations.

    Covers two scenarios:
    - orbit_correction_off_momentum: Verify reconstruction with corrected orbits
    - magnet_perturbations_on_momentum: Verify robustness to random magnet errors
    """
    json_path = json_b1_corrected
    test_id = f"test_{delta_p}_{do_apply_magnet_perturbations}"

    tracking_df, truth, tws = _setup_xsuite_simulation(
        delta_p,
        do_apply_magnet_perturbations,
        magnet_seed,
        json_path,
        sequence_file,
        tmp_path,
        test_id,
    )

    _verify_pz_reconstruction(
        tracking_df,
        truth,
        tws,
        px_clean_max,
        py_clean_max,
        px_noisy_min,
        px_noisy_max,
        py_noisy_min,
        py_noisy_max,
        px_divisor,
        py_divisor,
        rng_seed=42,
        subtract_mean=True,
    )
