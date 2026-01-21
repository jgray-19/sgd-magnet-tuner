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
from aba_optimiser.xsuite.acd import run_ac_dipole_tracking_with_particles
from aba_optimiser.xsuite.env import create_xsuite_environment, initialise_env
from aba_optimiser.xsuite.monitors import process_tracking_data

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
    mad.mad["zero_twiss", "_"] = mad.mad.twiss(sequence="loaded_sequence")  # ty:ignore[invalid-assignment]

    mad.observe_elements()
    tws = mad.run_twiss()
    tws = tws.loc[tws.index.str.upper().str.contains("BPM")]
    mad.unobserve_elements(["BPM"])

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

    corrector_table = tfs.read(corrector_file)
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]

    env = initialise_env(
        matched_tunes,
        magnet_strengths,
        corrector_table,  # ty:ignore[invalid-argument-type]
        sequence_file=sequence_file,
        json_file=json_path,
        seq_name="lhcb1",
    )

    baseline_line = env["lhcb1"].copy()  # ty:ignore[not-subscriptable]
    xsuite_tws = baseline_line.twiss(method="4d", delta0=delta_p)

    qx = float(xsuite_tws.qx % 1)
    qy = float(xsuite_tws.qy % 1)
    assert np.isclose(qx, 0.28, atol=5e-4, rtol=5e-4)
    assert np.isclose(qy, 0.31, atol=5e-4, rtol=5e-4)

    ramp_turns = 1000
    flattop_turns = 100

    # Use generalized tracking function
    particle_coords = {
        "x": [0],
        "px": [0],
        "y": [0],
        "py": [0],
        "delta": [delta_p],
    }

    monitored_line = run_ac_dipole_tracking_with_particles(
        line=baseline_line,
        tws=xsuite_tws,
        beam=1,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=[0.27, 0.322],
        bpm_pattern=r"(?i)bpm.*",
        particle_coords=particle_coords,
    )

    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        delta_p=delta_p,
        add_variance_columns=False,
    )
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0
    tracking_df["kick_plane"] = "both"

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )

    return tracking_df, truth, tws


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(seq_b1, tmp_path):
    """Test that calculate_pz reconstructs true momenta for on-momentum beam."""
    env = create_xsuite_environment(
        json_file=tmp_path / "lhcb1.json",
        sequence_file=seq_b1,
        seq_name="lhcb1",
    )

    baseline_line: Line = env["lhcb1"].copy()  # ty:ignore[not-subscriptable]
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

    # Use generalized tracking function
    particle_coords = {
        "x": [0],
        "px": [0],
        "y": [0],
        "py": [0],
        "delta": [0.0],
    }

    monitored_line = run_ac_dipole_tracking_with_particles(
        line=baseline_line,
        tws=tws,
        beam=1,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=[qxd, qyd],
        bpm_pattern=r"(?i)bpm.*",
        particle_coords=particle_coords,
    )

    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        delta_p=0.0,
        add_variance_columns=False,
    )
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
        px_nonoise_max=1.4e-7,
        py_nonoise_max=1.25e-7,
        px_noisy_min=1e-6,
        px_noisy_max=2.4e-6,
        py_noisy_min=1e-6,
        py_noisy_max=2.4e-6,
        px_divisor=4.5,
        py_divisor=4.5,
        rng_seed=42,
    )


def _verify_pz_reconstruction(
    tracking_df,
    truth,
    tws,
    px_nonoise_max,
    py_nonoise_max,
    px_noisy_min,
    px_noisy_max,
    py_noisy_min,
    py_noisy_max,
    px_divisor,
    py_divisor,
    rng_seed=42,
):
    """Wrapper around shared verify_pz_reconstruction for backward compatibility."""
    verify_pz_reconstruction(
        tracking_df,
        truth,
        tws,
        calculate_pz,
        px_nonoise_max,
        py_nonoise_max,
        px_noisy_min,
        px_noisy_max,
        py_noisy_min,
        py_noisy_max,
        px_divisor,
        py_divisor,
        rng_seed,
    )


@pytest.mark.parametrize(
    "delta_p, do_apply_magnet_perturbations, magnet_seed",
    [
        pytest.param(
            2e-4,
            False,
            None,
            id="orbit_correction_off_momentum",
        ),
        pytest.param(
            0.0,
            True,
            123,
            id="magnet_perturbations_on_momentum",
        ),
    ],
)
@pytest.mark.slow
def test_calculate_pz_with_corrections_and_perturbations(
    delta_p,
    do_apply_magnet_perturbations,
    magnet_seed,
    seq_b1,
    tmp_path,
    xsuite_json_path,
):
    """Test momentum reconstruction with orbit correction and/or magnet perturbations.

    Covers two scenarios:
    - orbit_correction_off_momentum: Verify reconstruction with corrected orbits
    - magnet_perturbations_on_momentum: Verify robustness to random magnet errors
    """
    # DO NOT EVER INCREASE THESE TOLERANCES, IF THE TESTS START FAILING, FIX THE UNDERLYING ISSUE
    tolerance_values = {
        (2e-4, False, None): {
            "px_nonoise_max": 1.6e-6,
            "py_nonoise_max": 2e-7,
            "px_noisy_min": 2e-6,
            "px_noisy_max": 3e-6,
            "py_noisy_min": 2e-6,
            "py_noisy_max": 3e-6,
            "px_divisor": 1.7,
            "py_divisor": 4.6,
        },
        (0.0, True, 123): {
            "px_nonoise_max": 2e-6,
            "py_nonoise_max": 2e-7,
            "px_noisy_min": 2e-6,
            "px_noisy_max": 3e-6,
            "py_noisy_min": 2e-6,
            "py_noisy_max": 3e-6,
            "px_divisor": 1.7,
            "py_divisor": 4.6,
        },
    }
    json_path = xsuite_json_path("lhcb1.seq")
    test_id = f"test_{delta_p}_{do_apply_magnet_perturbations}"

    tracking_df, truth, tws = _setup_xsuite_simulation(
        delta_p,
        do_apply_magnet_perturbations,
        magnet_seed,
        json_path,
        seq_b1,
        tmp_path,
        test_id,
    )

    tol_dict = tolerance_values[(delta_p, do_apply_magnet_perturbations, magnet_seed)]
    _verify_pz_reconstruction(
        tracking_df,
        truth,
        tws,
        **tol_dict,
        rng_seed=42,
    )
