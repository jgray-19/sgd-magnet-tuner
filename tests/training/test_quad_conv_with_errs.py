"""
Integration test for quadrupole convergence with errors using the measurement-processing pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")

import tfs
from omc3.scripts.fake_measurement_from_model import generate as fake_measurement
from pymadng_utils.io.utils import save_knobs
from tmom_recon import ACDipoleConfig
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.monitors import line_to_dataframes

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.measurements.optimise_squeeze_quads import (
    get_ac_dipole_bpm_points,
    window_from_attrs,
)
from aba_optimiser.measurements.reconstruction import process_single_dataframe
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    MeasurementDetails,
    OutputConfig,
    SequenceConfig,
)
from aba_optimiser.training.controller import Controller
from tests.training.helpers import generate_xsuite_env_with_errors, get_twiss_without_errors

if TYPE_CHECKING:
    from pathlib import Path

    from xtrack import xt

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.serial


def _should_plot() -> bool:
    """Determine whether to generate plots based on environment variable."""
    return os.getenv("PLOT_TEST_OUTPUT", "0") == "1"


def _normalise_knob_values(knobs: dict[str, object]) -> dict[str, float]:
    """Convert MAD/NumPy scalar-like knob values into plain Python floats."""
    normalised: dict[str, float] = {}
    for name, value in knobs.items():
        normalised[name] = float(np.asarray(value).reshape(-1)[0])
    return normalised


def _write_ac_dipole_measurements(
    env: xt.Environment,
    flattop_turns: int,
    acd_ramp: int,
    lags: list[float],
) -> list[pd.DataFrame]:
    """Generate raw AC-dipole turn-by-turn frames for each requested lag."""
    driven_tunes = [0.27, 0.322]
    line: xt.Line = env["lhcb1"]
    tws = line.twiss(method="4d", delta0=0)
    output_frames: list[pd.DataFrame] = []

    for idx, lag in enumerate(lags):
        monitored_line = run_ac_dipole_tracking_with_particles(
            line=line,
            tws=tws,
            acd_marker="mkqa.6l4.b1",
            sequence_name="lhcb1",
            ramp_turns=acd_ramp,
            flattop_turns=flattop_turns,
            driven_tunes=driven_tunes,
            lag=lag,
            bpm_pattern="bpm.*[^k]",
            particle_coords={
                "x": [0.0],
                "px": [0.0],
                "y": [0.0],
                "py": [0.0],
                "delta": [0.0],
            },
        )

        track_df = line_to_dataframes(monitored_line)[0]
        track_df = track_df[track_df["turn"] > acd_ramp].copy()
        track_df["turn"] = track_df["turn"] - acd_ramp - 1
        track_df = track_df[track_df["turn"] < flattop_turns].copy()
        track_df["bunch_number"] = idx
        track_df = track_df[
            ~track_df["name"].str.contains("bpmcs\\.", case=False, regex=True)
        ].copy()
        track_df["name"] = track_df["name"].str.upper()
        output_frames.append(track_df[["name", "turn", "x", "y"]].copy())

    return output_frames


def _generate_measurement_twiss(tmp_path: Path, loaded_interface: AbaMadInterface) -> Path:
    """Create a fake measurement folder that provides the Twiss inputs for reconstruction."""
    analysis_dir = tmp_path / "analysis"
    loaded_interface.observe_elements()
    twiss = loaded_interface.run_twiss(coupling=True)
    twiss.columns = [col.upper() for col in twiss.columns]
    twiss.rename(columns={"MU1": "MUX", "MU2": "MUY"}, inplace=True)
    twiss.headers = {key.upper(): value for key, value in twiss.headers.items()}
    fake_measurement(twiss=twiss, outputdir=analysis_dir)
    return analysis_dir


def _make_optimiser_config_bend() -> OptimiserConfig:
    return OptimiserConfig(
        max_epochs=50,
        warmup_epochs=20,
        warmup_lr_start=4e-6,
        max_lr=4e-7,
        min_lr=4e-7,
        gradient_converged_value=1e-10,
        optimiser_type="adam",
    )


@pytest.mark.skipif(
    os.cpu_count() is not None and (os.cpu_count() < 32),  # ty:ignore[unsupported-operator]
    reason="Requires at least 32 CPU cores for parallel processing",
)
@pytest.mark.slow
def test_controller_bend_opt_simple(
    tmp_path: Path,
    seq_b1: Path,
    model_dir_b1: Path,
    loaded_interface: AbaMadInterface,
) -> None:
    """Test bend optimisation using the same measurement processing flow as optimise_squeeze_quads."""
    flattop_turns = 300
    turns_per_batch = 50
    acd_ramp = 2_000

    corrector_file = tmp_path / "corrector_track_off_magnet.tfs"
    tune_knobs_file = tmp_path / "tune_knobs_track_off_magnet.json"

    env, magnet_strengths, matched_tunes, _ = generate_xsuite_env_with_errors(
        loaded_interface,
        dpp_value=0,
        corrector_file=corrector_file,
        perturb_quads=True,
        perturb_bends=True,
    )
    twiss_errs = loaded_interface.run_twiss(observe=0)
    matched_tunes = _normalise_knob_values(matched_tunes)
    save_knobs(matched_tunes, tune_knobs_file)

    measurement_sources = _write_ac_dipole_measurements(
        env=env,
        flattop_turns=flattop_turns,
        acd_ramp=acd_ramp,
        lags=np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist(),
    )
    analysis_dir = _generate_measurement_twiss(tmp_path, loaded_interface)
    tws_no_err = get_twiss_without_errors(seq_b1, just_bpms=True)

    ac_dipole_model = ACDipoleMadDriver(
        accelerator=LHC(beam=1, sequence_file=seq_b1, kinetic_energy=6800.0),
        pt=0.0,
        observed_elements=loaded_interface.accelerator.get_ac_dipole_marker(),
        discard_mad_output=True,
    )
    ac_dipole_config = ACDipoleConfig(
        ac_dipole_marker=loaded_interface.accelerator.get_ac_dipole_marker(),
        model=ac_dipole_model,
        dpx_tune=0.27,
        dpy_tune=0.322,
    )

    processed_dir = tmp_path / "processed_measurements"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_measurements: list[pd.DataFrame] = []
    bad_bpms: list[str] = []
    try:
        for idx, measurement_df in enumerate(measurement_sources):
            _, processed_df = process_single_dataframe(
                df_with_index=(idx, measurement_df),
                twiss=tws_no_err,
                bad_bpms=bad_bpms,
                analysis_dir=analysis_dir,
                use_uniform_vars=False,
                beam=1,
                ac_dipole_config_factory=lambda _idx: ac_dipole_config,
            )
            processed_measurements.append(processed_df)
    finally:
        ac_dipole_model.close()

    measurement_files = []
    for idx, processed_df in enumerate(processed_measurements):
        output_path = processed_dir / f"pz_data_{idx}.parquet"
        processed_df.to_parquet(output_path, index=False)
        measurement_files.append(output_path)

    first_measurement = processed_measurements[0]
    ac_dipole_window = window_from_attrs(first_measurement.attrs)
    if ac_dipole_window is None:
        raise ValueError("AC-dipole reconstruction did not return BPM marker metadata.")
    reference_deltap = float(
        np.mean([measurement.attrs["DPP_EST"] for measurement in processed_measurements])
    )
    machine_deltaps = [
        float(measurement.attrs["DPP_EST"] - reference_deltap)
        for measurement in processed_measurements
    ]
    corrector_files = [corrector_file] * len(measurement_files)
    tune_knobs_files = [tune_knobs_file] * len(measurement_files)

    optimiser_config = _make_optimiser_config_bend()
    all_estimates = {}

    def _run_optimisation_for_range(
        magnet_range: str,
        start_points: list[str],
        end_points: list[str],
        optimise_quadrupoles: bool,
        optimise_other_quadrupoles: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        tracks_per_worker = flattop_turns
        num_batches = int(np.ceil(tracks_per_worker / turns_per_batch))

        lhc_accelerator = LHC(
            beam=1,
            sequence_file=seq_b1,
            optimise_correctors=False,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_other_quadrupoles=optimise_other_quadrupoles,
        )

        sim_config = SimulationConfig(
            tracks_per_worker=tracks_per_worker,
            num_batches=num_batches,
            num_workers=1,
            optimise_momenta=False,
            use_fixed_bpm=True,
        )

        sequence_config = SequenceConfig(
            magnet_range=magnet_range,
            bad_bpms=bad_bpms,
            first_bpm="MSIA.EXIT.B1",
        )

        measurement_config = MeasurementConfig(
            {
                measurement_file: MeasurementDetails(
                    interface_options={
                        "corrector_strengths": corrector,
                        "tune_knobs_file": tune_knobs,
                    },
                    machine_deltap=deltap,
                )
                for measurement_file, corrector, tune_knobs, deltap in zip(
                    measurement_files, corrector_files, tune_knobs_files, machine_deltaps
                )
            }
        )

        ctrl = Controller(
            accelerator=lhc_accelerator,
            optimiser_config=optimiser_config,
            simulation_config=sim_config,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=start_points,
            bpm_end_points=end_points,
            output_config=OutputConfig(),
            true_strengths=magnet_strengths,
            debug=True,
        )
        return ctrl.run()

    magnet_range, bpm_start_points, bpm_end_points = get_ac_dipole_bpm_points(
        beam=1,
        window=ac_dipole_window,
    )
    estimate, _ = _run_optimisation_for_range(
        magnet_range=magnet_range,
        start_points=bpm_start_points,
        end_points=bpm_end_points,
        optimise_quadrupoles=True,
        optimise_other_quadrupoles=False,
    )
    all_estimates.update(estimate)
    plt.close("all")

    estimated_strengths_file = tmp_path / "estimated_quad_strengths.json"
    with estimated_strengths_file.open("w") as f:
        json.dump(all_estimates, f)

    tws_no_err = get_twiss_without_errors(seq_b1, just_bpms=False)

    tws_errs_betax = (twiss_errs.loc[:, "beta11"] - tws_no_err.loc[:, "beta11"]) / tws_no_err.loc[
        :, "beta11"
    ]
    tws_errs_betay = (twiss_errs.loc[:, "beta22"] - tws_no_err.loc[:, "beta22"]) / tws_no_err.loc[
        :, "beta22"
    ]

    beta_beating_before = pd.DataFrame(
        {
            "s": twiss_errs.loc[:, "s"],
            "name": twiss_errs.index,
            "betax_error_percent": tws_errs_betax * 100,
            "betay_error_percent": tws_errs_betay * 100,
        }
    )
    beta_beating_before_file = tmp_path / "beta_beating_before_correction.tfs"
    tfs.write(beta_beating_before_file, beta_beating_before)

    tws_est = get_twiss_without_errors(
        seq_b1,
        just_bpms=False,
        estimated_magnets=all_estimates,
        corrector_file=corrector_file,
        tune_knobs_file=tune_knobs_file,
    )
    tws_est_betax = (twiss_errs.loc[:, "beta11"] - tws_est.loc[:, "beta11"]) / tws_est.loc[
        :, "beta11"
    ]
    tws_est_betay = (twiss_errs.loc[:, "beta22"] - tws_est.loc[:, "beta22"]) / tws_est.loc[
        :, "beta22"
    ]

    beta_beating_after = pd.DataFrame(
        {
            "s": tws_est["s"],
            "name": tws_est.index,
            "betax_error_percent": tws_est_betax * 100,
            "betay_error_percent": tws_est_betay * 100,
        }
    )
    beta_beating_after_file = tmp_path / "beta_beating_after_correction.tfs"
    tfs.write(beta_beating_after_file, beta_beating_after)

    if _should_plot():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(tws_errs_betax * 100, label="BetaX error before correction")
        plt.plot(tws_est_betax * 100, label="BetaX error after correction")
        plt.xlabel("Element index")
        plt.ylabel("BetaX error (%)")
        plt.title("BetaX beating before and after correction")
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(tws_errs_betay * 100, label="BetaY error before correction")
        plt.plot(tws_est_betay * 100, label="BetaY error after correction")
        plt.xlabel("Element index")
        plt.ylabel("BetaY error (%)")
        plt.title("BetaY beating before and after correction")
        plt.legend()
        plt.grid()

        plot_file = tmp_path / "beta_beating_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.show()

    assert all(tws_est_betax.abs() < 0.0025), "BetaX errors exceed 0.25% after optimisation"
    assert all(tws_est_betay.abs() < 0.005), "BetaY errors exceed 0.5% after optimisation"
    assert any(tws_errs_betax.abs() > 0.005), "Original BetaX errors were not larger than 0.5%"
    assert any(tws_errs_betay.abs() > 0.01), "Original BetaY errors were not larger than 1%"
