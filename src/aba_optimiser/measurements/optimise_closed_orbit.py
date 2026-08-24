"""Closed-orbit optimisation workflow for arc-based measurement sets."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nxcals.spark_session_builder import get_or_create
from omc3.machine_data_extraction.nxcals_knobs import get_energy
from pymadng_utils.io.utils import save_knobs
from pymadng_utils.madx import make_madx_sequence

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import (
    MEASUREMENTS_ARTIFACTS_ROOT,
    PROTON_MASS,
    OptimiserConfig,
    SimulationConfig,
)
from aba_optimiser.measurements.ac_dipole import (
    ACDipoleOptimisationWindow,
    window_from_attrs,
)
from aba_optimiser.measurements.arc_config import (
    MeasurementSetupConfig,
    RangeConfig,
    arc_ranges,
)
from aba_optimiser.measurements.b2_errors import resolve_b2_error_table
from aba_optimiser.measurements.create_datafile import (
    ACDipoleReconstructionConfig,
    process_measurements,
)
from aba_optimiser.measurements.online_knobs import save_online_knobs
from aba_optimiser.measurements.orbit_averaging import compute_three_turn_averages
from aba_optimiser.measurements.output import measurement_output_config
from aba_optimiser.measurements.reference import reconstruction_frame
from aba_optimiser.measurements.sequence import (
    get_or_make_sequence,
    make_machine_settings_knobs_file,
)
from aba_optimiser.physics.deltap import deltap_wrt_reference_total_energy
from aba_optimiser.training.config.helpers import create_arc_measurement_config
from aba_optimiser.training.config.models import SequenceConfig
from aba_optimiser.training.tracking_fitter import ArcByArcFitter

if TYPE_CHECKING:
    from tmom_recon import ReconstructionFrame

logger = logging.getLogger(__name__)


def create_ac_dipole_full_ring_config(beam: int, window: ACDipoleOptimisationWindow) -> RangeConfig:
    """Create a single full-ring range anchored around the AC dipole BPM window."""
    suffix = f".B{beam}"
    if not window.bpm_upstream.endswith(suffix):
        raise ValueError(
            f"Upstream BPM {window.bpm_upstream} does not match expected beam suffix {suffix}"
        )
    if not window.bpm_downstream.endswith(suffix):
        raise ValueError(
            f"Downstream BPM {window.bpm_downstream} does not match expected beam suffix {suffix}"
        )

    return RangeConfig(
        magnet_ranges=[f"{window.bpm_downstream}/{window.bpm_upstream}"],
        bpm_starts=[[window.bpm_downstream]],
        bpm_end_points=[[window.bpm_upstream]],
    )


def weighted_mean(values: list[float], uncertainties: list[float]) -> float:
    """Compute weighted mean where weights are 1/sigma^2."""
    finite_pairs = [(v, u) for v, u in zip(values, uncertainties) if u > 0]
    if not finite_pairs:
        raise ValueError("Cannot compute weighted mean without positive uncertainties")
    weights = [1 / (u**2) for _, u in finite_pairs]
    numerator = sum(v * w for (v, _), w in zip(finite_pairs, weights))
    return numerator / sum(weights)


def prepare_sequence_file(
    beam: int,
    model_dir: Path,
    output_dir: Path,
    time: str | None = None,
) -> Path:
    """Prepare the sequence file used throughout one closed-orbit run.

    Beam 2 must use the beam-4 MAD-X convention consistently. Generate that
    sequence explicitly in the per-run analysis directory so preprocessing and
    optimisation both reuse the same file.
    """
    if beam != 2:
        return get_or_make_sequence(beam, model_dir, time=time)

    sequence_file = output_dir / "lhcb2_saved.seq"
    logger.info("Generating beam-4-adapted LHCB2 sequence in %s", sequence_file)

    post_optics_madx_files = None
    if time is not None:
        post_optics_madx_files = [
            make_machine_settings_knobs_file(output_dir / "machine_settings_knobs.madx", time)
        ]

    make_madx_sequence(
        model_dir,
        seq_outdir=output_dir,
        beam4=True,
        post_optics_madx_files=post_optics_madx_files,
    )
    return sequence_file


def optimise_ranges(
    range_config: RangeConfig,
    range_type: str,
    beam: int,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    sequence_path: Path,
    corrector_knobs: Path,
    tune_knobs: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    kinetic_energy: float,
    output_dir: Path,
    write_tensorboard_logs: bool = True,
    b2_errors: Path | None = None,
) -> tuple[list[float], list[float], list[float], float]:
    """Optimise for a given range configuration.

    Returns:
        Tuple of reference-relative deltaps, uncertainties, machine deltaps, and reference kinetic energy.
    """
    results = []
    uncertainties = []
    fitted_machine_deltaps = []
    num_ranges = len(range_config.magnet_ranges)
    reference_kinetic_energy = (
        6800 if abs(kinetic_energy - 6800) < abs(kinetic_energy - 450) else 450
    )
    for i in range(num_ranges):
        logger.info(f"Starting optimisation for {range_type} {i + 1}/{num_ranges} for {title}")

        # Create LHC accelerator instance
        accelerator = LHC(
            beam=beam,
            sequence_file=sequence_path,
            kinetic_energy=kinetic_energy,
            optimise_energy=True,
        )

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
        )

        measurement_config = create_arc_measurement_config(
            measurement_file,
            corrector_knobs=corrector_knobs,
            tune_knobs=tune_knobs,
            first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
            b2_errors=b2_errors,
        )

        logger.info(
            f"Found the start and end BPMs for this range: {range_config.bpm_starts[i]} to {range_config.bpm_end_points[i]}"
        )

        fitter = ArcByArcFitter(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            measurement_config,
            range_config.bpm_starts[i],
            range_config.bpm_end_points[i],
            initial_knob_strengths=None,
            true_strengths=None,
            output_config=measurement_output_config(
                output_dir,
                f"{title}_{range_type}_{i + 1}",
                include_uncertainty=True,
                write_tensorboard_logs=write_tensorboard_logs,
                parallel_hessian=True,
            ),
        )
        _, uncs = fitter.run()
        optimised_pt = fitter.optimisation_loop.best_knobs["pt"]
        fitted_machine_deltap = fitter.config_manager.mad_iface.pt2dp(optimised_pt)
        fitted_machine_deltaps.append(fitted_machine_deltap)
        deltap_wrt_ref = deltap_wrt_reference_total_energy(
            kinetic_energy,
            fitted_machine_deltap,
            reference_kinetic_energy,
            PROTON_MASS,
        )
        results.append(deltap_wrt_ref)
        uncertainties.append(uncs["pt"])
        logger.info(
            "%s %d: optimised pt = %s, machine deltap = %s, reference deltap = %s",
            range_type.capitalize(),
            i + 1,
            optimised_pt,
            fitted_machine_deltap,
            results[-1],
        )
        logger.info(f"Finished optimisation for {range_type} {i + 1}/{num_ranges} for {title}")
    return results, uncertainties, fitted_machine_deltaps, reference_kinetic_energy


def optimise_corrector_ranges(
    range_config: RangeConfig,
    range_type: str,
    beam: int,
    optimiser_config: OptimiserConfig,
    simulation_config: SimulationConfig,
    sequence_path: Path,
    corrector_knobs: Path,
    tune_knobs: Path,
    measurement_file: Path,
    bad_bpms: list[str],
    title: str,
    kinetic_energy: float,
    machine_deltap: float,
    output_dir: Path,
    b2_errors: Path | None = None,
) -> list[dict[str, float]]:
    """Optimise correctors for a given range configuration."""
    results = []
    num_ranges = len(range_config.magnet_ranges)
    for i in range(num_ranges):
        logger.info(
            f"Starting corrector optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )

        # Create LHC accelerator instance
        accelerator = LHC(
            beam=beam,
            sequence_file=sequence_path,
            kinetic_energy=kinetic_energy,
            optimise_correctors=True,
        )

        sequence_config = SequenceConfig(
            magnet_range=range_config.magnet_ranges[i],
            bad_bpms=bad_bpms,
        )

        meas_config = create_arc_measurement_config(
            measurement_file,
            machine_deltap=machine_deltap,
            corrector_knobs=corrector_knobs,
            tune_knobs=tune_knobs,
            first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
            b2_errors=b2_errors,
        )

        fitter = ArcByArcFitter(
            accelerator,
            optimiser_config,
            simulation_config,
            sequence_config,
            meas_config,
            range_config.bpm_starts[i],
            range_config.bpm_end_points[i],
            initial_knob_strengths=None,
            true_strengths=None,
            output_config=measurement_output_config(
                output_dir,
                f"{title}_{range_type}_{i + 1}_correctors",
                include_uncertainty=False,
                parallel_hessian=True,
            ),
        )
        final_knobs, _ = fitter.run()
        results.append(final_knobs)
        logger.info(
            f"Finished corrector optimisation for {range_type} {i + 1}/{num_ranges} for {title}"
        )
    return results


def create_beam1_configs(
    folder: str, name_prefix: str, fixed_bpm: bool
) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 1."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    skip_step = 3 if fixed_bpm else 5
    arc_config_b1 = arc_ranges(
        beam=1,
        start_indices=range(9, 35, skip_step),
        end_indices=range(9, 34, skip_step),
    )

    return [
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_53_05_820", "07_54_13_858"],
            title="0",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_08_02_826", "08_09_11_940"],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_11_13_745", "08_12_25_817"],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_18_09_980", "08_19_16_847"],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_23_20_980", "08_24_32_020"],
            title="m0p2",
        ),
    ]


def create_beam2_configs(
    folder: str, name_prefix: str, use_fixed_bpm: bool
) -> list[MeasurementSetupConfig]:
    """Create measurement configurations for beam 2."""
    model_dir_b2 = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    )
    # Arc settings
    skip_step = 3 if use_fixed_bpm else 5
    arc_config_b2 = arc_ranges(
        beam=2,
        start_indices=range(9, 34, skip_step),
        end_indices=range(9, 35, skip_step),
    )

    return [
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_35_27_940", "07_36_39_380", "07_38_44_035"],
            # times=["07_36_39_380", "07_38_44_035"],
            title="0",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["07_57_30_885", "08_00_44_900"],
            title="0p2",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_04_55_798", "08_06_06_900", "08_07_13_900"],
            # times=["08_06_06_900", "08_07_13_900"],
            title="0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_15_06_860", "08_16_13_980"],
            title="m0p1",
        ),
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=["08_19_35_860", "08_22_57_752", "08_18_27_900"],
            title="m0p2",
        ),
    ]


def _summarise_arc_deltaps(
    results_arcs: list[float],
    uncs_arcs: list[float],
    machine_deltaps: list[float],
) -> tuple[float, float]:
    """Return weighted-mean reference and machine deltaps, falling back to plain means."""
    try:
        mean_arcs = weighted_mean(results_arcs, uncs_arcs)
    except ValueError:
        mean_arcs = float(np.mean(results_arcs))
        logger.warning(
            "Falling back to unweighted mean for arcs due to non-positive uncertainties."
        )
    try:
        mean_machine_deltap = weighted_mean(machine_deltaps, uncs_arcs)
    except ValueError:
        mean_machine_deltap = float(np.mean(machine_deltaps))
        logger.warning(
            "Falling back to unweighted mean for machine deltaps due to non-positive uncertainties."
        )
    return mean_arcs, mean_machine_deltap


def _write_arc_results_file(
    results_file: Path,
    results_arcs: list[float],
    mean_arcs: float,
) -> None:
    """Write per-arc reference deltaps and their summary statistics to ``results_file``."""
    with results_file.open("w") as f:
        f.write("range\treference_deltap\n")

    with results_file.open("a") as f:
        for i, dp in enumerate(results_arcs):
            f.write(f"arc{i + 1}\t{dp}\n")
        f.write(f"MeanArcs\t{mean_arcs}\n")
        std_arcs = float(np.std(results_arcs))
        f.write(f"StdDevArcs\t{std_arcs}\n")
        stderr = std_arcs / np.sqrt(len(results_arcs)) if len(results_arcs) > 0 else 0.0
        f.write(f"StdErrArcs\t{stderr}\n")


def process_single_config(
    config: MeasurementSetupConfig,
    temp_analysis_dir: Path,
    date: str,
    skip_reload: bool,
    optimise_correctors: bool,
    frame: ReconstructionFrame,
    use_fixed_bpm: bool = True,
    acdipole_n_bpms_each_side: int = 1,
    sequence_time: str | None = None,
    acd_mode: bool = False,
) -> None:
    """Process a single measurement configuration.

    Args:
        config: Measurement configuration for this run
        temp_analysis_dir: Temporary directory for analysis outputs
        date: Date string in YYYY-MM-DD format
        skip_reload: If True, skip reloading strengths from LSA and reuse existing analysis
        optimise_correctors: If True, optimise correctors after energy optimisation
        use_fixed_bpm: If True (default), use fixed reference BPM approach.
                       If False, create all combinations of start/end BPMs (Cartesian product).
        acd_mode: If True, additionally run the AC-dipole full-ring optimisation anchored
                  on the reconstructed AC-dipole window. Default False keeps the workflow
                  purely arc-by-arc.
    """
    results_dir = MEASUREMENTS_ARTIFACTS_ROOT / "results" / f"b{config.beam}co_results"
    tune_knobs = results_dir / f"tune_knobs_{config.title}.txt"
    corrector_knobs = results_dir / f"corrector_knobs_{config.title}.txt"
    results_dir.mkdir(exist_ok=True)

    # Delete temp_analysis_dir if it exists
    temp_analysis_dir.mkdir(exist_ok=True)

    bad_bpms_file = results_dir / f"bad_bpms_{config.title}.txt"
    measurement_filename = "pz_data.parquet"
    measurement_file = temp_analysis_dir / measurement_filename
    sequence_file = prepare_sequence_file(
        config.beam,
        Path(config.model_dir),
        temp_analysis_dir,
        time=sequence_time,
    )

    # Compute meas_time always
    if not config.times:
        logger.warning(f"No times specified for config {config.title}, skipping.")
        return
    earliest_time = min(config.times)
    # e.g., "07_53_05_820" -> "07:53:05"
    time_str = earliest_time.replace("_", ":")[:8]
    start_str = f"{date} {time_str}"

    tz = ZoneInfo("UTC")
    meas_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)

    # Get beam energy from NXCALS always
    spark = get_or_create()
    kinetic_energy, _ = get_energy(spark, meas_time, beam=config.beam)
    spark.stop()
    del spark

    bad_bpms: list[str] | None = None
    if skip_reload:
        # Read the bad bpms from the file
        with bad_bpms_file.open("r") as f:
            bad_bpms = [line.strip() for line in f.readlines()]

    if not skip_reload:
        save_online_knobs(
            meas_time,
            beam=config.beam,
            tune_knobs=tune_knobs,
            corrector_knobs=corrector_knobs,
            energy=float(kinetic_energy),
        )

    # Generate files from times
    files = [Path(f"{config.folder}/{config.name_prefix}{time}.sdds") for time in config.times]

    ac_dipole_reconstruction_config = (
        ACDipoleReconstructionConfig(
            n_bpms_each_side=acdipole_n_bpms_each_side,
            tune_knobs_files=[tune_knobs] * len(files),
            corrector_knobs_files=[corrector_knobs] * len(files),
        )
        if acd_mode
        else None
    )
    accelerator = LHC(
        beam=config.beam,
        sequence_file=sequence_file,
        kinetic_energy=float(kinetic_energy),
    )

    # Known LHC dipole b2 errors are applied to both the reconstruction model and
    # the arc optimisation; the online tune knobs saved above compensate the tune
    # shift they introduce.
    b2_errors = resolve_b2_error_table(config.beam, float(kinetic_energy))

    pzs_dict, bad_bpms, output_paths, _ = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        accelerator=accelerator,
        frame=frame,
        filename=None,
        bad_bpms=bad_bpms,
        b2_errors=b2_errors,
        ac_dipole_reconstruction_config=ac_dipole_reconstruction_config,
    )
    pzs = pzs_dict["combined"]

    ana_dir = output_paths["combined"]

    file_path = ana_dir / measurement_filename

    new_df = compute_three_turn_averages(pzs)

    # Overwrite the measurement file
    new_df.to_parquet(file_path)

    if not skip_reload:
        # Save the bad bpms to a file
        with bad_bpms_file.open("w") as f:
            for bpm in bad_bpms:
                f.write(f"{bpm}\n")

    optimiser_config = OptimiserConfig(
        max_epochs=1000,
        warmup_epochs=5,
        warmup_lr_start=5e-2,
        max_lr=1e0,
        min_lr=1e0,
        gradient_converged_value=1e-9,
        optimiser_type="lbfgs",
    )
    simulation_config = SimulationConfig(
        # For pre trimmed data
        num_batches=1,
        num_workers=1,
        use_fixed_bpm=use_fixed_bpm,
        optimise_momenta=False,
        bpm_loss_outlier_sigma=5,
    )

    results_arcs, uncs_arcs, machine_deltaps, _ = optimise_ranges(
        config.arc_config,
        "arc",
        config.beam,
        optimiser_config,
        simulation_config,
        sequence_file,
        corrector_knobs,
        tune_knobs,
        measurement_file,
        bad_bpms,
        config.title,
        kinetic_energy,
        results_dir,
        b2_errors=b2_errors,
    )

    if acd_mode:
        ac_dipole_window = window_from_attrs(pzs.attrs)
        if ac_dipole_window is None:
            raise ValueError(
                "AC-dipole reconstruction did not return an optimisation window. "
                "Expected attrs 'ac_dipole_bpm_upstream' and 'ac_dipole_bpm_downstream'."
            )
        # Opt-in AC-dipole full-ring optimisation, anchored on the reconstructed
        # AC-dipole window. Off by default: the closed-orbit workflow is arc-by-arc.
        ac_dipole_full_ring_config = create_ac_dipole_full_ring_config(
            config.beam, ac_dipole_window
        )
        ac_results, ac_uncertainties, _, _ = optimise_ranges(
            ac_dipole_full_ring_config,
            "acdipole-fullring",
            config.beam,
            optimiser_config,
            simulation_config,
            sequence_file,
            corrector_knobs,
            tune_knobs,
            measurement_file,
            bad_bpms,
            config.title,
            kinetic_energy,
            results_dir,
            write_tensorboard_logs=False,
            b2_errors=b2_errors,
        )
        if not ac_results:
            raise RuntimeError("AC-dipole full-ring optimisation did not produce a result.")
        logger.info("AC-dipole full-ring deltap: %s +/- %s", ac_results[0], ac_uncertainties[0])

    logger.info(f"All arc optimisations complete for {config.title}.")
    logger.info("Final reference deltaps for each arc:")
    for i, dp in enumerate(results_arcs):
        logger.info(f"Arc {i + 1}: deltap = {dp}")

    if not results_arcs:
        logger.warning("No arc results produced; skipping summary output.")
        return

    mean_arcs, mean_machine_deltap = _summarise_arc_deltaps(
        results_arcs, uncs_arcs, machine_deltaps
    )
    logger.info(f"Weighted mean deltap arcs: {mean_arcs}")
    logger.info(f"Std dev of deltap arcs: {np.std(results_arcs)}")
    logger.info(f"Weighted mean fitted machine deltap: {mean_machine_deltap}")

    # Write the results to a file
    results_file = results_dir / f"{config.title}.txt"
    _write_arc_results_file(results_file, results_arcs, mean_arcs)

    if not optimise_correctors:
        return

    corrector_optimiser_config = replace(
        optimiser_config,
        max_epochs=3000,
        warmup_epochs=500,
        min_lr=1e-1,
        warmup_lr_start=1e-4,
    )
    corrector_results = optimise_corrector_ranges(
        config.arc_config,
        "arc",
        config.beam,
        corrector_optimiser_config,
        simulation_config,
        sequence_file,
        corrector_knobs,
        tune_knobs,
        measurement_file,
        bad_bpms,
        config.title,
        kinetic_energy,
        mean_machine_deltap,
        results_dir,
        b2_errors=b2_errors,
    )

    combined_correctors: dict[str, float] = {}
    for arc_idx, arc_knobs in enumerate(corrector_results, start=1):
        for knob, value in arc_knobs.items():
            combined_correctors[knob] = value

    combined_correctors_file = results_dir / f"corrector_knobs_{config.title}_optimised.txt"
    save_knobs(combined_correctors, combined_correctors_file)
    logger.info("Saved combined corrector knobs to %s", combined_correctors_file)


def main():
    """Main function to run the measurement processing loop."""
    # Set logging level
    logging.basicConfig(level=logging.INFO)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, choices=[1, 2], help="Beam number 1 or 2", default=2)
    parser.add_argument(
        "--orbit-zero",
        type=Path,
        required=True,
        help="Parquet table containing the measured setting-zero x/y orbit.",
    )
    parser.add_argument(
        "--skip-reload",
        action="store_true",
        help="Skip reloading strengths from LSA and redoing analysis",
    )
    parser.add_argument(
        "--no-fixed-bpm",
        action="store_true",
        help="Disable fixed BPM for start/end points, pair BPMs element-wise instead",
    )
    parser.add_argument(
        "--optimise-correctors",
        action="store_true",
        help="Optimise correctors after energy optimisation",
    )
    parser.add_argument(
        "--acdipole-n-bpms-each-side",
        type=int,
        default=1,
        help="Number of BPMs per side for AC-dipole momentum reconstruction",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help=(
            "Optional machine-settings extraction time for sequence generation. "
            "Uses the omc3 knob_extractor time format; ISO strings must include timezone."
        ),
    )
    parser.add_argument(
        "--acd-mode",
        "--acd-on",
        action="store_true",
        dest="acd_mode",
        default=False,
        help=(
            "Also run the AC-dipole full-ring optimisation anchored on the reconstructed "
            "AC-dipole window. Off by default (arc-by-arc optimisation only)."
        ),
    )
    args = parser.parse_args()
    frame = reconstruction_frame(pd.read_parquet(args.orbit_zero), dynamic_planes=("x", "y"))

    # Define date
    date = "2025-11-07"
    # folder = "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/11259/BPM"
    folder = "/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11259/BPM"
    name_prefix = f"Beam{args.beam}@BunchTurn@{date.replace('-', '_')}@"

    # Determine use_fixed_bpm from args
    use_fixed_bpm: bool = not args.no_fixed_bpm

    # Get configurations based on beam
    if args.beam == 1:
        configs = create_beam1_configs(folder, name_prefix, use_fixed_bpm)
    else:
        configs = create_beam2_configs(folder, name_prefix, use_fixed_bpm)
    print(configs)

    # Temporary analysis directory
    temp_analysis_dir = MEASUREMENTS_ARTIFACTS_ROOT / "temp" / f"temp_analysis_co_{args.beam}"

    # Process each configuration
    for config in configs:
        process_single_config(
            config,
            temp_analysis_dir,
            date,
            args.skip_reload,
            args.optimise_correctors,
            frame,
            use_fixed_bpm,
            args.acdipole_n_bpms_each_side,
            args.time,
            args.acd_mode,
        )


if __name__ == "__main__":
    main()
