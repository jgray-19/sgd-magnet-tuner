"""Arc-by-arc deltap optimisation over measurement campaigns (importable).

Builds per-campaign measurement configurations and runs the closed-orbit deltap
arc-by-arc optimisation for each. This logic previously lived in the
``create_datafile_loop`` and ``create_datafile_b2`` entry scripts; it is now
importable so it can be driven from notebooks, tests, or other scripts. The
shared 8-arc :class:`ArcByArcFitter` loop lives in :func:`optimise_arcs_for_deltap`.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import (
    DPP_OPTIMISER_CONFIG,
    DPP_SIMULATION_CONFIG,
    MEASUREMENTS_ARTIFACTS_ROOT,
    PROJECT_ROOT,
)
from aba_optimiser.measurements.arc_config import (
    MeasurementSetupConfig,
    RangeConfig,
    arc_ranges,
)
from aba_optimiser.measurements.create_datafile import process_measurements
from aba_optimiser.measurements.online_knobs import save_online_knobs
from aba_optimiser.measurements.output import measurement_output_config
from aba_optimiser.measurements.sequence import get_or_make_sequence
from aba_optimiser.training.config.models import (
    MeasurementConfig,
    MeasurementDetails,
    SequenceConfig,
)
from aba_optimiser.training.tracking_fitter import ArcByArcFitter

logger = logging.getLogger(__name__)

# The batch campaigns use the first five BPMs (indices 9..13) at each arc boundary.
_LOOP_BPM_INDICES = range(9, 14)
_MEASUREMENT_FILENAME = "pz_data.parquet"


def first_bpm_for_beam(beam: int) -> str:
    """Return the reference first BPM for the arc measurement config."""
    return "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"


def optimise_arcs_for_deltap(
    accelerator: LHC,
    arc_config: RangeConfig,
    measurement_config: MeasurementConfig,
    bad_bpms: list[str],
    results_file: Path,
    *,
    title: str = "",
) -> list[float]:
    """Run the 8-arc deltap optimisation and write per-arc results.

    Shared by every campaign runner: for each arc it builds a
    :class:`SequenceConfig` over that arc's magnet range and runs a
    :class:`ArcByArcFitter`, collecting the fitted ``deltap`` per arc. Writes a
    ``range/deltap`` table plus mean/stddev to ``results_file`` and returns the
    per-arc deltaps.
    """
    with results_file.open("w") as f:
        f.write("Arc\tDeltap\n")

    results: list[float] = []
    for arc in range(8):
        logger.info("Starting optimisation for arc %d/8 %s", arc + 1, title)
        sequence_config = SequenceConfig(
            magnet_range=arc_config.magnet_ranges[arc],
            bad_bpms=bad_bpms,
        )
        fitter = ArcByArcFitter(
            accelerator=accelerator,
            optimiser_config=DPP_OPTIMISER_CONFIG,
            simulation_config=DPP_SIMULATION_CONFIG,
            sequence_config=sequence_config,
            measurement_config=measurement_config,
            bpm_start_points=arc_config.bpm_starts[arc],
            bpm_end_points=arc_config.bpm_end_points[arc],
            initial_knob_strengths=None,
            true_strengths=None,
            output_config=measurement_output_config(
                results_file.parent,
                f"{title}_arc_{arc + 1}",
                include_uncertainty=True,
                parallel_hessian=True,
            ),
        )
        final_knobs, _ = fitter.run()
        # The fitter reports controller-space energy as ``pt``. This standalone
        # measurement writes physical dp/p, so convert only at this boundary.
        results.append(fitter.config_manager.mad_iface.pt2dp(final_knobs["pt"]))
        with results_file.open("a") as f:
            f.write(f"{arc + 1}\t{results[-1]}\n")
        logger.info("Arc %d: deltap = %s", arc + 1, results[-1])

    logger.info("All arc optimisations complete %s.", title)
    logger.info("Mean deltap: %s", np.mean(results))
    logger.info("Std dev of deltap: %s", np.std(results))
    with results_file.open("a") as f:
        f.write(f"Mean\t{np.mean(results)}\n")
        f.write(f"StdDev\t{np.std(results)}\n")
    return results


def _load_bad_bpms(bad_bpms_file: Path, bad_bpms: list[str]) -> list[str]:
    """Persist and reload the bad-BPM list so it round-trips through disk."""
    with bad_bpms_file.open("w") as f:
        for bpm in bad_bpms:
            f.write(f"{bpm}\n")
    with bad_bpms_file.open("r") as f:
        return [line.strip() for line in f.readlines()]


def _measurement_time(date: str, times: list[str]) -> datetime:
    """UTC datetime for the earliest acquisition in a campaign group."""
    time_str = min(times).replace("_", ":")[:8]  # "07_53_05_820" -> "07:53:05"
    return datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("UTC")
    )


def create_beam1_configs(folder: str, name_prefix: str) -> list[MeasurementSetupConfig]:
    """Beam-1 measurement configurations for the 2025-11-07 campaign."""
    model_dir_b1 = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    arc_config_b1 = arc_ranges(beam=1, start_indices=_LOOP_BPM_INDICES, end_indices=_LOOP_BPM_INDICES)
    times_by_title = {
        "0": ["07_53_05_820", "07_54_13_858"],
        "0p2": ["08_08_02_826", "08_09_11_940"],
        "0p1": ["08_11_13_745", "08_12_25_817"],
        "m0p1": ["08_18_09_980", "08_19_16_847"],
        "m0p2": ["08_23_20_980", "08_24_32_020"],
    }
    return [
        MeasurementSetupConfig(
            beam=1,
            model_dir=model_dir_b1,
            arc_config=arc_config_b1,
            folder=folder,
            name_prefix=name_prefix,
            times=times,
            title=title,
        )
        for title, times in times_by_title.items()
    ]


def create_beam2_configs(folder: str, name_prefix: str) -> list[MeasurementSetupConfig]:
    """Beam-2 measurement configurations for the 2025-11-07 campaign."""
    model_dir_b2 = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB2/Models/2025-11-07_B2_12cm"
    )
    arc_config_b2 = arc_ranges(beam=2, start_indices=_LOOP_BPM_INDICES, end_indices=_LOOP_BPM_INDICES)
    # Note: the "0" setting (times ["07_35_27_940", ...]) is intentionally omitted.
    times_by_title = {
        "0p2": ["07_57_30_885", "08_00_44_900"],
        "0p1": ["08_04_55_798", "08_06_06_900", "08_07_13_900"],
        "m0p1": ["08_15_06_860", "08_16_13_980"],
        "m0p2": ["08_19_35_860", "08_22_57_752"],
    }
    return [
        MeasurementSetupConfig(
            beam=2,
            model_dir=model_dir_b2,
            arc_config=arc_config_b2,
            folder=folder,
            name_prefix=name_prefix,
            times=times,
            title=title,
        )
        for title, times in times_by_title.items()
    ]


def process_single_config(
    config: MeasurementSetupConfig,
    temp_analysis_dir: Path,
    date: str,
) -> list[float]:
    """Reconstruct one campaign setting and run its arc-by-arc deltap optimisation.

    SDDS files are resolved from the flat ``<folder>/<name_prefix><time>.sdds``
    layout. Returns the per-arc deltaps.
    """
    results_dir = MEASUREMENTS_ARTIFACTS_ROOT / "results" / f"b{config.beam}_results"
    results_dir.mkdir(exist_ok=True)

    if temp_analysis_dir.exists():
        shutil.rmtree(temp_analysis_dir)
    temp_analysis_dir.mkdir()

    if not config.times:
        logger.warning("No times specified for config %s, skipping.", config.title)
        return []

    measurement_file = temp_analysis_dir / _MEASUREMENT_FILENAME
    bad_bpms_file = temp_analysis_dir / "bad_bpms.txt"

    save_online_knobs(_measurement_time(date, config.times), beam=config.beam)
    accelerator = LHC(
        beam=config.beam,
        sequence_file=get_or_make_sequence(config.beam, Path(config.model_dir)),
        kinetic_energy=6800,
    )

    files = [Path(f"{config.folder}/{config.name_prefix}{time}.sdds") for time in config.times]
    _, bad_bpms, _, _ = process_measurements(
        files,
        temp_analysis_dir,
        config.model_dir,
        accelerator=accelerator,
        # No blank orbit is acquired by this campaign, so the model closed orbit is
        # the reference. See aba_optimiser.measurements.reference for the cost.
        reference_closed_orbit="model",
        filename=_MEASUREMENT_FILENAME,
    )
    bad_bpms = _load_bad_bpms(bad_bpms_file, bad_bpms)

    measurement_config = MeasurementConfig(
        {measurement_file: MeasurementDetails(first_bpm=first_bpm_for_beam(config.beam))}
    )
    results = optimise_arcs_for_deltap(
        accelerator,
        config.arc_config,
        measurement_config,
        bad_bpms,
        results_dir / f"{config.title}.txt",
        title=f"for {config.title}",
    )
    shutil.rmtree(temp_analysis_dir)
    return results


def run_beam_campaign(
    beam: int,
    *,
    date: str = "2025-11-07",
    folder: str = "/nfs/cs-ccr-nfs4/lhc_data/OP_DATA/FILL_DATA/11259/BPM",
    name_prefix: str | None = None,
) -> None:
    """Run the full 2025-11-07 arc-by-arc deltap optimisation for one beam."""
    if name_prefix is None:
        name_prefix = f"Beam{beam}@BunchTurn@{date.replace('-', '_')}@"
    configs = (
        create_beam1_configs(folder, name_prefix)
        if beam == 1
        else create_beam2_configs(folder, name_prefix)
    )
    temp_analysis_dir = MEASUREMENTS_ARTIFACTS_ROOT / "temp" / "temp_analysis"
    for config in configs:
        process_single_config(config, temp_analysis_dir, date)


def run_beam2_2025_04_09_campaign(
    analysis_dir: Path = PROJECT_ROOT / "analysis_b2",
) -> list[float]:
    """Run the older 2025-04-09 beam-2 campaign (per-time subfolder SDDS layout)."""
    model_dir = (
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB2/Models/2025_LHCB2_0p18m"
    )
    arc_config = arc_ranges(beam=2, start_indices=range(9, 14), end_indices=range(9, 14))
    folder = Path("/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB2/Measurements/")
    name_prefix = "Beam2@BunchTurn@2025_04_09@"
    times = ["18_48_02_383", "18_49_07_430", "18_50_10_785"]
    files = [folder / f"{name_prefix}{t}/{name_prefix}{t}.sdds" for t in times]

    analysis_dir.mkdir(parents=True, exist_ok=True)
    meas_time = datetime.strptime("2025-04-09 18:47:50", "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=ZoneInfo("UTC")
    )
    save_online_knobs(meas_time, beam=2)

    measurement_file = analysis_dir / _MEASUREMENT_FILENAME
    bad_bpms_file = analysis_dir / "bad_bpms.txt"
    accelerator = LHC(
        beam=2,
        sequence_file=get_or_make_sequence(2, Path(model_dir)),
        kinetic_energy=6800,
    )

    _, bad_bpms, _, _ = process_measurements(
        files,
        analysis_dir,
        model_dir,
        accelerator=accelerator,
        # No blank orbit is acquired by this campaign, so the model closed orbit is
        # the reference. See aba_optimiser.measurements.reference for the cost.
        reference_closed_orbit="model",
        filename=_MEASUREMENT_FILENAME,
    )
    bad_bpms = _load_bad_bpms(bad_bpms_file, bad_bpms)

    measurement_config = MeasurementConfig(
        {measurement_file: MeasurementDetails(first_bpm=first_bpm_for_beam(2))}
    )
    return optimise_arcs_for_deltap(
        accelerator,
        arc_config,
        measurement_config,
        bad_bpms,
        analysis_dir / "deltap_results.txt",
    )
