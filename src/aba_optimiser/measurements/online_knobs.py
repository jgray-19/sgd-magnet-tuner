"""NXCALS online knob downloading utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from aba_optimiser.config import CORRECTOR_STRENGTHS, TUNE_KNOBS_FILE

LOGGER = logging.getLogger(__name__)


def build_dict_from_nxcal_result(result: list) -> dict[str, float]:
    """Convert a list of NXCALSResult objects to a {name: value} dict."""
    return {res.name: res.value for res in result}


def save_online_knobs(
    meas_time: datetime,
    beam: int,
    tune_knobs_file: Path | None = None,
    corrector_knobs_file: Path | None = None,
    energy: float | None = None,
) -> float:
    """Download and save knob data from NXCALS for the given measurement time.

    Returns the beam energy in GeV.
    """
    try:
        from nxcals.spark_session_builder import get_or_create
        from omc3.machine_data_extraction.mqt_extraction import get_mqt_vals
        from omc3.machine_data_extraction.nxcals_knobs import get_energy

        from aba_optimiser.measurements import knob_extraction
    except ImportError as e:
        raise ImportError(
            "nxcals is required for save_online_knobs but is not installed."
        ) from e

    from pymadng_utils.io.utils import save_knobs

    spark = get_or_create()
    if energy is None:
        energy, _ = get_energy(spark, meas_time)

    mq_results = knob_extraction.get_mq_vals(spark, meas_time, beam, energy=energy)
    mqt_results = get_mqt_vals(spark, meas_time, beam, energy=energy)
    ms_results = knob_extraction.get_ms_vals(spark, meas_time, beam, energy=energy)
    mb_results = knob_extraction.get_mb_vals(spark, meas_time, beam, energy=energy)
    corrector_results = knob_extraction.get_mcb_vals(spark, meas_time, beam, energy=energy)
    # Stop Spark to avoid conflicts with multiprocessing
    spark.stop()
    del spark

    main_magnet_knobs = {
        **build_dict_from_nxcal_result(mq_results),
        **build_dict_from_nxcal_result(mqt_results),
        **build_dict_from_nxcal_result(ms_results),
        **build_dict_from_nxcal_result(mb_results),
    }
    corrector_knobs = build_dict_from_nxcal_result(corrector_results)

    save_knobs(main_magnet_knobs, tune_knobs_file or TUNE_KNOBS_FILE)
    save_knobs(corrector_knobs, corrector_knobs_file or CORRECTOR_STRENGTHS)

    return float(energy)
