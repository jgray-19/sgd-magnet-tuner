"""Helper functions for creating controller configurations.

This module provides convenience functions to reduce duplication when
creating controller configuration objects for common use cases.
"""

from __future__ import annotations

from pathlib import Path

from aba_optimiser.training.config.models import MeasurementConfig, MeasurementDetails


def create_arc_measurement_config(
    measurement_file: Path,
    machine_deltap: float = 0.0,
    corrector_strengths: Path | None = None,
    tune_knobs_file: Path | None = None,
    first_bpm: str | None = None,
) -> MeasurementConfig:
    """Build a single-file MeasurementConfig.

    Convenience for the common one-measurement case. Multi-file runs build the
    ``{path: MeasurementDetails(...)}`` mapping directly. The bunch structure is
    read from the parquet's ``bunch_number`` column, so it is not configured here.
    ``first_bpm`` names the BPM the recorded turns begin at; leave it ``None`` to
    use the file's own first recorded BPM.
    """
    interface_options: dict = {}
    if corrector_strengths is not None:
        interface_options["corrector_strengths"] = corrector_strengths
    if tune_knobs_file is not None:
        interface_options["tune_knobs_file"] = tune_knobs_file
    return MeasurementConfig(
        {
            Path(measurement_file): MeasurementDetails(
                interface_options=interface_options,
                machine_deltap=machine_deltap,
                first_bpm=first_bpm,
            )
        }
    )
