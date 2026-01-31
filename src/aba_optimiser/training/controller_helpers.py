"""Helper functions for creating controller configurations.

This module provides convenience functions to reduce duplication when
creating controller configuration objects for common use cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.training.controller_config import BPMConfig, MeasurementConfig

if TYPE_CHECKING:
    from pathlib import Path


def create_arc_measurement_config(
    measurement_file: Path,
    machine_deltap: float = 0.0,
    num_tracks: int = 3,
    flattop_turns: int = 6600,
    corrector_files: list[Path | None] | Path | None = None,
    tune_knobs_files: list[Path | None] | Path | None = None,
) -> MeasurementConfig:
    """Create a MeasurementConfig for arc-by-arc processing.

    Args:
        measurement_file: Path to measurement file
        machine_deltap: Machine momentum offset
        num_tracks: Number of particle tracks per measurement file
        flattop_turns: Number of turns recorded on the flat top
        corrector_files: Optional corrector strength file(s)
        tune_knobs_files: Optional tune knob file(s)

    Returns:
        MeasurementConfig configured for arc processing
    """
    return MeasurementConfig(
        measurement_files=measurement_file,
        corrector_files=corrector_files,
        tune_knobs_files=tune_knobs_files,
        machine_deltaps=machine_deltap,
        bunches_per_file=num_tracks,
        flattop_turns=flattop_turns,
    )


def create_arc_bpm_config(
    bpm_starts: list[str],
    bpm_ends: list[str],
) -> BPMConfig:
    """Create a BPMConfig for arc processing.

    Args:
        bpm_starts: List of starting BPM names
        bpm_ends: List of ending BPM names

    Returns:
        BPMConfig configured with the specified BPM ranges
    """
    return BPMConfig(
        start_points=bpm_starts,
        end_points=bpm_ends,
    )
