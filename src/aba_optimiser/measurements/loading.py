"""Raw measurement loading helpers."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from turn_by_turn import TbtData, read_tbt

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


# =============================================================================
# !! LHC BPM HORIZONTAL SIGN CONVENTION — READ THIS BEFORE TOUCHING BPM DATA !!
# =============================================================================
#
# The LHC BPMs report positions in the beam-2 reference frame, but this codebase
# tracks with a beam-4 sequence (we auto-generate beam 4, never beam 2). Beam 4
# runs the machine in the opposite horizontal sense to beam 2, so a beam-2 BPM
# reading has the OPPOSITE horizontal sign to the beam-4 tracking convention:
#
#   BPM hardware (beam-2 frame):  positive x  ↔  beam-4 tracking:  negative x
#
# Therefore: flip x for beam 2 (x_tracking = -x_bpm); do NOT flip for beam 1.
#
# ALWAYS load LHC BPM data through `read_lhc_bpm_tbt` (defined below), NEVER call
# `read_tbt(..., datatype="lhc")` directly elsewhere. The sign correction and the
# mm→m unit conversion are applied once, here, so every downstream consumer works
# in the beam-4 tracking frame, in metres, without any further correction.
#
# History: the sign correction used to be threaded as a `flip_x` flag through
# every layer, keyed (incorrectly) on `beam == 4` — always false, since the
# accelerator is labelled beam 2. That never flipped anything. The authoritative
# correction now lives in `read_lhc_bpm_tbt` and keys on `beam == 2`.
# =============================================================================


def read_lhc_bpm_tbt(path: Path, *, beam: int) -> TbtData:
    """Load LHC turn-by-turn data in the beam-4 tracking frame, in metres.

    Reads the raw SDDS, negates all horizontal BPM readings for beam 2 (see the
    block comment above — beam-2 BPMs report in the opposite horizontal sense to
    the beam-4 sequence we track), and converts mm→m. Beam 1 is never flipped.

    Use this everywhere instead of calling ``read_tbt`` directly.
    """
    LOGGER.info("Loading data from %s", path)
    tbt_data = read_tbt(path, datatype="lhc")
    x_sign = -1.0 if beam == 2 else 1.0
    # Build new matrices rather than mutating in place: the underlying numpy array
    # may be read-only (e.g. memory-mapped from the SDDS file).
    corrected = [replace(m, X=(m.X * x_sign) / 1000.0, Y=m.Y / 1000.0) for m in tbt_data.matrices]
    return TbtData(
        matrices=corrected,
        nturns=tbt_data.nturns,
        bunch_ids=tbt_data.bunch_ids,
        meta=tbt_data.meta,
    )


def build_dataframe_file_indices(measurements: list[TbtData]) -> list[int]:
    """Map each converted dataframe back to the source measurement-file index."""
    file_indices: list[int] = []
    for file_idx, measurement in enumerate(measurements):
        file_indices.extend([file_idx] * len(measurement.matrices))
    return file_indices


def load_measurement_files(files: list[Path], *, beam: int) -> list[TbtData]:
    """Load multiple SDDS turn-by-turn files into beam-4-frame, metres TbtData objects."""
    return [read_lhc_bpm_tbt(file, beam=beam) for file in files]


def tbt_xy_to_long_dataframe(
    x_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    *,
    turn_offset: int = 0,
) -> pd.DataFrame:
    """Reshape a single TBT X/Y frame pair into a long-form (name, turn, x, y) DataFrame.

    The frames must already be in the beam-4 tracking frame and in metres — i.e.
    loaded via :func:`read_lhc_bpm_tbt`. This is a pure reshape; it applies no
    sign flip or unit conversion.
    """
    if list(x_frame.index) != list(y_frame.index):
        raise ValueError("X and Y frames have different BPM ordering")
    bpm_names = [str(n) for n in x_frame.index]
    n_turns = min(x_frame.shape[1], y_frame.shape[1])
    n_bpms = len(bpm_names)
    x_arr = x_frame.to_numpy(dtype=float)[:, :n_turns]
    y_arr = y_frame.to_numpy(dtype=float)[:, :n_turns]
    return pd.DataFrame(
        {
            "name": np.repeat(bpm_names, n_turns),
            "turn": np.tile(np.arange(n_turns) + turn_offset, n_bpms),
            "x": x_arr.ravel(),
            "y": y_arr.ravel(),
        }
    )


def convert_tbt_to_dataframes(
    measurements: list[TbtData],
    bad_bpms: list[str] | None = None,
    *,
    combine_measurements: bool = True,
) -> list[pd.DataFrame]:
    """Convert turn-by-turn matrices into long-form dataframes.

    ``measurements`` must already be beam-4-frame, metres data (loaded via
    :func:`load_measurement_files` / :func:`read_lhc_bpm_tbt`).
    """
    excluded_bpms = set(bad_bpms or [])
    converted: list[pd.DataFrame] = []
    turn_offset = 1
    bunch_number = 0

    for measurement in measurements:
        if not combine_measurements:
            turn_offset = 1

        for bunch in measurement.matrices:
            combined = tbt_xy_to_long_dataframe(bunch.X, bunch.Y, turn_offset=turn_offset)
            combined["bunch_number"] = bunch_number

            original_order = [str(n) for n in bunch.X.index]
            combined["name"] = pd.Categorical(combined["name"], categories=original_order)

            if excluded_bpms:
                combined = combined[~combined["name"].isin(excluded_bpms)]

            converted.append(combined.sort_values(["turn", "name"]).reset_index(drop=True))
            turn_offset += bunch.X.shape[1]
            bunch_number += 1

    return converted
