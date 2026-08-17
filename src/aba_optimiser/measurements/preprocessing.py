"""Measurement preprocessing helpers."""

from __future__ import annotations

import logging
import warnings
from typing import TypeAlias

import pandas as pd
from tmom_recon.kicker.core import find_kick, subtract_closed_orbit

LOGGER = logging.getLogger(__name__)

ClosedOrbitMapping: TypeAlias = dict[str, dict[str, float] | pd.Series]
ClosedOrbitInput: TypeAlias = str | pd.DataFrame | ClosedOrbitMapping | None
ClosedOrbitReferenceInput: TypeAlias = str | pd.DataFrame | ClosedOrbitMapping


def preprocess_measurement_dataframe(
    df: pd.DataFrame,
    twiss: pd.DataFrame,
    *,
    remove_closed_orbit: ClosedOrbitInput = None,
    n_turns_free: int = 1000,
    kicker_name: str | None = None,
) -> pd.DataFrame:
    """Apply optional measurement preprocessing before momentum reconstruction."""
    if remove_closed_orbit is None:
        return df

    if isinstance(remove_closed_orbit, pd.DataFrame | dict):
        return _subtract_closed_orbit_reference(
            df,
            _normalise_closed_orbit_reference(remove_closed_orbit, twiss),
        )

    if remove_closed_orbit == "average":
        return _subtract_average_closed_orbit_and_trim_to_kick(
            df,
            twiss,
            n_turns_free=n_turns_free,
            kicker_name=kicker_name,
        )

    return _subtract_closed_orbit_reference(df, _normalise_closed_orbit_reference(remove_closed_orbit, twiss))


def _normalise_closed_orbit_reference(
    closed_orbit: ClosedOrbitReferenceInput,
    twiss: pd.DataFrame,
) -> pd.DataFrame:
    """Return a BPM-indexed reference table with x/y and optional px/py."""
    if isinstance(closed_orbit, pd.DataFrame):
        reference = _ensure_name_index(closed_orbit)
    elif isinstance(closed_orbit, dict):
        reference = pd.DataFrame.from_dict(closed_orbit, orient="index")
        reference.index.name = "name"
    elif closed_orbit == "twiss":
        reference = twiss.copy()
    else:
        raise ValueError(
            "remove_closed_orbit must be None, 'twiss', 'average', a dataframe, or a BPM dict."
        )

    reference.columns = [str(column).lower() for column in reference.columns]
    _validate_closed_orbit_columns(reference)
    keep_columns = ["x", "y"] + (["px", "py"] if {"px", "py"} <= set(reference.columns) else [])
    return reference.loc[:, keep_columns]


def _validate_closed_orbit_columns(reference: pd.DataFrame) -> None:
    missing_xy = [column for column in ("x", "y") if column not in reference.columns]
    if missing_xy:
        raise ValueError(
            "Closed-orbit reference must provide both x and y columns; "
            f"missing {missing_xy}."
        )

    has_px = "px" in reference.columns
    has_py = "py" in reference.columns
    if has_px != has_py:
        raise ValueError("Closed-orbit reference must provide both px and py, or neither.")
    if not has_px:
        warnings.warn(
            "Closed-orbit reference does not provide px/py; only x/y will be subtracted.",
            stacklevel=3,
        )


def _ensure_name_index(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.index.name, str) and data.index.name.lower() == "name":
        return data.copy()

    for column in ("name", "NAME"):
        if column in data.columns:
            return data.set_index(column, drop=True).copy()

    raise ValueError("Closed-orbit dataframe must be indexed by BPM or contain a 'name'/'NAME' column.")


def _subtract_closed_orbit_reference(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Subtract a BPM-indexed closed-orbit reference from a measurement table."""
    result = df.copy()
    for column in ("x", "y"):
        result[column] = result[column] - result["name"].map(reference[column])

    if {"px", "py"} <= set(reference.columns):
        _subtract_momentum_reference(result, reference)

    return result


def _subtract_momentum_reference(df: pd.DataFrame, reference: pd.DataFrame) -> None:
    if not {"px", "py"} <= set(df.columns):
        raise ValueError("Closed-orbit reference provides px/py but the measurement dataframe does not.")
    for column in ("px", "py"):
        df[column] = df[column] - df["name"].map(reference[column])


def _subtract_average_closed_orbit_and_trim_to_kick(
    df: pd.DataFrame,
    twiss: pd.DataFrame,
    *,
    n_turns_free: int,
    kicker_name: str | None,
) -> pd.DataFrame:
    """Subtract the pre-kick average orbit and keep only post-kick samples.

    The subtraction is a per-BPM turn mean, so it removes every static contribution
    *exactly*: the dispersive orbit, the error orbit, BPM reading offsets, and any
    constant per-BPM momentum bias. That exactness is the point --- the turn-varying
    part needs no reference orbit, no momentum estimate and no dispersion model --- but
    it also means no downstream check can ever see those quantities. A real per-BPM
    constant px bias of ~5.9e-4 rad, larger than the horizontal signal itself, hid
    under precisely this subtraction. Anything that has to be diagnosed in absolute
    terms must be looked at before this runs.
    """
    if _starts_at_kicker(df, kicker_name):
        LOGGER.info("Skipping kick search because the dataframe already starts at kicker %s.", kicker_name)
        return df.copy()

    indexed = df.set_index("name", drop=False)
    kick_bpm, kick_turn = find_kick(indexed.copy(), n_turns_free=n_turns_free)
    orbit_subtracted, _, _ = subtract_closed_orbit(indexed.copy(), n_turns_free=n_turns_free)
    trimmed = orbit_subtracted.reset_index(drop=True)
    trimmed = trimmed[trimmed["turn"] >= kick_turn].copy()
    trimmed = _drop_upstream_rows_on_kick_turn(trimmed, twiss, kick_bpm, kick_turn)
    trimmed["turn"] = (trimmed["turn"] - kick_turn + 1).astype(df["turn"].dtype)
    return trimmed.reset_index(drop=True)


def _starts_at_kicker(df: pd.DataFrame, kicker_name: str | None) -> bool:
    return kicker_name is not None and not df.empty and str(df.iloc[0]["name"]).upper() == kicker_name.upper()


def _drop_upstream_rows_on_kick_turn(
    df: pd.DataFrame,
    twiss: pd.DataFrame,
    kick_bpm: str,
    kick_turn: int,
) -> pd.DataFrame:
    kick_s = float(twiss.loc[kick_bpm, "s"])
    bpm_s = df["name"].map(twiss["s"])
    keep_mask = ~((df["turn"] == kick_turn) & (bpm_s < kick_s))
    return df.loc[keep_mask].copy()
