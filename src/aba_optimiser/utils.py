from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import tfs


def filter_out_marker(tbl: tfs.TfsDataFrame, marker_name: str) -> tfs.TfsDataFrame:
    """
    Filter out markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.

    Returns:
        A TFS DataFrame without markers.
    """
    tbl = tbl.copy()
    if tbl.index.name != "name":
        return tbl[tbl["name"] != marker_name]
    return tbl[tbl.index != marker_name]


def filter_out_markers(
    tbl: tfs.TfsDataFrame, marker_names: list[str]
) -> tfs.TfsDataFrame:
    """
    Filter out markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.
        marker_names: A list of marker names to filter out.

    Returns:
        A TFS DataFrame without the specified markers.
    """
    tbl = tbl.copy()
    if tbl.index.name != "name":
        return tbl[~tbl["name"].isin(marker_names)]
    return tbl[~tbl.index.isin(marker_names)]


def select_markers(
    tbl: tfs.TfsDataFrame, marker_name: str | list[str]
) -> tfs.TfsDataFrame:
    """
    Select markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.

    Returns:
        A TFS DataFrame with only the specified markers.
    """
    if isinstance(marker_name, str):
        marker_name = [marker_name]
    tbl = tbl.copy()
    if tbl.index.name != "name":
        return tbl[tbl["name"].isin(marker_name)]
    return tbl[tbl.index.isin(marker_name)]


def read_elem_names(path: str) -> list[str]:
    """
    Read element names from a text file, it contains minimum one element name per line.
    The file is tab seperated and no header is expected.
    The file is expected to be in the format:
    ```
    element_name, element_name, ..., element_name
    ```
    element_name is the name of the element and ... is any other aliases of the element.

    Args:
        path: Path to the text file containing one element name per line.

    Returns:
        A list of non-empty, stripped element name strings.
    """
    names: list[list[str]] = []
    spos: list[float] = []
    with Path(path).open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            assert len(parts) >= 1, f"Invalid line in {path}: {line.strip()}"
            names.append(parts[1:])
            spos.append(float(parts[0]))

    return spos, names


def read_knobs(path: str) -> dict[str, float]:
    """
    Read knob strengths from a tab-delimited file.

    Args:
        path: Path to the file where each line is "knob_name\tstrength".

    Returns:
        A dictionary mapping knob names to their true float strengths.
    """
    strengths: dict[str, float] = {}
    with Path(path).open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            knob, val = parts
            strengths[knob] = float(val)
    return strengths


def scientific_notation(num: float, precision: int = 2) -> str:
    """
    Format a number into scientific notation with a given precision.

    Args:
        num: The number to format.
        precision: Number of decimal places for the mantissa.

    Returns:
        A string of the form "m*10^e" or "0" if num is zero.
    """
    if num == 0:
        return "0"
    import math

    exponent = int(math.floor(math.log10(abs(num))))
    mantissa = num / (10**exponent)
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    return f"${mantissa:.{precision}f}\\times10^{{{exponent}}}$"


def save_results(
    knob_names: list[str],
    knob_strengths: dict[str, float],
    uncertainties: list[float],
    output_path: str,
) -> None:
    """
    Save the final knob strengths and uncertainties to a file.

    Args:
        knob_names: List of knob names.
        knob_strengths: List of knob strengths.
        uncertainties: List of uncertainties for each knob.
        output_path: Path to the output file.
    """
    with open(output_path, "w") as f:
        f.write("Knob Name\tStrength\tUncertainty\n")
        for idx, knob in enumerate(knob_names):
            strength = knob_strengths[knob]
            uncertainty = uncertainties[idx]
            f.write(f"{knob}\t{strength:.15e}\t{uncertainty:.15e}\n")


def read_results(file_path: str) -> tuple[list[str], list[float], list[float]]:
    """
    Read the results from a file.

    Args:
        file_path: Path to the file containing the results.

    Returns:
        A tuple containing:
            - A list of knob names.
            - A list of knob strengths.
            - A list of uncertainties.
    """
    knob_names = []
    knob_strengths = []
    uncertainties = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            # Skip the header line
            if parts[0] == "Knob Name":
                continue

            knob, strength, uncertainty = parts
            knob_names.append(knob)
            knob_strengths.append(float(strength))
            uncertainties.append(float(uncertainty))

    return knob_names, knob_strengths, uncertainties


# -------------------------------------------------------------------------
# HELPER: find the previous BPM ~pi/2 phase advance
# -------------------------------------------------------------------------


def prev_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to pi/2 phase advance.
    Returns a DataFrame indexed by BPM_i with columns:
      - prev_bpm : name of BPM_j
      - delta    : (mu_i - mu_j - 0.25) signed error in turns
    """
    v = mu.to_numpy(float)
    n = len(v)

    # backward differences mod Q: (mu_i - mu_j) ∈ [0, tune)
    backward = (v.reshape(n, 1) - v.reshape(1, n) + tune) % tune
    np.fill_diagonal(backward, np.nan)

    # pick j minimizing |Δ_ij - target|. target = 0.25 (*2pi) -> pi/2
    idx = np.nanargmin(np.abs(backward - 0.25), axis=1)
    delta = backward[np.arange(n), idx] - 0.25
    names = mu.index[idx]

    return pd.DataFrame({"prev_bpm": names, "delta": delta}, index=mu.index)


# 1) define your new forward-phase helper
def next_bpm_to_pi_2(mu: pd.Series, tune: float) -> pd.DataFrame:
    """
    For each BPM_i find the *next* BPM_j whose forward phase advance
    (mu_j - mu_i) mod Q is closest to pi/2 phase advance.
    """
    v = mu.to_numpy(float)
    n = len(v)

    # forward differences mod Q: (mu_j - mu_i) ∈ [0, tune)
    forward = (v.reshape(1, n) - v.reshape(n, 1) + tune) % tune
    np.fill_diagonal(forward, np.nan)

    # pick j minimizing |Δ_ij - target|
    idx = np.nanargmin(np.abs(forward - 0.25), axis=1)
    delta = forward[np.arange(n), idx] - 0.25
    names = mu.index[idx]

    return pd.DataFrame({"next_bpm": names, "delta": delta}, index=mu.index)
