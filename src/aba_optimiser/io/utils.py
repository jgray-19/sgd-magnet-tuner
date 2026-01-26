from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aba_optimiser.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np

def get_lhc_file_path(beam: int) -> Path:
    """
    Get the path to the LHC sequence file for the given beam.

    Args:
        beam: The LHC beam number (1 or 2).

    Returns:
        Path to the LHC sequence file.
    """
    return (
        # PROJECT_ROOT / "src" / "aba_optimiser" / "mad" / "sequences" / f"lhcb{beam}.seq"
        PROJECT_ROOT / "models" / f"lhcb{beam}_12cm" / f"lhcb{beam}_saved.seq"
    )


def read_knobs(file_path: str | Path) -> dict[str, float]:
    """
    Read knob strengths from a tab-delimited file.

    Args:
        path: Path to the file where each line is "knob_name\tstrength".

    Returns:
        A dictionary mapping knob names to their true float strengths.
    """
    path = Path(file_path)
    logger.info(f"Reading knobs from {path}")
    strengths: dict[str, float] = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            knob, val = parts
            strengths[knob] = float(val)
    logger.debug(f"Read {len(strengths)} knobs from {path}")
    return strengths


def save_knobs(knobs: dict[str, float], filepath: Path) -> None:
    """
    Save matched tunes to file.

    Args:
        matched_tunes: Dictionary of tune knobs and values
        filepath: Path to save file
    """
    logger.info(f"Saving {len(knobs)} knobs to {filepath}")
    with filepath.open("w") as f:
        for key, val in knobs.items():
            f.write(f"{key}\t{val: .15e}\n")


def save_results(
    knob_names: list[str],
    knob_strengths: dict[str, float],
    uncertainties: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save the final knob strengths and uncertainties to a file.

    Args:
        knob_names: List of knob names.
        knob_strengths: List of knob strengths.
        uncertainties: List of uncertainties for each knob.
        output_path: Path to the output file.
    """
    with Path(output_path).open("w") as f:
        f.write("Knob Name\tStrength\tUncertainty\n")
        for idx, knob in enumerate(knob_names):
            strength = knob_strengths[knob]
            uncertainty = uncertainties[idx]
            f.write(f"{knob}\t{strength:.15e}\t{uncertainty:.15e}\n")
    logger.info(f"Saved results for {len(knob_names)} knobs to {output_path}")


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

    logger.info(f"Reading results from {file_path}")
    with Path(file_path).open("r") as f:
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
    logger.debug(f"Read {len(knob_names)} knobs from {file_path}")

    return knob_names, knob_strengths, uncertainties


def scientific_notation(num: float, precision: int = 2) -> str:
    """
    Format a number into scientific notation with a given precision.

    Args:
        num: The number to format.
        precision: Number of decimal places for the mantissa.

    Returns:
        A string of the form "m*10^e" or "0" if num is zero.
    """
    import math

    # Guard against non-numeric inputs and special float values
    try:
        fnum = float(num)
    except (TypeError, ValueError):
        # If it's not convertible to float, fall back to its string repr
        return str(num)

    # Handle zero explicitly
    if fnum == 0.0:
        return "0"

    # Handle NaN and infinite values robustly
    if not math.isfinite(fnum):
        return str(fnum)

    exponent = int(math.floor(math.log10(abs(fnum))))
    mantissa = fnum / (10**exponent)
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    return f"${mantissa:.{precision}f}\\times10^{{{exponent}}}$"
