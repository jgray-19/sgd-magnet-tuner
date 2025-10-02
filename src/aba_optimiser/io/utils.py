from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def read_knobs(path: str) -> dict[str, float]:
    """
    Read knob strengths from a tab-delimited file.

    Args:
        path: Path to the file where each line is "knob_name\tstrength".

    Returns:
        A dictionary mapping knob names to their true float strengths.
    """
    LOGGER.info(f"Reading knobs from {path}")
    strengths: dict[str, float] = {}
    with Path(path).open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            knob, val = parts
            strengths[knob] = float(val)
    LOGGER.debug(f"Read {len(strengths)} knobs from {path}")
    return strengths


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
    with Path(output_path).open("w") as f:
        f.write("Knob Name\tStrength\tUncertainty\n")
        for idx, knob in enumerate(knob_names):
            strength = knob_strengths[knob]
            uncertainty = uncertainties[idx]
            f.write(f"{knob}\t{strength:.15e}\t{uncertainty:.15e}\n")
    LOGGER.info(f"Saved results for {len(knob_names)} knobs to {output_path}")


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

    LOGGER.info(f"Reading results from {file_path}")
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
    LOGGER.debug(f"Read {len(knob_names)} knobs from {file_path}")

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
