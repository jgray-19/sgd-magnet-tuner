from typing import List, Dict
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
    else:
        return tbl[tbl.index != marker_name]
    
def filter_out_markers(tbl: tfs.TfsDataFrame, marker_names: List[str]) -> tfs.TfsDataFrame:
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
    else:
        return tbl[~tbl.index.isin(marker_names)]
    
def select_marker(tbl: tfs.TfsDataFrame, marker_name: str) -> tfs.TfsDataFrame:
    """
    Select markers from a TFS DataFrame.

    Args:
        tbl: A TFS DataFrame containing the data.

    Returns:
        A TFS DataFrame with only the specified markers.
    """
    tbl = tbl.copy()
    if tbl.index.name != "name":
        return tbl[tbl["name"] == marker_name]
    else:
        return tbl[tbl.index == marker_name]

def read_elem_names(path: str) -> List[str]:
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
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split("\t")
            assert len(parts) >= 1, f"Invalid line in {path}: {line.strip()}"
            names.append(parts[1:])
            spos.append(float(parts[0]))

    return spos, names

def read_knobs(path: str) -> Dict[str, float]:
    """
    Read knob strengths from a tab-delimited file.

    Args:
        path: Path to the file where each line is "knob_name\tstrength".

    Returns:
        A dictionary mapping knob names to their true float strengths.
    """
    strengths: Dict[str, float] = {}
    with open(path, 'r') as f:
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
    knob_names: List[str],
    knob_strengths: Dict[str, float],
    uncertainties: List[float],
    output_path: str
) -> None:
    """
    Save the final knob strengths and uncertainties to a file.

    Args:
        knob_names: List of knob names.
        knob_strengths: List of knob strengths.
        uncertainties: List of uncertainties for each knob.
        output_path: Path to the output file.
    """
    with open(output_path, 'w') as f:
        f.write("Knob Name\tStrength\tUncertainty\n")
        for idx, knob in enumerate(knob_names):
            strength = knob_strengths[knob]
            uncertainty = uncertainties[idx]
            f.write(f"{knob}\t{strength:.15e}\t{uncertainty:.15e}\n")

def read_results(
        file_path: str
) -> tuple[list[str], list[float], list[float]]:
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

    with open(file_path, 'r') as f:
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
