"""
Estimate dispersion at corrector magnets using optics analysis data and MAD-NG tracking.

This module provides functionality to estimate horizontal and vertical dispersion
at corrector locations by propagating optics parameters from nearby BPMs using
MAD-NG's differential algebra tracking capabilities.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import tfs
from omc3.model.constants import TWISS_ELEMENTS_DAT
from omc3.optics_measurements.constants import BETA_NAME, DISPERSION_NAME, EXT

from aba_optimiser.mad.base_mad_interface import BaseMadInterface

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)


def find_common_bpms(*dataframes: pd.DataFrame) -> list[str]:
    """Find common BPMs across multiple dataframes.

    Args:
        *dataframes: Variable number of DataFrames with BPM names as index

    Returns:
        List of common BPM names in the order they appear in the first dataframe
    """
    if not dataframes:
        return []

    # Get common BPMs across all dataframes
    common = set(dataframes[0].index)
    for df in dataframes[1:]:
        common &= set(df.index)

    # Return in order of first dataframe
    return [bpm for bpm in dataframes[0].index if bpm in common]


def load_optics_files(
    optics_dir: Path,
    file_specs: dict[str, tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    """Load multiple TFS optics files and return as a dictionary.

    Args:
        optics_dir: Directory containing the optics TFS files
        file_specs: Dict mapping keys to (prefix, suffix) tuples
            Example: {"beta_x": ("beta_phase_", "x")} -> beta_phase_x.tfs

    Returns:
        Dictionary mapping keys to loaded DataFrames

    Raises:
        FileNotFoundError: If any required file is missing
    """
    loaded = {}
    for key, (prefix, suffix) in file_specs.items():
        file_path = optics_dir / f"{prefix}{suffix}{EXT}"
        if not file_path.exists():
            raise FileNotFoundError(f"Required TFS file not found: {file_path}")
        loaded[key] = tfs.read(file_path, index="NAME")

    return loaded


def find_closest_bpms_for_correctors(
    correctors: Sequence[str],
    bpms: Sequence[str],
    twiss_elements: pd.DataFrame,
    num_closest: int = 5,
    beam: int = 1,
) -> dict[str, list[tuple[str, int]]]:
    """For each corrector, find the closest BPMs, considering ring wrapping.

    Args:
        correctors: List of corrector element names
        bpms: List of BPM element names
        twiss_elements: DataFrame with element positions (must have 'S' column and 'LENGTH' header)
        num_closest: Number of closest BPMs to find for each corrector
        beam: Beam number (1 or 2), affects direction calculation for beam 2

    Returns:
        Dictionary mapping corrector names to list of (bpm_name, direction) tuples,
        where direction is 1 for forward (corrector downstream of BPM) or
        -1 for backward (corrector upstream of BPM)
    """
    length = twiss_elements.headers["LENGTH"]
    corrector_bpms: dict[str, list[tuple[str, int]]] = {}

    for corrector in correctors:
        s_c = twiss_elements.loc[corrector, "S"]
        bpm_distances = []

        for bpm in bpms:
            s_b = twiss_elements.loc[bpm, "S"]
            # Distance from BPM to corrector (always positive)
            d_forward = (s_c - s_b) % length
            d_backward = (s_b - s_c) % length

            if d_forward <= d_backward:
                min_d = d_forward
                sdir = 1  # Track forward from BPM to corrector
            else:
                min_d = d_backward
                sdir = -1  # Track backward from BPM to corrector

            # For beam 2, flip the direction due to reversed sequence
            if beam == 2:
                sdir = -sdir

            bpm_distances.append((bpm, sdir, min_d))

        # Sort by distance and take the closest
        bpm_distances.sort(key=lambda x: x[2])
        corrector_bpms[corrector] = [(bpm, sdir) for bpm, sdir, _ in bpm_distances[:num_closest]]

    return corrector_bpms


def _has_valid_optics_at_bpm(
    bpm: str,
    beta_x: pd.DataFrame,
    beta_y: pd.DataFrame,
    alfa_x: pd.DataFrame,
    alfa_y: pd.DataFrame,
    disp_x: pd.DataFrame,
    disp_y: pd.DataFrame,
) -> bool:
    """Check if all optics values at a BPM are finite (not inf/nan)."""
    optics_data = [
        (beta_x, ["BETX"]),
        (beta_y, ["BETY"]),
        (alfa_x, ["ALFX"]),
        (alfa_y, ["ALFY"]),
        (disp_x, ["DX", "DPX"]),
        (disp_y, ["DY", "DPY"]),
    ]
    return all(np.isfinite(df.loc[bpm, col]) for df, cols in optics_data for col in cols)


def _process_corrector_worker(
    corrector: str,
    bpm_info: list[tuple[str, int]],
    beta_x: pd.DataFrame,
    beta_y: pd.DataFrame,
    alfa_x: pd.DataFrame,
    alfa_y: pd.DataFrame,
    disp_x: pd.DataFrame,
    disp_y: pd.DataFrame,
    sequence_file: Path,
    seq_name: str,
    beam: int,
    beam_energy_gev: float,
    particle: str,
    plane: str,
) -> tuple[str, list[float]]:
    """Process a single corrector in a parallel worker.

    This function is called by each worker process to estimate dispersion
    at one corrector using all its nearby BPMs.

    Args:
        corrector: Name of the corrector element
        bpm_info: List of (bpm_name, direction) tuples for this corrector
        beta_x: DataFrame with horizontal beta function at BPMs
        beta_y: DataFrame with vertical beta function at BPMs
        alfa_x: DataFrame with horizontal alpha function at BPMs
        alfa_y: DataFrame with vertical alpha function at BPMs
        disp_x: DataFrame with horizontal dispersion at BPMs
        disp_y: DataFrame with vertical dispersion at BPMs
        sequence_file: Path to the MAD-X sequence file
        seq_name: Name of the sequence
        beam_energy_gev: Beam energy in GeV
        particle: Particle type
        plane: Dispersion plane ("x" or "y")

    Returns:
        Tuple of (corrector_name, list_of_estimates)
    """
    # Create MAD instance for this worker
    mad_interface = BaseMadInterface()
    mad_interface.mad.MADX[f"b{beam}_re_ip7_knob"] = 0.0  # To avoid warnings
    mad_interface.mad.MADX[f"b{beam}_im_ip7_knob"] = 0.0  # To avoid warnings
    mad_interface.load_sequence(sequence_file, seq_name)
    mad_interface.setup_beam(beam_energy=beam_energy_gev, particle=particle)

    mad = mad_interface.mad
    mad.send("""
coord_names = {"x", "px", "y", "py", "t", "pt"}
da_x0_base = MAD.damap{nv=#coord_names, mo=1, vn=coord_names}
observed = MAD.element.flags.observed
loaded_sequence:deselect(observed)
loaded_sequence:select(observed, {pattern="MCB"})
    """)

    estimates = []
    for bpm, sdir in bpm_info:
        # Skip BPMs with invalid optics data
        if not _has_valid_optics_at_bpm(bpm, beta_x, beta_y, alfa_x, alfa_y, disp_x, disp_y):
            continue

        tracking_range = f"{bpm}/{corrector}"
        mad["tracking_range"] = tracking_range
        mad["sdir"] = sdir

        # Initialize differential algebra map at BPM with measured optics
        mad.send("""
shush()
    local B0 = MAD.beta0 {
        beta11=py:recv(),
        beta22=py:recv(),
        alfa11=py:recv(),
        alfa22=py:recv(),
        dx=py:recv(),
        dpx=py:recv(),
        dy=py:recv(),
        dpy=py:recv(),
        sdir = sdir,
    }
    da_x0 = MAD.gphys.bet2map(B0, da_x0_base)
unshush()
        """)
        mad.send(beta_x.loc[bpm, "BETX"])
        mad.send(beta_y.loc[bpm, "BETY"])
        mad.send(alfa_x.loc[bpm, "ALFX"])
        mad.send(alfa_y.loc[bpm, "ALFY"])
        mad.send(disp_x.loc[bpm, "DX"])
        mad.send(disp_x.loc[bpm, "DPX"])
        mad.send(disp_y.loc[bpm, "DY"])
        mad.send(disp_y.loc[bpm, "DPY"])

        # Track from BPM to corrector and extract dispersion
        mad.send(f"""
local _, flw = MAD.track{{
    sequence = loaded_sequence,
    range = tracking_range,
    observe=1,
    dir = sdir,
    X0 = da_x0,
    save=false,
    savemap=false,
}}
B1 = MAD.gphys.map2bet(flw[1], 6, nil, nil, sdir)
py:send(B1.d{plane})
""")
        estimated_dispersion: float = mad.receive()  # ty:ignore[invalid-assignment]
        estimates.append(estimated_dispersion)

    mad.send("shush()")
    del mad_interface
    return corrector, estimates


def estimate_corrector_dispersion(
    corrector_bpms: dict[str, list[tuple[str, int]]],
    beta_x: pd.DataFrame,
    beta_y: pd.DataFrame,
    alfa_x: pd.DataFrame,
    alfa_y: pd.DataFrame,
    disp_x: pd.DataFrame,
    disp_y: pd.DataFrame,
    sequence_file: Path,
    seq_name: str = "lhcb1",
    beam: Literal[1, 2] = 1,
    beam_energy_gev: float = 6800,
    particle: str = "proton",
    plane: str = "x",
    n_processes: int | None = None,
) -> dict[str, list[float]]:
    """Estimate dispersion at correctors using MAD-NG differential algebra tracking.

    For each corrector, this function:
    1. Initializes a differential algebra map at each nearby BPM using measured optics
    2. Tracks the map from the BPM to the corrector
    3. Extracts the dispersion at the corrector from the tracked map

    This function uses parallel processing to speed up the computation, with each
    worker process having its own MAD-NG instance.

    Args:
        corrector_bpms: Mapping of corrector names to list of (bpm, direction) tuples
        beta_x: DataFrame with horizontal beta function at BPMs (must have 'BETX' column)
        beta_y: DataFrame with vertical beta function at BPMs (must have 'BETY' column)
        alfa_x: DataFrame with horizontal alpha function at BPMs (must have 'ALFX' column)
        alfa_y: DataFrame with vertical alpha function at BPMs (must have 'ALFY' column)
        disp_x: DataFrame with horizontal dispersion at BPMs (must have 'DX', 'DPX' columns)
        disp_y: DataFrame with vertical dispersion at BPMs (must have 'DY', 'DPY' columns)
        sequence_file: Path to the MAD-X sequence file
        seq_name: Name of the sequence in the MAD-X file
        beam_energy_gev: Beam energy in GeV
        particle: Particle type (e.g., "proton")
        plane: Dispersion plane to extract ("x" or "y")
        n_processes: Number of parallel processes to use (default: number of CPU cores)

    Returns:
        Dictionary mapping corrector names to lists of estimated dispersion values
        (one estimate per nearby BPM)
    """
    if n_processes is None:
        n_processes = max(int(mp.cpu_count() / 2), 1)  # Use half available cores by default

    logger.info(
        f"Processing {len(corrector_bpms)} correctors using {n_processes} parallel processes"
    )

    # Prepare arguments for parallel processing
    args_list = [
        (
            corrector,
            bpm_info,
            beta_x,
            beta_y,
            alfa_x,
            alfa_y,
            disp_x,
            disp_y,
            sequence_file,
            seq_name,
            beam,
            beam_energy_gev,
            particle,
            plane,
        )
        for corrector, bpm_info in corrector_bpms.items()
    ]

    # Process correctors in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = pool.starmap(_process_corrector_worker, args_list)

    # Combine results
    return dict(results)


def calculate_dispersion_statistics(
    corrector_dispersion_estimates: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate mean and standard deviation of dispersion estimates, removing outliers.

    Uses IQR (Interquartile Range) method to filter outliers before computing statistics.

    Args:
        corrector_dispersion_estimates: Dictionary mapping corrector names to lists of estimates

    Returns:
        Tuple of (mean_dispersions, std_dispersions) dictionaries
    """
    final_corrector_dispersion: dict[str, float] = {}
    corrector_dispersion_stds: dict[str, float] = {}

    for corrector, estimates in corrector_dispersion_estimates.items():
        estimates_series = pd.Series(estimates)

        # Remove outliers using IQR method
        q1 = estimates_series.quantile(0.25)
        q3 = estimates_series.quantile(0.75)
        iqr = q3 - q1
        filtered_estimates = estimates_series[
            (estimates_series >= q1 - 1.5 * iqr) & (estimates_series <= q3 + 1.5 * iqr)
        ]

        final_corrector_dispersion[corrector] = filtered_estimates.mean()
        corrector_dispersion_stds[corrector] = filtered_estimates.std()

    return final_corrector_dispersion, corrector_dispersion_stds


def estimate_corrector_dispersions(
    optics_dir: Path,
    sequence_file: Path,
    model_dir: Path,
    seq_name: str = "lhcb1",
    beam: Literal[1, 2] = 1,
    beam_energy_gev: float = 6800,
    particle: str = "proton",
    num_closest_bpms: int = 10,
    plane: str = "x",
    n_processes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate dispersion at all correctors using optics analysis data.

    This is the main entry point for dispersion estimation. It loads optics data,
    finds nearby BPMs for each corrector, estimates dispersion via tracking,
    and returns the results with statistics.

    Args:
        optics_dir: Directory containing optics analysis TFS files
            (beta_phase_x.tfs, beta_phase_y.tfs, dispersion_x.tfs, dispersion_y.tfs)
        sequence_file: Path to the MAD-X sequence file
        model_dir: Directory containing the model twiss_elements.dat file
        seq_name: Name of the sequence in the MAD-X file
        beam: Beam number (1 or 2)
        beam_energy_gev: Beam energy in GeV
        particle: Particle type (e.g., "proton")
        num_closest_bpms: Number of nearby BPMs to use for each corrector estimate
        plane: Dispersion plane to extract ("x" or "y")
        n_processes: Number of parallel processes to use (default: number of CPU cores)

    Returns:
        Tuple of (dispersion_df, statistics_df) where:
        - dispersion_df: DataFrame with columns ['CORRECTOR', 'S', 'DISPERSION', 'STD']
        - statistics_df: DataFrame with all individual estimates for debugging

    Raises:
        FileNotFoundError: If required optics or model files are missing
    """
    logger.info(f"Loading optics data from {optics_dir}")

    # Define which files to load
    x_plane = "x"
    y_plane = "y"
    file_specs = {
        "beta_x": (BETA_NAME, x_plane),
        "beta_y": (BETA_NAME, y_plane),
        "alfa_x": (BETA_NAME, x_plane),
        "alfa_y": (BETA_NAME, y_plane),
        "disp_x": (DISPERSION_NAME, x_plane),
        "disp_y": (DISPERSION_NAME, y_plane),
    }

    # Load optics files
    optics_data = load_optics_files(optics_dir, file_specs)
    beta_x = optics_data["beta_x"]
    beta_y = optics_data["beta_y"]
    alfa_x = optics_data["alfa_x"]
    alfa_y = optics_data["alfa_y"]
    disp_x = optics_data["disp_x"]
    disp_y = optics_data["disp_y"]

    # Find common BPMs across all measurements
    logger.info("Finding common BPMs across all optics measurements")
    bpm_list = find_common_bpms(beta_x, beta_y, disp_x, disp_y, alfa_x, alfa_y)
    logger.info(f"Found {len(bpm_list)} common BPMs")

    # Filter to common BPMs
    beta_x = beta_x.loc[bpm_list]
    beta_y = beta_y.loc[bpm_list]
    alfa_x = alfa_x.loc[bpm_list]
    alfa_y = alfa_y.loc[bpm_list]
    disp_x = disp_x.loc[bpm_list]
    disp_y = disp_y.loc[bpm_list]

    # Load model twiss
    logger.info(f"Loading model twiss from {model_dir}")
    twiss_elements = tfs.read(model_dir / TWISS_ELEMENTS_DAT, index="NAME")

    # Get corrector list
    corrector_list = twiss_elements[twiss_elements.index.str.match(r"MCB[A-Z]?[HV]\.")].index  # ty:ignore[unresolved-attribute]
    logger.info(f"Found {len(corrector_list)} correctors")

    # Find closest BPMs for each corrector
    logger.info(f"Finding {num_closest_bpms} closest BPMs for each corrector")
    corrector_bpms = find_closest_bpms_for_correctors(
        corrector_list,
        bpm_list,
        twiss_elements,
        num_closest=num_closest_bpms,
        beam=beam,
    )

    # Set sequence name based on beam
    seq_name = f"lhcb{beam}"

    # Estimate dispersion
    logger.info("Estimating dispersion at correctors via MAD-NG tracking")
    corrector_dispersion_estimates = estimate_corrector_dispersion(
        corrector_bpms,
        beta_x,
        beta_y,
        alfa_x,
        alfa_y,
        disp_x,
        disp_y,
        sequence_file,
        seq_name=seq_name,
        beam=beam,
        beam_energy_gev=beam_energy_gev,
        particle=particle,
        plane=plane,
        n_processes=n_processes,
    )

    # Calculate statistics
    logger.info("Calculating dispersion statistics")
    final_dispersions, dispersion_stds = calculate_dispersion_statistics(
        corrector_dispersion_estimates
    )

    # Create output dataframes
    results = []
    for corrector in corrector_list:
        results.append(
            {
                "CORRECTOR": corrector,
                "S": twiss_elements.loc[corrector, "S"],
                "DISPERSION": final_dispersions.get(corrector, np.nan),
                "STD": dispersion_stds.get(corrector, np.nan),
            }
        )

    dispersion_df = pd.DataFrame(results).set_index("CORRECTOR")

    # Create detailed statistics dataframe
    detailed_results = []
    for corrector, estimates in corrector_dispersion_estimates.items():
        for i, est in enumerate(estimates):
            detailed_results.append(
                {
                    "CORRECTOR": corrector,
                    "BPM": corrector_bpms[corrector][i][0],
                    "DIRECTION": corrector_bpms[corrector][i][1],
                    "ESTIMATE": est,
                }
            )

    statistics_df = pd.DataFrame(detailed_results)

    logger.info(f"Dispersion estimation complete for {len(final_dispersions)} correctors")

    return dispersion_df, statistics_df
