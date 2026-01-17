from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import (
    ALPHA,
    BETA,
    BETA_NAME,
    DISPERSION,
    DISPERSION_NAME,
    ERR,
    EXT,
    MOMENTUM_DISPERSION,
    NAME,
    ORBIT,
    ORBIT_NAME,
    PHASE,
    PHASE_ADV,
    PHASE_NAME,
    S,
)


def build_twiss_from_measurements(
    measurement_dir: Path,
    include_errors: bool = False
) -> pd.DataFrame:
    """
    Builds a twiss file from measurement files in the given directory.

    Args:
        measurement_dir: Path to the directory containing measurement .tfs files.
        include_errors: If True, include error columns in the twiss dataframe.

    Returns:
        pd.DataFrame with twiss data.
    """
    # Read beta measurements
    beta_x_path = measurement_dir / f"{BETA_NAME}x{EXT}"
    beta_y_path = measurement_dir / f"{BETA_NAME}y{EXT}"
    if beta_x_path.exists() and beta_y_path.exists():
        beta_x = tfs.read(beta_x_path, index=NAME)
        beta_y = tfs.read(beta_y_path, index=NAME)
    else:
        raise FileNotFoundError("Beta measurement files not found.")

    # Read dispersion measurements
    disp_x_path = measurement_dir / f"{DISPERSION_NAME}x{EXT}"
    disp_y_path = measurement_dir / f"{DISPERSION_NAME}y{EXT}"
    if disp_x_path.exists() and disp_y_path.exists():
        disp_x = tfs.read(disp_x_path, index=NAME)
        disp_y = tfs.read(disp_y_path, index=NAME)
    else:
        raise FileNotFoundError("Dispersion measurement files not found.")

    # Read phase measurements
    phase_x_path = measurement_dir / f"{PHASE_NAME}x{EXT}"
    phase_y_path = measurement_dir / f"{PHASE_NAME}y{EXT}"
    if phase_x_path.exists() and phase_y_path.exists():
        phase_x = tfs.read(phase_x_path, index=NAME)
        phase_y = tfs.read(phase_y_path, index=NAME)
    else:
        raise FileNotFoundError("Phase measurement files not found.")

    orbit_x_path = measurement_dir / f"{ORBIT_NAME}x{EXT}"
    orbit_y_path = measurement_dir / f"{ORBIT_NAME}y{EXT}"
    if orbit_x_path.exists() and orbit_y_path.exists():
        orbit_x = tfs.read(orbit_x_path, index=NAME)
        orbit_y = tfs.read(orbit_y_path, index=NAME)
    else:
        raise FileNotFoundError("Orbit measurement files not found.")


    # Sort by S to ensure order
    sorted_index = beta_x.index

    # Build the twiss dataframe
    twiss_df = tfs.TfsDataFrame(index=sorted_index)

    # S position
    twiss_df[S] = beta_x.loc[sorted_index, S]

    # Closed orbit
    twiss_df["X"] = orbit_x.loc[sorted_index, f"{ORBIT}X"]
    twiss_df["Y"] = orbit_y.loc[sorted_index, f"{ORBIT}Y"]

    # Beta functions
    twiss_df[f"{BETA}X"] = beta_x.loc[sorted_index, f"{BETA}X"]
    twiss_df[f"{BETA}Y"] = beta_y.loc[sorted_index, f"{BETA}Y"]

    # Alpha functions
    twiss_df[f"{ALPHA}X"] = beta_x.loc[sorted_index, f"{ALPHA}X"]
    twiss_df[f"{ALPHA}Y"] = beta_y.loc[sorted_index, f"{ALPHA}Y"]

    # Dispersion
    twiss_df[f"{DISPERSION}X"] = disp_x.loc[sorted_index, f"{DISPERSION}X"]
    twiss_df[f"{DISPERSION}Y"] = disp_y.loc[sorted_index, f"{DISPERSION}Y"]
    twiss_df[f"{MOMENTUM_DISPERSION}X"] = disp_x.loc[sorted_index, f"{MOMENTUM_DISPERSION}X"]
    twiss_df[f"{MOMENTUM_DISPERSION}Y"] = disp_y.loc[sorted_index, f"{MOMENTUM_DISPERSION}Y"]

    # Phase advances (cumulative from start)
    mu_x, err_mu_x = _compute_cumulative_phase(phase_x, sorted_index, f"{PHASE}X")
    mu_y, err_mu_y = _compute_cumulative_phase(phase_y, sorted_index, f"{PHASE}Y")
    twiss_df[f"{PHASE_ADV}X"] = mu_x
    twiss_df[f"{PHASE_ADV}Y"] = mu_y

    # Optionally include errors
    if include_errors:
        twiss_df[f"{ERR}X"] = orbit_x.loc[sorted_index, f"{ERR}X"]
        twiss_df[f"{ERR}Y"] = orbit_y.loc[sorted_index, f"{ERR}Y"]
        twiss_df[f"{ERR}{BETA}X"] = beta_x.loc[sorted_index, f"{ERR}{BETA}X"]
        twiss_df[f"{ERR}{BETA}Y"] = beta_y.loc[sorted_index, f"{ERR}{BETA}Y"]
        twiss_df[f"{ERR}{ALPHA}X"] = beta_x.loc[sorted_index, f"{ERR}{ALPHA}X"]
        twiss_df[f"{ERR}{ALPHA}Y"] = beta_y.loc[sorted_index, f"{ERR}{ALPHA}Y"]
        twiss_df[f"{ERR}{DISPERSION}X"] = disp_x.loc[sorted_index, f"{ERR}{DISPERSION}X"]
        twiss_df[f"{ERR}{DISPERSION}Y"] = disp_y.loc[sorted_index, f"{ERR}{DISPERSION}Y"]
        twiss_df[f"{ERR}{MOMENTUM_DISPERSION}X"] = disp_x.loc[sorted_index, f"{ERR}{MOMENTUM_DISPERSION}X"]
        twiss_df[f"{ERR}{MOMENTUM_DISPERSION}Y"] = disp_y.loc[sorted_index, f"{ERR}{MOMENTUM_DISPERSION}Y"]
        twiss_df[f"{ERR}{PHASE_ADV}X"] = err_mu_x
        twiss_df[f"{ERR}{PHASE_ADV}Y"] = err_mu_y

    # Headers: tunes if available
    headers: dict[str, float] = {}
    if "Q1" in beta_x.headers:
        headers["Q1"] = beta_x.headers["Q1"]
    if "Q2" in beta_x.headers:
        headers["Q2"] = beta_x.headers["Q2"]
    twiss_df.headers = headers
    return twiss_df


def _compute_cumulative_phase(phase_df: tfs.TfsDataFrame, sorted_index: list, phase_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative phase advance from the start of the lattice."""
    mu = np.zeros(len(sorted_index))
    err_mu = np.zeros(len(sorted_index))
    current_mu = 0.0
    current_err_sq = 0.0
    for i, name in enumerate(sorted_index):
        mu[i] = current_mu
        err_mu[i] = np.sqrt(current_err_sq)
        # Find the segment starting from this name
        segment = phase_df.loc[phase_df.index == name]
        if not segment.empty:
            phase_val = segment.iloc[0][phase_col]
            err_val = segment.iloc[0][f"{ERR}{phase_col}"]
            current_mu += phase_val
            current_err_sq += err_val ** 2
    return mu, err_mu
