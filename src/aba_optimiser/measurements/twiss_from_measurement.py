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
    twiss_df.index.name = NAME

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
    mu_x, var_mu_x, total_var_x = _compute_cumulative_phase(phase_x, sorted_index, f"{PHASE}X")
    mu_y, var_mu_y, total_var_y = _compute_cumulative_phase(phase_y, sorted_index, f"{PHASE}Y")
    twiss_df[f"{PHASE_ADV}X"] = mu_x
    twiss_df[f"{PHASE_ADV}Y"] = mu_y

    # Always store variance; cheap and avoids later mistakes
    twiss_df["mu1_var"] = var_mu_x
    twiss_df["mu2_var"] = var_mu_y

    # Optionally include errors
    if include_errors:
        twiss_df[f"{ERR}{ORBIT}X"] = orbit_x.loc[sorted_index, f"{ERR}{ORBIT}X"]
        twiss_df[f"{ERR}{ORBIT}Y"] = orbit_y.loc[sorted_index, f"{ERR}{ORBIT}Y"]
        twiss_df[f"{ERR}{BETA}X"] = beta_x.loc[sorted_index, f"{ERR}{BETA}X"]
        twiss_df[f"{ERR}{BETA}Y"] = beta_y.loc[sorted_index, f"{ERR}{BETA}Y"]
        twiss_df[f"{ERR}{ALPHA}X"] = beta_x.loc[sorted_index, f"{ERR}{ALPHA}X"]
        twiss_df[f"{ERR}{ALPHA}Y"] = beta_y.loc[sorted_index, f"{ERR}{ALPHA}Y"]
        twiss_df[f"{ERR}{DISPERSION}X"] = disp_x.loc[sorted_index, f"{ERR}{DISPERSION}X"]
        twiss_df[f"{ERR}{DISPERSION}Y"] = disp_y.loc[sorted_index, f"{ERR}{DISPERSION}Y"]
        twiss_df[f"{ERR}{MOMENTUM_DISPERSION}X"] = disp_x.loc[sorted_index, f"{ERR}{MOMENTUM_DISPERSION}X"]
        twiss_df[f"{ERR}{MOMENTUM_DISPERSION}Y"] = disp_y.loc[sorted_index, f"{ERR}{MOMENTUM_DISPERSION}Y"]
        twiss_df[f"{ERR}{PHASE_ADV}X"] = np.sqrt(var_mu_x)
        twiss_df[f"{ERR}{PHASE_ADV}Y"] = np.sqrt(var_mu_y)

    # Headers: tunes if available and total variances
    headers: dict[str, float] = {}
    headers["MU1_TOTAL_VAR"] = total_var_x
    headers["MU2_TOTAL_VAR"] = total_var_y
    if "Q1" in beta_x.headers:
        headers["Q1"] = beta_x.headers["Q1"]
    if "Q2" in beta_x.headers:
        headers["Q2"] = beta_x.headers["Q2"]
    twiss_df.headers = headers
    return twiss_df


def _compute_cumulative_phase(
    phase_df: pd.DataFrame, sorted_index: pd.Index, phase_col: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute cumulative phase (turns) and cumulative variance (turns^2)
    assuming each row provides the phase advance from NAME -> NAME2 and its error.

    I don't like this function because it recalculates the phase advances from differences,
    instead we could just use these differences directly when deciding the bpms with the correct
    phase advance (pi/2 or pi). For now this is only used in testing so it's acceptable.

    Returns:
        mu: cumulative phase at each BPM in sorted_index (turns)
        var_mu: cumulative variance of mu at each BPM (turns^2)
        total_var: total variance around ring (turns^2)

    """
    # Step phase advance for each BPM in lattice order
    step = phase_df[phase_col].reindex(sorted_index).to_numpy(float)
    step_err = phase_df[f"{ERR}{phase_col}"].reindex(sorted_index).to_numpy(float)

    # If there can be missing BPMs, decide policy:
    # - missing step means "no information": treat as 0 with 0 error
    step = np.nan_to_num(step, nan=0.0)
    step_err = np.nan_to_num(step_err, nan=0.0)

    step_var = step_err**2
    total_var = float(step_var.sum())

    # mu at BPM i is sum of steps from BPM 0 up to i-1
    # so shift the cumulative sum by 1
    mu = np.zeros(len(sorted_index), dtype=float)
    var_mu = np.zeros(len(sorted_index), dtype=float)
    if len(sorted_index) > 1:
        mu[1:] = np.cumsum(step[:-1])
        var_mu[1:] = np.cumsum(step_var[:-1])

    return mu, var_mu, total_var
