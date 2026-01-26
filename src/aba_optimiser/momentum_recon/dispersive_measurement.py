from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aba_optimiser.momentum_recon.measurement_pipeline import (
    aggregate_results,
    attach_errors_inplace,
    calculate_momenta,
    prepare_data,
    process_twiss,
    run_diagnostics,
    setup_momentum_calculation,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def calculate_pz_measurement(
    orig_data: pd.DataFrame,
    measurement_folder: str | Path,
    model_tws: pd.DataFrame,
    info: bool = True,
    include_errors: bool = False,
    include_optics_errors: bool = False,
) -> pd.DataFrame:
    """Calculate transverse momenta from dispersive measurements.

    Automatically handles closed orbit removal before calculation and restoration after,
    along with addition of reference momenta from the twiss.

    Args:
        orig_data: Original tracking data.
        measurement_folder: Path to measurement files.
        model_tws: Model twiss data for closed orbit restoration.
        info: Whether to print diagnostic info.
        include_errors: Whether to include error columns from measurements.
        include_optics_errors: Whether to include optical function uncertainties in error propagation.

    Returns:
        DataFrame with calculated px and py columns, with closed orbit and reference
        momenta restored.

    Raises:
        ValueError: If error columns are inconsistent.
        RuntimeError: If dispersion maps not initialized correctly.
    """
    LOGGER.info(
        "Calculating dispersive transverse momentum from measurements - measurement_folder=%s",
        measurement_folder,
    )

    # Stage 1: Prepare data
    data, features = prepare_data(orig_data)

    # Get BPM list and filter data
    bpm_list = data["name"].unique().tolist()

    # Stage 2: Process twiss
    tws, has_errors, dispersion_found = process_twiss(
        Path(measurement_folder), bpm_list, include_errors
    )

    # Filter data to only BPMs present in the twiss
    data = data[data["name"].isin(tws.index)]

    # Check error consistency
    if include_errors and not has_errors:
        raise ValueError("include_errors=True but no error columns found in measurements")

    # Stage 3: Set up momentum calculation
    data_p, data_n, dpp_est = setup_momentum_calculation(
        data, tws, model_tws, dispersion_found, info
    )

    # Stage 4: Attach errors if they exist
    if has_errors:
        attach_errors_inplace(data_p, data_n, tws)

    # Stage 5: Calculate momenta
    data_p, data_n = calculate_momenta(data_p, data_n, dpp_est, include_optics_errors)

    # Stage 6: Aggregate results
    data_avg = aggregate_results(data_p, data_n, model_tws, dpp_est)

    # Stage 7: Run diagnostics
    run_diagnostics(orig_data, data_p, data_n, data_avg, info, features)

    return data_avg
