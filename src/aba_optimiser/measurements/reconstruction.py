"""Per-dataframe momentum reconstruction helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tmom_recon import ACDipoleConfig, calculate_pz_measurement
from tmom_recon.svd import svd_clean_measurements

from aba_optimiser.measurements.preprocessing import (
    ClosedOrbitInput,
    preprocess_measurement_dataframe,
)
from aba_optimiser.measurements.variances import (
    assign_known_noise_variances,
    assign_uniform_variances,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd

LOGGER = logging.getLogger(__name__)


def process_single_dataframe(
    df_with_index: tuple[int, pd.DataFrame],
    twiss: pd.DataFrame,
    bad_bpms: list[str],
    analysis_dir: Path,
    use_uniform_vars: bool,
    beam: int,
    ac_dipole_config_factory: Callable[[int], ACDipoleConfig | None] | None = None,
    machine_deltap: float | None = None,
    remove_closed_orbit: ClosedOrbitInput = None,
    n_turns_free: int = 1000,
    kicker_name: str | None = None,
    nan_variance_patterns: str | list[str] | None = None,
    accelerator_type: str = "lhc",
) -> tuple[int, pd.DataFrame]:
    """Preprocess, weight, and reconstruct one measurement dataframe."""
    index, df = df_with_index
    ac_dipole_config = (
        ac_dipole_config_factory(index) if ac_dipole_config_factory is not None else None
    )

    df = preprocess_measurement_dataframe(
        df,
        twiss,
        remove_closed_orbit=remove_closed_orbit,
        n_turns_free=n_turns_free,
        kicker_name=kicker_name,
    )
    df = svd_clean_measurements(df)
    df = df[df["name"].isin(twiss.index)]
    df = _assign_variances(
        df,
        bad_bpms,
        use_uniform_vars=use_uniform_vars,
        nan_variance_patterns=nan_variance_patterns,
        accelerator_type=accelerator_type,
    )

    df = calculate_pz_measurement(
        df,
        analysis_dir,
        model_tws=twiss,
        include_errors=True,
        include_optics_errors=True,
        reverse_meas_tws=beam == 2,
        dpp_override=machine_deltap if machine_deltap is not None else 0.0,
        ac_dipole_config=ac_dipole_config,
    )
    df = _scale_position_variances_after_svd(df)
    df = _drop_nan_momenta(df, dataframe_index=index)
    return index, df


def _assign_variances(
    df: pd.DataFrame,
    bad_bpms: list[str],
    *,
    use_uniform_vars: bool,
    nan_variance_patterns: str | list[str] | None,
    accelerator_type: str,
) -> pd.DataFrame:
    if use_uniform_vars:
        return assign_uniform_variances(df, bad_bpms)

    noise_input = df.set_index("name")
    weighted = assign_known_noise_variances(
        noise_input,
        bad_bpms,
        nan_variance_patterns=nan_variance_patterns,
        accelerator_type=accelerator_type,
    )
    return weighted.reset_index()


def _scale_position_variances_after_svd(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["var_x"] = result["var_x"] / 100.0
    result["var_y"] = result["var_y"] / 100.0
    return result


def _drop_nan_momenta(df: pd.DataFrame, *, dataframe_index: int) -> pd.DataFrame:
    if df["px"].isna().any() or df["py"].isna().any():
        LOGGER.warning("NaN values found in px or py for dataframe %s, dropping rows.", dataframe_index)
        return df.dropna(subset=["px", "py"])
    return df
