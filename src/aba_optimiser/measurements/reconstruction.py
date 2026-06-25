"""Per-dataframe momentum reconstruction helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from pymadng_utils.physics import dp2pt
from tmom_recon import ACDipoleConfig, calculate_pz
from tmom_recon.svd import weighted_svd_clean_measurements

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

LOGGER = logging.getLogger(__name__)
PROTON_MASS_GEV = 0.93827208816


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
    # Each input dataframe is a single bunch; preserve its identifier across the
    # reconstruction transforms (some of which rebuild the frame and drop columns).
    bunch_number = int(df["bunch_number"].iloc[0])
    # Check every row has the same bunch number, otherwise the input dataframe is malformed.
    if not (df["bunch_number"] == bunch_number).all():
        raise ValueError(f"Input dataframe {index} contains multiple bunch numbers")
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
    df = df[df["name"].isin(twiss.index)]
    df = _assign_variances(
        df,
        bad_bpms,
        use_uniform_vars=use_uniform_vars,
        nan_variance_patterns=nan_variance_patterns,
        accelerator_type=accelerator_type,
    )
    df = weighted_svd_clean_measurements(df)

    machine_pt = _machine_deltap_to_pt(machine_deltap, twiss)
    df = calculate_pz(
        df,
        measurement_dir=analysis_dir,
        model_tws=twiss,
        # include_errors=True,
        # include_optics_errors=True,
        reverse_meas_tws=beam == 2,
        pt_override=machine_pt,
        acd=ac_dipole_config,
    )
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Reconstruction returned unexpected type {type(df)} for dataframe")
    df = _scale_position_variances_after_svd(df)
    df = _drop_nan_momenta(df, dataframe_index=index)
    df["bunch_number"] = bunch_number
    return index, df


def _machine_deltap_to_pt(machine_deltap: float | None, twiss: pd.DataFrame) -> float:
    """Convert machine ``dp/p`` metadata to MAD-NG ``pt`` for dispersion-aware reconstruction."""
    if machine_deltap is None:
        return 0.0
    headers = {str(key).lower(): value for key, value in getattr(twiss, "headers", {}).items()}
    beta0 = headers.get("beta")
    if beta0 is not None:
        return dp2pt(machine_deltap, float(beta0))
    energy = headers.get("energy")
    if energy is not None:
        beta0 = (1.0 - (PROTON_MASS_GEV / float(energy)) ** 2) ** 0.5
        return dp2pt(machine_deltap, beta0)
    LOGGER.warning(
        "Twiss headers do not contain beam beta or energy; converting machine_deltap to pt with beta=1."
    )
    return dp2pt(machine_deltap)


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
