"""Create parquet training data from raw turn-by-turn measurements.

The functions in this module convert operational or simulated turn-by-turn
files into the parquet format consumed by the optimisation fitters. The
pipeline also saves supporting corrector and tune-knob files alongside the
processed tracking data.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from pymadng_utils.model_creator.madng_utils import update_model_with_madng
from tmom_recon import ACDipoleConfig, ModelDetails

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad import GenericMadInterface
from aba_optimiser.measurements.ac_dipole import infer_ac_dipole_s
from aba_optimiser.measurements.analysis import run_measurement_analysis
from aba_optimiser.measurements.b2_errors import b2_errors_to_magnet_strengths, read_b2_error_table
from aba_optimiser.measurements.loading import (
    build_dataframe_file_indices,
    convert_tbt_to_dataframes,
    load_measurement_files,
)
from aba_optimiser.measurements.reconstruction import process_single_dataframe
from aba_optimiser.measurements.sequence import extract_tunes_from_job_file

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tmom_recon import ReconstructionFrame
LOGGER = logging.getLogger(__name__)

AC_DIPOLE_ATTR_KEYS = (
    "ac_dipole_marker",
    "ac_dipole_bpm_upstream",
    "ac_dipole_bpm_downstream",
    "ac_dipole_n_bpms_each_side",
    "ac_dipole_smooth_lambda",
    "ac_dipole_barrier_s",
)


@dataclass(frozen=True)
class ACDipoleReconstructionConfig:
    """Configuration for AC-dipole assisted px/py reconstruction."""

    n_bpms_each_side: int = 1
    tune_knobs_files: list[Path | None] | None = None
    corrector_knobs_files: list[Path | None] | None = None


def _tune_knobs_cache_key(
    tune_knobs: Mapping[str, float] | str | Path | None,
) -> str | tuple[tuple[str, float], ...] | None:
    """A hashable identity for a knob set given either as values or as a file."""
    if tune_knobs is None:
        return None
    if isinstance(tune_knobs, (str, Path)):
        return str(Path(tune_knobs))
    return tuple(sorted((str(k), float(v)) for k, v in tune_knobs.items()))


def copy_ac_dipole_attrs(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Copy AC-dipole metadata attrs from source dataframe to target."""
    for key in AC_DIPOLE_ATTR_KEYS:
        if key in source.attrs:
            target.attrs[key] = source.attrs[key]


def average_required_attr(attrs: list[dict], key: str) -> float:
    """Average a numeric DataFrame attr required on every input frame."""
    missing = [idx for idx, item in enumerate(attrs) if key not in item]
    if missing:
        raise ValueError(f"Processed dataframe attrs missing required {key!r}: indices {missing}")
    return float(sum(item[key] for item in attrs) / len(attrs))


def validate_ac_dipole_reconstruction_config(
    config: ACDipoleReconstructionConfig | None,
    file_count: int,
) -> None:
    """Validate per-file AC-dipole reconstruction inputs."""
    if config is None:
        return
    if config.tune_knobs_files is not None and len(config.tune_knobs_files) != file_count:
        raise ValueError(
            "ac_dipole_reconstruction_config.tune_knobs_files must match files length: "
            f"{len(config.tune_knobs_files)} != {file_count}"
        )
    if config.corrector_knobs_files is not None and len(config.corrector_knobs_files) != file_count:
        raise ValueError(
            "ac_dipole_reconstruction_config.corrector_knobs_files must match files length: "
            f"{len(config.corrector_knobs_files)} != {file_count}"
        )


def expand_machine_deltaps(
    machine_deltaps: float | list[float | None] | None,
    file_count: int,
) -> list[float | None]:
    """Return one machine deltap value per measurement file.

    These are converted to the momentum offset used with the explicit
    setting-zero reconstruction frame.
    """
    if machine_deltaps is None:
        return [None] * file_count
    if isinstance(machine_deltaps, (int, float)):
        return [float(machine_deltaps)] * file_count

    per_file_machine_deltaps = list(machine_deltaps)
    if len(per_file_machine_deltaps) != file_count:
        raise ValueError(
            "machine_deltaps must match files length when provided as a list: "
            f"{len(per_file_machine_deltaps)} != {file_count}"
        )
    return per_file_machine_deltaps


# def write_datafile(data: pd.DataFrame, output_file: str | Path) -> None:
#     """Write the combined DataFrame to a Parquet file."""
#     data.to_parquet(output_file)
def detect_bad_bpms(
    pzs: pd.DataFrame | list[pd.DataFrame],
    all_bpms: set[str],
    bad_bpms: list[str],
    log_individual: bool = True,
) -> None:
    """Detect and add bad BPMs to the list.

    Args:
        pzs: DataFrame or list of DataFrames with processed data
        all_bpms: Set of all expected BPM names
        bad_bpms: List to extend with bad BPMs
        log_individual: Whether to log individual bad BPMs
    """
    if isinstance(pzs, pd.DataFrame):
        pzs = [pzs]

    for pz in pzs:
        # BPMs with NaN in px or py
        mask = pz["px"].isna() | pz["py"].isna()
        bad_bpms_mask = mask.groupby(pz["name"], observed=False).any()
        new_bad = bad_bpms_mask[bad_bpms_mask].index.tolist()
        bad_bpms.extend(new_bad)
        if log_individual:
            for bpm in new_bad:
                LOGGER.info(f"BPM {bpm}: has_nan=True")

        # BPMs with infinite variance in both planes (x/y)
        zero_mask = ((np.isinf(pz["var_x"])) | (np.isinf(pz["var_px"]))) & (
            (np.isinf(pz["var_y"])) | (np.isinf(pz["var_py"]))
        )
        bad_bpms_zero = zero_mask.groupby(pz["name"], observed=False).any()
        new_bad_zero = bad_bpms_zero[bad_bpms_zero].index.tolist()
        bad_bpms.extend(new_bad_zero)
        if log_individual:
            for bpm in new_bad_zero:
                LOGGER.info(f"BPM {bpm}: zero_weight=True")

    # Missing BPMs
    all_unique_bpms = set.union(*(set(pz["name"].unique()) for pz in pzs))
    missing_bpms = all_bpms - all_unique_bpms
    bad_bpms.extend(missing_bpms)
    if log_individual:
        for bpm in missing_bpms:
            LOGGER.info(f"BPM {bpm}: missing from data")

    # Remove duplicates
    bad_bpms[:] = list(set(bad_bpms))


def build_madng_twiss_table(
    model_dir: Path,
    accelerator: LHC,
    output_dir: Path,
    nattunes: list[float],
    tunes: list[float],
) -> pd.DataFrame:
    """Create a MAD-NG Twiss DataFrame for the given model directory and accelerator.

    Args:
        model_dir: Directory containing the MAD-NG model files
        accelerator: LHC accelerator carrying beam, energy and sequence metadata
        output_dir: Directory to save the generated Twiss table
    Returns:
        Twiss DataFrame with optics parameters
    """
    tws_file = output_dir / "twiss_ac.dat"
    if not tws_file.exists():
        LOGGER.info(
            f"Generating MAD-NG Twiss tables with extracted model tunes: {nattunes}, {tunes}"
        )
        natural_tunes = nattunes[:2]
        driven_tunes = tunes[:2]
        update_model_with_madng(
            accelerator,
            output_dir,
            tunes=natural_tunes,
            drv_tunes=driven_tunes,
            convert_to_madx=False,
        )
    return tfs.read(tws_file)


def process_measurements(
    files: list[Path],
    output_dir: Path,
    model_dir: str | Path,
    accelerator: LHC,
    frame: ReconstructionFrame,
    filename: str | None = "pz_data.parquet",
    bad_bpms: list[str] | None = None,
    b2_errors: Path | None = None,
    previous_analysis_dir: str | Path | None = None,
    use_uniform_vars: bool = False,
    num_workers: int | None = None,
    combine_files: bool = True,
    nattunes: list[float] | None = None,
    tunes: list[float] | None = None,
    machine_deltaps: float | list[float] | None = None,
    ac_dipole_reconstruction_config: ACDipoleReconstructionConfig | None = None,
    trim_to_kick: bool = False,
    n_turns_free: int = 1000,
    kicker_name: str | None = None,
    nan_variance_patterns: str | list[str] | None = None,
    accelerator_type: str = "lhc",
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Path], pd.DataFrame]:
    """Process measurement files to compute pz data and identify bad BPMs.

    Args:
        files: List of measurement file paths
        output_dir: Directory for analysis outputs
        model_dir: Directory containing model files
        accelerator: LHC accelerator carrying beam, sequence_file and pc
        frame: Measured orbit-zero frame used by every reconstruction.
        filename: Output filename for parquet file (None to skip saving)
        bad_bpms: List of bad BPM names (None to run analysis)
        b2_errors: Optional LHC dipole b2 error table applied to each AC-dipole
                reconstruction model (dknl[2]). Requires per-file tune knobs, since
                b2 errors shift the tunes. Note: only the reconstruction *model* is
                perturbed here; the model twiss read for BPM optics is still nominal.
        use_uniform_vars: If True, use uniform variances instead of noise-based
        num_workers: Number of parallel workers (None for auto)
        combine_files: If True, combine all processed dataframes into one dict entry with key 'combined';
                      if False, return dict with file paths as keys
        nattunes: Natural tunes [Qx, Qy, Qz] (None to extract from model)
        tunes: Driven tunes [Qx, Qy, Qz] (None to extract from model)
        machine_deltaps: Optional machine momentum offsets used during px/py reconstruction.
                If a list, must match files length and will be expanded per bunch.
        trim_to_kick: Trim raw data to the detected kick without subtracting its orbit.
        n_turns_free: Number of pre-kick turns used for kick detection.
        kicker_name: Optional kicker marker name used to detect already-aligned input.
        nan_variance_patterns: Optional regex pattern or patterns for names that should receive
                NaN variances instead of failing the known-noise lookup.
        accelerator_type: Noise-table accelerator key used for known-noise variances.

    Returns:
        Tuple of (dict mapping file paths to dataframes, bad_bpms_list, dict mapping keys to output paths, twiss_df)
    """
    if nattunes is None or tunes is None:
        job_file = Path(model_dir) / "job.create_model_nominal.madx"
        nat_x, nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
        if nattunes is None:
            nattunes = [nat_x, nat_y, 0.0]
        if tunes is None:
            tunes = [drv_x, drv_y, 0.0]
        LOGGER.info(f"Extracted tunes: nattunes={nattunes}, tunes={tunes}")

    if nattunes is None or tunes is None:
        raise ValueError("Could not determine natural and driven tunes for measurement processing.")

    validate_ac_dipole_reconstruction_config(ac_dipole_reconstruction_config, len(files))
    if b2_errors is not None and not isinstance(accelerator, LHC):
        raise ValueError("b2_errors are only supported for LHC reconstruction models.")
    ac_dipole_barrier_s = (
        infer_ac_dipole_s(Path(model_dir), accelerator.ac_dipole_name)
        if ac_dipole_reconstruction_config is not None
        else None
    )
    acd_thread_local = threading.local()
    b2_magnet_strengths = (
        b2_errors_to_magnet_strengths(read_b2_error_table(b2_errors)) if b2_errors is not None else None
    )
    default_model_details = ModelDetails(accelerator=accelerator, pt=0.0)

    if bad_bpms is None or previous_analysis_dir is None:
        bad_bpms = run_measurement_analysis(
            output_dir,
            model_dir,
            files,
            beam=accelerator.beam,
            nattunes=nattunes,
            tunes=tunes,
        )
        LOGGER.warning(
            "Previous analysis directory not provided; ran analysis for processing measurements."
        )
        analysis_dir = output_dir
    else:
        analysis_dir = Path(previous_analysis_dir)
        if not analysis_dir.exists():
            raise FileNotFoundError(
                f"Provided previous_analysis_dir {analysis_dir} does not exist."
            )

    data = load_measurement_files(files, beam=accelerator.beam)
    combined = convert_tbt_to_dataframes(data, bad_bpms, combine_measurements=combine_files)
    dataframe_file_indices = build_dataframe_file_indices(data)

    per_file_machine_deltaps = expand_machine_deltaps(machine_deltaps, len(files))

    def machine_deltap_for_dataframe(df_idx: int) -> float | None:
        return per_file_machine_deltaps[dataframe_file_indices[df_idx]]

    def get_thread_local_ac_dipole_inputs(
        df_idx: int,
    ) -> tuple[ModelDetails, ACDipoleConfig] | None:
        if ac_dipole_reconstruction_config is None:
            return None
        cfg_cache = getattr(acd_thread_local, "config_cache", None)
        if cfg_cache is None:
            cfg_cache = {}
            acd_thread_local.config_cache = cfg_cache

        file_idx = dataframe_file_indices[df_idx]
        tune_knobs = (
            ac_dipole_reconstruction_config.tune_knobs_files[file_idx]
            if ac_dipole_reconstruction_config.tune_knobs_files is not None
            else None
        )
        machine_deltap = machine_deltap_for_dataframe(df_idx)
        # ``tune_knobs`` is either a path or the knob values themselves, and a
        # mapping is neither hashable nor a Path -- key on the values in that
        # case so two different knob sets cannot share one cached config.
        cache_key = (
            _tune_knobs_cache_key(tune_knobs),
            machine_deltap,
        )
        cached = cfg_cache.get(cache_key)
        if cached is not None:
            return cached

        if b2_magnet_strengths is not None and tune_knobs is None:
            raise ValueError(
                "The tune knobs are designed to compensate for the known b2 errors."
                "Therefore it makes no sense to apply b2 errors without also applying the tune knobs."
            )

        acd_accelerator = accelerator.copy_with()
        machine_pt = acd_accelerator.dp2pt(machine_deltap or 0.0)
        # LHC only: perturb the reconstruction model with the known b2 dipole
        # errors (via dknl[2]) so the AC-dipole momentum jump is transported
        # through the real error lattice. Tune knobs are applied first, so the
        # tune shift from these errors is already compensated when we add them.
        model_details = ModelDetails(
            accelerator=acd_accelerator,
            pt=machine_pt,
            magnet_strengths=b2_magnet_strengths,
            tune_knobs=tune_knobs,
        )
        acd_config = ACDipoleConfig(
            ac_dipole_marker=accelerator.ac_dipole_name,
            driven_tunes=(float(tunes[0]), float(tunes[1])),
            barrier_s=ac_dipole_barrier_s,
        )
        cached = (model_details, acd_config)
        cfg_cache[cache_key] = cached
        return cached

    if any(dpp is not None for dpp in per_file_machine_deltaps):
        LOGGER.info("Using provided machine deltaps for measurement processing.")

    LOGGER.info(f"Combined data has {len(combined)} DataFrames from different files/bunches.")
    if len(dataframe_file_indices) != len(combined):
        raise ValueError(
            "Converted dataframe count does not match source-file mapping: "
            f"{len(combined)} != {len(dataframe_file_indices)}"
        )
    tws = build_madng_twiss_table(Path(model_dir), accelerator, output_dir, nattunes, tunes)
    tws.columns = [col.lower() for col in tws.columns]
    tws = tws.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws.headers = {k.lower(): v for k, v in tws.headers.items()}
    tws = tws.set_index("name")

    # Process DataFrames in parallel using threads to avoid Spark context inheritance issues
    # ThreadPoolExecutor shares memory space and doesn't inherit problematic global state like ProcessPoolExecutor
    LOGGER.info(f"Processing {len(combined)} DataFrames in parallel with threads...")
    processed_results: list[pd.DataFrame | None] = [None] * len(combined)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid Spark context conflicts
    # Limit to max 9 threads to avoid overloading the system
    effective_workers = min(num_workers or len(combined), 9)
    if effective_workers > 0:
        try:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_dataframe,
                        df_with_index=(i, df),
                        twiss=tws,
                        bad_bpms=bad_bpms,
                        analysis_dir=analysis_dir,
                        use_uniform_vars=use_uniform_vars,
                        beam=accelerator.beam,
                        model_details=default_model_details,
                        frame=frame,
                        ac_dipole_inputs_factory=get_thread_local_ac_dipole_inputs,
                        machine_deltap=machine_deltap_for_dataframe(i),
                        trim_to_kick=trim_to_kick,
                        n_turns_free=n_turns_free,
                        kicker_name=kicker_name,
                        nan_variance_patterns=nan_variance_patterns,
                        accelerator_type=accelerator_type,
                    ): i
                    for i, df in enumerate(combined)
                }

                for future in as_completed(futures):
                    try:
                        idx, processed_df = future.result(timeout=600)  # 10 minute timeout per task
                        processed_results[idx] = processed_df
                        LOGGER.info(f"Completed processing dataframe {idx + 1}/{len(combined)}")
                    except Exception as e:
                        idx = futures[future]
                        LOGGER.error(f"Error processing dataframe {idx}: {e}")
                        raise
        except KeyboardInterrupt:
            LOGGER.warning("Keyboard interrupt received, shutting down gracefully...")
            # The ThreadPoolExecutor context manager will handle cleanup
            raise
    else:
        for i, df in enumerate(combined):
            idx, processed_df = process_single_dataframe(
                df_with_index=(i, df),
                twiss=tws,
                bad_bpms=bad_bpms,
                analysis_dir=analysis_dir,
                use_uniform_vars=use_uniform_vars,
                beam=accelerator.beam,
                model_details=default_model_details,
                frame=frame,
                ac_dipole_inputs_factory=get_thread_local_ac_dipole_inputs,
                machine_deltap=machine_deltap_for_dataframe(i),
                trim_to_kick=trim_to_kick,
                n_turns_free=n_turns_free,
                kicker_name=kicker_name,
                nan_variance_patterns=nan_variance_patterns,
                accelerator_type=accelerator_type,
            )
            processed_results[idx] = processed_df
            LOGGER.info(f"Completed processing dataframe {idx + 1}/{len(combined)}")

    if any(res is None for res in processed_results):
        raise RuntimeError("Some dataframes failed to process.")

    combined: list[pd.DataFrame] = processed_results  # ty:ignore[invalid-assignment]
    if combine_files:
        combined_attrs = [df.attrs.copy() for df in combined]
        # Remove the attrs per file and combine all processed DataFrames into one
        for df in combined:
            df.attrs.clear()
        pzs_combined = pd.concat(combined, ignore_index=True)
        pzs_combined["name"] = pzs_combined["name"].astype("category")
        pzs_combined["turn"] = pzs_combined["turn"].astype("int32")
        pzs_combined["bunch_number"] = pzs_combined["bunch_number"].astype("int32")
        pzs_combined.attrs["PT_EST"] = average_required_attr(combined_attrs, "PT_EST")
        pzs_combined.attrs.update(
            {
                key: combined_attrs[0][key]
                for key in AC_DIPOLE_ATTR_KEYS
                if key in combined_attrs[0]
            }
        )
        pzs_dict: dict[str, pd.DataFrame] = {"combined": pzs_combined}
    else:
        # Group each file's bunches using the source-file mapping. The bunch_number
        # column already distinguishes bunches within each written parquet, so no
        # count inference is needed.
        file_groups: dict[int, list[pd.DataFrame]] = {}
        for df, file_idx in zip(combined, dataframe_file_indices):
            file_groups.setdefault(file_idx, []).append(df)
        pzs_dict: dict[str, pd.DataFrame] = {}
        for file_idx, file_dfs in file_groups.items():
            file_pzs = pd.concat(file_dfs, ignore_index=True)
            file_pzs["name"] = file_pzs["name"].astype("category")
            file_pzs["turn"] = file_pzs["turn"].astype("int32")
            file_pzs["bunch_number"] = file_pzs["bunch_number"].astype("int32")
            file_pzs.attrs["PT_EST"] = average_required_attr(
                [df.attrs for df in file_dfs], "PT_EST"
            )
            copy_ac_dipole_attrs(file_dfs[0], file_pzs)
            pzs_dict[str(files[file_idx])] = file_pzs

    mad_iface = GenericMadInterface(accelerator)
    all_bpms = set(mad_iface.all_bpms)
    del mad_iface

    if combine_files:
        pzs_combined = pzs_dict["combined"]
        pzs_combined["name"] = pzs_combined["name"].astype("category")

        detect_bad_bpms(pzs_combined, all_bpms, bad_bpms, log_individual=True)

        LOGGER.info(f"Total bad BPMs: {len(bad_bpms)}")

        if filename:
            file_path = output_dir / filename
            pzs_combined.to_parquet(file_path)
            output_paths = {"combined": file_path}
        else:
            output_paths = {"combined": output_dir}

        return pzs_dict, bad_bpms, output_paths, tws

    detect_bad_bpms(list(pzs_dict.values()), all_bpms, bad_bpms, log_individual=False)

    LOGGER.info(f"Total bad BPMs: {len(bad_bpms)}")

    if filename:
        output_paths: dict[str, Path] = {}
        for i, (file_key, pz) in enumerate(pzs_dict.items()):
            file_path = output_dir / f"{Path(filename).stem}_{i}.parquet"
            pz.to_parquet(file_path)
            output_paths[file_key] = file_path
    else:
        output_paths = dict.fromkeys(pzs_dict, output_dir)

    return pzs_dict, bad_bpms, output_paths, tws
