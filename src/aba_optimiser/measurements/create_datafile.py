"""Create parquet training data from raw turn-by-turn measurements.

The functions in this module convert operational or simulated turn-by-turn
files into the parquet format consumed by the optimisation controllers. The
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
from omc3.hole_in_one import hole_in_one_entrypoint
from pymadng_utils.model_creator.madng_utils import update_model_with_madng
from tmom_recon import ACDipoleConfig, calculate_pz_measurement
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from tmom_recon.svd import svd_clean_measurements

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import (
    DPP_OPTIMISER_CONFIG,
    DPP_SIMULATION_CONFIG,
    PROJECT_ROOT,
)
from aba_optimiser.mad import GenericMadInterface
from aba_optimiser.measurements.online_knobs import build_dict_from_nxcal_result, save_online_knobs
from aba_optimiser.measurements.squeeze_helpers import (
    extract_tunes_from_job_file,
    get_or_make_sequence,
)
from aba_optimiser.measurements.tbt_io import (
    _build_dataframe_file_indices,
    convert_measurements,
    load_files,
)
from aba_optimiser.noise import assign_bpm_variances
from aba_optimiser.training.controller import Controller
from aba_optimiser.training.controller_config import MeasurementConfig, OutputConfig, SequenceConfig

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)

AC_DIPOLE_ATTR_KEYS = (
    "ac_dipole_marker",
    "ac_dipole_bpm_upstream",
    "ac_dipole_bpm_downstream",
    "ac_dipole_n_bpms_each_side",
    "ac_dipole_smooth_lambda",
)


@dataclass(frozen=True)
class ACDipoleReconstructionConfig:
    """Configuration for AC-dipole assisted px/py reconstruction."""

    n_bpms_each_side: int = 1
    tune_knobs_files: list[Path | None] | None = None
    corrector_knobs_files: list[Path | None] | None = None


def compute_uniform_vars(
    combined: pd.DataFrame, bad_bpms: list[str], var_value: float = (1e-4) ** 2
) -> pd.DataFrame:
    """Set uniform variances for all BPMs.

    Args:
        combined: DataFrame with BPM measurements
        bad_bpms: List of bad BPM names to set to infinite variance
        var_value: Uniform variance value to use (default: (1e-4) ** 2)

    Returns:
        DataFrame with var_x, var_y, var_px, var_py columns added
    """
    combined["var_x"] = var_value
    combined["var_y"] = var_value
    combined["var_px"] = var_value
    combined["var_py"] = var_value
    combined.loc[combined["name"].isin(bad_bpms), "var_x"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_y"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_px"] = float("inf")
    combined.loc[combined["name"].isin(bad_bpms), "var_py"] = float("inf")
    return combined


def compute_vars_from_known_noise(combined: pd.DataFrame, bad_bpms: list[str]) -> pd.DataFrame:
    """Compute variances for x and y planes based on known noise levels."""
    return assign_bpm_variances(combined, accelerator_type="lhc", bad_bpms=bad_bpms)


def run_analysis(
    analysis_dir: str | Path,
    model_dir: str | Path,
    files: list[Path],
    beam: int,
    nattunes: list[float],
    tunes: list[float],
) -> list[str]:
    """Load, combine, and process data from multiple files."""
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    # bunches = [32, 1228, 2000] if beam == 2 else [0, 1138, 1968]
    hole_in_one_entrypoint(
        harpy=True,
        files=files,
        outputdir=analysis_dir / "lin_files",
        unit="mm",
        driven_excitation="acd",
        first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        is_free_kick=False,
        keep_exact_zeros=False,
        max_peak=0.02,
        nattunes=nattunes,
        num_svd_iterations=3,
        opposite_direction=beam == 2,
        output_bits=10,
        peak_to_peak=1e-08,
        resonances=4,
        sing_val=12,
        svd_dominance_limit=0.925,
        to_write=["lin", "spectra", "full_spectra", "bpm_summary"],
        tune_clean_limit=1e-05,
        tunes=tunes,
        turn_bits=14,
        model_dir=model_dir,
        turns=[0, 50000],
        clean=True,
    )
    # Find all the bunch IDs that were created
    linfile_dir = analysis_dir / "lin_files"
    analysed_files: list[Path] = [
        created_file.with_suffix("") for created_file in linfile_dir.glob("*_bunchID*.linx")
    ]

    hole_in_one_entrypoint(
        optics=True,
        files=analysed_files,
        outputdir=analysis_dir,
        analyse_dpp=0,
        # calibrationdir="/afs/cern.ch/eng/sl/lintrack/LHC_commissioning2017/Calibration_factors_2017/Calibration_factors_2017_beam1",
        chromatic_beating=False,
        compensation="equation",
        coupling_method=2,
        coupling_pairing=0,
        isolation_forest=False,
        nonlinear=[],
        only_coupling=False,
        range_of_bpms=11,
        second_order_dispersion=False,
        three_bpm_method=False,
        three_d_excitation=False,
        union=False,
        accel="lhc",
        ats=False,
        beam=beam,
        dpp=0.0,
        model_dir=model_dir,
        xing=False,
        year="2025",
    )

    bad_bpms: list[str] = []
    for file in analysed_files:
        bpm_summary_file_x = file.parent / (file.name + ".bad_bpms_x")
        bpm_summary_file_y = file.parent / (file.name + ".bad_bpms_y")
        if bpm_summary_file_x.exists():
            with bpm_summary_file_x.open("r") as f:
                bad_bpms.extend([line.split(" ")[0] for line in f.readlines()])
        if bpm_summary_file_y.exists():
            with bpm_summary_file_y.open("r") as f:
                bad_bpms.extend([line.split(" ")[0] for line in f.readlines()])
    bad_bpms = list(set(bad_bpms))
    LOGGER.info(f"Identified {len(bad_bpms)} bad BPMs from analysis.")
    return bad_bpms


def copy_ac_dipole_attrs(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Copy AC-dipole metadata attrs from source dataframe to target."""
    for key in AC_DIPOLE_ATTR_KEYS:
        if key in source.attrs:
            target.attrs[key] = source.attrs[key]


# def write_datafile(data: pd.DataFrame, output_file: str | Path) -> None:
#     """Write the combined DataFrame to a Parquet file."""
#     data.to_parquet(output_file)


def process_single_dataframe(
    df_with_index: tuple[int, pd.DataFrame],
    tws: pd.DataFrame,
    bad_bpms: list[str],
    analysis_dir: Path,
    use_uniform_vars: bool,
    beam: int,
    ac_dipole_config_factory: Callable[[int], ACDipoleConfig | None] | None = None,
    machine_deltap: float | None = None,
) -> tuple[int, pd.DataFrame]:
    """Process a single DataFrame: clean, compute vars, and calculate pz.

    Args:
        df_with_index: Tuple of (index, dataframe)
        tws: Twiss DataFrame with optics parameters
        bad_bpms: List of bad BPM names
        use_uniform_vars: Whether to use uniform variances

    Returns:
        Tuple of (original_index, processed_dataframe)
    """
    i, df = df_with_index
    ac_dipole_config = ac_dipole_config_factory(i) if ac_dipole_config_factory is not None else None

    # SVD clean
    df = svd_clean_measurements(df)

    # Remove BPMs not in twiss data
    df: pd.DataFrame = df[df["name"].isin(tws.index)]

    # Compute variances
    if use_uniform_vars:
        df = compute_uniform_vars(df, bad_bpms)
    else:
        df.set_index("name", inplace=True)
        df = compute_vars_from_known_noise(df, bad_bpms)
        df.reset_index(inplace=True)

    # Calculate pz
    df = calculate_pz_measurement(
        df,
        analysis_dir,
        model_tws=tws,
        include_errors=True,
        include_optics_errors=True,
        reverse_meas_tws=beam == 2,
        dpp_override=machine_deltap if machine_deltap is not None else 0.0,
        ac_dipole_config=ac_dipole_config,
    )

    # Divide the variances by 10 because of the svd cleaning reducing noise
    df["var_x"] = df["var_x"] / 100
    df["var_y"] = df["var_y"] / 100

    # Handle NaN values
    if df["px"].isna().any() or df["py"].isna().any():
        LOGGER.warning(f"NaN values found in px or py for dataframe {i}, dropping rows.")
        df = df.dropna(subset=["px", "py"])

    return i, df


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
    tws_file = output_dir / "twiss.dat"
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
    filename: str | None = "pz_data.parquet",
    bad_bpms: list[str] | None = None,
    previous_analysis_dir: str | Path | None = None,
    use_uniform_vars: bool = False,
    num_workers: int | None = None,
    combine_files: bool = True,
    nattunes: list[float] | None = None,
    tunes: list[float] | None = None,
    machine_deltaps: float | list[float] | None = None,
    ac_dipole_reconstruction_config: ACDipoleReconstructionConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Path], pd.DataFrame]:
    """Process measurement files to compute pz data and identify bad BPMs.

    Args:
        files: List of measurement file paths
        output_dir: Directory for analysis outputs
        model_dir: Directory containing model files
        accelerator: LHC accelerator carrying beam, sequence_file and pc
        filename: Output filename for parquet file (None to skip saving)
        bad_bpms: List of bad BPM names (None to run analysis)
        use_uniform_vars: If True, use uniform variances instead of noise-based
        num_workers: Number of parallel workers (None for auto)
        combine_files: If True, combine all processed dataframes into one dict entry with key 'combined';
                      if False, return dict with file paths as keys
        nattunes: Natural tunes [Qx, Qy, Qz] (None to extract from model)
        tunes: Driven tunes [Qx, Qy, Qz] (None to extract from model)
        machine_deltaps: Optional machine momentum offsets used during px/py reconstruction.
                If a list, must match files length and will be expanded per bunch.

    Returns:
        Tuple of (dict mapping file paths to dataframes, bad_bpms_list, dict mapping keys to output paths, twiss_df)
    """
    beam = accelerator.beam

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

    # Build one isolated AC-dipole reconstruction model per worker thread.
    sequence_for_acd = accelerator.sequence_file
    ac_dipole_marker = accelerator.get_ac_dipole_marker()
    acd_model_lock = threading.Lock()
    acd_models: list[ACDipoleMadDriver] = []
    acd_thread_local = threading.local()

    if ac_dipole_reconstruction_config is not None:
        tune_knobs_files = ac_dipole_reconstruction_config.tune_knobs_files
        corrector_knobs_files = ac_dipole_reconstruction_config.corrector_knobs_files
        if tune_knobs_files is not None and len(tune_knobs_files) != len(files):
            raise ValueError(
                "ac_dipole_reconstruction_config.tune_knobs_files must match files length: "
                f"{len(tune_knobs_files)} != {len(files)}"
            )
        if corrector_knobs_files is not None and len(corrector_knobs_files) != len(files):
            raise ValueError(
                "ac_dipole_reconstruction_config.corrector_knobs_files must match files length: "
                f"{len(corrector_knobs_files)} != {len(files)}"
            )

    def get_thread_local_ac_dipole_config(df_idx: int) -> ACDipoleConfig | None:
        if ac_dipole_reconstruction_config is None:
            return None
        cfg_cache = getattr(acd_thread_local, "config_cache", None)
        if cfg_cache is None:
            cfg_cache = {}
            acd_thread_local.config_cache = cfg_cache

        file_idx = dataframe_file_indices[df_idx]
        tune_knobs_file = (
            ac_dipole_reconstruction_config.tune_knobs_files[file_idx]
            if ac_dipole_reconstruction_config.tune_knobs_files is not None
            else None
        )
        corrector_knobs_file = (
            ac_dipole_reconstruction_config.corrector_knobs_files[file_idx]
            if ac_dipole_reconstruction_config.corrector_knobs_files is not None
            else None
        )
        machine_deltap = per_dataframe_machine_deltaps[df_idx]
        cache_key = (
            None if tune_knobs_file is None else str(Path(tune_knobs_file)),
            None if corrector_knobs_file is None else str(Path(corrector_knobs_file)),
            machine_deltap,
        )
        cfg = cfg_cache.get(cache_key)
        if cfg is not None:
            return cfg

        acd_accelerator = LHC(
            beam=beam, sequence_file=sequence_for_acd, kinetic_energy=accelerator.kinetic_energy
        )
        model = ACDipoleMadDriver(
            accelerator=acd_accelerator,
            deltap=machine_deltap if machine_deltap is not None else 0.0,
            observed_elements=ac_dipole_marker,
            tune_knobs_file=tune_knobs_file,
            corrector_knobs_file=corrector_knobs_file,
            discard_mad_output=True,
        )
        cfg = ACDipoleConfig(
            ac_dipole_marker=ac_dipole_marker,
            model=model,
            tune_knobs_file=tune_knobs_file,
            corrector_knobs_file=corrector_knobs_file,
        )
        cfg_cache[cache_key] = cfg
        with acd_model_lock:
            acd_models.append(model)
        return cfg

    if bad_bpms is None or previous_analysis_dir is None:
        bad_bpms = run_analysis(output_dir, model_dir, files, beam, nattunes, tunes)
        LOGGER.warning(
            "Previous analysis directory not provided; ran analysis for processing measurements."
        )
        analysis_dir = output_dir
    else:
        if previous_analysis_dir is None:
            raise ValueError(
                "previous_analysis_dir must be provided if bad_bpms is given to calculate the pz from measurements."
            )
        analysis_dir = Path(previous_analysis_dir)
        if not analysis_dir.exists():
            raise FileNotFoundError(
                f"Provided previous_analysis_dir {analysis_dir} does not exist."
            )

    data = load_files(files)
    combined = convert_measurements(data, bad_bpms, combine_measurements=combine_files)
    dataframe_file_indices = _build_dataframe_file_indices(data)

    if machine_deltaps is None:
        per_file_machine_deltaps = [None] * len(files)
    elif isinstance(machine_deltaps, int | float):
        per_file_machine_deltaps = [float(machine_deltaps)] * len(files)
    else:
        machine_deltaps_list = list(machine_deltaps)
        if len(machine_deltaps_list) != len(files):
            raise ValueError(
                "machine_deltaps must match files length when provided as a list: "
                f"{len(machine_deltaps_list)} != {len(files)}"
            )
        per_file_machine_deltaps = machine_deltaps_list

    per_dataframe_machine_deltaps = [
        per_file_machine_deltaps[file_idx] for file_idx in dataframe_file_indices
    ]

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

    try:
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid Spark context conflicts
        # Limit to max 9 threads to avoid overloading the system
        effective_workers = min(num_workers or len(combined), 9)
        if effective_workers > 0:
            try:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {
                        executor.submit(
                            process_single_dataframe,
                            (i, df),
                            tws,
                            bad_bpms,
                            analysis_dir,
                            use_uniform_vars,
                            beam,
                            get_thread_local_ac_dipole_config,
                            per_dataframe_machine_deltaps[i],
                        ): i
                        for i, df in enumerate(combined)
                    }

                    for future in as_completed(futures):
                        try:
                            idx, processed_df = future.result(
                                timeout=600
                            )  # 10 minute timeout per task
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
                    tws=tws,
                    bad_bpms=bad_bpms,
                    analysis_dir=analysis_dir,
                    use_uniform_vars=use_uniform_vars,
                    beam=beam,
                    ac_dipole_config_factory=get_thread_local_ac_dipole_config,
                    machine_deltap=per_dataframe_machine_deltaps[i],
                )
                processed_results[idx] = processed_df
                LOGGER.info(f"Completed processing dataframe {idx + 1}/{len(combined)}")
    finally:
        for model in acd_models:
            if hasattr(model, "close"):
                model.close()

    if any(res is None for res in processed_results):
        raise RuntimeError("Some dataframes failed to process.")

    combined: list[pd.DataFrame] = processed_results  # ty:ignore[invalid-assignment]
    if combine_files:
        pzs_combined = pd.concat(combined, ignore_index=True)
        pzs_combined["name"] = pzs_combined["name"].astype("category")
        pzs_combined["turn"] = pzs_combined["turn"].astype("int32")
        # Add the average dpp estimate to the headers
        dpp_est = sum(proc_res.attrs["DPP_EST"] for proc_res in combined) / len(combined)
        pzs_combined.attrs["DPP_EST"] = dpp_est
        copy_ac_dipole_attrs(combined[0], pzs_combined)
        pzs_dict: dict[str, pd.DataFrame] = {"combined": pzs_combined}
    else:
        # Group by file: each file has multiple bunches combined
        num_files = len(files)
        num_bunches_per_file = len(combined) // num_files
        pzs_dict: dict[str, pd.DataFrame] = {}
        for i in range(num_files):
            start = i * num_bunches_per_file
            end = (i + 1) * num_bunches_per_file
            file_dfs = combined[start:end]
            file_pzs = pd.concat(file_dfs, ignore_index=True)
            file_pzs["name"] = file_pzs["name"].astype("category")
            file_pzs["turn"] = file_pzs["turn"].astype("int32")
            file_pzs.attrs["DPP_EST"] = sum(df.attrs["DPP_EST"] for df in file_dfs) / len(file_dfs)
            copy_ac_dipole_attrs(file_dfs[0], file_pzs)
            pzs_dict[str(files[i])] = file_pzs

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
