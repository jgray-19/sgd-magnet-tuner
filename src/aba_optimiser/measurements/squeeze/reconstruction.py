"""Standalone AC-dipole momentum reconstruction for the squeeze (ACD-mode) scenario.

Reconstructs px/py from raw LHC turn-by-turn files and emits the ``<acd>_before`` /
``<acd>_after`` marker rows plus the AC-dipole optimisation window consumed by the
squeeze quadrupole pipeline (:mod:`aba_optimiser.measurements.squeeze.pipeline`). The
arc-by-arc closed-orbit workflow uses the per-dataframe reconstruction in
:mod:`aba_optimiser.measurements.reconstruction` instead.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pandas as pd
import tfs
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz
from tmom_recon.acd.integration import apply_precomputed_ac_dipole_bpm_overrides
from tmom_recon.physics.pt_calculation import estimate_pt_from_model
from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements

from aba_optimiser.accelerators import LHC
from aba_optimiser.measurements.loading import read_lhc_bpm_tbt, tbt_xy_to_long_dataframe
from aba_optimiser.measurements.reference import model_closed_orbit_reference
from aba_optimiser.measurements.sequence import extract_tunes_from_job_file
from aba_optimiser.noise import assign_bpm_variances

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _fill_acd_momenta(bpm_table: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Write reconstructed px/py into the two AC-dipole adjacent BPMs."""
    return apply_precomputed_ac_dipole_bpm_overrides(bpm_table, reconstructed)


def _append_acd_marker_rows(bpm_table: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Append the marker-side ACD state rows emitted by tmom-recon."""
    marker_rows = reconstructed.loc[
        reconstructed["name"].astype(str).str.endswith(("_before", "_after"))
    ].copy()
    if marker_rows.empty:
        logger.warning("ACD reconstruction carried no _before/_after marker rows to append.")
        return bpm_table
    marker_rows = marker_rows.reindex(columns=bpm_table.columns)
    combined = pd.concat([bpm_table, marker_rows], ignore_index=True, sort=False)
    logger.info(
        "Appended %d AC-dipole marker row(s) (%s) to the reconstruction.",
        len(marker_rows),
        ", ".join(sorted(marker_rows["name"].astype(str).unique())),
    )
    return combined


def reconstruct_ac_dipole_measurements(
    measurement_files: list[Path],
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    energy: float,
    use_weighted_svd: bool = True,
    tune_knobs_files: list[Path | None] | None = None,
    corrector_knobs_files: list[Path | None] | None = None,
    magnet_strengths: dict[str, float] | None = None,
    num_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Reconstruct AC-dipole momentum from raw LHC turn-by-turn measurement files.

    Returns a dict mapping each measurement file stem to a reconstructed DataFrame
    with columns (name, turn, x, y, var_x, var_y, px, py) and attrs
    DPP_EST, ac_dipole_marker, ac_dipole_bpm_upstream, ac_dipole_bpm_downstream.
    """
    model_twiss_file = model_dir / "twiss.dat"
    if not model_twiss_file.exists():
        raise FileNotFoundError(f"Model twiss not found: {model_twiss_file}")
    if not sequence_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")
    if tune_knobs_files is not None and len(tune_knobs_files) != len(measurement_files):
        raise ValueError(
            "tune_knobs_files must match measurement_files length: "
            f"{len(tune_knobs_files)} != {len(measurement_files)}"
        )
    if corrector_knobs_files is not None and len(corrector_knobs_files) != len(measurement_files):
        raise ValueError(
            "corrector_knobs_files must match measurement_files length: "
            f"{len(corrector_knobs_files)} != {len(measurement_files)}"
        )

    model_twiss = tfs.read(model_twiss_file, index="NAME")
    model_twiss.columns = [col.lower() for col in model_twiss.columns]
    job_file = model_dir / "job.create_model_nominal.madx"
    _nat_x, _nat_y, drv_x, drv_y = extract_tunes_from_job_file(job_file)
    lhc_accel = LHC(beam=beam, kinetic_energy=energy, sequence_file=sequence_path)
    svd_clean = weighted_svd_clean_measurements if use_weighted_svd else svd_clean_measurements

    def process_single_measurement(
        file_idx: int,
        measurement_file: Path,
    ) -> tuple[str, pd.DataFrame]:
        logger.info(f"Processing {measurement_file.name}")

        if measurement_file.stat().st_size == 0:
            raise ValueError(f"Empty measurement file: {measurement_file}")

        try:
            tbt_data = read_lhc_bpm_tbt(measurement_file, beam=beam)
        except Exception as e:
            raise ValueError(f"Failed to read TBT data from {measurement_file}: {e}") from e

        if not getattr(tbt_data, "matrices", None):
            raise ValueError(f"No TBT matrices found in {measurement_file}")

        x_frame = tbt_data.matrices[0].X
        y_frame = tbt_data.matrices[0].Y
        if x_frame.empty or y_frame.empty:
            raise ValueError(f"Empty X or Y frame in {measurement_file}")

        orig_data = assign_bpm_variances(tbt_xy_to_long_dataframe(x_frame, y_frame), "lhc")
        orig_data = svd_clean(orig_data)

        lattice_names = set(model_twiss.index.str.upper())
        unknown_bpms = set(orig_data["name"].str.upper().unique()) - lattice_names
        if unknown_bpms:
            logger.warning(
                "%s: dropping %d BPM(s) not in model twiss: %s",
                measurement_file.name,
                len(unknown_bpms),
                sorted(unknown_bpms),
            )
            orig_data = orig_data[~orig_data["name"].str.upper().isin(unknown_bpms)].copy()

        # acd_only=True: calculate_pz returns before reconstruct_momenta, so this
        # reference is only consumed by estimate_pt_from_model above.
        reference_co = model_closed_orbit_reference(model_twiss)
        pt_est = float(
            estimate_pt_from_model(
                orig_data.copy(deep=True),
                model_twiss,
                reference=reference_co,
            )
        )
        dpp_est = float(lhc_accel.pt2dp(pt_est))
        tune_knobs = tune_knobs_files[file_idx] if tune_knobs_files else None
        corrector_knobs = corrector_knobs_files[file_idx] if corrector_knobs_files else None

        model_details = ModelDetails(
            accelerator=lhc_accel,
            pt=pt_est,
            magnet_strengths=magnet_strengths,
            tune_knobs=tune_knobs,
            corrector_knobs=corrector_knobs,
        )
        acd_config = ACDipoleConfig(
            ac_dipole_marker=lhc_accel.ac_dipole_name,
            driven_tunes=(drv_x, drv_y),
        )
        reconstructed = calculate_pz(
            orig_data,
            model_details,
            reference=reference_co,
            acd=acd_config,
            acd_only=True,
            info=False,
        )

        if not isinstance(reconstructed, pd.DataFrame) or reconstructed.empty:
            raise ValueError(
                f"Reconstruction failed for {measurement_file}, got {type(reconstructed)} with shape {getattr(reconstructed, 'shape', None)}"
            )

        bpm_table = orig_data.copy(deep=True)
        bpm_table["px"] = 0.0
        bpm_table["py"] = 0.0
        bpm_table["var_px"] = 1.0
        bpm_table["var_py"] = 1.0
        bpm_table = _fill_acd_momenta(bpm_table, reconstructed)
        bpm_table = _append_acd_marker_rows(bpm_table, reconstructed)
        bpm_table = bpm_table.reset_index(drop=True)
        # Each reconstructed measurement file is a single bunch; the optimiser reads
        # this required column to group turns per bunch (one bunch per parquet here).
        bpm_table["bunch_number"] = 0

        upstream_name = str(reconstructed.attrs["bpm_upstream"])
        downstream_name = str(reconstructed.attrs["bpm_downstream"])
        bpm_table.attrs.update(
            {
                "DPP_EST": dpp_est,
                "PT_EST": pt_est,
                "ac_dipole_marker": lhc_accel.ac_dipole_name,
                "ac_dipole_bpm_upstream": upstream_name,
                "ac_dipole_bpm_downstream": downstream_name,
            }
        )

        logger.info(
            "Reconstructed %s: PT_EST=%.6e DPP_EST=%.6e",
            measurement_file.name,
            pt_est,
            dpp_est,
        )
        return measurement_file.stem, bpm_table

    results: dict[str, pd.DataFrame] = {}
    logger.info(f"Processing {len(measurement_files)} measurement files with {num_workers} workers")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_measurement, idx, mfile): idx
            for idx, mfile in enumerate(measurement_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                stem, result_df = future.result()
                results[stem] = result_df
            except Exception as e:
                logger.error(f"Failed to process measurement {idx}: {e}")
                raise

    logger.info(f"Successfully reconstructed {len(results)} measurements")
    return results
