"""Helper functions and constants for squeeze step measurements and analysis.

This module centralizes common code used across multiple measurement and analysis scripts,
reducing duplication and ensuring consistency across the codebase.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from pymadng_utils.madx import make_madx_sequence
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from tmom_recon.acd.reconstruction import calculate_ac_dipole_momentum
from tmom_recon.physics.dpp_calculation import estimate_dpp_from_model
from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements
from turn_by_turn import read_tbt

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import AbaMadInterface
from aba_optimiser.measurements.squeeze_config import (
    ANALYSIS_DIRS,
    BETABEAT_DIR,
    MODEL_DIRS,
    get_measurement_date,
)
from aba_optimiser.noise import assign_bpm_variances

logger = logging.getLogger(__name__)


# ==================== HELPER FUNCTIONS ====================
_MACHINE_SETTINGS_KNOBS_FILENAME = "machine_settings_knobs.madx"


def make_machine_settings_knobs_file(output_file: Path, time: str) -> Path:
    """Extract a single MAD-X knobs file for the requested machine-settings time."""
    from omc3.knob_extractor import main as extract_knobs

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    extract_knobs({"time": time, "output": output_file})
    if not output_file.exists():
        raise FileNotFoundError(
            f"Expected machine-settings knobs file was not created: {output_file}"
        )
    logger.info("Saved machine-settings knobs for %s to %s", time, output_file)
    return output_file


def get_or_make_sequence(beam: int, madng_model_dir: Path, time: str | None = None) -> Path:
    """Get cached sequence or generate a new one.

    Args:
        beam: Beam number (1 or 2)
        madng_model_dir: Path to MAD-NG model directory
        time: Optional machine-settings extraction time. When provided, a
            machine-settings knobs file is generated and applied after the
            optics modifiers during sequence creation.

    Returns:
        Path to sequence file
    """
    if time is None:
        knobs_file = None
    else:
        knobs_file = madng_model_dir / f"machine_settings_{time}.madx"
        if not knobs_file.exists():
            make_machine_settings_knobs_file(knobs_file, time)
        knobs_file = [Path(knobs_file)]
    expected_sequence_file = madng_model_dir / f"lhcb{beam}_saved.seq"
    if expected_sequence_file.exists():
        logger.info(f"Found existing sequence file for beam {beam} at {expected_sequence_file}")
        return expected_sequence_file
    return make_madx_sequence(madng_model_dir, post_optics_madx_files=knobs_file)

def load_estimates_and_uncertainties(
    estimates_file: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load quadrupole estimates and uncertainties from file.

    Handles both formats:
    - JSON format: ``{"Arc 1": {"knob": {"value": ..., "uncertainty": ...}}}``
    - Legacy text format: ``Arc X:`` headers followed by ``<magnet> <value>`` rows

    Args:
        estimates_file: Path to file containing magnet estimates

    Returns:
        Tuple of:
        - Dictionary mapping arcs to knob values
        - Dictionary mapping arcs to knob uncertainties
    """
    if estimates_file.suffix.lower() == ".json":
        with estimates_file.open() as f:
            payload = json.load(f)
        estimates = {
            str(arc): {
                str(magnet): float(knob_payload["value"])
                for magnet, knob_payload in arc_payload.items()
            }
            for arc, arc_payload in payload.items()
        }
        uncertainties = {
            str(arc): {
                str(magnet): float(knob_payload.get("uncertainty", 0.0))
                for magnet, knob_payload in arc_payload.items()
            }
            for arc, arc_payload in payload.items()
        }
        logger.info(
            f"Loaded {sum(len(v) for v in estimates.values())} magnet estimates from {estimates_file.name}"
        )
        return estimates, uncertainties

    estimates = {}
    uncertainties = {}
    current_arc = None

    with estimates_file.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("Arc"):
                current_arc = line.rstrip(":")  # remove trailing :
                if current_arc not in estimates:
                    estimates[current_arc] = {}
                    uncertainties[current_arc] = {}
            elif line and current_arc:
                parts = line.split()
                if len(parts) == 2:
                    magnet, value = parts
                    estimates[current_arc][magnet] = float(value)
                    uncertainties[current_arc][magnet] = 0.0

    if estimates:
        logger.info(
            f"Loaded {sum(len(v) for v in estimates.values())} magnet estimates from {estimates_file.name}"
        )
    return estimates, uncertainties


def load_estimates(estimates_file: Path) -> dict[str, dict[str, float]]:
    """Load quadrupole estimates from file."""
    estimates, _ = load_estimates_and_uncertainties(estimates_file)
    return estimates


def get_model_dir(beam: int, squeeze_step: str) -> Path:
    """Get model directory for a given beam and squeeze step.

    Args:
        beam: Beam number (1 or 2)
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")

    Returns:
        Path to model directory

    Raises:
        ValueError: If squeeze_step is not found for the beam
    """
    if squeeze_step not in MODEL_DIRS.get(beam, {}):
        raise ValueError(
            f"Model directory not defined for beam {beam}, squeeze_step {squeeze_step}"
        )

    meas_date = get_measurement_date(squeeze_step)
    model_dir = BETABEAT_DIR / meas_date / f"LHCB{beam}/Models/" / MODEL_DIRS[beam][squeeze_step]
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    logger.info(f"Using model directory: {model_dir}")
    return model_dir


def get_analysis_dir(beam: int, squeeze_step: str) -> Path:
    """Get analysis directory for a given beam and squeeze step.

    Args:
        beam: Beam number (1 or 2)
        squeeze_step: Squeeze step (e.g., "1.2m", "0.6m")

    Returns:
        Path to analysis directory

    Raises:
        ValueError: If squeeze_step is not found for the beam
    """
    if squeeze_step not in ANALYSIS_DIRS.get(beam, {}):
        raise ValueError(
            f"Analysis directory not defined for beam {beam}, squeeze_step {squeeze_step}"
        )

    meas_date = get_measurement_date(squeeze_step)
    analysis_dir = (
        BETABEAT_DIR / meas_date / f"LHCB{beam}/Results/" / ANALYSIS_DIRS[beam][squeeze_step]
    )
    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory not found: {analysis_dir}")

    logger.info(f"Using analysis directory: {analysis_dir}")
    return analysis_dir


def get_results_dir(beam: int) -> Path:
    """Get results directory for a given beam.

    Args:
        beam: Beam number (1 or 2)

    Returns:
        Path to results directory
    """
    results_dir = PROJECT_ROOT / f"b{beam}_squeeze_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_ir_bpm_ranges_from_model(
    model_dir: str | Path, beam: int, ip: int
) -> tuple[str, list[str], list[str]]:
    """Extract BPM ranges from twiss.dat file for IR optimisation.

    Args:
        model_dir: Path to the model directory containing twiss_elements.dat
        beam: Beam number (1 or 2)

    Returns:
        Tuple of (magnet_ranges, bpm_starts, bpm_end_points)
    """
    import re

    twiss_file = Path(model_dir) / "twiss_elements.dat"
    twiss_df = tfs.read(twiss_file, index="NAME")

    # Filter BPMs for this beam
    bpm_mask = twiss_df.index.str.startswith("BPM") & twiss_df.index.str.endswith(f".B{beam}")
    bpm_names = twiss_df.index[bpm_mask].tolist()

    # Regex to match BPM names: BPM.*.(IP)(L|R).*.B(beam)
    bpm_pattern = re.compile(r"BPM[A-Z]*\.(\d)([LR])(\d)\.B(\d+)")

    # Collect all matching BPMs with ip and side
    matches = [
        (bpm, int(match.group(3)), match.group(2), int(match.group(1)))
        for bpm in bpm_names
        if (match := bpm_pattern.match(bpm))
    ]

    before_side = "L" if beam == 1 else "R"
    after_side = "R" if beam == 1 else "L"
    # Include BPMs from position 4 onwards to get more measurement points
    min_from_ip = 4  # Adjust this threshold as needed to include more BPMs
    before_bpms = [
        bpm
        for bpm, ip_num, side, from_ip in matches
        if ip_num == ip and side == before_side and from_ip >= min_from_ip
    ]
    after_bpms = [
        bpm
        for bpm, ip_num, side, from_ip in matches
        if ip_num == ip and side == after_side and from_ip >= min_from_ip
    ]

    # Remove all bpms with W in their names
    # before_bpms = [bpm for bpm in before_bpms if "W" not in bpm]
    # after_bpms = [bpm for bpm in after_bpms if "W" not in bpm]

    magnet_range = f"BPM.9L{ip}.B1/BPM.9R{ip}.B1" if beam == 1 else f"BPM.9R{ip}.B2/BPM.9L{ip}.B2"
    return magnet_range, before_bpms, after_bpms


def extract_tunes_from_job_file(job_file_path: Path) -> tuple[float, float, float, float]:
    """Extract natural and driven tunes from the MAD-X job file.

    Args:
        job_file_path: Path to the job.create_model_nominal.madx file

    Returns:
        Tuple of (nat_x, nat_y, drv_x, drv_y)
    """
    import re

    with job_file_path.open("r") as f:
        content = f.read()

    # Regex to match twiss_ac_dipole(nat_x, nat_y, drv_x, drv_y, ...)
    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.search(
        rf"twiss_ac_dipole\(\s*({number_pattern})\s*,\s*({number_pattern})\s*,\s*({number_pattern})\s*,\s*({number_pattern})\s*,",
        content,
    )
    if not match:
        raise ValueError(f"Could not find twiss_ac_dipole call in {job_file_path}")

    nat_x = float(match.group(1))
    nat_y = float(match.group(2))
    drv_x = float(match.group(3))
    drv_y = float(match.group(4))

    logger.info(
        f"Extracted tunes from {job_file_path}: nat_x={nat_x}, nat_y={nat_y}, drv_x={drv_x}, drv_y={drv_y}"
    )
    return nat_x, nat_y, drv_x, drv_y


class _LHCACDipoleMadDriver(ACDipoleMadDriver, AbaMadInterface):
    """AC-dipole MAD driver with generic magnet setters for LHC squeeze measurements."""

    def __init__(self, **kwargs):
        ACDipoleMadDriver.__init__(self, **kwargs)


def _tbt_frames_to_dataframe(x_frame: pd.DataFrame, y_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert TBT X/Y frames to a long-form (name, turn, x, y) DataFrame."""
    if list(x_frame.index) != list(y_frame.index):
        raise ValueError("X and Y frames have different BPM ordering")
    bpm_names = [str(n) for n in x_frame.index]
    n_turns = min(x_frame.shape[1], y_frame.shape[1])
    n_bpms = len(bpm_names)
    x_arr = x_frame.to_numpy(dtype=float)[:, :n_turns] / 1000.0  # mm -> m
    y_arr = y_frame.to_numpy(dtype=float)[:, :n_turns] / 1000.0
    return pd.DataFrame(
        {
            "name": np.repeat(bpm_names, n_turns),
            "turn": np.tile(np.arange(n_turns), n_bpms),
            "x": x_arr.ravel(),
            "y": y_arr.ravel(),
        }
    )


def _fill_acd_momenta(bpm_table: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Write reconstructed px/py into the two AC-dipole adjacent BPMs."""
    turn_lookup = reconstructed.set_index("turn")
    for bpm_name, px_col, py_col in (
        (
            str(reconstructed.attrs["bpm_upstream"]),
            "px_bpm_upstream_cleaned",
            "py_bpm_upstream_cleaned",
        ),
        (
            str(reconstructed.attrs["bpm_downstream"]),
            "px_bpm_downstream_cleaned",
            "py_bpm_downstream_cleaned",
        ),
    ):
        mask = bpm_table["name"] == bpm_name
        turns = bpm_table.loc[mask, "turn"]
        bpm_table.loc[mask, "px"] = turns.map(turn_lookup[px_col]).to_numpy(dtype=float)
        bpm_table.loc[mask, "py"] = turns.map(turn_lookup[py_col]).to_numpy(dtype=float)
    return bpm_table


def reconstruct_ac_dipole_measurements(
    measurement_files: list[Path],
    model_dir: Path,
    sequence_path: Path,
    beam: int,
    energy: float,
    use_weighted_svd: bool = True,
    tune_knobs_files: list[Path | None] | None = None,
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

    model_twiss = tfs.read(model_twiss_file, index="name")
    lhc_accel = LHC(beam=beam, kinetic_energy=energy, sequence_file=sequence_path)
    ac_dipole_marker = lhc_accel.get_ac_dipole_marker()
    svd_clean = weighted_svd_clean_measurements if use_weighted_svd else svd_clean_measurements

    def process_single_measurement(
        file_idx: int,
        measurement_file: Path,
    ) -> tuple[str, pd.DataFrame]:
        logger.info(f"Processing {measurement_file.name}")

        if measurement_file.stat().st_size == 0:
            raise ValueError(f"Empty measurement file: {measurement_file}")

        try:
            tbt_data = read_tbt(measurement_file, datatype="lhc")
        except Exception as e:
            raise ValueError(f"Failed to read TBT data from {measurement_file}: {e}") from e

        if not getattr(tbt_data, "matrices", None):
            raise ValueError(f"No TBT matrices found in {measurement_file}")

        x_frame = tbt_data.matrices[0].X
        y_frame = tbt_data.matrices[0].Y
        if x_frame.empty or y_frame.empty:
            raise ValueError(f"Empty X or Y frame in {measurement_file}")

        orig_data = assign_bpm_variances(_tbt_frames_to_dataframe(x_frame, y_frame), "lhc")
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

        dpp_est = float(estimate_dpp_from_model(orig_data.copy(deep=True), model_twiss))
        tune_knobs_file = tune_knobs_files[file_idx] if tune_knobs_files else None

        model = None
        try:
            model = _LHCACDipoleMadDriver(
                accelerator=lhc_accel,
                deltap=dpp_est,
                observed_elements=ac_dipole_marker,
                discard_mad_output=True,
                tune_knobs_file=tune_knobs_file,
            )
            if magnet_strengths:
                model.set_magnet_strengths(magnet_strengths)

            reconstructed = calculate_ac_dipole_momentum(
                orig_data,
                model_twiss,
                ac_dipole_marker=ac_dipole_marker,
                model=model,
                inject_noise=False,
                use_immediate_neighbors_for_bpms=True,
            )
        finally:
            if model is not None and hasattr(model, "close"):
                model.close()

        bpm_table = orig_data.copy(deep=True)
        bpm_table["px"] = 0.0
        bpm_table["py"] = 0.0
        bpm_table["var_px"] = 1.0
        bpm_table["var_py"] = 1.0
        bpm_table = _fill_acd_momenta(bpm_table, reconstructed)
        bpm_table = bpm_table.reset_index(drop=True)

        upstream_name = str(reconstructed.attrs["bpm_upstream"])
        downstream_name = str(reconstructed.attrs["bpm_downstream"])
        bpm_table.attrs.update(
            {
                "DPP_EST": dpp_est,
                "ac_dipole_marker": ac_dipole_marker,
                "ac_dipole_bpm_upstream": upstream_name,
                "ac_dipole_bpm_downstream": downstream_name,
            }
        )

        logger.info(f"Reconstructed {measurement_file.name}: DPP_EST={dpp_est:.6f}")
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
