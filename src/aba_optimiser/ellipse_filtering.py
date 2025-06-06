import numpy as np
import pandas as pd
from tqdm import tqdm

from aba_optimiser.config import (
    STD_CUT,
    WINDOWS,
    SEQUENCE_FILE,
    BPM_RANGE,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.phase_space import PhaseSpaceDiagnostics


def filter_noisy_data(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index(["turn", "name"], inplace=True)
    mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
    tws = mad_iface.run_twiss()

    # Get Twiss data
    bpm_names = []
    for start_bpm, _ in WINDOWS:
        bpm_names.append(start_bpm)
    bpm_names = list(set(bpm_names))

    # Pre-split dataframe for efficiency
    bpm_groups = {
        bpm: data.xs(bpm, level="name")
        for bpm in bpm_names
        if bpm in data.index.get_level_values("name")
    }
    data = data.reset_index()
    data.set_index("turn", inplace=True)

    failed_turns = set()
    # Delete the turns for the failed BPMs
    for bpm in tqdm(bpm_names, desc="Filtering BPMs"):
        bpm_data = bpm_groups[bpm]
        if bpm_data.empty:
            print(f"Warning: No data available for BPM {bpm}")
            continue

        # Subset to existing rows after drop
        bpm_data = bpm_data[bpm_data.index.isin(data.index.get_level_values("turn"))]

        if bpm_data.empty:
            continue

        x, px, y, py = (
            bpm_data["x"].values,
            bpm_data["px"].values,
            bpm_data["y"].values,
            bpm_data["py"].values,
        )
        ps_diag = PhaseSpaceDiagnostics(bpm, x, px, y, py, tws=tws)
        residual_x, residual_y, std_x, std_y = ps_diag.compute_residuals()

        full_idx = bpm_data.index
        fail_x = full_idx[np.abs(residual_x) > std_x * STD_CUT]
        fail_y = full_idx[np.abs(residual_y) > std_y * STD_CUT]
        failed_turns.update(fail_x)
        failed_turns.update(fail_y)

    data.reset_index(inplace=True)
    filtered = data[
        ~data["turn"].isin(failed_turns)
    ].copy()  # Ensure filtered is a copy

    # Run through all the BPMs between the start and end, and set the weight_x and weight_y according to residuals
    start_bpm, end_bpm = BPM_RANGE.split("/")
    start_bpm_idx = tws.index.get_loc(start_bpm)
    end_bpm_idx = tws.index.get_loc(end_bpm)
    tws = tws.iloc[start_bpm_idx : end_bpm_idx + 1]

    all_bpms = tws.index.tolist()

    # Compute and set weight_x and weight_y for each BPM in range
    filtered.loc[:, "weight_x"] = np.nan
    filtered.loc[:, "weight_y"] = np.nan
    weight_x_dict = {}
    weight_y_dict = {}
    for bpm in tqdm(all_bpms, desc="Retrieving weight_x/weight_y"):
        bpm_mask = filtered["name"] == bpm
        bpm_data = filtered[bpm_mask]
        if bpm_data.empty:
            continue
        x, px, y, py = (
            bpm_data["x"].values,
            bpm_data["px"].values,
            bpm_data["y"].values,
            bpm_data["py"].values,
        )
        ps_diag = PhaseSpaceDiagnostics(bpm, x, px, y, py, tws=tws)
        residual_x, residual_y, std_x, std_y = ps_diag.compute_residuals()
        full_idx = bpm_data.index

        # Calculate weights as max(1/(10**(std/abs(residual))), 1)
        # Avoid division by zero and clamp exponent to avoid overflow
        abs_residual_x = np.abs(residual_x)
        abs_residual_y = np.abs(residual_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Clamp exponent to max 10 to avoid overflow
            exp_x = np.clip(abs_residual_x / std_x, 1, 100)
            exp_y = np.clip(abs_residual_y / std_y, 1, 100)
            weight_x = 1 / (10 ** (exp_x - 1))
            weight_y = 1 / (10 ** (exp_y - 1))
            assert np.all(weight_x >= 0) and np.all(weight_y >= 0), (
                "Weights must be non-negative"
            )
            assert np.all(weight_x <= 1) and np.all(weight_y <= 1), (
                "Weights must be less than or equal to 1"
            )

        weight_x_dict.update(dict(zip(full_idx, weight_x)))
        weight_y_dict.update(dict(zip(full_idx, weight_y)))

    filtered.loc[:, "weight_x"] = filtered.index.map(weight_x_dict)
    filtered.loc[:, "weight_y"] = filtered.index.map(weight_y_dict)

    print(f"Total failed turns: {len(failed_turns)}")
    return filtered
