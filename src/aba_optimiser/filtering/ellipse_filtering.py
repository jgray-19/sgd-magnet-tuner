import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from aba_optimiser.config import BPM_START_POINTS, MAGNET_RANGE, SEQUENCE_FILE
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.physics.phase_space import PhaseSpaceDiagnostics

LOGGER = logging.getLogger(__name__)
regexpr = r"BPM\.([0-9]+)[RL][34]\.B1"


def is_x_bpm(name: str) -> bool:
    # Odd R3 and odd L4 BPMs are X BPMs
    bpm_num = re.match(regexpr, name)
    if bpm_num:
        bpm_num = int(bpm_num.group(1))
        return bpm_num % 2 == 1
    raise ValueError(f"Invalid BPM name: {name}")


def filter_noisy_data(data: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info(f"Starting ellipse filtering on {len(data)} data points")
    data = data.copy()
    data.set_index(["turn", "name"], inplace=True)
    mad_iface = OptimisationMadInterface(
        SEQUENCE_FILE,
        use_real_strengths=False,
        bpm_pattern="BPM",
    )
    tws = mad_iface.run_twiss()
    data = data.reset_index()
    data.set_index("turn", inplace=True)

    failed_turns = set()
    # Delete the turns for the failed BPMs
    # for bpm in tqdm(bpm_names, desc="Filtering BPMs"):
    #     bpm_data = bpm_groups[bpm]
    #     if bpm_data.empty:
    #         print(f"Warning: No data available for BPM {bpm}")
    #         continue

    #     # Subset to existing rows after drop
    #     bpm_data = bpm_data[bpm_data.index.isin(data.index.get_level_values("turn"))]

    #     if bpm_data.empty:
    #         continue

    #     x, px, y, py = (
    #         bpm_data["x"].values,
    #         bpm_data["px"].values,
    #         bpm_data["y"].values,
    #         bpm_data["py"].values,
    #     )
    #     ps_diag = PhaseSpaceDiagnostics(bpm, x, px, y, py, tws=tws)
    #     residual_x, residual_y, std_x, std_y = ps_diag.compute_residuals()

    #     full_idx = bpm_data.index
    #     fail_y = full_idx[np.abs(residual_y) > std_y]
    #     fail_x = full_idx[np.abs(residual_x) > std_x]
    #     failed_turns.update(fail_x)
    #     failed_turns.update(fail_y)

    data.reset_index(inplace=True)
    filtered = data[
        ~data["turn"].isin(failed_turns)
    ].copy()  # Ensure filtered is a copy

    all_bpms = filtered["name"].unique()
    # BPMs in the range MAGNET_RANGE + start BPMs
    start_bpm, end_bpm = MAGNET_RANGE.split("/")
    start_bpm_idx = np.where(all_bpms == start_bpm)[0][0]
    end_bpm_idx = np.where(all_bpms == end_bpm)[0][0] + 1
    if start_bpm_idx < end_bpm_idx:
        reduced_bpms = all_bpms[start_bpm_idx:end_bpm_idx].tolist()
    else:
        reduced_bpms = (
            all_bpms[start_bpm_idx:].tolist() + all_bpms[:end_bpm_idx].tolist()
        )
    reduced_bpms = set(reduced_bpms) | set(BPM_START_POINTS)

    # Compute and set x_weight and y_weight for each BPM in range
    filtered.loc[:, "x_weight"] = np.nan
    filtered.loc[:, "y_weight"] = np.nan
    x_weight_dict = {}
    y_weight_dict = {}
    for bpm in tqdm(reduced_bpms, desc="Retrieving x_weight/y_weight"):
        bpm_data = filtered[filtered["name"] == bpm]
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

        # Calculate weights as max(1/(10**(std/abs(residual))), 1)
        # Avoid division by zero and clamp exponent to avoid overflow
        abs_residual_x = np.abs(residual_x)
        abs_residual_y = np.abs(residual_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Clamp exponent to max 10 to avoid overflow
            exp_x = np.clip(abs_residual_x / std_x, 1, 100)
            exp_y = np.clip(abs_residual_y / std_y, 1, 100)
            x_weights = 1 / (10 ** (exp_x - 1))
            y_weights = 1 / (10 ** (exp_y - 1))
            assert np.all(x_weights >= 0) and np.all(y_weights >= 0), (
                "Weights must be non-negative"
            )
            assert np.all(x_weights <= 1) and np.all(y_weights <= 1), (
                "Weights must be less than or equal to 1"
            )

        x_weight_dict.update(dict(zip(bpm_data.index, x_weights)))
        y_weight_dict.update(dict(zip(bpm_data.index, y_weights)))

    filtered.loc[:, "x_weight"] = filtered.index.map(x_weight_dict)
    filtered.loc[:, "y_weight"] = filtered.index.map(y_weight_dict)

    print(f"Total failed turns: {len(failed_turns)}")
    return filtered
