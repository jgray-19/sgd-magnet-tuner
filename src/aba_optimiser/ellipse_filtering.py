import numpy as np
import pandas as pd
from tqdm import tqdm

from aba_optimiser.config import (
    STD_CUT,
    WINDOWS,
)
from aba_optimiser.phase_space import PhaseSpaceDiagnostics

def filter_noisy_data(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index(['turn', 'name'], inplace=True)
    
    # Get Twiss data
    bpm_names = []
    for start_bpm, _ in WINDOWS:
        bpm_names.append(start_bpm)
    bpm_names = list(set(bpm_names))

    # Pre-split dataframe for efficiency
    bpm_groups = {bpm: data.xs(bpm, level='name') for bpm in bpm_names if bpm in data.index.get_level_values('name')}

    failed_turns = set()
    # Process remaining BPMs
    for bpm in tqdm(bpm_names, desc="Filtering BPMs"):
        bpm_data = bpm_groups[bpm]
        if bpm_data.empty:
            print(f"Warning: No data available for BPM {bpm}")
            continue

        # Subset to existing rows after drop
        bpm_data = bpm_data[bpm_data.index.isin(data.index.get_level_values("turn"))]

        if bpm_data.empty:
            continue

        x, px, y, py = bpm_data["x"].values, bpm_data["px"].values, bpm_data["y"].values, bpm_data["py"].values
        ps_diag = PhaseSpaceDiagnostics(bpm, x, px, y, py)
        std_x, std_y = ps_diag.compute_residuals()

        gamma_x = (1 + ps_diag.alfax**2) / ps_diag.betax
        inv_x = gamma_x * x**2 + 2 * ps_diag.alfax * x * px + ps_diag.betax * px**2
        gamma_y = (1 + ps_diag.alfay**2) / ps_diag.betay
        inv_y = gamma_y * y**2 + 2 * ps_diag.alfay * y * py + ps_diag.betay * py**2

        residual_x = (inv_x/2 - ps_diag.emit_x) / ps_diag.emit_x
        residual_y = (inv_y/2 - ps_diag.emit_y) / ps_diag.emit_y

        full_idx = bpm_data.index
        fail_x = full_idx[np.abs(residual_x) > std_x * STD_CUT]
        fail_y = full_idx[np.abs(residual_y) > std_y * STD_CUT]
        failed_turns.update(fail_x)
        failed_turns.update(fail_y)

    data.reset_index(inplace=True)
    filtered = data[~data['turn'].isin(failed_turns)]
    return filtered
