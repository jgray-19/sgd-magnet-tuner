from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import tfs
from tqdm import tqdm

from aba_optimiser.config import (
    BPM_START_POINTS,
    FILTERED_FILE,
    KALMAN_FILE,
    NOISE_FILE,
    TRACK_DATA_FILE,
)

# from aba_optimiser.ellipse_filtering import filter_noisy_data
# from aba_optimiser.kalman_filtering import BPMKalmanFilter
from aba_optimiser.physics.transverse_momentum import calculate_pz

# from aba_optimiser.filtering.phase_space import PhaseSpaceDiagnostics

if (
    not NOISE_FILE.exists()
    or TRACK_DATA_FILE.stat().st_mtime > NOISE_FILE.stat().st_mtime
):
    print(f"Creating noisy TFS data file: {NOISE_FILE}")
    orig_data = pd.read_parquet(TRACK_DATA_FILE)
    _, _, data_avg = calculate_pz(orig_data, low_noise_bpms=BPM_START_POINTS)
    data_avg.to_feather(NOISE_FILE, compression="lz4")
    import sys

    sys.exit(0)  # Exit early to avoid running the rest of the script
else:
    print("Noisy data already exists and is up-to-date. Creating filtered data only.")
    data_p = pd.read_parquet(NOISE_FILE)

filtered_data = filter_noisy_data(data_p)
filtered_data.to_feather(FILTERED_FILE, compression="lz4")
print("→ Saved filtered data:", FILTERED_FILE)

bpm_groups = dict(tuple(data_p.groupby("name", observed=False)))
BPMs = list(bpm_groups.keys())
phase_spaces = []


def make_ps(bpm):
    bpm_data = bpm_groups[bpm]
    return PhaseSpaceDiagnostics(
        bpm, bpm_data["x"], bpm_data["px"], bpm_data["y"], bpm_data["py"], num_points=20
    )


with ThreadPoolExecutor() as executor:
    phase_spaces = list(
        tqdm(
            executor.map(make_ps, BPMs),
            total=len(BPMs),
            desc="Creating phase space diagnostics",
        )
    )

kalman_filter = BPMKalmanFilter()
kalman_data = kalman_filter.run(data_p)
# kalman_data = filter_noisy_data(kalman_data)

kalman_data.to_feather(KALMAN_FILE, compression="lz4")
print("→ Saved Kalman-filtered data:", KALMAN_FILE)

orig_data = pd.read_parquet(TRACK_DATA_FILE)
diff_p = data_p[["x", "px", "y", "py"]].sub(orig_data[["x", "px", "y", "py"]])
print("x_diff mean (prev w/ k)", diff_p["x"].abs().mean(), "±", diff_p["x"].std())
print("y_diff mean (prev w/ k)", diff_p["y"].abs().mean(), "±", diff_p["y"].std())
print("px_diff mean (prev w/ k)", diff_p["px"].abs().mean(), "±", diff_p["px"].std())
print("py_diff mean (prev w/ k)", diff_p["py"].abs().mean(), "±", diff_p["py"].std())

filtered_diff = filtered_data[["x", "px", "y", "py"]].sub(
    orig_data[["x", "px", "y", "py"]]
)
print(
    "x_diff mean (filtered)",
    filtered_diff["x"].abs().mean(),
    "±",
    filtered_diff["x"].std(),
)
print(
    "y_diff mean (filtered)",
    filtered_diff["y"].abs().mean(),
    "±",
    filtered_diff["y"].std(),
)
print(
    "px_diff mean (filtered)",
    filtered_diff["px"].abs().mean(),
    "±",
    filtered_diff["px"].std(),
)
print(
    "py_diff mean (filtered)",
    filtered_diff["py"].abs().mean(),
    "±",
    filtered_diff["py"].std(),
)

kalman_diff = kalman_data[["x", "px", "y", "py"]].sub(orig_data[["x", "px", "y", "py"]])
print(
    "x_diff mean (kalman)", kalman_diff["x"].abs().mean(), "±", kalman_diff["x"].std()
)
print(
    "y_diff mean (kalman)", kalman_diff["y"].abs().mean(), "±", kalman_diff["y"].std()
)
print(
    "px_diff mean (kalman)",
    kalman_diff["px"].abs().mean(),
    "±",
    kalman_diff["px"].std(),
)
print(
    "py_diff mean (kalman)",
    kalman_diff["py"].abs().mean(),
    "±",
    kalman_diff["py"].std(),
)
