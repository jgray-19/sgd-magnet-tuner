import pandas as pd
import tfs

from aba_optimiser.config import (
    KALMAN_FILE,
    NO_NOISE_FILE,
    NOISY_FILE,
)
from aba_optimiser.kalman_filtering import BPMKalmanFilter

# from aba_optimiser.make_noisy_track_data import make_noisy_track_data

# orig_data = pd.read_parquet(TRACK_DATA_FILE)
# data_p, data_n = make_noisy_track_data(orig_data)
# data_p.to_feather(NOISE_FILE, compression="lz4")
data_p = pd.read_parquet(NOISY_FILE)

bpm_groups = dict(tuple(data_p.groupby("name", observed=False)))
BPMs = list(bpm_groups.keys())
phase_spaces = []

kalman_filter = BPMKalmanFilter()
kalman_data = kalman_filter.run(data_p)
# kalman_data = filter_noisy_data(kalman_data)

kalman_data.to_feather(KALMAN_FILE, compression="lz4")
print("→ Saved Kalman-filtered data:", KALMAN_FILE)

orig_data = pd.read_parquet(NO_NOISE_FILE)
diff_p = data_p[["x", "px", "y", "py"]].sub(orig_data[["x", "px", "y", "py"]])
print("x_diff mean (prev w/ k)", diff_p["x"].abs().mean(), "±", diff_p["x"].std())
print("y_diff mean (prev w/ k)", diff_p["y"].abs().mean(), "±", diff_p["y"].std())
print("px_diff mean (prev w/ k)", diff_p["px"].abs().mean(), "±", diff_p["px"].std())
print("py_diff mean (prev w/ k)", diff_p["py"].abs().mean(), "±", diff_p["py"].std())

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


# Plot the std per BPM (group it) as a line graph
import matplotlib.pyplot as plt
