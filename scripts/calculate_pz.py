import numpy as np
import tfs

from aba_optimiser.config import (
    BPM_RANGE,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
    TRUE_STRENGTHS,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import select_marker, read_knobs

mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
knob_names = mad_iface.knob_names
true_strengths = read_knobs(TRUE_STRENGTHS)
for key, value in true_strengths.items():
    mad_iface.mad.send(f"MADX.lhcb1['{key}'] = {value}")

# Run a twiss and to get the beta functions
tws = mad_iface.run_twiss()
print("Found tunes:", tws.q1, tws.q2)

sqrt_beta_x = np.sqrt(tws["beta11"].to_numpy())
sqrt_beta_y = np.sqrt(tws["beta22"].to_numpy())
mu_x = tws["mu1"].to_numpy()
mu_y = tws["mu2"].to_numpy()

init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
# Remove all rows that are not the BPM s.ds.r3.b1
start_bpm, end_bpm = BPM_RANGE.split("/")
start_coords = select_marker(init_coords, start_bpm)

nbpm_to_next_x = 2
nbpm_to_next_y = 2
next_bpm_x = tws.index[tws.index.get_loc(start_bpm) + nbpm_to_next_x]
next_bpm_y = tws.index[tws.index.get_loc(start_bpm) + nbpm_to_next_y]
next_coords_x = select_marker(init_coords, next_bpm_x)
next_coords_y = select_marker(init_coords, next_bpm_y)

start_bpm_idx = tws.index.get_loc(start_bpm)
next_bpm_idx_x = start_bpm_idx + nbpm_to_next_x
next_bpm_idx_y = start_bpm_idx + nbpm_to_next_y
turn = 2000

start_x = start_coords["x"].iloc[turn] / sqrt_beta_x[start_bpm_idx]
start_y = start_coords["y"].iloc[turn] / sqrt_beta_y[start_bpm_idx]
print("Start coordinates:", start_x, start_y)

next_x = next_coords_x["x"].iloc[turn] / sqrt_beta_x[next_bpm_idx_x]
next_y = next_coords_y["y"].iloc[turn] / sqrt_beta_y[next_bpm_idx_y]
print("Next coordinates:", next_x, next_y)

alpha_next_x = tws["alfa11"].iloc[next_bpm_idx_x]
alpha_next_y = tws["alfa22"].iloc[next_bpm_idx_y]

mu_next_x  = mu_x[next_bpm_idx_x]
mu_next_y  = mu_y[next_bpm_idx_y]
mu_start_x = mu_x[start_bpm_idx]
mu_start_y = mu_y[start_bpm_idx]
delta_x = (mu_next_x - mu_start_x - 0.25) * 2 * np.pi
delta_y = (mu_next_y - mu_start_y - 0.25) * 2 * np.pi

px_next = -1 * (
    start_x * (np.cos(delta_x) + np.sin(delta_x) * np.tan(delta_x))
    + next_x * (np.tan(delta_x) + alpha_next_x)
) / sqrt_beta_x[next_bpm_idx_x] 


py_next = -1 * (
    start_y * (np.cos(delta_y) + np.sin(delta_y) * np.tan(delta_y))
    + next_y * (np.tan(delta_y) + alpha_next_y)
) / sqrt_beta_y[next_bpm_idx_y]


print("Difference in px calculation:", px_next - next_coords_x["px"][turn])
print("Difference in py calculation:", py_next - next_coords_y["py"][turn])
print("Relative difference in px calculation:", (px_next - next_coords_x["px"][turn]) / next_coords_x["px"][turn] * 100, " %")
print("Relative difference in py calculation:", (py_next - next_coords_y["py"][turn]) / next_coords_y["py"][turn] * 100, " %")

