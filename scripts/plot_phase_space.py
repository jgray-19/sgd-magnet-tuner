import matplotlib.pyplot as plt
# import numpy as np
import tfs
# from matplotlib.patches import Ellipse
# from skimage.measure import EllipseModel, ransac

from aba_optimiser.config import (
    ACD_ON,
    BPM_RANGE,
    FILTERED_FILE,
    NOISE_FILE,
    RAMP_UP_TURNS,
    TRACK_DATA_FILE,
)
from aba_optimiser.utils import select_marker
import pandas as pd

# Read non-noisy data (TRACK_DATA_FILE)
init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
non_noisy_other = init_coords.copy()
start_bpm, _ = BPM_RANGE.split("/")
other_bpm = "BPM.14R3.B1"  # Example of another BPM

non_noisy_start = select_marker(init_coords, start_bpm)
non_noisy_other = select_marker(non_noisy_other, other_bpm)
if ACD_ON:
    non_noisy_start = non_noisy_start[non_noisy_start.index > RAMP_UP_TURNS]
    non_noisy_other = non_noisy_other[non_noisy_other.index > RAMP_UP_TURNS]

# Read noisy data (NOISE_FILE)
noise_init = pd.read_feather(NOISE_FILE).set_index("turn")
noise_other = noise_init.copy()
noisy_start = select_marker(noise_init, start_bpm)
noise_other = select_marker(noise_other, other_bpm)
if ACD_ON:
    noisy_start = noisy_start[noisy_start.index > RAMP_UP_TURNS]
    noise_other = noise_other[noise_other.index > RAMP_UP_TURNS]

filtered_start = pd.read_feather(FILTERED_FILE).set_index("turn")
filtered_other = filtered_start.copy()

filtered_start = select_marker(filtered_start, start_bpm)
filtered_other = select_marker(filtered_other, other_bpm)

if ACD_ON:
    filtered_start = filtered_start[filtered_start.index > RAMP_UP_TURNS]
    filtered_other = filtered_other[filtered_other.index > RAMP_UP_TURNS]

# Create a 2x2 subplot for phase space plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 0,0: x vs px for start BPM
axs[0, 0].scatter(noisy_start["x"], noisy_start["px"], s=1, color="blue", label="Noisy")
axs[0, 0].scatter(
    non_noisy_start["x"], non_noisy_start["px"], s=1, color="red", label="Non-noisy"
)
axs[0, 0].scatter(
    filtered_start["x"], filtered_start["px"], s=1, color="green", label="Filtered"
)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("px")
axs[0, 0].set_title(f"x, px Phase Space ({start_bpm})")
axs[0, 0].grid()

# Subplot 0,1: y vs py for start BPM
axs[0, 1].scatter(noisy_start["y"], noisy_start["py"], s=1, color="blue", label="Noisy")
axs[0, 1].scatter(
    non_noisy_start["y"], non_noisy_start["py"], s=1, color="red", label="Non-noisy"
)
axs[0, 1].scatter(
    filtered_start["y"], filtered_start["py"], s=1, color="green", label="Filtered"
)
axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("py")
axs[0, 1].set_title(f"y, py Phase Space ({start_bpm})")
axs[0, 1].grid()


# Subplot 1,0: x vs px for end BPM
axs[1, 0].scatter(noise_other["x"], noise_other["px"], s=1, color="blue", label="Noisy")
axs[1, 0].scatter(
    non_noisy_other["x"], non_noisy_other["px"], s=1, color="red", label="Non-noisy"
)
axs[1, 0].scatter(
    filtered_other["x"], filtered_other["px"], s=1, color="green", label="Filtered"
)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("px")
axs[1, 0].set_title(f"x, px Phase Space ({other_bpm})")
axs[1, 0].grid()

# Subplot 1,1: y vs py for end BPM
axs[1, 1].scatter(noise_other["y"], noise_other["py"], s=1, color="blue", label="Noisy")
axs[1, 1].scatter(
    non_noisy_other["y"], non_noisy_other["py"], s=1, color="red", label="Non-noisy"
)
axs[1, 1].scatter(
    filtered_other["y"], filtered_other["py"], s=1, color="green", label="Filtered"
)
axs[1, 1].set_xlabel("y")
axs[1, 1].set_ylabel("py")
axs[1, 1].set_title(f"y, py Phase Space ({other_bpm})")
axs[1, 1].grid()

# Create one global legend for the entire figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("merged_phase_space_comparison.png", dpi=300)
plt.show()

import numpy as np
print(np.sqrt(filtered_other['var_x']))
print(np.sqrt(filtered_other['var_px']))
print(np.sqrt(filtered_other['var_y']))
print(np.sqrt(filtered_other['var_py']))

# In the filtered data, we have var_x, var_px, var_y, var_py
# We can plot the variance of x and px, and y and py in the same figure
# Create a new figure for variance plots
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
# Variance of x and px for start BPM
axs[0].scatter(
    np.sqrt(filtered_other["var_x"]), 
    np.sqrt(filtered_other["var_px"]), s=1, color="green", label="Filtered"
)
axs[0].set_xlabel("Variance of x")
axs[0].set_ylabel("Variance of px")
axs[0].set_title(f"Variance of x and px ({start_bpm})")
axs[0].grid()
# Variance of y and py for start BPM
axs[1].scatter(
    np.sqrt(filtered_other["var_y"]), np.sqrt(filtered_other["var_py"]), s=1, color="green", label="Filtered"
)
axs[1].set_xlabel("Variance of y")
axs[1].set_ylabel("Variance of py")
axs[1].set_title(f"Variance of y and py ({start_bpm})")
axs[1].grid()
# Create one global legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))
plt.suptitle("Variance Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("variance_comparison.png", dpi=300)
plt.show()
