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
    MIN_FRACTION_MAX,
    XY_MIN,
    PXPY_MIN,
)
from aba_optimiser.utils import select_marker
from aba_optimiser.phase_space import PhaseSpaceDiagnostics
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

max_x_start = noisy_start["x"].abs().max()
x_start_lim = XY_MIN
ps_diag = PhaseSpaceDiagnostics(
    bpm=start_bpm,
    x_data=noisy_start["x"], px_data=noisy_start["px"],
    y_data=noisy_start["y"], py_data=noisy_start["py"]
)

# Get analytical ellipse
x_ellipse, px_ellipse, y_ellipse, py_ellipse = ps_diag.ellipse_points()

# Get +1 sigma ellipse
x_upper, px_upper, y_upper, py_upper = ps_diag.ellipse_sigma(sigma_level=1.0)
x_lower, px_lower, y_lower, py_lower = ps_diag.ellipse_sigma(sigma_level=-1.0)

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
# PLot the max x line +/-
axs[0, 0].axvline(
    x_start_lim, color="black", linestyle="--", label="Max x"
)
axs[0, 0].axvline(
    -x_start_lim, color="black", linestyle="--"
)
axs[0, 0].axhline(
    PXPY_MIN, color="black", linestyle="--", label="Min px"
)
axs[0, 0].axhline(
    -PXPY_MIN, color="black", linestyle="--"
)
axs[0, 0].plot(x_ellipse, px_ellipse, color="orange", label="Analytical Ellipse")
axs[0, 0].plot(x_upper, px_upper, color="orange", linestyle="--", label="+1 Sigma Ellipse")
axs[0, 0].plot(x_lower, px_lower, color="orange", linestyle="--", label="-1 Sigma Ellipse")

axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("px")
axs[0, 0].set_title(f"x, px Phase Space ({start_bpm})")
axs[0, 0].grid()

# Subplot 0,1: y vs py for start BPM
axs[0, 1].scatter(noisy_start["y"], noisy_start["py"], s=1, color="blue", label="Noisy", alpha=0.5)
axs[0, 1].scatter(
    non_noisy_start["y"], non_noisy_start["py"], s=1, color="red", label="Non-noisy", alpha=0.5
)
axs[0, 1].scatter(
    filtered_start["y"], filtered_start["py"], s=1, color="green", label="Filtered", alpha=0.5
)
# Plot the y points where the x points are above the max x 
max_x_mask = noisy_start["x"].abs() > x_start_lim
axs[0, 1].scatter(
    noisy_start["y"][max_x_mask],
    noisy_start["py"][max_x_mask],
    s=1, color="cyan", label="Noisy (x > max x)"
)
axs[0, 1].scatter(
    non_noisy_start["y"][max_x_mask],
    non_noisy_start["py"][max_x_mask],
    s=1, color="orange", label="Non-noisy (x > max x)"
)
axs[0, 1].scatter(
    filtered_start["y"][max_x_mask],
    filtered_start["py"][max_x_mask],
    s=1, color="purple", label="Filtered (x > max x)"
)
axs[0, 1].axvline(
    XY_MIN/2, color="black", linestyle="--", label="Min y"
)
axs[0, 1].axvline(
    -XY_MIN/2, color="black", linestyle="--"
)
axs[0, 1].axhline(
    PXPY_MIN, color="black", linestyle="--", label="Min py"
)
axs[0, 1].axhline(
    -PXPY_MIN, color="black", linestyle="--"
)
axs[0, 1].plot(y_ellipse, py_ellipse, color="orange", label="Analytical Ellipse")
axs[0, 1].plot(y_upper, py_upper, color="orange", linestyle="--", label="+1 Sigma Ellipse")
axs[0, 1].plot(y_lower, py_lower, color="orange", linestyle="--", label="-1 Sigma Ellipse")

axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("py")
axs[0, 1].set_title(f"y, py Phase Space ({start_bpm})")
axs[0, 1].grid()


max_y_other = noise_other["y"].abs().max()
y_other_lim = max_y_other * MIN_FRACTION_MAX
# Calculate analytical ellipse for the other BPM
ps_diag_other = PhaseSpaceDiagnostics(
    bpm=other_bpm,
    x_data=noise_other["x"], px_data=noise_other["px"],
    y_data=noise_other["y"], py_data=noise_other["py"]
)
# Get analytical ellipse for the other BPM
ana_x_other, ana_px_other, ana_y_other, ana_py_other = ps_diag_other.ellipse_points()
# Get +1 sigma ellipse for the other BPM
ana_x_upper_other, ana_px_upper_other, ana_y_upper_other, ana_py_upper_other = ps_diag_other.ellipse_sigma(sigma_level=1.0)
ana_x_lower_other, ana_px_lower_other, ana_y_lower_other, ana_py_lower_other = ps_diag_other.ellipse_sigma(sigma_level=-1.0)

# Subplot 1,0: x vs px for end BPM
axs[1, 0].scatter(noise_other["x"], noise_other["px"], s=1, color="blue", label="Noisy", alpha=0.5)
axs[1, 0].scatter(
    non_noisy_other["x"], non_noisy_other["px"], s=1, color="red", label="Non-noisy", alpha=0.5
)
axs[1, 0].scatter(
    filtered_other["x"], filtered_other["px"], s=1, color="green", label="Filtered", alpha=0.5
)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("px")
axs[1, 0].set_title(f"x, px Phase Space ({other_bpm})")
axs[1, 0].grid()
# Plot the x points where the y points are above the max y
y_other_mask = noise_other["y"].abs() > y_other_lim
axs[1, 0].scatter(
    noise_other["x"][y_other_mask],
    noise_other["px"][y_other_mask],
    s=1, color="cyan", label="Noisy (y > max y)"
)
axs[1, 0].scatter(
    non_noisy_other["x"][y_other_mask],
    non_noisy_other["px"][y_other_mask],
    s=1, color="orange", label="Non-noisy (y > max y)"
)
axs[1, 0].scatter(
    filtered_other["x"][y_other_mask],
    filtered_other["px"][y_other_mask],
    s=1, color="purple", label="Filtered (y > max y)"
)
axs[1, 0].axvline(
    XY_MIN/2, color="black", linestyle="--", label="Min x"
)
axs[1, 0].axvline(
    -XY_MIN/2, color="black", linestyle="--"
)
axs[1, 0].axhline(
    PXPY_MIN, color="black", linestyle="--", label="Min px"
)
axs[1, 0].axhline(
    -PXPY_MIN, color="black", linestyle="--"
)
axs[1, 0].plot(ana_x_other, ana_px_other, color="orange", label="Analytical Ellipse")
axs[1, 0].plot(ana_x_upper_other, ana_px_upper_other, color="orange", linestyle="--", label="+1 Sigma Ellipse")
axs[1, 0].plot(ana_x_lower_other, ana_px_lower_other, color="orange", linestyle="--", label="-1 Sigma Ellipse")


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
axs[1, 1].axvline(
    y_other_lim, color="black", linestyle="--", label="Max y"
)
axs[1, 1].axvline(
    -y_other_lim, color="black", linestyle="--"
)
axs[1, 1].axhline(
    PXPY_MIN, color="black", linestyle="--", label="Min py"
)
axs[1, 1].axhline(
    -PXPY_MIN, color="black", linestyle="--"
)
axs[1, 1].plot(ana_y_other, ana_py_other, color="orange", label="Analytical Ellipse")
axs[1, 1].plot(ana_y_upper_other, ana_py_upper_other, color="orange", linestyle="--", label="+1 Sigma Ellipse")
axs[1, 1].plot(ana_y_lower_other, ana_py_lower_other, color="orange", linestyle="--", label="-1 Sigma Ellipse")

# Create one global legend for the entire figure
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("merged_phase_space_comparison.png", dpi=300)
plt.show()
from aba_optimiser.config import XY_MIN, PXPY_MIN
x_mask_start = ((noisy_start["px"].abs() > PXPY_MIN) & (abs(noisy_start["x"]) > XY_MIN))
y_mask_start = ((noisy_start["py"].abs() > PXPY_MIN) & (abs(noisy_start["y"]) > XY_MIN/2))
mask_start = x_mask_start & y_mask_start

x_mask_other = ((noise_other["px"].abs() > PXPY_MIN) & (abs(noise_other["x"]) > XY_MIN/2))
y_mask_other = ((noise_other["py"].abs() > PXPY_MIN) & (abs(noise_other["y"]) > XY_MIN))
mask_other = x_mask_other & y_mask_other

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(
    abs(noisy_start["x"] - non_noisy_start["x"])[mask_start],
    abs(noisy_start["px"] - non_noisy_start["px"])[mask_start],
    s=1, color="blue", label="Noisy"
)
axs[0, 0].scatter(
    abs(filtered_start["x"] - non_noisy_start["x"])[mask_start],
    abs(filtered_start["px"] - non_noisy_start["px"])[mask_start],
    s=1, color="green", label="Filtered"
)
axs[0, 0].set_xlabel("Absolute difference of x")
axs[0, 0].set_ylabel("Absolute difference of px")
axs[0, 0].set_title(f"Difference of x and px ({start_bpm})")
axs[0, 0].grid()
axs[0, 1].scatter(
    abs(noisy_start["y"] - non_noisy_start["y"])[mask_start],
    abs(noisy_start["py"] - non_noisy_start["py"])[mask_start],
    s=1, color="blue", label="Noisy"
)
axs[0, 1].scatter(
    abs(filtered_start["y"] - non_noisy_start["y"])[mask_start],
    abs(filtered_start["py"] - non_noisy_start["py"])[mask_start],
    s=1, color="green", label="Filtered"
)
axs[0, 1].set_xlabel("Absolute difference of y")
axs[0, 1].set_ylabel("Absolute difference of py")
axs[0, 1].set_title(f"Difference of y and py ({start_bpm})")
axs[0, 1].grid()


axs[1, 0].scatter(
    abs(noise_other["x"] - non_noisy_other["x"])[mask_other],
    abs(noise_other["px"] - non_noisy_other["px"])[mask_other],
    s=1, color="blue", label="Noisy"
)
axs[1, 0].scatter(
    abs(filtered_other["x"] - non_noisy_other["x"])[mask_other],
    abs(filtered_other["px"] - non_noisy_other["px"])[mask_other],
    s=1, color="green", label="Filtered"
)
axs[1, 0].set_xlabel("Absolute difference of x")
axs[1, 0].set_ylabel("Absolute difference of px")
axs[1, 0].set_title(f"Difference of x and px ({start_bpm})")
axs[1, 0].grid()

axs[1, 1].scatter(
    abs(noise_other["y"] - non_noisy_other["y"])[mask_other],
    abs(noise_other["py"] - non_noisy_other["py"])[mask_other],
    s=1, color="blue", label="Noisy"
)
axs[1, 1].scatter(
    abs(filtered_other["y"] - non_noisy_other["y"])[mask_other],
    abs(filtered_other["py"] - non_noisy_other["py"])[mask_other],
    s=1, color="green", label="Filtered"
)
axs[1, 1].set_xlabel("Absolute difference of y")
axs[1, 1].set_ylabel("Absolute difference of py")
axs[1, 1].set_title(f"Difference of y and py ({start_bpm})")
axs[1, 1].grid()
# Create one global legend for the entire figure

fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))
plt.suptitle("Differences Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("differences_comparison.png", dpi=300)
# plt.show()

# Now one final plot that plots the relationship between the amplitude of x or y and the difference in x, px or y, py
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(
    abs(noisy_start["x"][mask_start]),
    abs((noisy_start["px"][mask_start] - non_noisy_start["px"][mask_start]) / non_noisy_start["px"][mask_start]),
    s=1, color="blue", label="Noisy"
)
axs[0, 0].scatter(
    abs(filtered_start["x"][mask_start]),
    abs((filtered_start["px"][mask_start] - non_noisy_start["px"][mask_start]) / non_noisy_start["px"][mask_start]),
    s=1, color="green", label="Filtered"
)

axs[0, 0].set_xlabel("Amplitude of x")
axs[0, 0].set_ylabel("Relative difference of px")
axs[0, 0].set_title(f"Amplitude of x vs Difference of px ({start_bpm})")
axs[0, 1].scatter(
    abs(noisy_start["y"][mask_start]),
    abs((noisy_start["py"][mask_start] - non_noisy_start["py"][mask_start]) / non_noisy_start["py"][mask_start]),
    s=1, color="blue", label="Noisy", alpha=0.5
)
axs[0, 1].scatter(
    abs(filtered_start["y"][mask_start]),
    abs((filtered_start["py"][mask_start] - non_noisy_start["py"][mask_start]) / non_noisy_start["py"][mask_start]),
    s=1, color="green", label="Filtered", alpha=0.5
)
axs[0, 1].set_xlabel("Amplitude of y")
axs[0, 1].set_ylabel("Relative difference of py")
axs[0, 1].set_title(f"Amplitude of y vs Difference of py ({start_bpm})")

axs[1, 0].scatter(
    abs(noise_other["x"][mask_other]),
    abs((noise_other["px"][mask_other] - non_noisy_other["px"][mask_other]) / non_noisy_other["px"][mask_other]),
    s=1, color="blue", label="Noisy", alpha=0.5
)
axs[1, 0].scatter(
    abs(filtered_other["x"][mask_other]),
    abs((filtered_other["px"][mask_other] - non_noisy_other["px"][mask_other]) / non_noisy_other["px"][mask_other]),
    s=1, color="green", label="Filtered", alpha=0.5
)
axs[1, 0].set_xlabel("Amplitude of x")
axs[1, 0].set_ylabel("Relative difference of px")
axs[1, 0].set_title(f"Amplitude of x vs Difference of px ({other_bpm})")
axs[1, 1].scatter(
    abs(noise_other["y"][mask_other]),
    abs((noise_other["py"][mask_other] - non_noisy_other["py"][mask_other]) / non_noisy_other["py"][mask_other]),
    s=1, color="blue", label="Noisy"
)
axs[1, 1].scatter(
    abs(filtered_other["y"][mask_other]),
    abs((filtered_other["py"][mask_other] - non_noisy_other["py"][mask_other]) / non_noisy_other["py"][mask_other]),
    s=1, color="green", label="Filtered"
)
axs[1, 1].set_xlabel("Amplitude of y")
axs[1, 1].set_ylabel("Relative difference of py")
axs[1, 1].set_title(f"Amplitude of y vs Difference of py ({other_bpm})")
for ax in axs.flat:
    ax.grid()
    ax.set_yscale("log")  # Set y-axis to logarithmic scale for better visibility of differences
# Create one global legend for the entire figure
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))
plt.suptitle("Amplitude vs Differences Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("amplitude_vs_differences_comparison.png", dpi=300)

# Create a new phase space plot of x, px and y, py for the start and end BPMs. 
# Instead of plotting all points, only plot the points where the relative difference is above the median of the relative difference

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(
    noisy_start["x"][mask_start],
    noisy_start["px"][mask_start],
    s=1, color="blue", label="Noisy"
)
axs[0, 0].scatter(
    filtered_start["x"][mask_start],
    filtered_start["px"][mask_start],
    s=1, color="green", label="Filtered (x > max x)"
)
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("px")
axs[0, 0].set_title(f"x, px Phase Space ({start_bpm})")
axs[0, 0].grid()
axs[0, 1].scatter(
    noisy_start["y"][mask_start],
    noisy_start["py"][mask_start],
    s=1, color="blue", label="Noisy", alpha=0.5
)
axs[0, 1].scatter(
    filtered_start["y"][mask_start],
    filtered_start["py"][mask_start],
    s=1, color="green", label="Filtered", alpha=0.5
)
axs[0, 1].set_xlabel("y")
axs[0, 1].set_ylabel("py")
axs[0, 1].set_title(f"y, py Phase Space ({start_bpm})")
axs[1, 0].scatter(
    noise_other["x"][mask_other],
    noise_other["px"][mask_other],
    s=1, color="blue", label="Noisy", alpha=0.5
)
axs[1, 0].scatter(
    filtered_other["x"][mask_other],
    filtered_other["px"][mask_other],
    s=1, color="green", label="Filtered", alpha=0.5
)
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("px")
axs[1, 0].set_title(f"x, px Phase Space ({other_bpm})")
axs[1, 0].grid()
axs[1, 1].scatter(
    noise_other["y"][mask_other],
    noise_other["py"][mask_other],
    s=1, color="blue", label="Noisy"
)
axs[1, 1].scatter(
    filtered_other["y"][mask_other],
    filtered_other["py"][mask_other],
    s=1, color="green", label="Filtered"
)
axs[1, 1].set_xlabel("y")
axs[1, 1].set_ylabel("py")
axs[1, 1].set_title(f"y, py Phase Space ({other_bpm})")
for ax in axs.flat:
    ax.grid()
plt.tight_layout()


plt.show()
