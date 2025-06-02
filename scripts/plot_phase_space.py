import matplotlib.pyplot as plt
import tfs
import pandas as pd

from aba_optimiser.config import (
    ACD_ON, BPM_RANGE, FILTERED_FILE, NOISE_FILE,
    RAMP_UP_TURNS, TRACK_DATA_FILE, MIN_FRACTION_MAX,
    XY_MIN, PXPY_MIN,
)
from aba_optimiser.utils import select_marker
from aba_optimiser.phase_space import PhaseSpaceDiagnostics

# Extract BPM names
start_bpm, _ = BPM_RANGE.split("/")
other_bpm = "BPM.14R3.B1"

# Load non-noisy data
init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
non_noisy_start = select_marker(init_coords, start_bpm)
non_noisy_other = select_marker(init_coords, other_bpm)

# Load noisy data
noise_init = pd.read_feather(NOISE_FILE).set_index("turn")
noisy_start = select_marker(noise_init, start_bpm)
noise_other = select_marker(noise_init, other_bpm)

# Load filtered data
filtered_start_full = pd.read_feather(FILTERED_FILE).set_index("turn")
filtered_start = select_marker(filtered_start_full, start_bpm)
filtered_other = select_marker(filtered_start_full, other_bpm)

# Apply ramp-up filter if ACD_ON
if ACD_ON:
    non_noisy_start = non_noisy_start[non_noisy_start.index > RAMP_UP_TURNS]
    non_noisy_other = non_noisy_other[non_noisy_other.index > RAMP_UP_TURNS]
    noisy_start = noisy_start[noisy_start.index > RAMP_UP_TURNS]
    noise_other = noise_other[noise_other.index > RAMP_UP_TURNS]
    filtered_start = filtered_start[filtered_start.index > RAMP_UP_TURNS]
    filtered_other = filtered_other[filtered_other.index > RAMP_UP_TURNS]

# Prepare analytical ellipses for start BPM
ps_diag_start = PhaseSpaceDiagnostics(
    bpm=start_bpm,
    x_data=noisy_start["x"], px_data=noisy_start["px"],
    y_data=noisy_start["y"], py_data=noisy_start["py"]
)
x_ellipse, px_ellipse, y_ellipse, py_ellipse = ps_diag_start.ellipse_points()
x_upper, px_upper, y_upper, py_upper = ps_diag_start.ellipse_sigma(sigma_level=1.0)
x_lower, px_lower, y_lower, py_lower = ps_diag_start.ellipse_sigma(sigma_level=-1.0)

# Prepare analytical ellipses for other BPM
ps_diag_other = PhaseSpaceDiagnostics(
    bpm=other_bpm,
    x_data=noise_other["x"], px_data=noise_other["px"],
    y_data=noise_other["y"], py_data=noise_other["py"]
)
ana_x_other, ana_px_other, ana_y_other, ana_py_other = ps_diag_other.ellipse_points()
ana_x_upper_other, ana_px_upper_other, ana_y_upper_other, ana_py_upper_other = ps_diag_other.ellipse_sigma(sigma_level=1.0)
ana_x_lower_other, ana_px_lower_other, ana_y_lower_other, ana_py_lower_other = ps_diag_other.ellipse_sigma(sigma_level=-1.0)

# Prepare limits
max_x_start = noisy_start["x"].abs().max()
x_start_lim = XY_MIN

max_y_other = noise_other["y"].abs().max()
y_other_lim = max_y_other * MIN_FRACTION_MAX

# Prepare masks for filtering points (used in later difference plots)
x_mask_start = ((noisy_start["px"].abs() > PXPY_MIN) & (abs(noisy_start["x"]) > XY_MIN))
y_mask_start = ((noisy_start["py"].abs() > PXPY_MIN) & (abs(noisy_start["y"]) > XY_MIN/2))
mask_start = x_mask_start & y_mask_start

filtered_mask_start = ((filtered_start["px"].abs() > PXPY_MIN) & (abs(filtered_start["x"]) > XY_MIN))
filtered_mask_start &= ((filtered_start["py"].abs() > PXPY_MIN) & (abs(filtered_start["y"]) > XY_MIN/2))

x_mask_other = ((noise_other["px"].abs() > PXPY_MIN) & (abs(noise_other["x"]) > XY_MIN/2))
y_mask_other = ((noise_other["py"].abs() > PXPY_MIN) & (abs(noise_other["y"]) > XY_MIN))
mask_other = x_mask_other & y_mask_other

filtered_mask_other = ((filtered_other["px"].abs() > PXPY_MIN) & (abs(filtered_other["x"]) > XY_MIN/2))
filtered_mask_other &= ((filtered_other["py"].abs() > PXPY_MIN) & (abs(filtered_other["y"]) > XY_MIN))

# === First plot: full phase space plots with ellipses ===
# === Simplified first plot with functions and loops ===

def plot_phase_space(ax, noisy, non_noisy, filtered, coord1, coord2, title,
                     limits=None, ellipse=None, extra_mask=None, extra_labels=None):
    """General phase space plot function"""

    ax.scatter(noisy[coord1], noisy[coord2], s=1, color="blue", label="Noisy")
    ax.scatter(non_noisy[coord1], non_noisy[coord2], s=1, color="red", label="Non-noisy")
    ax.scatter(filtered[coord1], filtered[coord2], s=1, color="green", label="Filtered")

    if extra_mask is not None and extra_labels is not None:
        ax.scatter(noisy[coord1][extra_mask], noisy[coord2][extra_mask], s=1, color="cyan", label=extra_labels[0])
        ax.scatter(non_noisy[coord1][extra_mask], non_noisy[coord2][extra_mask], s=1, color="orange", label=extra_labels[1])
        ax.scatter(filtered[coord1][extra_mask], filtered[coord2][extra_mask], s=1, color="purple", label=extra_labels[2])

    if limits is not None:
        for (v, axis, label) in limits:
            if axis == 'x':
                ax.axvline(v, color="black", linestyle="--", label=label)
                ax.axvline(-v, color="black", linestyle="--")
            if axis == 'y':
                ax.axhline(v, color="black", linestyle="--", label=label)
                ax.axhline(-v, color="black", linestyle="--")

    if ellipse is not None:
        # ellipse: (center, upper, lower) tuples
        center_x, center_y = ellipse[0]
        upper_x, upper_y = ellipse[1]
        lower_x, lower_y = ellipse[2]
        ax.plot(center_x, center_y, color="orange", label="Analytical Ellipse")
        ax.plot(upper_x, upper_y, color="orange", linestyle="--", label="+1 Sigma Ellipse")
        ax.plot(lower_x, lower_y, color="orange", linestyle="--", label="-1 Sigma Ellipse")

    ax.set_xlabel(coord1)
    ax.set_ylabel(coord2)
    ax.set_title(title)
    ax.grid()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Settings for each subplot
plot_configs = [
    {
        "ax": axs[0, 0],
        "noisy": noisy_start, "non_noisy": non_noisy_start, "filtered": filtered_start,
        "coord1": "x", "coord2": "px", "title": f"x, px Phase Space ({start_bpm})",
        "limits": [(x_start_lim, 'x', "Max x"), (PXPY_MIN, 'y', "Min px")],
        "ellipse": [(x_ellipse, px_ellipse), (x_upper, px_upper), (x_lower, px_lower)]
    },
    {
        "ax": axs[0, 1],
        "noisy": noisy_start, "non_noisy": non_noisy_start, "filtered": filtered_start,
        "coord1": "y", "coord2": "py", "title": f"y, py Phase Space ({start_bpm})",
        "limits": [(XY_MIN/2, 'x', "Min y"), (PXPY_MIN, 'y', "Min py")],
        "ellipse": [(y_ellipse, py_ellipse), (y_upper, py_upper), (y_lower, py_lower)],
        "extra_mask": noisy_start["x"].abs() > x_start_lim,
        "extra_labels": ["Noisy (x > max x)", "Non-noisy (x > max x)", "Filtered (x > max x)"]
    },
    {
        "ax": axs[1, 0],
        "noisy": noise_other, "non_noisy": non_noisy_other, "filtered": filtered_other,
        "coord1": "x", "coord2": "px", "title": f"x, px Phase Space ({other_bpm})",
        "limits": [(XY_MIN/2, 'x', "Min x"), (PXPY_MIN, 'y', "Min px")],
        "ellipse": [(ana_x_other, ana_px_other), (ana_x_upper_other, ana_px_upper_other), (ana_x_lower_other, ana_px_lower_other)],
        "extra_mask": noise_other["y"].abs() > y_other_lim,
        "extra_labels": ["Noisy (y > max y)", "Non-noisy (y > max y)", "Filtered (y > max y)"]
    },
    {
        "ax": axs[1, 1],
        "noisy": noise_other, "non_noisy": non_noisy_other, "filtered": filtered_other,
        "coord1": "y", "coord2": "py", "title": f"y, py Phase Space ({other_bpm})",
        "limits": [(y_other_lim, 'x', "Max y"), (PXPY_MIN, 'y', "Min py")],
        "ellipse": [(ana_y_other, ana_py_other), (ana_y_upper_other, ana_py_upper_other), (ana_y_lower_other, ana_py_lower_other)]
    }
]

# Loop over subplots
for cfg in plot_configs:
    plot_phase_space(**cfg)

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("merged_phase_space_comparison.png", dpi=300)
# plt.show()

# === Second plot: absolute differences ===

def plot_absolute_difference(ax, noisy, filtered, non_noisy, coord1, coord2, mask, title):
    """Plot absolute difference for a given pair of coordinates."""
    # Noisy differences
    ax.scatter(
        abs(noisy[coord1] - non_noisy[coord1])[mask],
        abs(noisy[coord2] - non_noisy[coord2])[mask],
        s=1, color="blue", label="Noisy"
    )
    # Filtered differences
    ax.scatter(
        abs(filtered[coord1] - non_noisy[coord1])[mask],
        abs(filtered[coord2] - non_noisy[coord2])[mask],
        s=1, color="green", label="Filtered"
    )
    ax.set_xlabel(f"Absolute difference of {coord1}")
    ax.set_ylabel(f"Absolute difference of {coord2}")
    ax.set_title(title)
    ax.grid()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Settings for each subplot
difference_configs = [
    {
        "ax": axs[0, 0],
        "noisy": noisy_start, "filtered": filtered_start, "non_noisy": non_noisy_start,
        "coord1": "x", "coord2": "px", "mask": mask_start,
        "title": f"Difference of x and px ({start_bpm})"
    },
    {
        "ax": axs[0, 1],
        "noisy": noisy_start, "filtered": filtered_start, "non_noisy": non_noisy_start,
        "coord1": "y", "coord2": "py", "mask": mask_start,
        "title": f"Difference of y and py ({start_bpm})"
    },
    {
        "ax": axs[1, 0],
        "noisy": noise_other, "filtered": filtered_other, "non_noisy": non_noisy_other,
        "coord1": "x", "coord2": "px", "mask": mask_other,
        "title": f"Difference of x and px ({other_bpm})"
    },
    {
        "ax": axs[1, 1],
        "noisy": noise_other, "filtered": filtered_other, "non_noisy": non_noisy_other,
        "coord1": "y", "coord2": "py", "mask": mask_other,
        "title": f"Difference of y and py ({other_bpm})"
    }
]

# Loop over subplots
for cfg in difference_configs:
    plot_absolute_difference(**cfg)

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Differences Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("differences_comparison.png", dpi=300)
# plt.show()

# === Third plot: amplitude vs relative difference ===

def plot_relative_difference(ax, noisy, filtered, non_noisy, coord_amp, coord_diff, mask_noisy, mask_filtered, title):
    """Plot amplitude vs relative difference."""
    
    # Noisy relative difference
    ax.scatter(
        abs(noisy[coord_amp][mask_noisy]),
        abs((noisy[coord_diff][mask_noisy] - non_noisy[coord_diff][mask_noisy]) / non_noisy[coord_diff][mask_noisy]),
        s=1, color="blue", label="Noisy"
    )

    # Filtered relative difference
    # Align non_noisy to filtered turns
    filtered_turns = filtered.index
    non_noisy_filtered_aligned = non_noisy.loc[filtered_turns]

    ax.scatter(
        abs(filtered[coord_amp][mask_filtered]),
        abs((filtered[coord_diff][mask_filtered] - non_noisy_filtered_aligned[coord_diff][mask_filtered]) / 
            non_noisy_filtered_aligned[coord_diff][mask_filtered]),
        s=1, color="green", label="Filtered"
    )

    ax.set_xlabel(f"Amplitude of {coord_amp}")
    ax.set_ylabel(f"Relative difference of {coord_diff}")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Settings for each subplot
relative_configs = [
    {
        "ax": axs[0, 0],
        "noisy": noisy_start, "filtered": filtered_start, "non_noisy": non_noisy_start,
        "coord_amp": "x", "coord_diff": "px",
        "mask_noisy": mask_start, "mask_filtered": filtered_mask_start,
        "title": f"Amplitude of x vs Difference of px ({start_bpm})"
    },
    {
        "ax": axs[0, 1],
        "noisy": noisy_start, "filtered": filtered_start, "non_noisy": non_noisy_start,
        "coord_amp": "y", "coord_diff": "py",
        "mask_noisy": mask_start, "mask_filtered": filtered_mask_start,
        "title": f"Amplitude of y vs Difference of py ({start_bpm})"
    },
    {
        "ax": axs[1, 0],
        "noisy": noise_other, "filtered": filtered_other, "non_noisy": non_noisy_other,
        "coord_amp": "x", "coord_diff": "px",
        "mask_noisy": mask_other, "mask_filtered": filtered_mask_other,
        "title": f"Amplitude of x vs Difference of px ({other_bpm})"
    },
    {
        "ax": axs[1, 1],
        "noisy": noise_other, "filtered": filtered_other, "non_noisy": non_noisy_other,
        "coord_amp": "y", "coord_diff": "py",
        "mask_noisy": mask_other, "mask_filtered": filtered_mask_other,
        "title": f"Amplitude of y vs Difference of py ({other_bpm})"
    }
]

# Loop over subplots
for cfg in relative_configs:
    plot_relative_difference(**cfg)

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Amplitude vs Differences Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("amplitude_vs_differences_comparison.png", dpi=300)
# plt.show()

# === Fourth plot: filtered phase space after masks ===

def plot_masked_phase_space(ax, noisy, filtered, coord1, coord2, mask, title):
    """Plot phase space for noisy and filtered data after applying mask."""
    
    ax.scatter(noisy[coord1][mask], noisy[coord2][mask], s=1, color="blue", label="Noisy")
    ax.scatter(filtered[coord1][mask], filtered[coord2][mask], s=1, color="green", label="Filtered")
    
    ax.set_xlabel(coord1)
    ax.set_ylabel(coord2)
    ax.set_title(title)
    ax.grid()

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Settings for each subplot
masked_configs = [
    {
        "ax": axs[0, 0],
        "noisy": noisy_start, "filtered": filtered_start,
        "coord1": "x", "coord2": "px",
        "mask": mask_start,
        "title": f"x, px Phase Space ({start_bpm})"
    },
    {
        "ax": axs[0, 1],
        "noisy": noisy_start, "filtered": filtered_start,
        "coord1": "y", "coord2": "py",
        "mask": mask_start,
        "title": f"y, py Phase Space ({start_bpm})"
    },
    {
        "ax": axs[1, 0],
        "noisy": noise_other, "filtered": filtered_other,
        "coord1": "x", "coord2": "px",
        "mask": mask_other,
        "title": f"x, px Phase Space ({other_bpm})"
    },
    {
        "ax": axs[1, 1],
        "noisy": noise_other, "filtered": filtered_other,
        "coord1": "y", "coord2": "py",
        "mask": mask_other,
        "title": f"y, py Phase Space ({other_bpm})"
    }
]

# Loop over subplots
for cfg in masked_configs:
    plot_masked_phase_space(**cfg)

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

plt.suptitle("Masked Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("masked_phase_space_comparison.png", dpi=300)
plt.show()
