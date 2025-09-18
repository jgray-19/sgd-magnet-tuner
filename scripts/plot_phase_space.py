import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

from aba_optimiser.config import (
    ACD_ON,
    # BPM_START,
    # BPM_RANGE,
    CLEANED_FILE,
    # EPLUS_NOISY_FILE,
    # EPLUS_NONOISE_FILE,
    NO_NOISE_FILE,
    # KALMAN_FILE,
    NOISY_FILE,
    RAMP_UP_TURNS,
)
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.physics.phase_space import PhaseSpaceDiagnostics
from aba_optimiser.plotting.utils import setup_scientific_formatting

# Extract BPM names
start_bpm = "BPM.10R4.B1"
other_bpm = "BPM.11R4.B1"

# Load non-noisy data
init_coords = pd.read_parquet(NO_NOISE_FILE).set_index("turn")
non_noisy_start = select_markers(init_coords, start_bpm)
non_noisy_other = select_markers(init_coords, other_bpm)

# Load noisy data
noise_init = pd.read_parquet(NOISY_FILE).set_index("turn")
noisy_start = select_markers(noise_init, start_bpm)
noise_other = select_markers(noise_init, other_bpm)

# Load filtered data
filtered_init = pd.read_parquet(CLEANED_FILE).set_index("turn")
filtered_start = select_markers(filtered_init, start_bpm)
filtered_other = select_markers(filtered_init, other_bpm)

# Apply ramp-up filter if ACD_ON
if ACD_ON:
    non_noisy_start = non_noisy_start[non_noisy_start.index > RAMP_UP_TURNS]
    non_noisy_other = non_noisy_other[non_noisy_other.index > RAMP_UP_TURNS]
    noisy_start = noisy_start[noisy_start.index > RAMP_UP_TURNS]
    noise_other = noise_other[noise_other.index > RAMP_UP_TURNS]
    filtered_start = filtered_start[filtered_start.index > RAMP_UP_TURNS]
    filtered_other = filtered_other[filtered_other.index > RAMP_UP_TURNS]
    # kalman_start = kalman_start[kalman_start.index > RAMP_UP_TURNS]
    # kalman_other = kalman_other[kalman_other.index > RAMP_UP_TURNS]

# Prepare analytical ellipses for start BPM
ps_diag_start = PhaseSpaceDiagnostics(
    bpm=start_bpm,
    x_data=noisy_start["x"],
    px_data=noisy_start["px"],
    y_data=noisy_start["y"],
    py_data=noisy_start["py"],
)
x_ellipse, px_ellipse, y_ellipse, py_ellipse = ps_diag_start.ellipse_points()
x_upper, px_upper, y_upper, py_upper = ps_diag_start.ellipse_sigma(sigma_level=1.0)
x_lower, px_lower, y_lower, py_lower = ps_diag_start.ellipse_sigma(sigma_level=-1.0)

# Prepare analytical ellipses for other BPM
ps_diag_other = PhaseSpaceDiagnostics(
    bpm=other_bpm,
    x_data=noise_other["x"],
    px_data=noise_other["px"],
    y_data=noise_other["y"],
    py_data=noise_other["py"],
    info=True,
)
x_ellipse_other, px_ellipse_other, y_ellipse_other, py_ellipse_other = (
    ps_diag_other.ellipse_points()
)
x_upper_other, px_upper_other, y_upper_other, py_upper_other = (
    ps_diag_other.ellipse_sigma(sigma_level=1.0)
)
x_lower_other, px_lower_other, y_lower_other, py_lower_other = (
    ps_diag_other.ellipse_sigma(sigma_level=-1.0)
)


def plot_phase_space(
    *,
    ax: plt.Axes,
    noisy,
    non_noisy,
    filtered,
    # kalman,
    coord1: str,
    coord2: str,
    title: str,
    ellipse=None,
    extra_mask=None,
    extra_labels=None,
):
    """General phase space plot function"""

    ax.scatter(noisy[coord1], noisy[coord2], s=1, color="blue", label="Noisy")
    ax.scatter(
        non_noisy[coord1], non_noisy[coord2], s=1, color="red", label="Non-noisy"
    )
    ax.scatter(filtered[coord1], filtered[coord2], s=1, color="green", label="Filtered")
    # ax.scatter(kalman[coord1], kalman[coord2], s=1, color="green", label="Kalman")

    if extra_mask is not None and extra_labels is not None:
        ax.scatter(
            noisy[coord1][extra_mask],
            noisy[coord2][extra_mask],
            s=1,
            color="cyan",
            label=extra_labels[0],
        )
        ax.scatter(
            non_noisy[coord1][extra_mask],
            non_noisy[coord2][extra_mask],
            s=1,
            color="orange",
            label=extra_labels[1],
        )
        ax.scatter(
            filtered[coord1][extra_mask],
            filtered[coord2][extra_mask],
            s=1,
            color="purple",
            label=extra_labels[2],
        )

    if ellipse is not None:
        # ellipse: (center, upper, lower) tuples
        center_x, center_y = ellipse[0]
        upper_x, upper_y = ellipse[1]
        lower_x, lower_y = ellipse[2]
        ax.plot(center_x, center_y, color="orange", label="Analytical Ellipse")
        ax.plot(
            upper_x, upper_y, color="orange", linestyle="--", label="+1 Sigma Ellipse"
        )
        ax.plot(
            lower_x, lower_y, color="orange", linestyle="--", label="-1 Sigma Ellipse"
        )

    coord1 = coord1.replace("p", "p_")
    coord2 = coord2.replace("p", "p_")
    ax.set_xlabel("$" + coord1 + "$")
    ax.set_ylabel("$" + coord2 + "$")
    ax.set_title(title)
    ax.grid()
    setup_scientific_formatting(ax, powerlimits=(-1, 1))


# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Settings for each subplot
plot_configs = [
    {
        "ax": axs[0, 0],
        "noisy": noisy_start,
        "non_noisy": non_noisy_start,
        "filtered": filtered_start,
        # "kalman": kalman_start,
        "coord1": "x",
        "coord2": "px",
        "title": f"x, px Phase Space ({start_bpm})",
        "ellipse": [(x_ellipse, px_ellipse), (x_upper, px_upper), (x_lower, px_lower)],
    },
    {
        "ax": axs[0, 1],
        "noisy": noisy_start,
        "non_noisy": non_noisy_start,
        "filtered": filtered_start,
        # "kalman": kalman_start,
        "coord1": "y",
        "coord2": "py",
        "title": f"y, py Phase Space ({start_bpm})",
        "ellipse": [(y_ellipse, py_ellipse), (y_upper, py_upper), (y_lower, py_lower)],
    },
    {
        "ax": axs[1, 0],
        "noisy": noise_other,
        "non_noisy": non_noisy_other,
        "filtered": filtered_other,
        # "kalman": kalman_other,
        "coord1": "x",
        "coord2": "px",
        "title": f"x, px Phase Space ({other_bpm})",
        "ellipse": [
            (x_ellipse_other, px_ellipse_other),
            (x_upper_other, px_upper_other),
            (x_lower_other, px_lower_other),
        ],
    },
    {
        "ax": axs[1, 1],
        "noisy": noise_other,
        "non_noisy": non_noisy_other,
        "filtered": filtered_other,
        # "kalman": kalman_other,
        "coord1": "y",
        "coord2": "py",
        "title": f"y, py Phase Space ({other_bpm})",
        "ellipse": [
            (y_ellipse_other, py_ellipse_other),
            (y_upper_other, py_upper_other),
            (y_lower_other, py_lower_other),
        ],
    },
]


# Loop over subplots
for cfg in plot_configs:
    plot_phase_space(**cfg)

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

# plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("phase_space_comparison.png", dpi=250)
plt.show()
