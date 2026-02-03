import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

from aba_optimiser.config import (
    CLEANED_FILE,
    NO_NOISE_FILE,
)
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.plotting.utils import setup_scientific_formatting

ACD_ON = False  # Whether the ACD was used or not (Ignores the ramp up turns)
RAMP_UP_TURNS = 1000  # Number of turns to ramp up the ACD
# Extract BPM names
start_bpm = "BPM.11R4.B1"
other_bpm = "BPM.12R4.B1"

# Load non-noisy data
# init_coords = pd.read_parquet(EMINUS_NONOISE_FILE).set_index("turn")
init_coords = pd.read_parquet(NO_NOISE_FILE).set_index("turn")
non_noisy_start = select_markers(init_coords, start_bpm)
non_noisy_other = select_markers(init_coords, other_bpm)

# Load noisy data
# noise_init = pd.read_parquet(EMINUS_CLEANED_FILE).set_index("turn")
noise_init = pd.read_parquet(CLEANED_FILE).set_index("turn")
noisy_start = select_markers(noise_init, start_bpm)
noise_other = select_markers(noise_init, other_bpm)

# Load filtered data
# filtered_init = pd.read_parquet(EMINUS_CLEANED_FILE).set_index("turn")
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
    extra_mask=None,
    extra_labels=None,
):
    """General phase space plot function"""

    # ax.scatter(noisy[coord1], noisy[coord2], s=1, color="blue", label="Noisy")
    ax.scatter(non_noisy[coord1], non_noisy[coord2], s=1, color="red", label="Non-noisy")
    # ax.scatter(filtered[coord1], filtered[coord2], s=1, color="green", label="Filtered")
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
    },
]


# Loop over subplots
for cfg in plot_configs:
    plot_phase_space(**cfg)  # ty:ignore[invalid-argument-type]

# Global legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 1))

# plt.suptitle("Phase Space Comparison", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("phase_space_comparison.png", dpi=250)
plt.show()
