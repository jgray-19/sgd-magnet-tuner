import argparse

import matplotlib.pyplot as plt
import pandas as pd

from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.plotting.utils import setup_scientific_formatting

# EXAMPLES
# python scripts/plot_pz_data.py analysis BPMYB.6L4.B1 BPMYA.6L4.B1


def main():
    parser = argparse.ArgumentParser(
        description="Plot phase space data from pz_data.parquet"
    )
    parser.add_argument("folder", help="Folder containing pz_data.parquet")
    parser.add_argument("before_bpm", help="BPM name before ACD")
    parser.add_argument("after_bpm", help="BPM name after ACD")
    args = parser.parse_args()

    folder = args.folder
    before_bpm = args.before_bpm
    after_bpm = args.after_bpm

    # Load data
    data = pd.read_parquet(f"{folder}/pz_data.parquet").set_index("turn")

    # Select data for BPMs
    data_before = select_markers(data, before_bpm)
    data_after = select_markers(data, after_bpm)

    # Limit to first 6600 turns
    data_before = data_before[data_before.index <= 6600]
    data_after = data_after[data_after.index <= 6600]

    # Compute differences
    diff_x = data_after["x"] - data_before["x"]
    diff_px = data_after["px"] - data_before["px"]
    diff_y = data_after["y"] - data_before["y"]
    diff_py = data_after["py"] - data_before["py"]

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot for before BPM x, px
    axs[0, 0].scatter(data_before["x"], data_before["px"], s=1, color="blue")
    axs[0, 0].set_xlabel("$x$")
    axs[0, 0].set_ylabel("$p_x$")
    axs[0, 0].set_title("x, px Phase Space (before_acd)")
    axs[0, 0].grid()
    setup_scientific_formatting(axs[0, 0], powerlimits=(-1, 1))

    # Plot for before BPM y, py
    axs[0, 1].scatter(data_before["y"], data_before["py"], s=1, color="blue")
    axs[0, 1].set_xlabel("$y$")
    axs[0, 1].set_ylabel("$p_y$")
    axs[0, 1].set_title("y, py Phase Space (before_acd)")
    axs[0, 1].grid()
    setup_scientific_formatting(axs[0, 1], powerlimits=(-1, 1))

    # Plot for after BPM x, px
    axs[1, 0].scatter(data_after["x"], data_after["px"], s=1, color="red")
    axs[1, 0].set_xlabel("$x$")
    axs[1, 0].set_ylabel("$p_x$")
    axs[1, 0].set_title("x, px Phase Space (after_acd)")
    axs[1, 0].grid()
    setup_scientific_formatting(axs[1, 0], powerlimits=(-1, 1))

    # Plot for after BPM y, py
    axs[1, 1].scatter(data_after["y"], data_after["py"], s=1, color="red")
    axs[1, 1].set_xlabel("$y$")
    axs[1, 1].set_ylabel("$p_y$")
    axs[1, 1].set_title("y, py Phase Space (after_acd)")
    axs[1, 1].grid()
    setup_scientific_formatting(axs[1, 1], powerlimits=(-1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("pz_phase_space_comparison.png", dpi=250)
    plt.show()

    # Separate figure for differences
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8))

    # Plot differences
    axs2[0, 0].plot(data_before.index, diff_x, color="blue")
    axs2[0, 0].set_xlabel("Turn")
    axs2[0, 0].set_ylabel("$\\Delta x$")
    axs2[0, 0].set_title("Difference in x (after - before)")
    axs2[0, 0].grid()
    setup_scientific_formatting(axs2[0, 0], powerlimits=(-1, 1))

    axs2[0, 1].plot(data_before.index, diff_px, color="red")
    axs2[0, 1].set_xlabel("Turn")
    axs2[0, 1].set_ylabel("$\\Delta p_x$")
    axs2[0, 1].set_title("Difference in px (after - before)")
    axs2[0, 1].grid()
    setup_scientific_formatting(axs2[0, 1], powerlimits=(-1, 1))

    axs2[1, 0].plot(data_before.index, diff_y, color="green")
    axs2[1, 0].set_xlabel("Turn")
    axs2[1, 0].set_ylabel("$\\Delta y$")
    axs2[1, 0].set_title("Difference in y (after - before)")
    axs2[1, 0].grid()
    setup_scientific_formatting(axs2[1, 0], powerlimits=(-1, 1))

    axs2[1, 1].plot(data_before.index, diff_py, color="purple")
    axs2[1, 1].set_xlabel("Turn")
    axs2[1, 1].set_ylabel("$\\Delta p_y$")
    axs2[1, 1].set_title("Difference in py (after - before)")
    axs2[1, 1].grid()
    setup_scientific_formatting(axs2[1, 1], powerlimits=(-1, 1))

    plt.tight_layout()
    plt.savefig("pz_differences.png", dpi=250)
    plt.show()


if __name__ == "__main__":
    main()
