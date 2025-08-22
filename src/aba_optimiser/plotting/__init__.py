import matplotlib.pyplot as plt

from aba_optimiser.plotting.strengths import (
    plot_strengths_comparison,
    plot_strengths_vs_position,
)


def show_plots():
    plt.show()


__all__ = [
    "plot_strengths_comparison",
    "plot_strengths_vs_position",
    "show_plots",
]
