from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def setup_scientific_formatting(
    ax: plt.Axes = None,
    use_math_text: bool = True,
    powerlimits: tuple[int, int] = (0, 0),
    plane: str = "xy",
):
    """
    Setup scientific notation formatting for both axes of a matplotlib Axes object.

    Parameters:
        ax (plt.Axes or list of plt.Axes, optional): The axes to format. If None, uses current axes.
        use_math_text (bool): Whether to use math text for scientific notation.
        powerlimits (tuple): Limits for scientific notation.
        plane (str): The plane to apply formatting to ("xy", "x", "y").
    Returns:
        formatter: The ScalarFormatter instance used.
    """
    if ax is None:
        ax = plt.gca()
    # Support for list/array of axes
    axes = ax if isinstance(ax, (list, tuple)) else [ax]
    for axis in axes:
        # Create independent ScalarFormatter instances per-axis so each
        # axis can pick its own exponent (offset) based on its data range.
        if "x" in plane:
            fmt_x = mticker.ScalarFormatter(useMathText=use_math_text)
            fmt_x.set_scientific(True)
            fmt_x.set_powerlimits(powerlimits)
            axis.xaxis.set_major_formatter(fmt_x)

        if "y" in plane:
            fmt_y = mticker.ScalarFormatter(useMathText=use_math_text)
            fmt_y.set_scientific(True)
            fmt_y.set_powerlimits(powerlimits)
            axis.yaxis.set_major_formatter(fmt_y)


def show_plots():
    plt.show()
