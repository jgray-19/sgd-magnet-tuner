from pathlib import Path

from omc3.plotting.plot_optics_measurements import plot

DATA_DIR = Path(__file__).parent.parent
figs = plot(
    folders=[DATA_DIR / "optics"],
    # output='output_directory',
    delta=True,  # delta from reference
    optics_parameters=[
        "orbit",
        "beta_phase",
        "beta_amplitude",
        "phase",
        "total_phase",
        "f1001_x",
        "f1010_x",
    ],
    x_axis="location",  # or 'phase-advance'
    ip_positions="LHCB1",
    suppress_column_legend=True,
    show=True,
    ncol_legend=2,
)
