from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from aba_optimiser.measurements import plot_quad_diffs_and_phases as plot_module

plt.switch_backend("Agg")

if TYPE_CHECKING:
    from pathlib import Path


def test_convert_estimates_to_optimisation_space_ignores_tune_and_corrector_context(
    tmp_path: Path,
) -> None:
    accelerator = SimpleNamespace()
    estimates = {"Arc 1": {"Q1.dk1l": 1.5}}
    converted = plot_module.convert_estimates_to_optimisation_space(
        accelerator,  # ty:ignore[invalid-argument-type]
        estimates,
        tmp_path / "tune_knobs.txt",
        tmp_path / "correctors.txt",
    )

    assert converted == estimates


def test_filter_estimates_by_max_uncertainty_discards_large_relative_and_nonfinite_values() -> None:
    estimates = {
        "Arc 1": {"k1": 1.0, "k2": 2.0, "k3": 0.0},
        "Arc 2": {"k4": 4.0},
    }
    uncertainties = {
        "Arc 1": {"k1": 0.1, "k2": 3.0, "k3": 0.1},
        "Arc 2": {"k4": 0.2},
    }
    actual = {
        "Arc 1": {"k1": 10.0, "k2": 20.0, "k3": 30.0},
        "Arc 2": {"k4": 40.0},
    }

    filtered_estimates, filtered_uncertainties, filtered_actual = (
        plot_module.filter_estimates_by_max_uncertainty(
            estimates,
            uncertainties,
            actual,
            max_uncertainty=1.0,
        )
    )

    assert filtered_estimates == {
        "Arc 1": {"k1": 1.0},
        "Arc 2": {"k4": 4.0},
    }
    assert filtered_uncertainties == {
        "Arc 1": {"k1": 0.1},
        "Arc 2": {"k4": 0.2},
    }
    assert filtered_actual == {
        "Arc 1": {"k1": 10.0},
        "Arc 2": {"k4": 40.0},
    }


def test_plot_quad_diffs_per_arc_writes_figure(tmp_path: Path) -> None:
    estimates = {
        "Arc 1": {"MQ.13R1.B1.dk1l": 1.5, "MQ.14R1.B1.dk1l": -0.5},
        "Arc 2": {"MQ.21R2.B1.dk1l": 0.2},
    }
    uncertainties = {
        "Arc 1": {"MQ.13R1.B1.dk1l": 0.1, "MQ.14R1.B1.dk1l": 0.2},
        "Arc 2": {"MQ.21R2.B1.dk1l": 0.05},
    }
    actual = {
        "Arc 1": {"MQ.13R1.B1.dk1l": 1.0, "MQ.14R1.B1.dk1l": 0.0},
        "Arc 2": {"MQ.21R2.B1.dk1l": 0.1},
    }

    plot_module.plot_quad_diffs(
        estimates,
        uncertainties,
        actual,
        squeeze_step="step0",
        results_dir=tmp_path,
        fullring=False,
        accelerator=None,
    )

    assert (tmp_path / "quad_diffs_step0.png").exists()


def test_plot_quad_diffs_fullring_writes_main_and_bends_figures(tmp_path: Path) -> None:
    arc1 = {
        "MQ.13R1.B1.dk1l": 1.5,
        "MB.A12L1.B1.dk0l": 0.3,
        "MS.13R1.B1.dk2l": 0.05,
        "MQ.13R1.B1.dx": 1.0e-3,
        "MQ.13R1.B1.dy": -2.0e-3,
    }
    estimates = {"Arc 1": dict(arc1)}
    uncertainties = {"Arc 1": dict.fromkeys(arc1, 0.01)}
    actual = {"Arc 1": dict.fromkeys(arc1, 0.0)}

    plot_module.plot_quad_diffs(
        estimates,
        uncertainties,
        actual,
        squeeze_step="step1",
        results_dir=tmp_path,
        fullring=True,
        accelerator=None,
    )

    assert (tmp_path / "quad_diffs_step1_fullring.png").exists()
    assert (tmp_path / "quad_diffs_step1_fullring_bends.png").exists()
