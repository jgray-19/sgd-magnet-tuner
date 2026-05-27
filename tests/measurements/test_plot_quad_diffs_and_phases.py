from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from aba_optimiser.measurements import plot_quad_diffs_and_phases as plot_module

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
