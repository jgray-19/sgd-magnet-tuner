"""Result handling for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.io.utils import save_results, scientific_notation
from aba_optimiser.plotting.strengths import (
    plot_deltap_comparison,
    plot_strengths_comparison,
    plot_strengths_vs_position,
)
from aba_optimiser.plotting.utils import show_plots

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


class ResultManager:
    """Manages result processing and output generation."""

    def __init__(
        self,
        knob_names: list[str],
        elem_spos: list[float],
        simulation_config: SimulationConfig,
        show_plots: bool = True,
        output_knobs_path: Path | None = None,
        knob_table_path: Path | None = None,
        plots_dir: Path | None = None,
    ):
        """Initialise result manager.

        Args:
            knob_names: List of knob names
            elem_spos: Element s-positions
            simulation_config: Simulation configuration settings
            show_plots: Whether to display plots
            output_knobs_path: Path to save final knobs (defaults to PROJECT_ROOT/data/final_knobs.txt)
            knob_table_path: Path to save knob table (defaults to PROJECT_ROOT/data/knob_strengths_table.txt)
            plots_dir: Directory to save plots (defaults to plots/)
        """
        self.knob_names = knob_names
        self.elem_spos = elem_spos
        self.show_plots = show_plots
        self.simulation_config = simulation_config

        # Lazy import defaults if not provided
        if output_knobs_path is None or knob_table_path is None:
            from aba_optimiser.config import KNOB_TABLE, OUTPUT_KNOBS

            self.output_knobs_path = output_knobs_path or OUTPUT_KNOBS
            self.knob_table_path = knob_table_path or KNOB_TABLE
        else:
            self.output_knobs_path = output_knobs_path
            self.knob_table_path = knob_table_path

        # Set plots directory with default
        from pathlib import Path
        self.plots_dir = Path(plots_dir) if plots_dir is not None else Path("plots")

    def save_results(
        self,
        current_knobs: dict[str, float],
        uncertainties: np.ndarray,
        true_strengths: dict[str, float],
    ) -> None:
        """Write final knob strengths and markdown table to file."""
        LOGGER.info("Writing final knob strengths and markdown table...")
        save_results(self.knob_names, current_knobs, uncertainties, self.output_knobs_path)

        # Prepare rows with index, knob, true, final, diff, relative difference, and uncertainty.
        rows = []
        for idx, knob in enumerate(self.knob_names):
            true_val = true_strengths.get(knob, np.nan)
            final_val = current_knobs[knob]
            diff = final_val - true_val
            rel_diff = diff / true_val if true_val != 0 else 0
            uncertainty_val = uncertainties[idx]
            rows.append(
                {
                    "index": idx,
                    "knob": knob,
                    "true": true_val,
                    "final": final_val,
                    "diff": diff,
                    "reldiff": rel_diff,
                    "uncertainty": uncertainty_val,
                    "rel_uncertainty": uncertainty_val / abs(true_val) if true_val != 0 else 0,
                }
            )

        # Order rows by relative difference (descending order)
        rows.sort(key=lambda row: abs(row["reldiff"]), reverse=True)

        with self.knob_table_path.open("w") as f:
            f.write(
                "| Index |   Knob   |   True   |   Final   |   Diff   | Uncertainty | Relative Diff | Relative Uncertainty |\n"
                "|-------|----------|----------|----------|----------|-------------|---------------|----------------------|\n"
            )
            for row in rows:
                f.write(
                    f"|{row['index']}|{row['knob']}|"
                    f"{scientific_notation(row['true'])}|"
                    f"{scientific_notation(row['final'])}|"
                    f"{scientific_notation(row['diff'])}|"
                    f"{scientific_notation(row['uncertainty'])}|"
                    f"{scientific_notation(row['reldiff'])}|"
                    f"{scientific_notation(row['rel_uncertainty'])}|\n"
                )
        LOGGER.info("Results saved successfully.")

    def generate_plots(
        self,
        current_knobs: dict[str, float],
        initial_strengths: np.ndarray,
        true_strengths: dict[str, float],
        uncertainties: np.ndarray,
    ) -> None:
        """Generate all plotting results using the plotting module."""
        LOGGER.info("Generating plots...")
        quad_unc = uncertainties.copy()
        knob_names = self.knob_names.copy()
        if self.simulation_config.optimise_energy:
            knob_names.remove("deltap")
            quad_unc = quad_unc[:-1]  # Remove uncertainty for deltap

        magnet_names = [knob[:-3] for knob in knob_names]
        initial_vals = np.array([initial_strengths[i] for i in range(len(knob_names))])
        final_vals = np.array([current_knobs[k] for k in knob_names])
        true_vals = np.array([true_strengths.get(k, np.nan) for k in knob_names])

        # Ensure plots directory exists
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        save_prefix = f"{self.plots_dir}/"
        show_errorbars = True

        if (
            self.simulation_config.optimise_quadrupoles or self.simulation_config.optimise_bends
        ):  # Relative difference comparison
            plot_strengths_comparison(
                magnet_names,
                final_vals,
                true_vals,
                quad_unc,
                initial_vals=initial_vals,
                show_errorbars=show_errorbars,
                plot_real=False,
                save_path=f"{save_prefix}relative_difference_comparison.png",
                unit="$m^{-1}$",
            )

            plot_strengths_vs_position(
                self.elem_spos,
                final_vals,
                true_vals,
                quad_unc,
                initial_vals=initial_vals,
                show_errorbars=show_errorbars,
                plot_real=False,
                save_path=f"{save_prefix}relative_difference_vs_position_comparison.png",
                magnet_names=magnet_names,
            )

        if "deltap" in current_knobs:  # Energy knob was optimised
            plot_deltap_comparison(
                true_strengths.get("deltap", 0),
                current_knobs["deltap"],
                uncertainties[-1],
            )
        LOGGER.info(f"Plots saved to {self.plots_dir.resolve()}")

        if self.show_plots:
            show_plots()
