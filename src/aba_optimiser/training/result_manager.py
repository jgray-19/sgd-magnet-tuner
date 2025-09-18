"""Result handling for the optimisation controller."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import KNOB_TABLE, OUTPUT_KNOBS
from aba_optimiser.io.utils import save_results, scientific_notation
from aba_optimiser.plotting.strengths import (
    plot_deltap_comparison,
    plot_strengths_comparison,
    plot_strengths_vs_position,
)
from aba_optimiser.plotting.utils import show_plots

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings


LOGGER = logging.getLogger(__name__)


class ResultManager:
    """Manages result processing and output generation."""

    def __init__(
        self,
        knob_names: list[str],
        elem_spos: np.ndarray,
        opt_settings: OptSettings,
        show_plots: bool = True,
    ):
        self.knob_names = knob_names
        self.elem_spos = elem_spos
        self.show_plots = show_plots
        self.opt_settings = opt_settings

    def save_results(
        self,
        current_knobs: dict[str, float],
        uncertainties: np.ndarray,
        true_strengths: dict[str, float],
    ) -> None:
        """Write final knob strengths and markdown table to file."""
        LOGGER.info("Writing final knob strengths and markdown table...")
        save_results(self.knob_names, current_knobs, uncertainties, OUTPUT_KNOBS)

        # Prepare rows with index, knob, true, final, diff, relative difference, and uncertainty.
        rows = []
        for idx, knob in enumerate(self.knob_names):
            true_val = true_strengths[knob]
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
                    "rel_uncertainty": uncertainty_val / abs(true_val)
                    if true_val != 0
                    else 0,
                }
            )

        # Order rows by relative difference (descending order)
        rows.sort(key=lambda row: abs(row["reldiff"]), reverse=True)

        with Path(KNOB_TABLE).open("w") as f:
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
        knob_names_wo_dpp = self.knob_names[:-1]
        quad_unc = uncertainties[:-1]

        magnet_names = [knob[:-3] for knob in knob_names_wo_dpp]
        initial_vals = np.array(
            [initial_strengths[i] for i in range(len(knob_names_wo_dpp))]
        )
        final_vals = np.array([current_knobs[k] for k in knob_names_wo_dpp])
        true_vals = np.array([true_strengths[k] for k in knob_names_wo_dpp])

        save_prefix = "plots/"
        show_errorbars = True

        if not self.opt_settings.only_energy:  # Relative difference comparison
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

        deltap_key = "deltap"  # self.knob_names[-1]
        if current_knobs[deltap_key] != 0:
            plot_deltap_comparison(
                true_strengths[deltap_key],
                current_knobs[deltap_key],
                uncertainties[-1],
            )
        else:
            LOGGER.info("Skipping delta-p plot as the final value is zero.")
        if self.show_plots:
            show_plots()
