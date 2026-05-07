"""Result handling for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.io.utils import save_results, scientific_notation

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator


LOGGER = logging.getLogger(__name__)


class ResultManager:
    """Manages result processing and output generation."""

    def __init__(
        self,
        knob_names: list[str],
        elem_spos: list[float],
        accelerator: Accelerator,
        output_knobs_path: Path | None = None,
        knob_table_path: Path | None = None,
        include_uncertainty: bool = True,
    ):
        """Initialise result manager.

        Args:
            knob_names: List of knob names
            elem_spos: Element s-positions
            accelerator: Accelerator instance for machine-specific info
            output_knobs_path: Path to save final knobs
            knob_table_path: Path to save knob table
            include_uncertainty: Whether to include uncertainty error bars
        """
        self.knob_names = knob_names
        self.elem_spos = elem_spos
        self.accelerator = accelerator
        self.output_knobs_path = output_knobs_path
        self.knob_table_path = knob_table_path
        self.include_uncertainty = include_uncertainty

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
        if self.knob_table_path is not None:
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

