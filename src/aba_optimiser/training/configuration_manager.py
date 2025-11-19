"""Configuration and setup management for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    BEAM_ENERGY,
)
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.workers.arc_by_arc import ArcByArcWorker

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings


LOGGER = logging.getLogger(__name__)


def circular_far_samples(
    arr: np.ndarray, k: int, start_offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick k indices as far apart as possible on a circular array.
    k must be <= len(arr). start_offset rotates the selection.
    """
    n = len(arr)
    if k > n:
        raise ValueError("k must be <= len(arr)")
    step = n / k
    # bin centers → unique indices even when step is non-integer
    idx = (np.floor((np.arange(k) + 0.5) * step + start_offset) % n).astype(int)
    return arr[idx], idx


class ConfigurationManager:
    """Manages configuration and setup for the optimisation process."""

    def __init__(
        self,
        opt_settings: OptSettings,
        magnet_range: str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
    ):
        self.mad_iface: OptimisationMadInterface | None = None
        self.knob_names: list[str] = []
        self.elem_spos: np.ndarray = np.array([])
        self.all_bpms: np.ndarray = np.array([])
        self.initial_strengths: np.ndarray = np.array([])

        self.start_bpms = bpm_start_points
        self.end_bpms = bpm_end_points
        self.magnet_range = magnet_range
        self.global_config = opt_settings
        self.bpm_ranges = [
            s + "/" + e for s in bpm_start_points for e in bpm_end_points
        ]

    def setup_mad_interface(
        self,
        sequence_file_path: str,
        bad_bpms: list[str] | None,
        seq_name: str | None = None,
        beam_energy: float = BEAM_ENERGY,
    ) -> None:
        """Initialise the MAD-NG interface and get basic model parameters."""
        self.mad_iface = OptimisationMadInterface(
            sequence_file_path,
            seq_name=seq_name,
            magnet_range=self.magnet_range,
            opt_settings=self.global_config,
            corrector_strengths=None,
            tune_knobs_file=None,
            bad_bpms=bad_bpms,
            beam_energy=beam_energy,
        )
        self.knob_names = self.mad_iface.knob_names

        self.elem_spos = self.mad_iface.elem_spos
        self.all_bpms = self.mad_iface.all_bpms

    def determine_worker_and_bpms(self) -> None:
        """Determine the worker type and BPM start points based on the run mode."""
        for bpm in self.start_bpms:
            if bpm not in self.all_bpms:
                raise ValueError(f"BPM {bpm} not found in the sequence.")

    def initialise_knob_strengths(
        self,
        true_strengths: dict[str, float],
        provided_initial_knobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Initialise knob strengths from MAD and filter true strengths."""
        if self.mad_iface is None:
            raise ValueError("MAD interface must be setup first")

        if provided_initial_knobs is not None:
            # Use provided initial knobs where available, fill missing ones from MAD
            LOGGER.info(
                "Using provided initial knob strengths from previous optimization"
            )
            # First get the default MAD values for all knobs
            self.mad_iface.update_knob_values(provided_initial_knobs)
        initial_strengths = self.mad_iface.receive_knob_values()

        self.initial_strengths = initial_strengths
        current_knobs = dict(zip(self.knob_names, initial_strengths))

        # Restrict true strengths to knobs we actually have in model
        if len(true_strengths) == 0:
            LOGGER.warning("No true strengths provided, skipping filtering")
            filtered_true_strengths = {}
        else:
            filtered_true_strengths = {
                knob: true_strengths[knob] for knob in self.knob_names
            }
        return current_knobs, filtered_true_strengths

    def calculate_n_data_points(self) -> dict[str, int]:
        """Calculate number of data points for each BPM start point."""
        n_data_points = {}
        for bpm_range in self.bpm_ranges:
            n_bpms, _ = self.mad_iface.count_bpms(bpm_range)
            n_data_points[bpm_range] = ArcByArcWorker.get_n_data_points(n_bpms)
            logging.info(f"{bpm_range}: {n_data_points[bpm_range]} data points")
        return n_data_points
