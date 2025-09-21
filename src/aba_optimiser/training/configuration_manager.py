"""Configuration and setup management for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    BPM_START_POINTS,
    MAGNET_RANGE,
    RUN_ARC_BY_ARC,
    SEQUENCE_FILE,
    USE_NOISY_DATA,
)
from aba_optimiser.mad.mad_interface import OptimizationMadInterface
from aba_optimiser.workers.arc_by_arc import ArcByArcWorker
from aba_optimiser.workers.ring import RingWorker

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings
    from aba_optimiser.workers.base_worker import BaseWorker


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

    def __init__(self, opt_settings: OptSettings):
        self.mad_iface: OptimizationMadInterface | None = None
        self.knob_names: list[str] = []
        self.elem_spos: np.ndarray = np.array([])
        self.all_bpms: np.ndarray = np.array([])
        self.start_bpms: list[str] = []
        self.Worker: type[BaseWorker] | None = None
        self.initial_strengths: np.ndarray = np.array([])

        self.global_config = opt_settings

    def setup_mad_interface(self) -> None:
        """Initialise the MAD-NG interface and get basic model parameters."""
        self.mad_iface = OptimizationMadInterface(
            SEQUENCE_FILE,
            MAGNET_RANGE,
            opt_settings=self.global_config,
            use_real_strengths="noisy" if USE_NOISY_DATA else True,
        )
        self.knob_names = self.mad_iface.knob_names

        self.elem_spos = self.mad_iface.elem_spos
        tws = self.mad_iface.run_twiss()
        LOGGER.info(f"Found tunes: {tws['q1']}, {tws['q2']}")
        self.all_bpms = tws.index.to_numpy()

    def determine_worker_and_bpms(self) -> None:
        """Determine the worker type and BPM start points based on the run mode."""
        if RUN_ARC_BY_ARC:
            LOGGER.warning(
                "Arc by arc chosen, ignoring N_RUN_TURNS, OBSERVE_TURNS_FROM, N_COMPARE_TURNS"
            )
            for bpm in BPM_START_POINTS:
                if bpm not in self.all_bpms:
                    raise ValueError(f"BPM {bpm} not found in the sequence.")
            self.start_bpms = BPM_START_POINTS
            self.Worker = ArcByArcWorker
        else:
            LOGGER.warning(
                "Whole ring chosen, BPM start points ignored, taking an even distribution based on NUM_WORKERS"
            )
            self.start_bpms, _ = circular_far_samples(
                self.all_bpms, min(self.global_config.num_workers, len(self.all_bpms))
            )
            self.Worker = RingWorker

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
        if RUN_ARC_BY_ARC:
            filtered_true_strengths = {
                knob: true_strengths[knob] for knob in self.knob_names
            }
        else:
            missing = set(self.knob_names) ^ set(true_strengths)
            if missing:
                raise ValueError(
                    f"Mismatch between model knobs and true strengths: {missing}"
                )
            filtered_true_strengths = true_strengths

        return current_knobs, filtered_true_strengths

    def calculate_n_data_points(self) -> dict[str, int]:
        """Calculate number of data points for each BPM start point."""
        if self.Worker is None:
            raise ValueError("Worker type must be determined first")

        n_data_points = {}
        for start_bpm in self.start_bpms:
            bpm_range = self.Worker.get_bpm_range(start_bpm)
            n_bpms = self.mad_iface.count_bpms(bpm_range)
            n_data_points[start_bpm] = self.Worker.get_n_data_points(n_bpms)
        return n_data_points
