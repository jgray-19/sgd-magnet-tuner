"""Data management for the optimisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import (
    ACD_ON,
    FLATTOP_TURNS,
    N_COMPARE_TURNS,
    N_RUN_TURNS,
    NOISE_FILE,
    NUM_TRACKS,
    NUM_WORKERS,
    POSITION_STD_DEV,
    RAMP_UP_TURNS,
    RUN_ARC_BY_ARC,
    TRACK_DATA_FILE,
    TRACKS_PER_WORKER,
    USE_NOISY_DATA,
)
from aba_optimiser.dataframes.utils import select_markers

if TYPE_CHECKING:
    import numpy as np

LOGGER = logging.getLogger(__name__)


class DataManager:
    """Manages track data loading and processing for optimisation."""

    def __init__(self, all_bpms: np.ndarray, bpm_start_points: list[str]):
        self.all_bpms = all_bpms
        self.bpm_start_points = bpm_start_points
        self.available_turns: list[int] = list(range(1, FLATTOP_TURNS * NUM_TRACKS + 1))

        # Will be set later
        self.comparison_data: pd.DataFrame | None = None
        self.turn_batches: list[list[int]] = []

        # For the hessian
        self.var_x = (POSITION_STD_DEV) ** 2
        self.var_y = (POSITION_STD_DEV) ** 2

    def load_track_data(self, needed_turns: set[int] | None = None) -> None:
        """Load and pre-process track/comparison data.

        If `needed_turns` is provided, read only those turns from the parquet to
        minimise memory usage.
        """
        LOGGER.info("Loading track data (filtered=%s)...", needed_turns is not None)

        source = NOISE_FILE if USE_NOISY_DATA else TRACK_DATA_FILE

        # Only read required columns; if a turn filter is supplied use parquet filters
        cols = ["turn", "name", "x", "px", "y", "py"]
        if needed_turns:
            # pandas accepts a list of tuples for filters; provide the in-list
            filters = [("turn", "in", list(needed_turns))]
            track_data = pd.read_parquet(source, columns=cols, filters=filters)
        else:
            track_data = pd.read_parquet(source, columns=cols)

        missing = [c for c in cols if c not in track_data.columns]
        if missing:
            raise ValueError(f"Missing columns in track data: {missing}")

        # Downcast and optimise memory
        track_data["turn"] = track_data["turn"].astype("int32", copy=False)
        track_data["name"] = track_data["name"].astype("category", copy=False)

        # Filter data to BPMs present in the sequence
        self.comparison_data = select_markers(track_data, self.all_bpms)
        del track_data

    def prepare_turn_batches(self) -> None:
        """Build the list of turns to be processed and validate availability."""
        LOGGER.info("Preparing turn batches for worker distribution")

        if self.comparison_data is None and self.available_turns is None:
            raise ValueError("Track data (or preview) must be loaded first")

        if not RUN_ARC_BY_ARC:
            # Remove the last N_RUN_TURNS
            LOGGER.debug(
                f"Removing last {N_RUN_TURNS} turns from available turns for RING mode"
            )
            self.available_turns = self.available_turns[:-N_RUN_TURNS]

        num_turns_needed = TRACKS_PER_WORKER * NUM_WORKERS // len(self.bpm_start_points)
        LOGGER.debug(
            f"Need {num_turns_needed} turns for {NUM_WORKERS} workers with {TRACKS_PER_WORKER} tracks each"
        )

        if len(self.available_turns) < num_turns_needed:
            raise ValueError(
                f"Not enough available turns. Need {num_turns_needed}, "
                f"but found {len(self.available_turns)} turns."
            )

        if ACD_ON:
            original_count = len(self.available_turns)
            self.available_turns = [
                turn for turn in self.available_turns if turn >= RAMP_UP_TURNS
            ]
            LOGGER.debug(
                f"ACD filtering: {original_count} -> {len(self.available_turns)} turns"
            )

        # Determine how many turns each batch should contain
        tracks_per_worker = min(
            len(self.available_turns) // NUM_WORKERS, TRACKS_PER_WORKER
        )
        num_batches = NUM_WORKERS // len(self.bpm_start_points)
        self.turn_batches = [
            self.available_turns[i * tracks_per_worker : (i + 1) * tracks_per_worker]
            for i in range(num_batches)
        ]

    def get_total_turns(self) -> int:
        """Calculate total number of turns to process."""
        n_compare = N_COMPARE_TURNS if not RUN_ARC_BY_ARC else 1
        tracks_per_worker = len(self.turn_batches[0]) if self.turn_batches else 0

        return NUM_WORKERS * tracks_per_worker * n_compare

    def get_indexed_comparison_data(self) -> pd.DataFrame:
        """Get comparison data with multi-index for efficient lookups."""
        if self.comparison_data is None:
            raise ValueError("Track data must be loaded first")
        return self.comparison_data.set_index(["turn", "name"])
