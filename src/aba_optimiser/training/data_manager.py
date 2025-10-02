"""Data management for the optimisation."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import (
    CLEAN_DATA,
    CLEANED_FILE,
    DIFFERENT_TURNS_PER_RANGE,
    EMINUS_NOISY_FILE,
    EMINUS_NONOISE_FILE,
    EPLUS_NOISY_FILE,
    EPLUS_NONOISE_FILE,
    FILE_COLUMNS,
    FLATTOP_TURNS,
    N_COMPARE_TURNS,
    N_RUN_TURNS,
    NO_NOISE_FILE,
    NOISY_FILE,
    NUM_TRACKS,
    POSITION_STD_DEV,
    RUN_ARC_BY_ARC,
    USE_NOISY_DATA,
)
from aba_optimiser.dataframes.utils import select_markers

if TYPE_CHECKING:
    import numpy as np

    from aba_optimiser.config import OptSettings
    from aba_optimiser.training.configuration_manager import ConfigurationManager

LOGGER = logging.getLogger(__name__)
cols_to_read = FILE_COLUMNS


class DataManager:
    """Manages track data loading and processing for optimisation."""

    def __init__(
        self,
        all_bpms: np.ndarray,
        opt_settings: OptSettings,
    ):
        self.all_bpms = all_bpms
        self.opt_settings = opt_settings

        # Available global "turn" ids (already include offsets if sextupoles are on)
        total_files = 3 if opt_settings.use_off_energy_data else 1
        self.available_turns: list[int] = list(
            range(1, FLATTOP_TURNS * NUM_TRACKS * total_files + 1)
        )

        self.turn_batches: list[list[int]] = []
        self.tracks_per_worker: int = 0  # set in prepare_turn_batches

        # For the Hessian
        self.var_x = POSITION_STD_DEV**2
        self.var_y = POSITION_STD_DEV**2

        # Only populated when sextupoles are optimised
        self.track_data: dict[str, pd.DataFrame] | None = None
        self.energy_map: dict[int, str] | None = None  # {turn -> "plus"|"minus"|"zero"}

    # ---------- Internals ----------

    def _reduce_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["turn"] = df["turn"].astype("int32", copy=False)
        df["name"] = df["name"].astype("category", copy=False)
        df["kick_plane"] = df["kick_plane"].astype("category", copy=False)
        # Copy because we drop non-selected markers and convert from view.
        return select_markers(df, self.all_bpms).copy()

    def _read_parquet(
        self, source: str, needed_turns: set[int] | None, offset: int
    ) -> pd.DataFrame:
        """Read a parquet with optional turn filtering and column validation."""
        if needed_turns:
            filtered_turns = [t - offset for t in needed_turns]
            filters = [("turn", "in", filtered_turns)]
            df = pd.read_parquet(source, columns=cols_to_read, filters=filters)
            df["turn"] += offset  # Create global ids
        else:
            df = pd.read_parquet(source, columns=cols_to_read)

        missing = [c for c in cols_to_read if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in track data: {missing}")
        return df

    def _select_file_0e(self) -> str:
        """Select the appropriate zero-energy file based on noise and filtering settings."""
        if USE_NOISY_DATA:
            if CLEAN_DATA:
                LOGGER.debug("Using filtered data for zero-energy tracks")
                return CLEANED_FILE

            LOGGER.debug("Using noisy data for zero-energy tracks")
            return NOISY_FILE

        LOGGER.debug("Using non-noisy data for zero-energy tracks")
        return NO_NOISE_FILE

    # ---------- Public API ----------

    def load_track_data(
        self, needed_turns: set[int] | None = None, use_off_energy_data: bool = False
    ) -> None:
        """Load track data for optimisation and build energy map.

        Note: When use_off_energy_data=False, only the 'zero' file is loaded.
        """
        LOGGER.info(
            "Loading energy track data (custom turns=%s, use_off_energy_data=%s)...",
            needed_turns is not None,
            use_off_energy_data,
        )

        # Pick source files based on noise setting
        sources = {
            "plus": EPLUS_NOISY_FILE if USE_NOISY_DATA else EPLUS_NONOISE_FILE,
            "minus": EMINUS_NOISY_FILE if USE_NOISY_DATA else EMINUS_NONOISE_FILE,
            "zero": self._select_file_0e(),
        }

        # Turn offsets per energy type (global turn space)
        offsets = {
            "zero": 0,
            "minus": FLATTOP_TURNS * NUM_TRACKS,
            "plus": 2 * FLATTOP_TURNS * NUM_TRACKS,
        }

        # Which energies to load
        energies = ("plus", "minus", "zero") if use_off_energy_data else ("zero",)

        # Load and reduce
        energy_tracks: dict[str, pd.DataFrame] = {}
        for e in energies:
            df = self._read_parquet(sources[e], needed_turns, offsets[e])
            energy_tracks[e] = self._reduce_dataframe(df)

        self.track_data = (
            energy_tracks if use_off_energy_data else {"zero": energy_tracks["zero"]}
        )

        # Build a fast energy map {turn -> energy}, using unique turn sets
        zero_turns = set(self.track_data["zero"]["turn"].unique())
        if use_off_energy_data:
            plus_turns = set(self.track_data["plus"]["turn"].unique())
            minus_turns = set(self.track_data["minus"]["turn"].unique())
        else:
            plus_turns = set()
            minus_turns = set()

        for df in self.track_data.values():
            df.set_index(["turn", "name"], inplace=True)

        # Priority: plus > minus > zero (disjoint in practice if offsets are correct)
        energy_map: dict[int, str] = {}
        for t in plus_turns | minus_turns | zero_turns:
            if t in plus_turns:
                energy_map[t] = "plus"
            elif t in minus_turns:
                energy_map[t] = "minus"
            else:
                energy_map[t] = "zero"

        self.energy_map = energy_map

        LOGGER.info(
            "Loaded track data: zero=%d, minus=%d, plus=%d unique turns",
            len(zero_turns),
            len(minus_turns),
            len(plus_turns),
        )

    def prepare_turn_batches(self, config_manager: ConfigurationManager) -> None:
        """Build the list of turns to be processed and validate availability."""
        LOGGER.info("Preparing turn batches for worker distribution")

        if not RUN_ARC_BY_ARC:
            # Drop the last N_RUN_TURNS in RING mode
            LOGGER.debug(
                "Removing last %d turns from available turns for RING mode",
                N_RUN_TURNS + 1,
            )
            self.available_turns = self.available_turns[: -(N_RUN_TURNS + 2)]
        else:
            to_remove = [(i + 1) * FLATTOP_TURNS for i in range(NUM_TRACKS)]
            self.available_turns = [
                t for t in self.available_turns if t not in to_remove
            ]

        num_turns_needed = self.opt_settings.total_tracks
        num_workers = self.opt_settings.num_workers
        tracks_per_worker = self.opt_settings.tracks_per_worker
        LOGGER.debug(
            "Need %d turns for %d workers with %d tracks each",
            num_turns_needed,
            num_workers,
            tracks_per_worker,
        )

        # If sextupoles are optimised, we effectively have 3 files worth of turns
        total_available_turns = len(self.available_turns)

        if total_available_turns < num_turns_needed:
            raise ValueError(
                f"Not enough available turns. Need {num_turns_needed}, "
                f"but found {total_available_turns}."
            )

        # Cap tracks per worker at configured TRACKS_PER_WORKER
        self.tracks_per_worker = min(
            total_available_turns // num_workers,
            tracks_per_worker,
        )

        if not DIFFERENT_TURNS_PER_RANGE:
            num_ranges = len(config_manager.bpm_ranges)
            self.tracks_per_worker *= num_ranges
            num_workers = num_workers // num_ranges

        # Randomly select turns for each batch
        random.shuffle(self.available_turns)
        LOGGER.info("Each worker will process %d turns", self.tracks_per_worker)
        self.turn_batches = [
            self.available_turns[
                i * self.tracks_per_worker : (i + 1) * self.tracks_per_worker
            ]
            for i in range(num_workers)
        ]

    def get_total_turns(self) -> int:
        """Calculate total number of turns to process."""
        n_compare = N_COMPARE_TURNS if not RUN_ARC_BY_ARC else 1
        return self.opt_settings.num_workers * self.tracks_per_worker * n_compare
