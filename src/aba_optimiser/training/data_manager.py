"""Data management for the optimisation."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import (
    FILE_COLUMNS,
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
        measurement_files: list[str],
        bpm_order: list[str],
        num_tracks: int,
        flattop_turns: int,
    ):
        self.all_bpms = all_bpms
        self.opt_settings = opt_settings
        self.measurement_files = measurement_files
        self.bpm_order = bpm_order
        self.num_tracks = num_tracks
        self.flattop_turns = flattop_turns

        # Available global "turn" ids (now include offsets for multiple files)
        total_files = len(measurement_files)
        self.available_turns: list[int] = list(
            range(1, self.flattop_turns * self.num_tracks * total_files + 1)
        )

        self.turn_batches: list[list[int]] = []
        self.tracks_per_worker: int = 0  # set in prepare_turn_batches

        # Track data per measurement file (indexed by file index)
        self.track_data: dict[int, pd.DataFrame] | None = None
        self.file_map: dict[int, int] | None = None  # {turn -> file_index}

    # ---------- Internals ----------

    def _get_file_and_bunch_index(self, turn: int) -> tuple[int, int]:
        """Get the file index and bunch index for a given global turn number.
        
        Args:
            turn (int): Global turn number (1-indexed)
            
        Returns:
            tuple[int, int]: (file_index, bunch_index) where bunch_index is 0-indexed
        """
        # Adjust to 0-indexed for calculations
        turn_0indexed = turn - 1
        turns_per_file = self.flattop_turns * self.num_tracks
        file_idx = turn_0indexed // turns_per_file
        
        # Within file: determine which bunch (track) this turn belongs to
        turn_in_file = turn_0indexed % turns_per_file
        bunch_idx = turn_in_file // self.flattop_turns
        
        return file_idx, bunch_idx

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
            filters = [("turn", "in", filtered_turns), ("name", "in", self.all_bpms)]
            df = pd.read_parquet(source, columns=cols_to_read, filters=filters)
            df["turn"] += offset  # Create global ids
        else:
            df = pd.read_parquet(source, columns=cols_to_read)

        missing = [c for c in cols_to_read if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in track data: {missing}")
        return df

    def _reorder_track_dataframes(self) -> None:
        """Reorder track dataframes to have turns in ascending order and BPMs in bpm_order."""
        for file_idx in self.track_data:
            all_turns = sorted(
                self.track_data[file_idx].index.get_level_values("turn").unique()
            )
            # reduce bpm order to only those present in the data
            if self.bpm_order is not None:
                bpm_order_filtered = [
                    bpm
                    for bpm in self.bpm_order
                    if bpm in self.track_data[file_idx].index.get_level_values("name")
                ]
            else:
                bpm_order_filtered = sorted(
                    self.track_data[file_idx].index.get_level_values("name").unique()
                )
            self.track_data[file_idx] = self.track_data[file_idx].reindex(
                pd.MultiIndex.from_product(
                    [all_turns, bpm_order_filtered], names=["turn", "name"]
                )
            )

    # ---------- Public API ----------

    def load_track_data(
        self, needed_turns: set[int] | None = None
    ) -> None:
        """Load track data from all measurement files and build file map.

        Each measurement file gets a unique file index and corresponding turn offset.
        """
        LOGGER.info(
            "Loading track data from %d measurement file(s) (custom turns=%s)...",
            len(self.measurement_files),
            needed_turns is not None,
        )

        # Determine source files - controller has already resolved None to actual files
        sources = []
        for mf in self.measurement_files:
            if mf is not None:
                sources.append(mf)
            else:
                raise ValueError("measurement_files should not contain None - controller should have resolved defaults")

        # Turn offsets per file (global turn space)
        offsets = {
            file_idx: file_idx * self.flattop_turns * self.num_tracks
            for file_idx in range(len(sources))
        }

        # Load and reduce
        file_tracks: dict[int, pd.DataFrame] = {}
        for file_idx, source in enumerate(sources):
            LOGGER.info(f"Loading file {file_idx}: {source}")
            df = self._read_parquet(source, needed_turns, offsets[file_idx])
            file_tracks[file_idx] = self._reduce_dataframe(df)

        # Handle NaN values in track data
        for df in file_tracks.values():
            # For each row that contains NaNs in the x, y or px, py columns,
            # set the x_weight and y_weight to 0
            nan_mask = df[["x", "y", "px", "py"]].isna().any(axis=1)
            if nan_mask.any():
                raise ValueError(
                    "Found NaN values in track data. Please clean the data before proceeding."
                )

        self.track_data = file_tracks

        # Build a fast file map {turn -> file_index}
        file_turn_sets = {
            file_idx: set(self.track_data[file_idx]["turn"].unique())
            for file_idx in file_tracks.keys()
        }

        for df in self.track_data.values():
            df.set_index(["turn", "name"], inplace=True)

        self._reorder_track_dataframes()

        # Build file map
        file_map: dict[int, int] = {}
        for file_idx, turns in file_turn_sets.items():
            for t in turns:
                file_map[t] = file_idx

        self.file_map = file_map

        LOGGER.info(
            "Loaded track data: %s",
            ", ".join(
                f"file_{idx}={len(file_turn_sets[idx])} turns"
                for idx in sorted(file_tracks.keys())
            ),
        )

    def prepare_turn_batches(self, config_manager: ConfigurationManager) -> None:
        """Build the list of turns to be processed and validate availability."""
        LOGGER.info("Preparing turn batches for worker distribution")

        # Remove boundary turns from each bunch to ensure loop data availability
        # First and last turn in each bunch cannot be selected as they need prev/next turns
        total_files = len(self.measurement_files)
        turns_to_remove = []
        
        for file_idx in range(total_files):
            for bunch_idx in range(self.num_tracks):
                # Calculate global turn numbers for first and last turn in this bunch
                base_turn = file_idx * self.flattop_turns * self.num_tracks + bunch_idx * self.flattop_turns + 1
                first_turn = base_turn  # First turn in bunch (1-indexed)
                last_turn = base_turn + self.flattop_turns - 1  # Last turn in bunch
                turns_to_remove.extend([first_turn, last_turn])
        
        # Remove boundary turns
        self.available_turns = [t for t in self.available_turns if t not in turns_to_remove]

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

        num_ranges = len(config_manager.bpm_ranges)
        if not self.opt_settings.different_turns_per_range:
            self.tracks_per_worker *= num_ranges
            num_workers = num_workers // num_ranges

        num_workers = num_workers // 2  # /2 for sdir = +/-1
        self.tracks_per_worker *= 2

        num_workers = max(1, num_workers)  # Ensure at least one worker
        LOGGER.info(
            f"Adjusted tracks per worker to {self.tracks_per_worker} "
            f"and number of workers to {num_workers} "
        )
        if num_workers * num_ranges * 2 > 60:  # * 2 for sdir = +/-1
            LOGGER.warning(
                "Total number of workers (%d) exceeds 60, which may lead to resource issues.",
                num_workers * num_ranges * 2,
            )

        # Group turns by (file, bunch) and shuffle within groups for variety
        turns_by_group: dict[tuple[int, int], list[int]] = {}
        for turn in self.available_turns:
            file_idx, bunch_idx = self._get_file_and_bunch_index(turn)
            key = (file_idx, bunch_idx)
            turns_by_group.setdefault(key, []).append(turn)
        
        # Shuffle within each group for varied data
        for turns_list in turns_by_group.values():
            random.shuffle(turns_list)
        
        # Distribute turns: fill each batch from one group, cycling through groups
        self.turn_batches = []
        groups = list(turns_by_group.keys())
        group_idx = 0
        
        LOGGER.info("Each worker will process %d turns from %d (file, bunch) groups", 
                    self.tracks_per_worker, len(groups))
        
        for _ in range(num_workers):
            # Try to get a full batch from current group
            for _ in range(len(groups)):  # Try all groups
                if groups[group_idx] in turns_by_group:
                    turns_list = turns_by_group[groups[group_idx]]
                    if len(turns_list) >= self.tracks_per_worker:
                        batch = turns_list[:self.tracks_per_worker]
                        turns_by_group[groups[group_idx]] = turns_list[self.tracks_per_worker:]
                        if not turns_by_group[groups[group_idx]]:
                            del turns_by_group[groups[group_idx]]
                        self.turn_batches.append(batch)
                        group_idx = (group_idx + 1) % len(groups)
                        break
                group_idx = (group_idx + 1) % len(groups)
            else:
                # No group has enough turns left
                break

    def get_total_turns(self) -> int:
        """Calculate total number of turns to process."""
        return self.opt_settings.num_workers * self.tracks_per_worker
