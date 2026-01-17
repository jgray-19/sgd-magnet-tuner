"""Data management for the optimisation."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.dataframes.utils import select_markers

if TYPE_CHECKING:
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.configuration_manager import ConfigurationManager

LOGGER = logging.getLogger(__name__)
cols_to_read = FILE_COLUMNS


class DataManager:
    """Manages track data loading and processing for optimisation."""

    def __init__(
        self,
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        measurement_files: list[str],
        num_bunches: int,
        flattop_turns: int,
    ):
        self.all_bpms = all_bpms
        self.simulation_config = simulation_config
        self.measurement_files = measurement_files
        self.num_bunches = num_bunches
        self.flattop_turns = flattop_turns

        # Available turns will be populated after loading track data
        self.available_turns: list[int]

        self.turn_batches: list[list[int]]
        self.tracks_per_worker: int  # set in prepare_turn_batches
        self.num_workers: int  # set in prepare_turn_batches

        # Track data per measurement file (indexed by file index)
        self.track_data: dict[int, pd.DataFrame]
        self.file_map: dict[int, int]  # {turn -> file_index}

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
            filters = [("turn", "in", filtered_turns), ("name", "in", self.all_bpms)]
            df = pd.read_parquet(source, columns=cols_to_read, filters=filters)
        else:
            df = pd.read_parquet(source, columns=cols_to_read)

        # Always apply offset to create global turn IDs
        df["turn"] = df["turn"] + offset

        missing = [c for c in cols_to_read if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in track data: {missing}")
        return df

    def _reorder_track_dataframes(self) -> None:
        """Reorder track dataframes to have turns in ascending order and BPMs in bpm_order."""
        for file_idx in self.track_data:
            all_turns = sorted(self.track_data[file_idx].index.get_level_values("turn").unique())
            # reduce bpm order to only those present in the data
            bpm_order_filtered = [
                bpm
                for bpm in self.all_bpms
                if bpm in self.track_data[file_idx].index.get_level_values("name")
            ]
            self.track_data[file_idx] = self.track_data[file_idx].reindex(
                pd.MultiIndex.from_product([all_turns, bpm_order_filtered], names=["turn", "name"])
            )

    # ---------- Public API ----------

    def load_track_data(self, needed_turns: set[int] | None = None) -> None:
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
                raise ValueError(
                    "measurement_files should not contain None - controller should have resolved defaults"
                )

        # Turn offsets per file (global turn space)
        offsets = {
            file_idx: file_idx * self.flattop_turns * self.num_bunches
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
            file_idx: set(self.track_data[file_idx]["turn"].unique()) for file_idx in file_tracks
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

        # Populate available_turns from all loaded turns
        all_turns = set()
        for turns in file_turn_sets.values():
            all_turns.update(turns)
        self.available_turns = sorted(all_turns)

        LOGGER.info(
            "Loaded track data: %s",
            ", ".join(
                f"file_{idx}={len(file_turn_sets[idx])} turns" for idx in sorted(file_tracks.keys())
            ),
        )

    def prepare_turn_batches(self, config_manager: ConfigurationManager) -> None:
        """Build the list of turns to be processed and validate availability."""
        if self.track_data is None:
            raise ValueError(
                "Track data must be loaded before preparing turn batches. Call load_track_data() first."
            )

        LOGGER.info("Preparing turn batches for worker distribution")

        # Remove boundary turns (first and last of each track)
        turns_to_remove = set()
        for file_idx, df in self.track_data.items():
            file_turns = sorted(df.index.get_level_values("turn").unique())
            LOGGER.debug(f"File {file_idx} has {len(file_turns)} turns")

            # Split into tracks and remove boundaries
            for track_idx in range(0, len(file_turns), self.flattop_turns):
                track_turns = file_turns[track_idx : track_idx + self.flattop_turns]
                if len(track_turns) > 1:
                    turns_to_remove.add(track_turns[0])  # First
                    turns_to_remove.add(track_turns[-1])  # Last

        self.available_turns = [t for t in self.available_turns if t not in turns_to_remove]
        LOGGER.info(
            f"Removed {len(turns_to_remove)} boundary turns, {len(self.available_turns)} available"
        )

        # Determine how many batches to create
        num_workers = self.simulation_config.num_workers
        num_files = len(self.track_data)
        tracks_per_worker = self.simulation_config.tracks_per_worker

        # Check if we have enough turns
        if len(self.available_turns) == 0:
            raise ValueError(
                "No turns available after removing boundary turns. Check that your flattop_turns setting leaves at least one turn per track."
            )

        if len(self.available_turns) < tracks_per_worker:
            LOGGER.warning(
                f"Only {len(self.available_turns)} turns available, but tracks_per_worker={tracks_per_worker}. "
                f"Reducing tracks_per_worker to {len(self.available_turns)}."
            )
            tracks_per_worker = len(self.available_turns)

        # Start with requested number, but ensure we use all files (for different deltaps)
        num_batches = max(num_workers, num_files)

        # Cap by available turns
        max_batches = len(self.available_turns) // tracks_per_worker
        num_batches = min(num_batches, max(1, max_batches))

        # Use configured or adjusted tracks_per_worker
        self.tracks_per_worker = tracks_per_worker

        num_starts = len(config_manager.start_bpms)
        num_ends = len(config_manager.end_bpms)
        actual_workers = num_batches * (num_starts + num_ends)
        LOGGER.info(
            f"Creating {num_batches} batches x ({num_starts} starts + {num_ends} ends) = {actual_workers} workers, {self.tracks_per_worker} turns/worker"
        )

        # Organise turns by file, then create batches round-robin
        turns_by_file: dict[int, list[int]] = {}
        for turn in self.available_turns:
            turns_by_file.setdefault(self.file_map[turn], []).append(turn)

        for turns in turns_by_file.values():
            random.shuffle(turns)

        # Create batches by round-robin from files
        self.turn_batches = []
        file_indices = sorted(turns_by_file.keys())
        file_idx_pos = 0  # Track which file to take from next

        for _ in range(num_batches):
            # Try each file starting from current position
            found = False
            for _ in range(len(file_indices)):
                if file_idx_pos >= len(file_indices):
                    file_idx_pos = 0

                file_idx = file_indices[file_idx_pos]
                if (
                    file_idx in turns_by_file
                    and len(turns_by_file[file_idx]) >= self.tracks_per_worker
                ):
                    # Take batch from this file
                    batch = turns_by_file[file_idx][: self.tracks_per_worker]
                    turns_by_file[file_idx] = turns_by_file[file_idx][self.tracks_per_worker :]
                    if not turns_by_file[file_idx]:
                        del turns_by_file[file_idx]
                        file_indices.remove(file_idx)
                    else:
                        file_idx_pos += 1
                    self.turn_batches.append(batch)
                    found = True
                    break
                file_idx_pos += 1

            if not found:
                LOGGER.warning(f"Only created {len(self.turn_batches)}/{num_batches} batches")
                break

        if len(self.turn_batches) == 0:
            raise ValueError(
                f"Failed to create any batches. Available turns: {len(self.available_turns)}, "
                f"required tracks_per_worker: {self.tracks_per_worker}. "
                "Consider reducing tracks_per_worker or increasing flattop_turns."
            )

        self.num_workers = len(self.turn_batches)
        LOGGER.info(f"Created {self.num_workers} batches from {len(self.track_data)} files")

        total_available_turns = len(self.available_turns)
        total_used_turns = sum(len(batch) for batch in self.turn_batches)
        unused_turns = total_available_turns - total_used_turns
        unused_percentage = (
            (unused_turns / total_available_turns * 100) if total_available_turns > 0 else 0
        )
        LOGGER.info(
            f"Unused turns: {unused_turns} out of {total_available_turns} ({unused_percentage:.1f}%)"
        )

    def get_total_turns(self) -> int:
        """Calculate total number of turns to process."""
        return self.num_workers * self.tracks_per_worker
