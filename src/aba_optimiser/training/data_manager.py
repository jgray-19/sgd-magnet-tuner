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
        bpms_in_range: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        measurement_files: list[str],
        num_bunches: int,
        flattop_turns: int,
    ):
        self.all_bpms = all_bpms
        self.bpms_in_range = bpms_in_range
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
        self.file_kick_planes: dict[int, str]

    # ---------- Internals ----------

    def _reduce_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["turn"] = df["turn"].astype("int32", copy=False)
        df["name"] = df["name"].astype("category", copy=False)
        # Copy because we drop non-selected markers and convert from view.
        return select_markers(df, self.bpms_in_range).copy()

    def _read_parquet(
        self, source: str, needed_turns: set[int] | None, offset: int
    ) -> pd.DataFrame:
        """Read a parquet with optional turn filtering and column validation."""
        if needed_turns:
            filtered_turns = [t - offset for t in needed_turns]
            filters = [("turn", "in", filtered_turns), ("name", "in", self.bpms_in_range)]
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
                for bpm in self.bpms_in_range
                if bpm in self.track_data[file_idx].index.get_level_values("name")
            ]
            self.track_data[file_idx] = self.track_data[file_idx].reindex(
                pd.MultiIndex.from_product([all_turns, bpm_order_filtered], names=["turn", "name"])
            )

    @staticmethod
    def _plane_span(df: pd.DataFrame, coord: str, momentum: str) -> float:
        coord_values = df[coord].dropna().to_numpy(dtype="float64", copy=False)
        momentum_values = df[momentum].dropna().to_numpy(dtype="float64", copy=False)
        coord_span = (
            float(coord_values.max() - coord_values.min()) if coord_values.size else 0.0
        )
        momentum_span = (
            float(momentum_values.max() - momentum_values.min()) if momentum_values.size else 0.0
        )
        return max(coord_span, momentum_span)

    @classmethod
    def infer_kick_plane(
        cls,
        df: pd.DataFrame,
        *,
        dominance_ratio: float = 10.0,
        minimum_span: float = 1e-12,
    ) -> str:
        """Infer whether a file is excited in x, y, or both planes."""
        x_span = cls._plane_span(df, "x", "px")
        y_span = cls._plane_span(df, "y", "py")

        if x_span <= minimum_span and y_span <= minimum_span:
            return "xy"
        if x_span <= minimum_span:
            return "y"
        if y_span <= minimum_span:
            return "x"

        ratio = max(x_span, y_span) / min(x_span, y_span)
        if ratio >= dominance_ratio:
            return "x" if x_span > y_span else "y"
        return "xy"

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
        file_kick_planes: dict[int, str] = {}
        for file_idx, source in enumerate(sources):
            LOGGER.info(f"Loading file {file_idx}: {source}")
            df = self._read_parquet(source, needed_turns, offsets[file_idx])
            file_tracks[file_idx] = self._reduce_dataframe(df)
            file_kick_planes[file_idx] = self.infer_kick_plane(file_tracks[file_idx])
            LOGGER.info(
                "File %d kick-plane classification: %s",
                file_idx,
                file_kick_planes[file_idx],
            )

        # Handle NaN values in track data coordinate-by-coordinate.
        # This is important for single-plane BPMs where one plane is intentionally missing:
        # mark only that coordinate as zero-weight (variance=inf), not the whole row.
        coord_to_var = {
            "x": "var_x",
            "y": "var_y",
            "px": "var_px",
            "py": "var_py",
        }
        for file_idx, df in file_tracks.items():
            nan_info = {col: df[col].isna().sum() for col in coord_to_var}
            # Inspect a representative SPS BPM before the MultiIndex is applied.
            if file_idx == 0:
                bpm_mask = df["name"] == "BPV.13108"
                if bpm_mask.any():
                    bpm_nan_info = {col: df.loc[bpm_mask, col].isna().sum() for col in coord_to_var}
                    LOGGER.info(f"BPV.13108 NaN counts in file {file_idx}: {bpm_nan_info}")
                    LOGGER.info(
                        "BPV.13108 coordinate sample:\n%s",
                        df.loc[bpm_mask, ["x", "px", "y", "py"]].head(),
                    )
            for coord_col, var_col in coord_to_var.items():
                nan_mask = df[coord_col].isna()
                if nan_mask.any():
                    df.loc[nan_mask, var_col] = float("inf")
                    df.loc[nan_mask, coord_col] = 0.0

            if file_idx == 0:
                bpm_mask = df["name"] == "BPV.13108"
                if bpm_mask.any():
                    bpm_nan_info = {col: df.loc[bpm_mask, col].isna().sum() for col in coord_to_var}
                    LOGGER.info(f"BPV.13108 NaN counts in file {file_idx}: {bpm_nan_info}")
                    LOGGER.info(
                        "BPV.13108 coordinate sample:\n%s",
                        df.loc[bpm_mask, ["x", "px", "y", "py"]].head(),
                    )

            LOGGER.info(f"File {file_idx} loaded with {len(df)} rows, NaN counts: {nan_info}")

        self.track_data = file_tracks
        self.file_kick_planes = file_kick_planes

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
                f"file_{idx}={len(file_turn_sets[idx])} turns ({self.file_kick_planes[idx]})"
                for idx in sorted(file_tracks.keys())
            ),
        )

    def prepare_turn_batches(self, config_manager: ConfigurationManager) -> None:
        """Build the list of turns to be processed and validate availability."""
        if self.track_data is None:
            raise ValueError(
                "Track data must be loaded before preparing turn batches. Call load_track_data() first."
            )

        LOGGER.info("Preparing turn batches for worker distribution")

        # Remove boundary turns to guarantee enough turn context per selected start turn.
        # In arc-by-arc mode this keeps the historical first/last removal behaviour.
        turns_per_sample = (
            1 if self.simulation_config.run_arc_by_arc else self.simulation_config.n_run_turns
        )
        boundary_margin = max(1, turns_per_sample)
        turns_to_remove = set()
        for file_idx, df in self.track_data.items():
            file_turns = sorted(df.index.get_level_values("turn").unique())
            LOGGER.debug(f"File {file_idx} has {len(file_turns)} turns")

            # Split into tracks and remove boundaries
            for track_idx in range(0, len(file_turns), self.flattop_turns):
                track_turns = file_turns[track_idx : track_idx + self.flattop_turns]
                if len(track_turns) <= 2 * boundary_margin:
                    turns_to_remove.update(track_turns)
                else:
                    turns_to_remove.update(track_turns[:boundary_margin])
                    turns_to_remove.update(track_turns[-boundary_margin:])

        self.available_turns = [t for t in self.available_turns if t not in turns_to_remove]
        LOGGER.info(
            "Removed %d boundary turns (margin=%d, n_run_turns=%d), %d available",
            len(turns_to_remove),
            boundary_margin,
            self.simulation_config.n_run_turns,
            len(self.available_turns),
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
        num_starts = len(config_manager.start_bpms)
        num_ends = len(config_manager.end_bpms)
        max_workers = num_workers // num_starts // 2
        num_batches = max(max_workers, num_files)

        # Cap by available turns
        max_batches = len(self.available_turns) // tracks_per_worker
        num_batches = min(num_batches, max(1, max_batches))

        # Use configured or adjusted tracks_per_worker
        self.tracks_per_worker = tracks_per_worker

        if not self.simulation_config.run_arc_by_arc:
            actual_workers = num_batches * num_starts * 2
            LOGGER.info(
                f"Creating 2 directions x {num_batches} batches x {num_starts} start BPMs = {actual_workers} workers, {self.tracks_per_worker} turns/worker"
            )
        elif self.simulation_config.use_fixed_bpm:
            actual_workers = num_batches * (num_starts + num_ends)
            symbol = "+"
            LOGGER.info(
                f"Creating 2 directions x {num_batches} batches x ({num_starts} starts {symbol} {num_ends} ends) = {actual_workers * 2} workers, {self.tracks_per_worker} turns/worker"
            )
        else:
            actual_workers = num_batches * (num_starts * num_ends)
            symbol = "x"
            LOGGER.info(
                f"Creating 2 directions x {num_batches} batches x ({num_starts} starts {symbol} {num_ends} ends) = {actual_workers * 2} workers, {self.tracks_per_worker} turns/worker"
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
