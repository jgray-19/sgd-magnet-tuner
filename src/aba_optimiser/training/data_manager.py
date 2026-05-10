"""Data management for the optimisation."""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.dataframes.utils import select_markers

if TYPE_CHECKING:
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.configuration_manager import ConfigurationManager

LOGGER = logging.getLogger(__name__)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return ceil(numerator / denominator) for positive integers."""
    return (numerator + denominator - 1) // denominator


def _get_range_spec_plan(
    *,
    run_arc_by_arc: bool,
    use_fixed_bpm: bool,
    num_starts: int,
    num_ends: int,
) -> tuple[int, str]:
    """Return number of range specs per batch and a human-readable description."""
    if not run_arc_by_arc:
        return num_starts * 2, f"2 directions x {num_starts} start BPMs"
    if use_fixed_bpm:
        # create_bpm_range_specs already encodes direction here
        return num_starts + num_ends, f"fixed pairs ({num_starts} starts + {num_ends} ends)"
    return num_starts * num_ends * 2, f"2 directions x {num_starts} starts x {num_ends} ends"


def _group_turns_by_file(
    available_turns: list[int], file_map: dict[int, int]
) -> dict[int, list[int]]:
    """Group available turns by measurement file index."""
    turns_by_file: dict[int, list[int]] = {}
    for turn in available_turns:
        turns_by_file.setdefault(file_map[turn], []).append(turn)
    return turns_by_file


def _boundary_turns_for_track(track_turns: list[int], margin: int) -> list[int]:
    """Return turns to remove at track boundaries for a given margin."""
    if len(track_turns) <= 2 * margin:
        return track_turns
    return track_turns[:margin] + track_turns[-margin:]


def _distribute_target_batches_by_file(
    turns_by_file: dict[int, list[int]],
    tracks_per_worker: int,
    num_turn_batches: int,
    per_worker_batches: int,
) -> tuple[dict[int, int], bool, int]:
    """Allocate target turn-batch counts per file for worker assignment.

    Returns:
        target_batches_by_file: number of batches to draw from each file
        use_balanced_sizing: whether to intentionally reduce turns/batch to hit targets
        effective_num_batches: achievable total batch count from the targets
    """
    file_ids = sorted(turns_by_file.keys())
    target_batches_by_file = dict.fromkeys(file_ids, 0)

    if per_worker_batches <= 0:
        raise ValueError(f"per_worker_batches must be positive, got {per_worker_batches}")
    if tracks_per_worker < per_worker_batches:
        raise ValueError(
            f"tracks_per_worker={tracks_per_worker} must be >= per_worker_batches={per_worker_batches}"
        )

    batches_to_assign = max(0, num_turn_batches)
    total_available_batches = sum(
        len(turns_by_file[file_idx]) // per_worker_batches for file_idx in file_ids
    )
    batches_to_assign = min(batches_to_assign, total_available_batches)

    # First pass: spread across files to avoid starving files when budget allows it.
    while batches_to_assign > 0:
        assigned_in_round = False
        for file_idx in file_ids:
            if batches_to_assign == 0:
                break
            if (
                target_batches_by_file[file_idx]
                < len(turns_by_file[file_idx]) // per_worker_batches
            ):
                target_batches_by_file[file_idx] += 1
                batches_to_assign -= 1
                assigned_in_round = True
        if not assigned_in_round:
            break

    min_needed_per_file = {
        file_idx: _ceil_div(
            len(turns_by_file[file_idx]),
            (tracks_per_worker // per_worker_batches) * per_worker_batches,
        )
        for file_idx in file_ids
    }
    use_balanced_sizing = any(
        target_batches_by_file[file_idx] > min_needed_per_file[file_idx]
        for file_idx in file_ids
    )
    effective_num_batches = sum(target_batches_by_file.values())
    return target_batches_by_file, use_balanced_sizing, effective_num_batches


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

        # Track data per measurement file (indexed by file index)
        self.track_data: dict[int, pd.DataFrame]
        self.boundary_turns_by_file: dict[int, set[int]]
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
            df = pd.read_parquet(source, columns=FILE_COLUMNS, filters=filters)
        else:
            df = pd.read_parquet(source, columns=FILE_COLUMNS)

        # Always apply offset to create global turn IDs
        df["turn"] = df["turn"] + offset

        missing = [c for c in FILE_COLUMNS if c not in df.columns]
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

    @staticmethod
    def _fill_missing_coordinates(df: pd.DataFrame) -> None:
        """Replace missing coordinates with zero-weight values in-place."""
        coord_to_var = {
            "x": "var_x",
            "y": "var_y",
            "px": "var_px",
            "py": "var_py",
        }
        for coord_col, var_col in coord_to_var.items():
            nan_mask = df[coord_col].isna()
            if nan_mask.any():
                df.loc[nan_mask, var_col] = float("inf")
                df.loc[nan_mask, coord_col] = 0.0

    # ---------- Public API ----------

    def _filter_boundary_turns(self) -> None:
        """Drop boundary turns so each selected start turn has sufficient context."""
        turns_per_sample = (
            1 if self.simulation_config.run_arc_by_arc else self.simulation_config.n_run_turns
        )
        boundary_margin = max(1, turns_per_sample)
        turns_to_remove = set()
        boundary_turns_by_file: dict[int, set[int]] = {}

        for file_idx, df in self.track_data.items():
            file_turns = sorted(df.index.get_level_values("turn").unique())
            LOGGER.debug(f"File {file_idx} has {len(file_turns)} turns")

            for track_idx in range(0, len(file_turns), self.flattop_turns):
                track_turns = file_turns[track_idx : track_idx + self.flattop_turns]
                boundary_turns = _boundary_turns_for_track(track_turns, boundary_margin)
                boundary_turns_by_file.setdefault(file_idx, set()).update(boundary_turns)
                turns_to_remove.update(boundary_turns)

        self.boundary_turns_by_file = boundary_turns_by_file
        self.available_turns = [t for t in self.available_turns if t not in turns_to_remove]
        LOGGER.info(
            "Removed %d boundary turns (margin=%d, n_run_turns=%d), %d available",
            len(turns_to_remove),
            boundary_margin,
            self.simulation_config.n_run_turns,
            len(self.available_turns),
        )

    def _build_turn_batches(
        self,
        turns_by_file: dict[int, list[int]],
        num_turn_batches: int,
        tracks_per_worker: int,
        per_worker_batches: int,
    ) -> list[list[int]]:
        """Materialise worker turn batches from grouped turns."""
        for turns in turns_by_file.values():
            random.shuffle(turns)

        target_batches_by_file, use_balanced_sizing, effective_num_batches = (
            _distribute_target_batches_by_file(
                turns_by_file,
                tracks_per_worker,
                num_turn_batches,
                per_worker_batches,
            )
        )
        if effective_num_batches != num_turn_batches:
            LOGGER.warning(
                "Could only allocate %d/%d worker turn batches across files while keeping >=1 turn per batch",
                effective_num_batches,
                num_turn_batches,
            )
            num_turn_batches = effective_num_batches

        turn_batches: list[list[int]] = []
        file_queue = deque(sorted(target_batches_by_file.keys()))

        for _ in range(num_turn_batches):
            if not file_queue:
                LOGGER.warning(
                    "Only created %d/%d worker turn batches",
                    len(turn_batches),
                    num_turn_batches,
                )
                break

            file_idx: int | None = None
            for _ in range(len(file_queue)):
                candidate = file_queue.popleft()
                if turns_by_file.get(candidate) and target_batches_by_file[candidate] > 0:
                    file_idx = candidate
                    break

            if file_idx is None:
                LOGGER.warning(
                    "Only created %d/%d worker turn batches",
                    len(turn_batches),
                    num_turn_batches,
                )
                break

            turns_left = len(turns_by_file[file_idx])
            batches_left_for_file = target_batches_by_file[file_idx]
            max_turns_per_batch = (tracks_per_worker // per_worker_batches) * per_worker_batches
            if use_balanced_sizing:
                batch_size = per_worker_batches * _ceil_div(
                    _ceil_div(turns_left, per_worker_batches),
                    batches_left_for_file,
                )
                batch_size = min(batch_size, max_turns_per_batch)
                max_assignable = turns_left - per_worker_batches * (batches_left_for_file - 1)
                batch_size = min(batch_size, (max_assignable // per_worker_batches) * per_worker_batches)
                batch_size = max(per_worker_batches, batch_size)
            else:
                batch_size = min(max_turns_per_batch, (turns_left // per_worker_batches) * per_worker_batches)

            # Round down to a multiple of num_batches so MAD can split evenly.
            # Only apply when batch_size >= num_batches; smaller batches cause the
            # worker to reduce num_batches to match, which is safe.
            mad_num_batches = self.simulation_config.num_batches
            if batch_size >= mad_num_batches:
                batch_size = (batch_size // mad_num_batches) * mad_num_batches

            batch = turns_by_file[file_idx][:batch_size]
            turns_by_file[file_idx] = turns_by_file[file_idx][batch_size:]
            target_batches_by_file[file_idx] -= 1
            turn_batches.append(batch)

            if target_batches_by_file[file_idx] > 0 and turns_by_file[file_idx]:
                file_queue.append(file_idx)

        return turn_batches

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
        LOGGER.info(f"Loading {len(sources)} measurement file(s)...")
        for file_idx, source in enumerate(sources):
            LOGGER.debug(f"Loading file {file_idx}: {source}")
            df = self._read_parquet(source, needed_turns, offsets[file_idx])
            file_tracks[file_idx] = self._reduce_dataframe(df)
            file_kick_planes[file_idx] = self.infer_kick_plane(file_tracks[file_idx])
            LOGGER.debug(
                "File %d kick-plane classification: %s",
                file_idx,
                file_kick_planes[file_idx],
            )

        # Handle NaN values in track data coordinate-by-coordinate.
        # This is important for single-plane BPMs where one plane is intentionally missing:
        # mark only that coordinate as zero-weight (variance=inf), not the whole row.
        for file_idx, df in file_tracks.items():
            self._fill_missing_coordinates(df)

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

        self._filter_boundary_turns()
        if len(self.available_turns) == 0:
            raise ValueError(
                "No turns available after removing boundary turns. Check that your flattop_turns setting leaves at least one turn per track."
            )

        turns_by_file: dict[int, list[int]] = _group_turns_by_file(
            self.available_turns, self.file_map
        )
        tracks_per_worker = self.simulation_config.tracks_per_worker

        num_workers = self.simulation_config.num_workers
        num_files = len(self.track_data)
        num_starts = len(config_manager.start_bpms)
        num_ends = len(config_manager.end_bpms)
        range_specs_per_batch, range_specs_desc = _get_range_spec_plan(
            run_arc_by_arc=self.simulation_config.run_arc_by_arc,
            use_fixed_bpm=self.simulation_config.use_fixed_bpm,
            num_starts=num_starts,
            num_ends=num_ends,
        )
        worker_turn_batches = max(1, num_workers // max(1, range_specs_per_batch))
        max_batches_by_turn_capacity = sum(
            _ceil_div(len(turns), tracks_per_worker) for turns in turns_by_file.values()
        )
        num_turn_batches = min(worker_turn_batches, max_batches_by_turn_capacity)

        planned_workers = num_turn_batches * range_specs_per_batch
        limiting_factor = "data capacity" if max_batches_by_turn_capacity < worker_turn_batches else "num_workers cap"
        LOGGER.info(
            "Worker planning: requested=%d workers, range_specs_per_batch=%d (%s), starts=%d, ends=%d",
            num_workers,
            range_specs_per_batch,
            range_specs_desc,
            num_starts,
            num_ends,
        )
        LOGGER.info(
            "Turn-batch planning: "
            "num_workers_cap→%d batches, "
            "data_capacity→%d batches (%d files x up to ceil(turns/%d) each), "
            "selected=%d batches [limited by %s]",
            worker_turn_batches,
            max_batches_by_turn_capacity,
            num_files,
            tracks_per_worker,
            num_turn_batches,
            limiting_factor,
        )
        LOGGER.info(
            "Planned workers: %d turn batches x %d range specs = %d workers "
            "(num_batches=%d is MAD-internal sub-batching, does not affect worker count)",
            num_turn_batches,
            range_specs_per_batch,
            planned_workers,
            self.simulation_config.num_batches,
        )

        self.turn_batches = self._build_turn_batches(
            turns_by_file,
            num_turn_batches,
            tracks_per_worker,
            per_worker_batches=1,
        )

        if len(self.turn_batches) == 0:
            raise ValueError(
                f"Failed to create any batches. Available turns: {len(self.available_turns)}, "
                f"required tracks_per_worker: {tracks_per_worker}. "
                "Consider reducing tracks_per_worker or increasing flattop_turns."
            )

        self.num_workers = len(self.turn_batches)
        LOGGER.info("Created %d batches from %d files", self.num_workers, len(self.track_data))
        LOGGER.info(
            "Expected worker count after range expansion: %d batches x %d range specs = %d workers",
            self.num_workers,
            range_specs_per_batch,
            self.num_workers * range_specs_per_batch,
        )

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
        """Calculate the number of turns that will actually be processed."""
        if not self.turn_batches:
            return 0

        return sum(len(batch) for batch in self.turn_batches)
