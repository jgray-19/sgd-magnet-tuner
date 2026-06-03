"""Data management for the optimisation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.training.workers.turn_planner import WorkerTurnPlanner

if TYPE_CHECKING:
    from collections.abc import Callable

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.manager import ConfigurationManager
    from aba_optimiser.training.config.tracking import TrackingPlan

    ShuffleTurns = Callable[[list[int]], None]

LOGGER = logging.getLogger(__name__)

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
        tracking_plan: TrackingPlan,
        extra_markers: list[str] | None = None,
        shuffle_turns: ShuffleTurns | None = None,
    ):
        """Create a data manager for one optimisation run.

        Args:
            shuffle_turns: Optional in-place turn ordering strategy forwarded to
                ``WorkerTurnPlanner``. This primarily supports deterministic tests.
        """
        self.all_bpms = all_bpms
        self.bpms_in_range = bpms_in_range
        self.simulation_config = simulation_config
        self.measurement_files = measurement_files
        self.num_bunches = num_bunches
        self.flattop_turns = flattop_turns
        self.tracking_plan = tracking_plan
        self.extra_markers = extra_markers or []
        self.shuffle_turns = shuffle_turns

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
        df["turn"] = df["turn"].astype("int32")
        df["name"] = df["name"].astype("category")
        markers = list(dict.fromkeys(self.bpms_in_range + self.extra_markers))
        # Copy because we drop non-selected markers and convert from view.
        return select_markers(df, markers).copy()

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
            marker_order = list(dict.fromkeys(self.bpms_in_range + self.extra_markers))
            bpm_order_filtered = [
                bpm
                for bpm in marker_order
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
        self.boundary_turns_by_file, self.available_turns = (
            self.tracking_plan.select_available_turns(
                track_data=self.track_data,
                flattop_turns=self.flattop_turns,
                simulation_config=self.simulation_config,
                available_turns=self.available_turns,
            )
        )
        if any(self.boundary_turns_by_file.values()):
            total_removed = sum(len(turns) for turns in self.boundary_turns_by_file.values())
            LOGGER.info(
                "Removed %d boundary turns (n_run_turns=%d), %d available",
                total_removed,
                self.simulation_config.n_run_turns,
                len(self.available_turns),
            )
        else:
            LOGGER.info(
                "Using turn starts directly with no boundary removal: %d available",
                len(self.available_turns),
            )

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

        batch_plan = WorkerTurnPlanner(
            self.tracking_plan,
            self.simulation_config,
            shuffle_turns=self.shuffle_turns,
        ).build_turn_batches(
            available_turns=self.available_turns,
            file_map=self.file_map,
            num_files=len(self.track_data),
            num_starts=len(config_manager.start_bpms),
            num_ends=len(config_manager.end_bpms),
        )
        self.turn_batches = batch_plan.turn_batches

        if len(self.turn_batches) == 0:
            raise ValueError(
                f"Failed to create any batches. Available turns: {len(self.available_turns)}, "
                f"required tracks_per_worker: {self.simulation_config.tracks_per_worker}. "
                "Consider reducing tracks_per_worker or increasing flattop_turns."
            )

        self.num_workers = len(self.turn_batches)
        LOGGER.info("Created %d batches from %d files", self.num_workers, len(self.track_data))
        LOGGER.info(
            "Expected worker count after range expansion: %d batches x %d range specs = %d workers",
            self.num_workers,
            batch_plan.range_specs_per_batch,
            self.num_workers * batch_plan.range_specs_per_batch,
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
