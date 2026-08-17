"""Data management for the optimisation."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import TYPE_CHECKING

import pandas as pd

from aba_optimiser.config import FILE_COLUMNS
from aba_optimiser.dataframes.utils import select_markers
from aba_optimiser.training.workers.turn_planner import WorkerTurnPlanner, group_turns_by_file

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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
        measurement_files: list[Path],
        tracking_plan: TrackingPlan,
        first_bpms: list[str | None] | None = None,
        extra_markers: list[str] | None = None,
        shuffle_turns: ShuffleTurns | None = None,
    ):
        """Create a data manager for one optimisation run.

        The per-file bunch structure (how turns group into bunches) is read from
        the ``bunch_number`` column of each measurement parquet rather than being
        inferred from a configured turns-per-bunch count.

        Args:
            shuffle_turns: Optional in-place turn ordering strategy forwarded to
                ``WorkerTurnPlanner``. This primarily supports deterministic tests.
        """
        self.all_bpms = all_bpms
        self.bpms_in_range = bpms_in_range
        self.simulation_config = simulation_config
        self.measurement_files = measurement_files
        self.tracking_plan = tracking_plan
        self.first_bpms = first_bpms or [None] * len(measurement_files)
        self.extra_markers = extra_markers or []
        self.shuffle_turns = shuffle_turns

        # Available turns will be populated after loading track data
        self.available_turns: list[int]

        self.turn_batches: list[list[int]]
        # Held-out validation turns grouped into per-file batches. These turns are
        # removed from ``turn_batches`` entirely so validation loss is genuinely
        # out-of-sample. Empty when validation is disabled or there is too little data.
        self.validation_turn_batches: list[list[int]] = []

        # Track data per measurement file (indexed by file index)
        self.track_data: dict[int, pd.DataFrame]
        # Per-file mapping {file_index -> {bunch_number -> sorted global turns}}
        self.bunch_turns_by_file: dict[int, dict[int, list[int]]]
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

    def _read_parquet(self, source: str) -> pd.DataFrame:
        """Read a parquet's in-range marker rows and validate the column schema."""
        markers = self.bpms_in_range + self.extra_markers
        filters: list = [("name", "in", markers)]
        df = pd.read_parquet(source, columns=FILE_COLUMNS, filters=filters)

        missing = [c for c in FILE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in track data {source}: {missing}. The 'bunch_number' "
                "column is required; regenerate the measurement parquet."
            )
        return df

    def _cycle_ring_to_first_bpm(
        self, file_idx: int, ring_bpms: list[str], appearance: list[str]
    ) -> list[str]:
        """Return ``ring_bpms`` rotated so it begins at this file's first BPM.

        Logs whether the dataframe is cycled to an explicit ``first_bpm`` or left
        in its recorded order (using its own first recorded BPM as the boundary).
        """
        if not ring_bpms:
            return []

        ring_set = set(ring_bpms)
        # The data's own boundary: the first recorded name that is a ring BPM.
        natural = next((b for b in appearance if b in ring_set), ring_bpms[0])
        requested = self.first_bpms[file_idx] if file_idx < len(self.first_bpms) else None

        if requested is not None and requested in ring_set:
            first_bpm = requested
            LOGGER.info(
                "File %d: cycling measurement data to start at first BPM %s",
                file_idx,
                first_bpm,
            )
        elif requested is not None and requested in self.all_bpms:
            # The turn-boundary BPM is not observed in this range. Project it to the
            # first in-range BPM that follows it in ring order: only when this range
            # straddles the boundary is that a different BPM from the range's own
            # start, so only then is the data actually cycled (a turn wrap placed).
            requested_idx = self.all_bpms.index(requested)
            first_bpm = next(
                (
                    self.all_bpms[(requested_idx + offset) % len(self.all_bpms)]
                    for offset in range(1, len(self.all_bpms) + 1)
                    if self.all_bpms[(requested_idx + offset) % len(self.all_bpms)]
                    in ring_set
                ),
                natural,
            )
            if first_bpm == ring_bpms[0]:
                LOGGER.debug(
                    "File %d: turn boundary %s is outside this range; no turn wrap "
                    "needed, leaving measurement data in ring order",
                    file_idx,
                    requested,
                )
            else:
                LOGGER.debug(
                    "File %d: this range straddles turn boundary %s; placing the "
                    "wrap at the next in-range BPM %s",
                    file_idx,
                    requested,
                    first_bpm,
                )
        elif requested is not None and requested in appearance:
            # A non-BPM start marker (e.g. a kicker): the data is already recorded
            # from it, so its first ring BPM is the boundary to cycle to.
            first_bpm = natural
            LOGGER.info(
                "File %d: measurement data recorded from marker %s; first BPM is %s",
                file_idx,
                requested,
                first_bpm,
            )
        else:
            if requested is not None:
                LOGGER.warning(
                    "File %d: requested first BPM %s is not in the measurement data; "
                    "not cycling, using %s as the first BPM",
                    file_idx,
                    requested,
                    natural,
                )
            else:
                LOGGER.info(
                    "File %d: not cycling measurement data, using %s as the first BPM",
                    file_idx,
                    natural,
                )
            first_bpm = natural

        pivot = ring_bpms.index(first_bpm)
        return ring_bpms[pivot:] + ring_bpms[:pivot]

    def _reorder_track_dataframes(self) -> None:
        """Reorder track dataframes into ring order, cycled to each file's first BPM.

        The payload builder infers per-turn range wraps from the element order,
        assuming it follows ring order so that a contiguous tracking range maps to
        a monotonic column sequence that crosses the ring boundary exactly once.
        The ring boundary that matters is the one the *measurement* was generated
        from: the BPM each recorded turn begins at. A tracking arc that straddles
        that boundary has its early BPMs at the end of one turn and its later BPMs
        at the start of the next, and only a wrap placed at the generation boundary
        reads each side from the right turn.

        The model's own ``$start`` is generally a different point, so we cycle the
        model ring order to begin at the file's first BPM before reindexing. That
        first BPM is taken from ``first_bpms`` when the caller supplied it (use this
        when the file's own row order is unreliable, e.g. ACD marker rows written
        after all the BPMs); otherwise it defaults to the file's first recorded BPM.
        Names present in the data but absent from ``all_bpms`` are kept, appended in
        their original appearance order.
        """
        for file_idx in self.track_data:
            all_turns = sorted(self.track_data[file_idx].index.get_level_values("turn").unique())
            appearance = list(
                dict.fromkeys(self.track_data[file_idx].index.get_level_values("name"))
            )
            appearance_rank = {name: idx for idx, name in enumerate(appearance)}
            ring_bpms = [b for b in self.all_bpms if b in appearance_rank]
            ring_cycle = self._cycle_ring_to_first_bpm(file_idx, ring_bpms, appearance)
            ring_rank = {name: idx for idx, name in enumerate(ring_cycle)}
            ordered = sorted(
                appearance,
                key=lambda name: (
                    ring_rank.get(name, len(ring_rank)),
                    appearance_rank[name],
                ),
            )
            self.track_data[file_idx] = self.track_data[file_idx].reindex(
                pd.MultiIndex.from_product([all_turns, ordered], names=["turn", "name"])
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
                bunch_turns_by_file=self.bunch_turns_by_file,
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

    def load_track_data(self) -> None:
        """Load track data from all measurement files and build the file/bunch maps.

        Each measurement file is read with its own (file-local) turn numbering, then
        shifted into a disjoint global turn block so turns are unique across files.
        The bunch structure of every file is read from the ``bunch_number`` column.
        """
        LOGGER.info(
            "Loading track data from %d measurement file(s)...", len(self.measurement_files)
        )

        # Determine source files - fitter has already resolved None to actual files
        sources = []
        for mf in self.measurement_files:
            if mf is not None:
                sources.append(mf)
            else:
                raise ValueError(
                    "measurement_files should not contain None - fitter should have resolved defaults"
                )

        # Read raw frames in parallel (file-local turn numbering, includes bunch_number).
        raw_tracks: dict[int, pd.DataFrame] = {}
        LOGGER.info(f"Loading {len(sources)} measurement file(s)...")

        def _load_one(args: tuple[int, str]) -> tuple[int, pd.DataFrame]:
            file_idx, source = args
            LOGGER.debug(f"Loading file {file_idx}: {source}")
            return file_idx, self._read_parquet(source)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            for file_idx, df in pool.map(_load_one, enumerate(sources)):
                raw_tracks[file_idx] = df

        # Shift each file into a disjoint global turn block and record its bunch
        # structure, then reduce to the observed markers.
        file_tracks: dict[int, pd.DataFrame] = {}
        file_kick_planes: dict[int, str] = {}
        self.bunch_turns_by_file = {}
        running_offset = 0
        for file_idx in range(len(sources)):
            df = raw_tracks[file_idx]
            local_min = int(df["turn"].min())
            df["turn"] = (df["turn"] - local_min + running_offset).astype("int32")
            running_offset = int(df["turn"].max()) + 1

            bunches: dict[int, list[int]] = {}
            per_turn = df[["turn", "bunch_number"]].drop_duplicates("turn")
            for turn, bunch in zip(per_turn["turn"], per_turn["bunch_number"]):
                bunches.setdefault(int(bunch), []).append(int(turn))
            self.bunch_turns_by_file[file_idx] = {
                bunch: sorted(turns) for bunch, turns in bunches.items()
            }

            reduced = self._reduce_dataframe(df.drop(columns=["bunch_number"]))
            file_tracks[file_idx] = reduced
            kick_plane = self.infer_kick_plane(reduced)
            LOGGER.debug("File %d kick-plane classification: %s", file_idx, kick_plane)
            file_kick_planes[file_idx] = kick_plane

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
                "No turns available after removing boundary turns. Check that each bunch in the measurement data has more than one turn."
            )

        train_turns, validation_turns = self._split_validation_turns()
        train_turns = self._sample_training_turns(train_turns)
        self.validation_turn_batches = self._build_validation_turn_batches(validation_turns)

        batch_plan = WorkerTurnPlanner(
            self.tracking_plan,
            self.simulation_config,
            shuffle_turns=self.shuffle_turns,
        ).build_turn_batches(
            available_turns=train_turns,
            file_map=self.file_map,
            num_files=len(self.track_data),
            num_starts=len(config_manager.start_bpms),
            num_ends=len(config_manager.end_bpms),
        )
        self.turn_batches = batch_plan.turn_batches

        if len(self.turn_batches) == 0:
            raise ValueError(
                f"Failed to create any training batches. Available turns: {len(self.available_turns)}, "
                f"training turns after validation split + data_fraction="
                f"{self.simulation_config.data_fraction}: {len(train_turns)}. "
                "Consider raising data_fraction, lowering validation_fraction, or using longer bunches."
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
        validation_used_turns = sum(len(batch) for batch in self.validation_turn_batches)
        # Turns neither trained nor validated: dropped by data_fraction sampling or
        # by MAD sub-batch trimming.
        unused_turns = total_available_turns - total_used_turns - validation_used_turns
        unused_percentage = (
            (unused_turns / total_available_turns * 100) if total_available_turns > 0 else 0
        )
        LOGGER.info(
            "Turn usage: %d training, %d held-out validation, %d unused (%.1f%%) of %d available",
            total_used_turns,
            validation_used_turns,
            unused_turns,
            unused_percentage,
            total_available_turns,
        )
        if validation_used_turns == 0 and self.tracking_plan.enable_validation:
            LOGGER.warning(
                "No held-out validation turns were reserved (validation_fraction=%.3f, too little "
                "data). Validation loss will fall back to training loss and CANNOT detect overfitting.",
                self.simulation_config.validation_fraction,
            )

    def _split_validation_turns(self) -> tuple[list[int], list[int]]:
        """Partition available turns into disjoint (training, validation) sets.

        Validation turns are reserved per file (stratified) so every file keeps its
        share of held-out data. At least one training turn is always retained per
        file. Returns two disjoint, sorted lists whose union is ``available_turns``.
        """
        if (
            not self.tracking_plan.enable_validation
            or self.simulation_config.validation_fraction <= 0.0
        ):
            return sorted(self.available_turns), []

        val_frac = self.simulation_config.validation_fraction
        turns_by_file = group_turns_by_file(self.available_turns, self.file_map)
        train_turns: list[int] = []
        validation_turns: list[int] = []
        for file_idx in sorted(turns_by_file):
            turns = list(turns_by_file[file_idx])
            self._shuffle_in_place(turns)
            n_val = min(round(val_frac * len(turns)), len(turns) - 1)
            n_val = max(0, n_val)
            validation_turns.extend(turns[:n_val])
            train_turns.extend(turns[n_val:])
        return sorted(train_turns), sorted(validation_turns)

    def _sample_training_turns(self, train_turns: list[int]) -> list[int]:
        """Keep ``data_fraction`` of the training turns, stratified per file."""
        frac = self.simulation_config.data_fraction
        if frac >= 1.0:
            return sorted(train_turns)

        turns_by_file = group_turns_by_file(train_turns, self.file_map)
        kept: list[int] = []
        for file_idx in sorted(turns_by_file):
            turns = list(turns_by_file[file_idx])
            self._shuffle_in_place(turns)
            n_keep = max(1, round(frac * len(turns)))
            kept.extend(turns[:n_keep])
        return sorted(kept)

    def _build_validation_turn_batches(self, validation_turns: list[int]) -> list[list[int]]:
        """Group held-out validation turns into one batch per file."""
        if not validation_turns:
            return []
        turns_by_file = group_turns_by_file(validation_turns, self.file_map)
        return [sorted(turns) for _idx, turns in sorted(turns_by_file.items()) if turns]

    def _shuffle_in_place(self, seq: list[int]) -> None:
        """Shuffle ``seq`` using the injected strategy (deterministic in tests)."""
        if self.shuffle_turns is not None:
            self.shuffle_turns(seq)
        else:
            import random

            random.shuffle(seq)

    def get_total_turns(self) -> int:
        """Calculate the number of tracked turns that will actually be processed."""
        if not self.turn_batches:
            return 0

        start_turns = sum(len(batch) for batch in self.turn_batches)
        if self.simulation_config.run_arc_by_arc:
            return start_turns
        return start_turns * self.simulation_config.n_run_turns
