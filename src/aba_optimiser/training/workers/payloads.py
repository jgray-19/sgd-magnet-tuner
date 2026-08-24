"""Worker payload construction helpers.

This module turns measurement data and observation plans into immutable worker
payloads. It keeps array manipulation and turn-stitching logic away from the
multiprocessing orchestration layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import PROTON_MASS
from aba_optimiser.physics.deltap import dp2pt
from aba_optimiser.training.utils import bpm_supports_both_planes, bpm_supports_plane
from aba_optimiser.workers import (
    PrecomputedTrackingWeights,
    TrackingData,
    WeightProcessor,
)

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.workers.setup import WorkerObservationPlan
    from aba_optimiser.workers import WorkerConfig

LOGGER = logging.getLogger(__name__)


class WorkerPayloadBuilder:
    """Build tracking payload arrays and shared weights for workers.

    The payload layer assumes worker planning has already removed BPMs that
    cannot measure the worker's plane. Single-plane workers therefore receive
    only same-plane BPMs, and dual-plane workers are valid only when every BPM
    in the plan is genuinely dual-plane.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        all_bpms: list[str],
        tracking_anchor_markers: list[str] | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.all_bpms = all_bpms
        self.tracking_anchor_markers = set(tracking_anchor_markers or [])
        self._pos_cache: dict[tuple, dict[tuple[int, str], int]] = {}
        self._layout_cache: dict[tuple, tuple[dict[int, int], dict[str, int], int]] = {}

    @staticmethod
    def _df_cache_key(df: pd.DataFrame) -> tuple:
        """Return a stable cache key derived from the DataFrame's MultiIndex structure."""
        turns = tuple(dict.fromkeys(df.index.get_level_values("turn")))
        bpms = tuple(dict.fromkeys(df.index.get_level_values("name")))
        return (turns, bpms)

    def compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Compute transverse momentum based on file index."""
        return dp2pt(machine_deltaps[file_idx], PROTON_MASS, energy=self.accelerator.energy)

    @staticmethod
    def freeze_payload_arrays(*arrays: np.ndarray) -> None:
        """Mark payload arrays as read-only before passing them to workers."""
        for array in arrays:
            array.setflags(write=False)

    def bpm_supports_plane(self, bpm: str, kick_plane: str) -> bool:
        """Return whether `bpm` can measure the requested kick plane."""
        if bpm in self.tracking_anchor_markers:
            return True
        return bpm_supports_plane(self.accelerator, bpm, kick_plane)

    def bpm_supports_both_planes(self, bpm: str) -> bool:
        """Return whether `bpm` can measure both transverse planes."""
        if bpm in self.tracking_anchor_markers:
            return True
        return bpm_supports_both_planes(self.accelerator, bpm)

    def validate_worker_bpm_names(self, bpm_names: list[str], kick_plane: str) -> None:
        """Validate that a worker only receives BPMs compatible with its plane."""
        if not bpm_names:
            raise ValueError(f"No BPMs available for {kick_plane!r} worker")

        if kick_plane == "xy":
            invalid = [bpm for bpm in bpm_names if not self.bpm_supports_both_planes(bpm)]
            if invalid:
                raise ValueError(
                    "Dual-plane worker received single-plane BPMs: "
                    + ", ".join(sorted(invalid))
                )
            return

        invalid = [bpm for bpm in bpm_names if not self.bpm_supports_plane(bpm, kick_plane)]
        if invalid:
            raise ValueError(
                f"Single-plane worker {kick_plane!r} received incompatible BPMs: "
                + ", ".join(sorted(invalid))
            )

    def extract_arrays(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Extract numpy arrays from DataFrame once for memory efficiency."""
        return {
            "x": df["x"].to_numpy(dtype="float64", copy=False),
            "y": df["y"].to_numpy(dtype="float64", copy=False),
            "px": df["px"].to_numpy(dtype="float64", copy=False),
            "py": df["py"].to_numpy(dtype="float64", copy=False),
            "var_x": df["var_x"].to_numpy(dtype="float64", copy=False),
            "var_y": df["var_y"].to_numpy(dtype="float64", copy=False),
            "var_px": df["var_px"].to_numpy(dtype="float64", copy=False),
            "var_py": df["var_py"].to_numpy(dtype="float64", copy=False),
        }

    def _get_pos(self, df: pd.DataFrame, turn: int, bpm: str) -> int:
        """Fast integer index position for MultiIndex (turn, bpm) with caching."""
        cache_key = self._df_cache_key(df)
        bucket = self._pos_cache.setdefault(cache_key, {})
        key = (turn, bpm)
        pos = bucket.get(key)
        if pos is None:
            turn_offsets, bpm_offsets, row_stride = self._layout_cache.get(cache_key, ({}, {}, 0))
            if not turn_offsets:
                turn_offsets, bpm_offsets, row_stride = self._build_layout_cache(df)
                self._layout_cache[cache_key] = (turn_offsets, bpm_offsets, row_stride)
            try:
                pos = turn_offsets[turn] + bpm_offsets[bpm]
            except KeyError:
                pos = int(df.index.get_loc((turn, bpm)))
            bucket[key] = pos
        return pos

    @staticmethod
    def _build_layout_cache(df: pd.DataFrame) -> tuple[dict[int, int], dict[str, int], int]:
        """Return row-offset caches for a turn-major full MultiIndex grid."""
        turns = list(dict.fromkeys(df.index.get_level_values("turn")))
        bpms = list(dict.fromkeys(df.index.get_level_values("name")))
        row_stride = len(bpms)
        turn_offsets = {int(turn): idx * row_stride for idx, turn in enumerate(turns)}
        bpm_offsets = {str(bpm): idx for idx, bpm in enumerate(bpms)}
        return turn_offsets, bpm_offsets, row_stride

    def get_turn(self, df: pd.DataFrame, pos: int) -> int:
        """Get the turn number from a DataFrame position."""
        return df.index[pos][0]

    def get_observation_positions(
        self,
        df: pd.DataFrame,
        bpm_names: list[str],
        sdir: int,
        turn: int,
        n_run_turns: int,
    ) -> np.ndarray:
        """Return explicit row positions for the observed BPM list across tracking turns."""
        return self.get_observation_positions_batch(df, bpm_names, sdir, [turn], n_run_turns)[0]

    def get_observation_positions_batch(
        self,
        df: pd.DataFrame,
        bpm_names: list[str],
        sdir: int,
        turns: list[int],
        n_run_turns: int,
    ) -> np.ndarray:
        """Return row positions for multiple starting turns at once.

        Returns shape ``(len(turns), len(bpm_names) * n_run_turns)``.  The
        per-turn offset is computed as a single outer sum over precomputed
        *fixed_offsets*, so the cost is O(n_turns + n_data) rather than the
        O(n_turns * n_data) Python loop that the single-turn helper used.
        """
        if not bpm_names:
            raise ValueError("No BPMs available for observation")

        cache_key = self._df_cache_key(df)
        turn_offsets, bpm_offsets, row_stride = self._layout_cache.get(cache_key, ({}, {}, 0))
        if not turn_offsets:
            turn_offsets, bpm_offsets, row_stride = self._build_layout_cache(df)
            self._layout_cache[cache_key] = (turn_offsets, bpm_offsets, row_stride)

        # Column offsets for the full repeated BPM sequence
        repeated = bpm_names * n_run_turns
        col_offsets = np.array([bpm_offsets[b] for b in repeated], dtype=np.int64)

        # Measurement rows stay in sequence-file order; detect physical turn wraps
        # from the BPM column order, even when tracking starts from a marker.
        diff = np.diff(col_offsets)
        if sdir == 1:
            wrap = np.concatenate([[0], (diff < 0).astype(np.int64)])
        else:
            wrap = np.concatenate([[0], (diff > 0).astype(np.int64)])
        turn_delta = np.cumsum(wrap) * sdir  # shape (n_data,)

        # fixed_offsets is the same for every starting turn; shape (n_data,)
        fixed_offsets = turn_delta * row_stride + col_offsets

        # Per-starting-turn base offsets; shape (n_turns,)
        starting_offsets = np.array([turn_offsets[t] for t in turns], dtype=np.int64)

        # Outer sum → shape (n_turns, n_data)
        return (starting_offsets[:, None] + fixed_offsets[None, :]).astype(np.int64)

    def get_measured_start_planes(self, init_bpm: str, kick_plane: str) -> tuple[bool, bool]:
        """Return which coordinates should be used for the initial condition."""
        if kick_plane == "x":
            return True, False
        if kick_plane == "y":
            return False, True
        if init_bpm in self.tracking_anchor_markers:
            return True, True
        plane = self.accelerator.infer_monitor_plane(init_bpm)
        return "H" in plane, "V" in plane

    def make_worker_payload(
        self,
        turn_batch: list[int],
        file_turn_map: dict[int, int],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bpm_names: list[str],
        kick_plane: str,
        machine_deltaps: list[float],
        arrays_cache: dict[int, dict[str, np.ndarray]],
        track_data: dict[int, pd.DataFrame],
        n_run_turns: int,
        init_marker: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the raw arrays for one worker payload."""
        n_turns = len(turn_batch)
        n_data_points = len(bpm_names) * n_run_turns
        if n_data_points == 0:
            raise ValueError(f"No active data points for worker range {start_bpm}/{end_bpm}")

        pos = np.zeros((n_turns, n_data_points, 2), dtype="float64")
        mom = np.zeros((n_turns, n_data_points, 2), dtype="float64")
        pos_var = np.full((n_turns, n_data_points, 2), np.inf, dtype="float64")
        mom_var = np.full((n_turns, n_data_points, 2), np.inf, dtype="float64")
        init_coords = np.empty((n_turns, 6), dtype="float64")
        pts = np.empty((n_turns,), dtype="float64")
        init_bpm = start_bpm if sdir == 1 else end_bpm
        init_marker = init_marker or init_bpm
        self.validate_worker_bpm_names(bpm_names, kick_plane)
        if init_marker not in self.all_bpms:
            has_x = kick_plane in ("x", "xy")
            has_y = kick_plane in ("y", "xy")
        else:
            has_x, has_y = self.get_measured_start_planes(init_bpm, kick_plane)

        # Group turns by file so we can do vectorised fancy-indexing per file.
        from collections import defaultdict

        turns_by_file: dict[int, list[int]] = defaultdict(list)
        indices_by_file: dict[int, list[int]] = defaultdict(list)
        for i, turn in enumerate(turn_batch):
            fi = file_turn_map[turn]
            turns_by_file[fi].append(turn)
            indices_by_file[fi].append(i)
            pts[i] = self.compute_pt(fi, machine_deltaps)

        for file_idx, file_turns in turns_by_file.items():
            idxs = indices_by_file[file_idx]
            cache = arrays_cache[file_idx]
            df = track_data[file_idx]
            df_cache_key = self._df_cache_key(df)
            turn_offsets, bpm_offsets, _row_stride = self._layout_cache.get(df_cache_key, ({}, {}, 0))
            if not bpm_offsets:
                turn_offsets, bpm_offsets, _row_stride = self._build_layout_cache(df)
                self._layout_cache[df_cache_key] = (turn_offsets, bpm_offsets, _row_stride)
            if init_marker not in bpm_offsets:
                raise ValueError(
                    f"Init marker '{init_marker}' not found in tracking data for file {file_idx}"
                )
            base_x, base_y = cache["x"], cache["y"]
            base_px, base_py = cache["px"], cache["py"]
            base_vx, base_vy = cache["var_x"], cache["var_y"]
            base_vpx, base_vpy = cache["var_px"], cache["var_py"]

            # All positions at once: shape (n_file_turns, n_data_points)
            all_pos = self.get_observation_positions_batch(
                df=df,
                bpm_names=bpm_names,
                sdir=sdir,
                turns=file_turns,
                n_run_turns=n_run_turns,
            )

            init_positions = self.get_observation_positions_batch(
                df=df,
                bpm_names=[init_marker],
                sdir=1,
                turns=file_turns,
                n_run_turns=1,
            )
            init_pos = init_positions[:, 0]
            if has_x:
                init_coords[idxs, 0] = base_x[init_pos]
                init_coords[idxs, 1] = base_px[init_pos]
            else:
                init_coords[idxs, 0] = 0.0
                init_coords[idxs, 1] = 0.0
            if has_y:
                init_coords[idxs, 2] = base_y[init_pos]
                init_coords[idxs, 3] = base_py[init_pos]
            else:
                init_coords[idxs, 2] = 0.0
                init_coords[idxs, 3] = 0.0
            init_coords[idxs, 4] = 0.0
            init_coords[idxs, 5] = pts[idxs]

            for local_i, global_i in enumerate(idxs):
                if np.all(init_coords[global_i, :] == 0.0):
                    raise ValueError(
                        f"Initial coordinates for turn {file_turns[local_i]} at BPM {init_bpm} are all zero"
                    )

            # Vectorised observable extraction: shape (n_file_turns, n_data_points)
            p = all_pos  # alias
            if kick_plane == "x":
                pos[idxs, :, 0] = base_x[p]
                mom[idxs, :, 0] = base_px[p]
                pos_var[idxs, :, 0] = base_vx[p]
                mom_var[idxs, :, 0] = base_vpx[p]
            elif kick_plane == "y":
                pos[idxs, :, 1] = base_y[p]
                mom[idxs, :, 1] = base_py[p]
                pos_var[idxs, :, 1] = base_vy[p]
                mom_var[idxs, :, 1] = base_vpy[p]
            else:
                pos[idxs, :, 0] = base_x[p]
                pos[idxs, :, 1] = base_y[p]
                mom[idxs, :, 0] = base_px[p]
                mom[idxs, :, 1] = base_py[p]
                pos_var[idxs, :, 0] = base_vx[p]
                pos_var[idxs, :, 1] = base_vy[p]
                mom_var[idxs, :, 0] = base_vpx[p]
                mom_var[idxs, :, 1] = base_vpy[p]

        return pos, mom, pos_var, mom_var, init_coords, pts

    def make_tracking_data(
        self,
        turn_batch: list[int],
        file_turn_map: dict[int, int],
        plan: WorkerObservationPlan,
        machine_deltaps: list[float],
        arrays_cache: dict[int, dict[str, np.ndarray]],
        track_data: dict[int, pd.DataFrame],
        n_run_turns: int,
    ) -> TrackingData:
        """Build the serialisable tracking payload for one worker plan."""
        pos, mom, pos_var, mom_var, init_coords, pts = self.make_worker_payload(
            turn_batch=turn_batch,
            file_turn_map=file_turn_map,
            start_bpm=plan.range_spec.start_bpm,
            end_bpm=plan.range_spec.end_bpm,
            sdir=plan.range_spec.sdir,
            bpm_names=plan.bpm_names,
            kick_plane=plan.kick_plane,
            init_marker=plan.init_marker,
            machine_deltaps=machine_deltaps,
            arrays_cache=arrays_cache,
            track_data=track_data,
            n_run_turns=n_run_turns,
        )
        self.freeze_payload_arrays(pos, mom, pos_var, mom_var)
        return TrackingData(
            position_comparisons=pos,
            momentum_comparisons=mom,
            position_variances=pos_var,
            momentum_variances=mom_var,
            init_coords=init_coords,
            init_pts=pts,
            precomputed_weights=None,
        )

    @staticmethod
    def attach_global_weights(
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        num_batches: int,
        *,
        optimise_momenta: bool = True,
    ) -> list[tuple[TrackingData, WorkerConfig, int]]:
        """Precompute globally normalised weights for all tracking workers."""
        if not payloads:
            return payloads

        def active_observables(config: WorkerConfig) -> tuple[str, ...]:
            kick_plane = getattr(config.kick_plane, "value", config.kick_plane)
            if kick_plane == "x":
                return ("x", "px") if optimise_momenta else ("x",)
            if kick_plane == "y":
                return ("y", "py") if optimise_momenta else ("y",)
            if kick_plane == "xy":
                return ("x", "y", "px", "py") if optimise_momenta else ("x", "y")
            raise ValueError(f"Unsupported kick plane {config.kick_plane!r}")

        observable_arrays = ("x", "y", "px", "py")
        payload_data: list[tuple[TrackingData, int, list[np.ndarray], tuple[str, ...]]] = []
        for data, config, file_idx in payloads:
            n_init = len(data.init_coords)
            if n_init <= 0:
                raise ValueError(
                    f"Worker payload for file {file_idx} has no initial coordinates"
                )
            var_slices = [
                data.position_variances[:n_init, :, 0],
                data.position_variances[:n_init, :, 1],
                data.momentum_variances[:n_init, :, 0],
                data.momentum_variances[:n_init, :, 1],
            ]
            payload_data.append((data, file_idx, var_slices, active_observables(config)))

        all_variances = [
            [var_slices[i] for _, _, var_slices, active in payload_data if observable in active]
            for i, observable in enumerate(observable_arrays)
        ]
        floors = [
            WeightProcessor.compute_variance_floor(
                np.concatenate([values.reshape(-1) for values in dim_vars])
            )
            if dim_vars
            else None
            for dim_vars in all_variances
        ]

        weight_cache: list[tuple[TrackingData, int, list[np.ndarray], tuple[str, ...]]] = []
        global_max = 0.0
        for data, file_idx, var_slices, active in payload_data:
            raw_weights = [
                WeightProcessor.variance_to_weight(
                    WeightProcessor.floor_variances(var_slice, floor_value=floor)
                )
                for var_slice, floor in zip(var_slices, floors, strict=True)
            ]
            active_maxima = [
                np.max(raw_weights[i]) if raw_weights[i].size else 0.0
                for i, observable in enumerate(observable_arrays)
                if observable in active
            ]
            global_max = max(global_max, max(active_maxima, default=0.0))
            weight_cache.append((data, file_idx, raw_weights, active))

        normaliser = global_max if global_max > 0.0 else 1.0
        if global_max == 0.0:
            LOGGER.warning("All computed weights are zero; skipping global normalisation")

        for data, file_idx, raw_weights, _active in weight_cache:
            normalised = [weights / normaliser for weights in raw_weights]
            data.precomputed_weights = PrecomputedTrackingWeights(
                x=normalised[0],
                y=normalised[1],
                px=normalised[2],
                py=normalised[3],
                hessian_x=WeightProcessor.aggregate_hessian_weights(raw_weights[0]),
                hessian_y=WeightProcessor.aggregate_hessian_weights(raw_weights[1]),
                hessian_px=WeightProcessor.aggregate_hessian_weights(raw_weights[2]),
                hessian_py=WeightProcessor.aggregate_hessian_weights(raw_weights[3]),
            )
            LOGGER.debug(
                "Attached precomputed weights to worker payload for file %d\n"
                "x max=%.3e, min=%.3e, mean=%.3e\n"
                "y max=%.3e, min=%.3e, mean=%.3e\n"
                "px max=%.3e, min=%.3e, mean=%.3e\n"
                "py max=%.3e, min=%.3e, mean=%.3e\n",
                file_idx,
                np.max(normalised[0]),
                np.min(normalised[0]),
                np.mean(normalised[0]),
                np.max(normalised[1]),
                np.min(normalised[1]),
                np.mean(normalised[1]),
                np.max(normalised[2]),
                np.min(normalised[2]),
                np.mean(normalised[2]),
                np.max(normalised[3]),
                np.min(normalised[3]),
                np.mean(normalised[3]),
            )

        LOGGER.info(
            "Global weight normalisation complete: max weight=%.3e across %d payloads",
            global_max,
            len(payloads),
        )
        return payloads

    @staticmethod
    def expand_bpm_mask(mask: np.ndarray, n_run_turns: int) -> np.ndarray:
        """Expand a per-BPM mask across repeated turns."""
        if n_run_turns <= 1:
            return mask
        return np.tile(mask, n_run_turns)

    @staticmethod
    def diagnostic_loss_per_bpm(
        loss_per_point: np.ndarray,
        bpm_names: list[str],
        n_run_turns: int,
        worker_id: int,
    ) -> np.ndarray:
        """Reduce point-wise diagnostic losses to one value per BPM."""
        expected_points = len(bpm_names) * n_run_turns
        if loss_per_point.size != expected_points:
            raise RuntimeError(
                f"Worker {worker_id}: diagnostics size mismatch "
                f"(got {loss_per_point.size}, expected {expected_points} = "
                f"{len(bpm_names)} BPMs x {n_run_turns} turns)"
            )
        if n_run_turns == 1:
            return loss_per_point
        return loss_per_point.reshape(n_run_turns, len(bpm_names)).sum(axis=0)
