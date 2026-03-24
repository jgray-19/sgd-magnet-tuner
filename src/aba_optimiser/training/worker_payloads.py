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
from aba_optimiser.workers import PrecomputedTrackingWeights, TrackingData, WeightProcessor

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.worker_setup import WorkerObservationPlan
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
        beam_energy: float,
    ) -> None:
        self.accelerator = accelerator
        self.all_bpms = all_bpms
        self.beam_energy = beam_energy
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}

    def compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Compute transverse momentum based on file index."""
        return dp2pt(machine_deltaps[file_idx], PROTON_MASS, self.beam_energy)

    @staticmethod
    def freeze_payload_arrays(*arrays: np.ndarray) -> None:
        """Mark payload arrays as read-only before passing them to workers."""
        for array in arrays:
            array.setflags(write=False)

    def bpm_supports_plane(self, bpm: str, kick_plane: str) -> bool:
        """Return whether `bpm` can measure the requested kick plane."""
        plane = self.accelerator.infer_monitor_plane(bpm)
        if kick_plane == "x":
            return "H" in plane
        if kick_plane == "y":
            return "V" in plane
        if kick_plane == "xy":
            return ("H" in plane) or ("V" in plane)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    def bpm_supports_both_planes(self, bpm: str) -> bool:
        """Return whether `bpm` can measure both transverse planes."""
        return self.bpm_supports_plane(bpm, "x") and self.bpm_supports_plane(bpm, "y")

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
        df_id = id(df)
        bucket = self._pos_cache.setdefault(df_id, {})
        key = (turn, bpm)
        pos = bucket.get(key)
        if pos is None:
            pos = int(df.index.get_loc((turn, bpm)))
            bucket[key] = pos
        return pos

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
        if not bpm_names:
            raise ValueError("No BPMs available for observation")

        bpm_indices = {bpm: idx for idx, bpm in enumerate(self.all_bpms)}
        repeated_bpm_names = bpm_names * n_run_turns
        observation_turn = turn
        previous_idx = bpm_indices[repeated_bpm_names[0]]
        positions = [self._get_pos(df, observation_turn, repeated_bpm_names[0])]

        for bpm in repeated_bpm_names[1:]:
            bpm_idx = bpm_indices[bpm]
            if sdir == 1 and bpm_idx < previous_idx:
                observation_turn += 1
            elif sdir == -1 and bpm_idx > previous_idx:
                observation_turn -= 1

            positions.append(self._get_pos(df, observation_turn, bpm))
            previous_idx = bpm_idx

        return np.asarray(positions, dtype=np.int64)

    def get_measured_start_planes(self, init_bpm: str, kick_plane: str) -> tuple[bool, bool]:
        """Return which coordinates should be used for the initial condition."""
        if kick_plane == "x":
            return True, False
        if kick_plane == "y":
            return False, True
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
        self.validate_worker_bpm_names(bpm_names, kick_plane)

        for i, turn in enumerate(turn_batch):
            file_idx = file_turn_map[turn]
            cache = arrays_cache[file_idx]
            df = track_data[file_idx]
            base_x, base_y, base_px, base_py = cache["x"], cache["y"], cache["px"], cache["py"]
            base_vx, base_vy, base_vpx, base_vpy = (
                cache["var_x"],
                cache["var_y"],
                cache["var_px"],
                cache["var_py"],
            )

            pts[i] = self.compute_pt(file_idx, machine_deltaps)
            positions = self.get_observation_positions(
                df=df,
                bpm_names=bpm_names,
                sdir=sdir,
                turn=turn,
                n_run_turns=n_run_turns,
            )

            init_pos = int(positions[0])
            has_x, has_y = self.get_measured_start_planes(init_bpm, kick_plane)
            x_val = base_x[init_pos] if has_x else 0.0
            px_val = base_px[init_pos] if has_x else 0.0
            y_val = base_y[init_pos] if has_y else 0.0
            py_val = base_py[init_pos] if has_y else 0.0

            init_coords[i, :] = [x_val, px_val, y_val, py_val, 0.0, pts[i]]
            if all(init_coords[i, :] == 0.0):
                raise ValueError(
                    f"Initial coordinates for turn {turn} at BPM {init_bpm} are all zero"
                )

            x_values = base_x[positions]
            y_values = base_y[positions]
            px_values = base_px[positions]
            py_values = base_py[positions]
            vx_values = base_vx[positions]
            vy_values = base_vy[positions]
            vpx_values = base_vpx[positions]
            vpy_values = base_vpy[positions]

            if kick_plane == "x":
                pos[i, :, 0] = x_values
                mom[i, :, 0] = px_values
                pos_var[i, :, 0] = vx_values
                mom_var[i, :, 0] = vpx_values
                continue
            if kick_plane == "y":
                pos[i, :, 1] = y_values
                mom[i, :, 1] = py_values
                pos_var[i, :, 1] = vy_values
                mom_var[i, :, 1] = vpy_values
                continue

            pos[i, :, 0] = x_values
            pos[i, :, 1] = y_values
            mom[i, :, 0] = px_values
            mom[i, :, 1] = py_values
            pos_var[i, :, 0] = vx_values
            pos_var[i, :, 1] = vy_values
            mom_var[i, :, 0] = vpx_values
            mom_var[i, :, 1] = vpy_values

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
    ) -> list[tuple[TrackingData, WorkerConfig, int]]:
        """Precompute globally normalised weights for all tracking workers."""
        if not payloads:
            return payloads

        payload_data: list[tuple[TrackingData, int, list[np.ndarray]]] = []
        for data, _config, file_idx in payloads:
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
            payload_data.append((data, file_idx, var_slices))

        all_variances = [[var_slices[i] for _, _, var_slices in payload_data] for i in range(4)]
        floors = [
            WeightProcessor.compute_variance_floor(
                np.concatenate([values.reshape(-1) for values in dim_vars])
            )
            for dim_vars in all_variances
        ]

        weight_cache: list[tuple[TrackingData, int, list[np.ndarray]]] = []
        global_max = 0.0
        for data, file_idx, var_slices in payload_data:
            raw_weights = [
                WeightProcessor.variance_to_weight(
                    WeightProcessor.floor_variances(var_slice, floor_value=floor)
                )
                for var_slice, floor in zip(var_slices, floors, strict=True)
            ]
            global_max = max(
                global_max,
                max((np.max(weights) if weights.size else 0.0) for weights in raw_weights),
            )
            weight_cache.append((data, file_idx, raw_weights))

        normaliser = global_max if global_max > 0.0 else 1.0
        if global_max == 0.0:
            LOGGER.warning("All computed weights are zero; skipping global normalisation")

        for data, file_idx, raw_weights in weight_cache:
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
