"""Worker management for the optimisation controller.

This module provides the WorkerManager class which handles the creation,
management, and coordination of worker processes for parallel optimisation
tasks in the ABA optimiser framework. It facilitates distributed computation
of loss functions and gradients across multiple worker processes.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import PROTON_MASS
from aba_optimiser.physics.deltap import dp2pt
from aba_optimiser.training.utils import create_bpm_range_specs, extract_bpm_range_names
from aba_optimiser.workers import (
    PrecomputedTrackingWeights,
    TrackingData,
    TrackingWorker,
    WeightProcessor,
    WorkerConfig,
)
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker processes for parallel optimisation.

    The WorkerManager is responsible for:
    - Creating and distributing work payloads to worker processes
    - Starting and managing multiprocessing workers
    - Collecting results and gradients from workers
    - Coordinating communication between the main process and workers
    - Handling worker termination and cleanup

    Attributes:
        Worker (type[BaseWorker]): The worker class to instantiate for each process
        n_data_points (dict[str, int]): Number of data points per BPM for payload creation
        parent_conns (list[Connection]): List of parent connections to worker processes
        workers (list[mp.Process]): List of worker process objects
    """

    def __init__(
        self,
        n_data_points: dict[tuple[str, str], int],
        ybpm: str,
        magnet_range: str,
        fixed_start: str,
        fixed_end: str,
        accelerator: Accelerator,
        corrector_strengths_files: list[Path],
        tune_knobs_files: list[Path],
        all_bpms: list[str],
        bad_bpms: list[str] | None = None,
        flattop_turns: int = 1000,
        num_tracks: int = 1,
        use_fixed_bpm: bool = True,
        debug: bool = False,
        mad_logfile: Path | None = None,
        optimise_knobs: list[str] | None = None,
    ):
        """Initialise the WorkerManager.

        Args:
            n_data_points: Number of data points per BPM pair
            ybpm: Y BPM name
            magnet_range: Range of magnets
            fixed_start: Fixed start BPM
            fixed_end: Fixed end BPM
            accelerator: Accelerator instance encapsulating machine parameters
            corrector_strengths_files: List of corrector strength files
            tune_knobs_files: List of tune knob files
            bad_bpms: List of bad BPMs
            flattop_turns: Number of flattop turns
            num_tracks: Number of tracks
            use_fixed_bpm: Whether to use fixed BPM
            debug: Debug mode
            mad_logfile: MAD log file path
            optimise_knobs: List of knobs to optimize
        """
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []
        self.y_bpm = ybpm
        self.magnet_range = magnet_range
        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self.accelerator = accelerator
        self.corrector_strengths_files = corrector_strengths_files
        self.tune_knobs_files = tune_knobs_files
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}
        self.bad_bpms = bad_bpms
        self.all_bpms = all_bpms
        self.use_fixed_bpm = use_fixed_bpm
        self.beam_energy = accelerator.beam_energy
        self.flattop_turns = flattop_turns
        self.num_tracks = num_tracks
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.optimise_knobs = optimise_knobs
        self.worker_metadata: list[dict[str, object]] = []

    def _compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Compute transverse momentum based on file index."""
        return dp2pt(machine_deltaps[file_idx], PROTON_MASS, self.beam_energy)

    def create_worker_payloads(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        machine_deltaps: list[float],
    ) -> list[tuple[TrackingData, WorkerConfig, int]]:
        """Create payloads for all workers.

        Args:
            track_data: Dictionary mapping file indices to track data DataFrames
            turn_batches: List of turn batches, each containing turn numbers for a worker
            file_turn_map: Mapping from turn numbers to file indices
            start_bpms: List of start BPMs (varying starts, track forward to fixed_end)
            end_bpms: List of end BPMs (varying ends, track backward from fixed_start)
            machine_deltaps: List of machine deltaps corresponding to each file
        """
        payloads: list[tuple[TrackingData, WorkerConfig, int]] = []
        arrays_cache = {idx: self._extract_arrays(df) for idx, df in track_data.items()}

        # Build range specs: (start, end, sdir)
        range_specs = create_bpm_range_specs(
            start_bpms, end_bpms, self.use_fixed_bpm, self.fixed_start, self.fixed_end
        )

        LOGGER.info(f"Creating {len(range_specs)} range specs × {len(turn_batches)} batches")

        for start_bpm, end_bpm, sdir in range_specs:
            for batch_idx, turn_batch in enumerate(turn_batches):
                if not turn_batch:
                    raise ValueError(f"Empty batch {batch_idx} for {start_bpm}/{end_bpm}")

                # All turns must be from same file
                primary_file_idx = file_turn_map[turn_batch[0]]
                if any(file_turn_map[t] != primary_file_idx for t in turn_batch):
                    raise ValueError(f"Batch {batch_idx} has turns from multiple files")

                # Create payload arrays
                pos, mom, pos_var, mom_var, init_coords, pts = self._make_worker_payload(
                    turn_batch,
                    file_turn_map,
                    start_bpm,
                    end_bpm,
                    self.n_data_points[(start_bpm, end_bpm)],
                    sdir,
                    machine_deltaps,
                    arrays_cache,
                    track_data,
                )

                for arr in (pos, mom, pos_var, mom_var):
                    arr.setflags(write=False)

                LOGGER.debug(
                    f"Worker {len(payloads)}: file={primary_file_idx}, range={start_bpm}/{end_bpm}, sdir={sdir}, turns={len(turn_batch)}"
                )
                data = TrackingData(
                    position_comparisons=pos,
                    momentum_comparisons=mom,
                    position_variances=pos_var,
                    momentum_variances=mom_var,
                    init_coords=init_coords,
                    init_pts=pts,
                    precomputed_weights=None,
                )
                config = WorkerConfig(
                    accelerator=self.accelerator,
                    start_bpm=start_bpm,
                    end_bpm=end_bpm,
                    magnet_range=self.magnet_range,
                    corrector_strengths=self.corrector_strengths_files[primary_file_idx],
                    tune_knobs_file=self.tune_knobs_files[primary_file_idx],
                    sdir=sdir,
                    bad_bpms=self.bad_bpms,
                    debug=self.debug,
                    mad_logfile=self.mad_logfile,
                )

                payloads.append((data, config, primary_file_idx))

        # Log summary of worker file assignments
        file_usage = {}
        for _, _, file_idx in payloads:
            file_usage[file_idx] = file_usage.get(file_idx, 0) + 1

        LOGGER.info(
            f"Created {len(payloads)} workers using files: "
            + ", ".join(f"file_{idx}={count} workers" for idx, count in sorted(file_usage.items()))
        )

        # Verify all files are used if we have enough batches
        num_files = len(self.corrector_strengths_files)
        if len(file_usage) < num_files:
            LOGGER.warning(
                f"Only {len(file_usage)}/{num_files} measurement files are being used by workers! "
                f"This may lead to poor optimisation if different files have different deltap values."
            )
        if len(file_usage) == 0:
            raise ValueError(
                "No worker payloads were created; check your input data and batch configuration"
            )

        return payloads

    def _generate_bpm_plane_masks(self, config: WorkerConfig) -> tuple[np.ndarray, np.ndarray]:
        """Generate plane masks for a worker's BPM range.

        Args:
            config: Worker configuration containing BPM range and bad BPMs

        Returns:
            Tuple of (h_mask, v_mask) as numpy arrays with shape (1, n_bpms)
        """
        # Order BPMs correctly for tracking direction
        bpm_list_ordered = extract_bpm_range_names(
            self.all_bpms, config.start_bpm, config.end_bpm, config.sdir
        )

        # Generate masks from accelerator
        bad_bpms = config.bad_bpms or []
        h_mask, v_mask = config.accelerator.get_bpm_plane_mask(bpm_list_ordered, bad_bpms)

        # Convert to numpy arrays with shape (1, n_bpms) for broadcasting
        h_mask_array = np.array(h_mask, dtype=np.float64).reshape(1, -1)
        v_mask_array = np.array(v_mask, dtype=np.float64).reshape(1, -1)

        return h_mask_array, v_mask_array

    def _attach_global_weights(
        self,
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        num_batches: int,
    ) -> list[tuple[TrackingData, WorkerConfig, int]]:
        """Precompute globally normalised weights for all tracking workers.

        Weighting is computed once across all worker payloads to ensure
        consistent scaling between workers. Also applies BPM plane masks for
        single-plane BPMs and per-plane bad BPM filtering.
        """
        if not payloads:
            return payloads

        # Extract variance slices for all payloads (shape: pos[:,:,0], pos[:,:,1], mom[:,:,0], mom[:,:,1])
        payload_data: list[tuple[TrackingData, WorkerConfig, int, list[np.ndarray]]] = []
        for data, config, file_idx in payloads:
            n_init = len(data.init_coords) - (len(data.init_coords) % num_batches)
            if n_init <= 0:
                raise ValueError(
                    f"Worker payload for file {file_idx} has no usable particles after batching"
                )
            var_slices = [
                data.position_variances[:n_init, :, 0],  # x
                data.position_variances[:n_init, :, 1],  # y
                data.momentum_variances[:n_init, :, 0],  # px
                data.momentum_variances[:n_init, :, 1],  # py
            ]
            payload_data.append((data, config, file_idx, var_slices))

        # Compute variance floors for all dimensions
        all_variances = [[var_slices[i] for _, _, _, var_slices in payload_data] for i in range(4)]
        floors = [
            WeightProcessor.compute_variance_floor(
                np.concatenate([v.reshape(-1) for v in dim_vars])
            )
            for dim_vars in all_variances
        ]

        # Compute weights and find global max (raw, unnormalised)
        weight_cache: list[tuple[TrackingData, WorkerConfig, int, list[np.ndarray]]] = []
        global_max = 0.0
        for data, config, file_idx, var_slices in payload_data:
            # Compute raw weights from variances
            raw_weights = [
                WeightProcessor.variance_to_weight(
                    WeightProcessor.floor_variances(var_slice, floor_value=floor)
                )
                for i, (var_slice, floor) in enumerate(zip(var_slices, floors))
            ]

            # Apply BPM plane masks (handles single-plane BPMs and per-plane bad BPM filtering)
            h_mask, v_mask = self._generate_bpm_plane_masks(config)
            raw_weights[0] *= h_mask  # x
            raw_weights[1] *= v_mask  # y
            raw_weights[2] *= h_mask  # px
            raw_weights[3] *= v_mask  # py

            global_max = max(
                global_max,
                max((np.max(w) if w.size else 0.0) for w in raw_weights),
            )
            weight_cache.append((data, config, file_idx, raw_weights))

        # Normalize and attach weights
        normaliser = global_max if global_max > 0.0 else 1.0
        if global_max == 0.0:
            LOGGER.warning("All computed weights are zero; skipping global normalisation")

        for data, config, file_idx, raw_weights in weight_cache:
            normalized = [w / normaliser for w in raw_weights]
            data.precomputed_weights = PrecomputedTrackingWeights(
                x=normalized[0],
                y=normalized[1],
                px=normalized[2],
                py=normalized[3],
                hessian_x=WeightProcessor.aggregate_hessian_weights(raw_weights[0]),
                hessian_y=WeightProcessor.aggregate_hessian_weights(raw_weights[1]),
                hessian_px=WeightProcessor.aggregate_hessian_weights(raw_weights[2]),
                hessian_py=WeightProcessor.aggregate_hessian_weights(raw_weights[3]),
            )
            LOGGER.debug(
                f"Attached precomputed weights to worker payload for file {file_idx}\n"
                f"x max={np.max(normalized[0]):.3e}, min={np.min(normalized[0]):.3e}, mean={np.mean(normalized[0]):.3e}\n"
                f"y max={np.max(normalized[1]):.3e}, min={np.min(normalized[1]):.3e}, mean={np.mean(normalized[1]):.3e}\n"
                f"px max={np.max(normalized[2]):.3e}, min={np.min(normalized[2]):.3e}, mean={np.mean(normalized[2]):.3e}\n"
                f"py max={np.max(normalized[3]):.3e}, min={np.min(normalized[3]):.3e}, mean={np.mean(normalized[3]):.3e}\n",
            )

        LOGGER.info(
            "Global weight normalisation complete: max weight=%.3e across %d payloads",
            global_max,
            len(payloads),
        )

        return payloads

    def _extract_arrays(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
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

    def _make_worker_payload(
        self,
        turn_batch: list[int],
        file_turn_map: dict[int, int],
        start_bpm: str,
        end_bpm: str,
        n_data_points: int,
        sdir: int,
        machine_deltaps: list[float],
        arrays_cache: dict[int, dict[str, np.ndarray]],
        track_data: dict[int, pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create arrays for worker payload."""
        n_turns = len(turn_batch)
        pos = np.empty((n_turns, n_data_points, 2), dtype="float64")
        mom = np.empty((n_turns, n_data_points, 2), dtype="float64")
        pos_var = np.empty((n_turns, n_data_points, 2), dtype="float64")
        mom_var = np.empty((n_turns, n_data_points, 2), dtype="float64")
        init_coords = np.empty((n_turns, 6), dtype="float64")
        pts = np.empty((n_turns,), dtype="float64")
        init_bpm = start_bpm if sdir == 1 else end_bpm

        for i, turn in enumerate(turn_batch):
            file_idx = file_turn_map[turn]
            cache = arrays_cache[file_idx]
            df = track_data[file_idx]
            arr_x, arr_y, arr_px, arr_py = cache["x"], cache["y"], cache["px"], cache["py"]
            arr_vx, arr_vy, arr_vpx, arr_vpy = (
                cache["var_x"],
                cache["var_y"],
                cache["var_px"],
                cache["var_py"],
            )

            pts[i] = self._compute_pt(file_idx, machine_deltaps)

            # Determine initial turn for backward tracking
            init_turn = turn
            if sdir == -1:
                start_pos = self._get_pos(df, turn, start_bpm)
                end_pos = start_pos + n_data_points - 1
                init_turn = self.get_turn(df, end_pos)

            init_pos = self._get_pos(df, init_turn, init_bpm)

            # Extract kick plane and zero out non-kicked planes in initial conditions
            kick_plane = df.iloc[init_pos]["kick_plane"]
            x_val = arr_x[init_pos] if "x" in kick_plane else 0.0
            px_val = arr_px[init_pos] if "x" in kick_plane else 0.0
            y_val = arr_y[init_pos] if "y" in kick_plane else 0.0
            py_val = arr_py[init_pos] if "y" in kick_plane else 0.0

            init_coords[i, :] = [x_val, px_val, y_val, py_val, 0.0, pts[i]]

            # Extract data slice
            sl = (
                slice(init_pos, init_pos - n_data_points, -1)
                if sdir == -1
                else slice(init_pos, init_pos + n_data_points)
            )
            pos[i, :, 0], pos[i, :, 1] = arr_x[sl], arr_y[sl]
            mom[i, :, 0], mom[i, :, 1] = arr_px[sl], arr_py[sl]
            pos_var[i, :, 0], pos_var[i, :, 1] = arr_vx[sl], arr_vy[sl]
            mom_var[i, :, 0], mom_var[i, :, 1] = arr_vpx[sl], arr_vpy[sl]

        return pos, mom, pos_var, mom_var, init_coords, pts

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

    def start_workers(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
    ):
        """Start worker processes.

        Args:
            track_data: Dictionary mapping file indices to track data DataFrames
            turn_batches: List of turn batches for worker distribution
            file_turn_map: Mapping from turn numbers to file indices
            start_bpms: List of start BPMs
            end_bpms: List of end BPMs
            simulation_config: Simulation configuration
            machine_deltaps: List of machine deltaps corresponding to each file
        """
        payloads = self.create_worker_payloads(
            track_data,
            turn_batches,
            file_turn_map,
            start_bpms,
            end_bpms,
            machine_deltaps,
        )
        payloads = self._attach_global_weights(payloads, simulation_config.num_batches)
        LOGGER.info(f"Starting {len(payloads)} workers...")

        # Select worker class based on whether momenta optimisation is enabled
        if simulation_config.optimise_momenta:
            worker_class = TrackingWorker
            LOGGER.info("Using TrackingWorker (position + momentum)")
        else:
            worker_class = PositionOnlyTrackingWorker
            LOGGER.info("Using PositionOnlyTrackingWorker (position only)")

        self.worker_metadata = []
        for worker_id, (data, config, _file_idx) in enumerate(payloads):
            parent, child = mp.Pipe()
            w = worker_class(child, worker_id, data, config, simulation_config, mode="arc-by-arc")
            w.start()
            self.parent_conns.append(parent)
            self.workers.append(w)
            parent.send(None)
            self.worker_metadata.append(
                {
                    "worker_id": worker_id,
                    "start_bpm": config.start_bpm,
                    "end_bpm": config.end_bpm,
                    "sdir": config.sdir,
                    "bpm_names": extract_bpm_range_names(
                        self.all_bpms,
                        config.start_bpm,
                        config.end_bpm,
                        config.sdir,
                    ),
                }
            )

    @staticmethod
    def _compute_positive_z_scores(values: np.ndarray) -> np.ndarray:
        """Compute positive-side z-scores; values below mean map to 0."""
        v = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(v)
        z = np.zeros_like(v, dtype=np.float64)
        if finite.sum() < 2:
            return z

        mean = float(np.mean(v[finite]))
        std = float(np.std(v[finite]))
        if std <= 0.0:
            return z

        z_vals = (v - mean) / std
        z[finite] = np.maximum(z_vals[finite], 0.0)
        return z

    def _request_worker_diagnostics(
        self, initial_knobs: dict[str, float]
    ) -> list[dict[str, object]]:
        """Request diagnostics from all workers and return validated payloads."""
        for conn in self.parent_conns:
            conn.send({"cmd": "diagnostics", "knobs": initial_knobs})

        diagnostics: list[dict[str, object]] = []
        for conn in self.parent_conns:
            result = conn.recv()
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected diagnostics payload from worker: {type(result)}")
            diagnostics.append(result)
        return diagnostics

    def _build_bpm_masks_from_diagnostics(
        self,
        diagnostics: list[dict[str, object]],
        bpm_sigma_threshold: float,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Build keep-masks from per-BPM losses and return adjusted worker losses."""
        bpm_masks: list[np.ndarray] = []
        adjusted_worker_losses: list[float] = []

        for meta, diag in zip(self.worker_metadata, diagnostics, strict=True):
            worker_id = int(diag["worker_id"])
            bpm_names = list(meta["bpm_names"])
            loss_per_bpm = np.asarray(diag["loss_per_bpm"], dtype=np.float64)
            if loss_per_bpm.size != len(bpm_names):
                raise RuntimeError(
                    f"Worker {worker_id}: diagnostics size mismatch "
                    f"({loss_per_bpm.size} losses vs {len(bpm_names)} BPMs)"
                )

            bpm_z = self._compute_positive_z_scores(loss_per_bpm)
            keep_mask = np.ones(loss_per_bpm.shape[0], dtype=bool)
            outlier_indices = np.where(bpm_z > bpm_sigma_threshold)[0]

            for bpm_idx in outlier_indices:
                keep_mask[bpm_idx] = False
                LOGGER.warning(
                    "Worker %d: loss at BPM %s is %.2f standard deviations away from the mean, ignoring for optimisation.",
                    worker_id,
                    bpm_names[bpm_idx],
                    bpm_z[bpm_idx],
                )

            bpm_masks.append(keep_mask)
            adjusted_worker_losses.append(float(np.sum(loss_per_bpm[keep_mask])))

        return bpm_masks, np.asarray(adjusted_worker_losses, dtype=np.float64)

    def _classify_worker_outliers(
        self,
        worker_losses: np.ndarray,
        worker_sigma_threshold: float,
    ) -> list[bool]:
        """Identify high-loss worker outliers from adjusted worker losses."""
        worker_z = self._compute_positive_z_scores(worker_losses)
        worker_disabled: list[bool] = []

        for idx, meta in enumerate(self.worker_metadata):
            worker_id = int(meta["worker_id"])
            start_bpm = str(meta["start_bpm"])
            z_score = float(worker_z[idx])
            disable = z_score > worker_sigma_threshold
            worker_disabled.append(disable)
            if disable:
                LOGGER.warning(
                    "Worker %d with starting BPM %s is %.2f standard deviations away from the mean, ignoring.",
                    worker_id,
                    start_bpm,
                    z_score,
                )

        return worker_disabled

    def _apply_screening_actions(
        self,
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
    ) -> None:
        """Send mask/disable settings to workers and verify acknowledgements."""
        for conn, keep_mask, disable in zip(
            self.parent_conns, bpm_masks, worker_disabled, strict=True
        ):
            conn.send(
                {
                    "cmd": "apply_mask",
                    "keep_bpm_mask": keep_mask.tolist(),
                    "disable_worker": disable,
                }
            )

        for conn in self.parent_conns:
            ack = conn.recv()
            if not isinstance(ack, dict) or ack.get("status") != "ok":
                raise RuntimeError(f"Failed to apply worker mask settings: {ack}")

    def screen_initial_outliers(
        self,
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> None:
        """Screen and mask outliers before optimisation starts.

        Performs two checks using diagnostics evaluated at initial knobs:
        1. Per-worker BPM losses: masks BPMs with z-score > bpm_sigma_threshold.
        2. Per-worker total losses: disables workers with z-score > worker_sigma_threshold.
        """
        if not self.parent_conns:
            LOGGER.warning("No workers available for pre-optimisation outlier screening")
            return

        LOGGER.info(
            "Running pre-optimisation outlier screening (BPM=%.1fσ, worker=%.1fσ)",
            bpm_sigma_threshold,
            worker_sigma_threshold,
        )

        diagnostics = self._request_worker_diagnostics(initial_knobs)
        bpm_masks, adjusted_worker_losses = self._build_bpm_masks_from_diagnostics(
            diagnostics,
            bpm_sigma_threshold,
        )
        worker_disabled = self._classify_worker_outliers(
            adjusted_worker_losses,
            worker_sigma_threshold,
        )
        self._apply_screening_actions(bpm_masks, worker_disabled)

        n_masked_bpms = sum(int((~mask).sum()) for mask in bpm_masks)
        n_disabled_workers = sum(worker_disabled)
        LOGGER.info(
            "Pre-optimisation screening complete: masked %d BPM entries across workers, disabled %d/%d workers",
            n_masked_bpms,
            n_disabled_workers,
            len(self.parent_conns),
        )

    def collect_worker_results(self, total_turns: int) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch.

        Aggregates gradients using per-knob averaging: each knob's gradient is
        averaged only over the workers that contributed a non-zero gradient for
        that knob. This prevents magnets at the edges of the BPM range (which
        are only visible to fewer workers) from being under-weighted compared
        to magnets in the middle (which contribute gradients from all workers).

        Args:
            total_turns (int): Total number of turns across all workers for normalisation

        Returns:
            tuple[float, np.ndarray]: A tuple containing:
                - total_loss: average loss across all turns
                - agg_grad: per-knob averaged gradient array
        """
        total_loss = 0.0
        agg_grad = np.zeros_like(self.optimise_knobs, dtype=np.float64)
        # contrib_count = None  # Count of non-zero contributions per knob
        if len(self.parent_conns) == 0:
            raise RuntimeError("No workers to collect results from!")

        for i, conn in enumerate(self.parent_conns):
            _, grad, loss = conn.recv()
            grad_flat = grad.flatten()

            if i == 0:
                agg_grad = grad_flat.copy()
                # contrib_count = (grad_flat != 0).astype(np.float64)
            else:
                agg_grad += grad_flat
                # contrib_count += (grad_flat != 0).astype(np.float64)

            total_loss += loss

        # Average each knob's gradient by the number of workers that contributed
        # to it (avoiding division by zero for knobs with no contributions)
        # contrib_count = np.maximum(contrib_count, 1.0)
        # avg_grad = agg_grad  / contrib_count

        return total_loss / total_turns, agg_grad

    def terminate_workers(self) -> None:
        """Terminate all workers and clean up processes."""
        LOGGER.info("Terminating workers...")
        for conn in self.parent_conns:
            with contextlib.suppress(BrokenPipeError, EOFError):
                conn.send((None, None))

        for w in self.workers:
            w.join()

    def termination_and_hessian(self, n_knobs: int) -> np.ndarray:
        """Terminate all workers and collect final Hessian information.

        Returns:
            np.ndarray: Global Hessian matrix computed as the sum of all worker Hessians

        Note:
            This method signals all workers to stop, collects their final Hessian
            contributions, and waits for all processes to finish before returning.
        """
        LOGGER.info("Terminating workers...")
        for conn in self.parent_conns:
            conn.send((None, None))

        hessians = []
        for conn in self.parent_conns:
            try:
                h = conn.recv()
            except EOFError:
                # Worker already terminated, use zero Hessian
                h = np.zeros((n_knobs, n_knobs))
            hessians.append(h)
        for w in self.workers:
            w.join()

        return sum(hessians)
