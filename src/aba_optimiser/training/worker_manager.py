"""Worker orchestration for tracking optimisation.

`WorkerManager` intentionally focuses on process orchestration, screening, and
result collection. Worker-range selection lives in :mod:`worker_setup`, and
payload construction lives in :mod:`worker_payloads`.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, cast

import numpy as np

from aba_optimiser.training.tracking_mode import ArcByArcTrackingPlan, TrackingPlan
from aba_optimiser.training.validation_selection import (
    payload_track_count,
    split_validation_payloads,
)
from aba_optimiser.training.worker_payloads import WorkerPayloadBuilder
from aba_optimiser.training.worker_setup import WorkerRuntimeMetadata, WorkerSetupHelper
from aba_optimiser.workers import (
    PositionOnlyValidationTrackingWorker,
    TrackingData,
    TrackingWorker,
    ValidationTrackingWorker,
    WorkerConfig,
)
from aba_optimiser.workers.protocol import WorkerChannels, raise_for_worker_error_payload
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker

# Maps (optimise_momenta, validation) -> worker class.
# Add new worker types here without touching WorkerManager.
_WORKER_CLASS_REGISTRY: dict[tuple[bool, bool], type] = {
    (True, False): TrackingWorker,
    (False, False): PositionOnlyTrackingWorker,
    (True, True): ValidationTrackingWorker,
    (False, True): PositionOnlyValidationTrackingWorker,
}

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


class WorkerManager:
    """Create worker payloads, launch processes, and manage runtime coordination."""

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
        file_kick_planes: dict[int, str] | None = None,
        bad_bpms: list[str] | None = None,
        flattop_turns: int = 1000,
        num_tracks: int = 1,
        use_fixed_bpm: bool = True,
        debug: bool = False,
        mad_logfile: Path | None = None,
        python_logfile: Path | None = None,
        optimise_knobs: list[str] | None = None,
        tracking_plan: TrackingPlan | None = None,
    ) -> None:
        # `n_data_points` is kept for constructor compatibility with existing callers.
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
        self.bad_bpms = bad_bpms
        self.all_bpms = all_bpms
        self.file_kick_planes = file_kick_planes or {}
        self.use_fixed_bpm = use_fixed_bpm
        self.kinetic_energy = accelerator.kinetic_energy
        self.flattop_turns = flattop_turns
        self.num_tracks = num_tracks
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.python_logfile = python_logfile
        self.optimise_knobs = optimise_knobs
        self.tracking_plan = tracking_plan if tracking_plan is not None else ArcByArcTrackingPlan()
        self.worker_metadata: list[WorkerRuntimeMetadata] = []
        self.validation_parent_conns: list[Connection] = []
        self.validation_workers: list[mp.Process] = []
        self.validation_channels: WorkerChannels | None = None
        self.validation_metadata: list[WorkerRuntimeMetadata] = []
        self.validation_loss_weights: list[float] = []
        self.channels: WorkerChannels | None = None
        self.track_data: dict[int, pd.DataFrame] = {}
        self.turn_batches: list[list[int]] = []
        self.file_turn_map: dict[int, int] = {}
        self.start_bpms: list[str] = []
        self.end_bpms: list[str] = []
        self.simulation_config: SimulationConfig | None = None
        self.machine_deltaps: list[float] = []

        self.setup_helper = WorkerSetupHelper(
            accelerator=accelerator,
            all_bpms=all_bpms,
            fixed_start=fixed_start,
            fixed_end=fixed_end,
            use_fixed_bpm=use_fixed_bpm,
            bad_bpms=bad_bpms,
            file_kick_planes=self.file_kick_planes,
            magnet_range=magnet_range,
            corrector_strengths_files=corrector_strengths_files,
            tune_knobs_files=tune_knobs_files,
            debug=debug,
            mad_logfile=mad_logfile,
            python_logfile=python_logfile,
            tracking_plan=self.tracking_plan,
        )
        self.payload_builder = WorkerPayloadBuilder(
            accelerator=accelerator,
            all_bpms=all_bpms,
        )

    def _sync_helpers(self) -> None:
        """Keep helper objects aligned with mutable manager attributes."""
        self.setup_helper.bad_bpms = self.bad_bpms
        self.setup_helper.file_kick_planes = self.file_kick_planes
        self.setup_helper.corrector_strengths_files = self.corrector_strengths_files
        self.setup_helper.tune_knobs_files = self.tune_knobs_files
        self.setup_helper.tracking_plan = self.tracking_plan
        self.payload_builder.all_bpms = self.all_bpms

    def _channels(self) -> WorkerChannels:
        """Return the active training-worker channels."""
        if self.channels is None:
            raise RuntimeError("Worker channels are not initialised")
        return self.channels

    def _validation_channels(self) -> WorkerChannels:
        """Return the active validation-worker channel."""
        if self.validation_channels is None:
            raise RuntimeError("Validation worker channel is not initialised")
        return self.validation_channels

    @staticmethod
    def _select_worker_class(
        kick_plane: str,
        optimise_momenta: bool,
        *,
        validation: bool = False,
    ):
        """Select the worker implementation for a payload."""
        if kick_plane not in {"xy", "x", "y"}:
            raise ValueError(f"Unsupported kick plane {kick_plane!r}")
        worker_class = _WORKER_CLASS_REGISTRY.get((optimise_momenta, validation))
        if worker_class is None:
            raise ValueError(
                f"No worker class registered for optimise_momenta={optimise_momenta}, validation={validation}"
            )
        return worker_class

    @staticmethod
    def _summarise_file_usage(
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        num_files: int,
    ) -> None:
        """Log measurement-file usage and validate that at least one worker exists."""
        file_usage: dict[int, int] = {}
        for _, _, file_idx in payloads:
            file_usage[file_idx] = file_usage.get(file_idx, 0) + 1

        LOGGER.info(
            "Created %d workers using files: %s",
            len(payloads),
            ", ".join(f"file_{idx}={count} workers" for idx, count in sorted(file_usage.items())),
        )

        if len(file_usage) < num_files:
            LOGGER.warning(
                "Only %d/%d measurement files are being used by workers! "
                "This may lead to poor optimisation if different files have different deltap values.",
                len(file_usage),
                num_files,
            )
        if not file_usage:
            raise ValueError(
                "No worker payloads were created; check your input data and batch configuration"
            )

    def create_worker_payloads(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
    ) -> list[tuple[TrackingData, WorkerConfig, int]]:
        """Create per-worker data/config payloads from measurement files."""
        self._sync_helpers()
        payloads: list[tuple[TrackingData, WorkerConfig, int]] = []
        arrays_cache = {idx: self.payload_builder.extract_arrays(df) for idx, df in track_data.items()}
        file_available_bpms = {
            idx: set(df.index.get_level_values("name")) for idx, df in track_data.items()
        }
        plan_cache: dict[tuple[str, str, int, int], list] = {}
        range_specs = self.setup_helper.build_range_specs(start_bpms, end_bpms, simulation_config)

        LOGGER.info("Creating %d range specs x %d batches", len(range_specs), len(turn_batches))

        for range_spec in range_specs:
            for batch_idx, turn_batch in enumerate(turn_batches):
                if not turn_batch:
                    raise ValueError(
                        f"Empty batch {batch_idx} for {range_spec.start_bpm}/{range_spec.end_bpm}"
                    )

                primary_file_idx = self.setup_helper.get_primary_file_idx(turn_batch, file_turn_map)
                cache_key = (
                    range_spec.start_bpm,
                    range_spec.end_bpm,
                    range_spec.sdir,
                    primary_file_idx,
                )
                plans = plan_cache.get(cache_key)
                if plans is None:
                    plans = self.setup_helper.build_observation_plans(
                        range_spec,
                        primary_file_idx,
                        available_bpms=file_available_bpms.get(primary_file_idx),
                    )
                    plan_cache[cache_key] = plans
                if not plans:
                    LOGGER.debug(
                        "Skipping worker for %s/%s sdir=%d on file %d: no valid observation plan",
                        range_spec.start_bpm,
                        range_spec.end_bpm,
                        range_spec.sdir,
                        primary_file_idx,
                    )
                    continue

                for plan in plans:
                    data = self.payload_builder.make_tracking_data(
                        turn_batch=turn_batch,
                        file_turn_map=file_turn_map,
                        plan=plan,
                        machine_deltaps=machine_deltaps,
                        arrays_cache=arrays_cache,
                        track_data=track_data,
                        n_run_turns=simulation_config.n_run_turns,
                    )
                    config = self.setup_helper.make_worker_config(plan)
                    payloads.append((data, config, primary_file_idx))

                    LOGGER.debug(
                        "Worker %d: file=%d, range=%s/%s, sdir=%d, turns=%d, kick_plane=%s, observed_bpms=%d",
                        len(payloads) - 1,
                        primary_file_idx,
                        range_spec.start_bpm,
                        range_spec.end_bpm,
                        range_spec.sdir,
                        len(turn_batch),
                        plan.kick_plane,
                        len(plan.bpm_names),
                    )

        self._summarise_file_usage(payloads, len(self.corrector_strengths_files))
        return payloads

    def _build_payload_split(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
    ):
        """Build weighted training/validation payloads from current track data."""
        payloads = self.create_worker_payloads(
            track_data,
            turn_batches,
            file_turn_map,
            start_bpms,
            end_bpms,
            simulation_config,
            machine_deltaps,
        )
        payloads = self.payload_builder.attach_global_weights(
            payloads,
            simulation_config.num_batches,
            optimise_momenta=simulation_config.optimise_momenta,
        )
        return split_validation_payloads(payloads, LOGGER)

    def start_workers(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
        initial_knobs: dict[str, float],
        enable_validation: bool = True,
    ) -> None:
        """Start training workers plus one separate validation worker."""
        self.track_data = {}
        self.turn_batches = turn_batches
        self.file_turn_map = file_turn_map
        self.start_bpms = start_bpms
        self.end_bpms = end_bpms
        self.simulation_config = simulation_config
        self.machine_deltaps = machine_deltaps

        n_run_turns = 1 if simulation_config.run_arc_by_arc else simulation_config.n_run_turns
        worker_mode = "arc-by-arc" if simulation_config.run_arc_by_arc else "multi-turn"

        training_payloads, validation_payloads, duplicated_validation_payload = (
            self._build_worker_payloads(track_data, simulation_config, enable_validation)
        )

        LOGGER.info(
            "Worker tracking mode: %s (n_run_turns=%d)",
            worker_mode,
            simulation_config.n_run_turns,
        )
        LOGGER.info(
            "Starting %d trn worker(s) + %d val worker(s)",
            len(training_payloads),
            len(validation_payloads),
        )
        if duplicated_validation_payload:
            LOGGER.warning(
                "Validation payloads duplicate training payloads because a clean split would leave no training workers."
            )

        self.parent_conns = []
        self.workers = []
        self.worker_metadata = []
        self.validation_parent_conns = []
        self.validation_workers = []
        self.validation_channels = None
        self.validation_metadata = []
        self.validation_loss_weights = []
        self._worker_particle_counts: list[int] = []

        self._spawn_training_workers(
            training_payloads, simulation_config, worker_mode, n_run_turns, initial_knobs
        )
        if validation_payloads:
            self._spawn_validation_workers(
                validation_payloads,
                len(training_payloads),
                simulation_config,
                worker_mode,
                n_run_turns,
                initial_knobs,
            )

        self.channels = WorkerChannels(self.parent_conns, self.workers)
        self.validation_channels = (
            WorkerChannels(self.validation_parent_conns, self.validation_workers)
            if self.validation_workers
            else None
        )

    def _build_worker_payloads(
        self,
        track_data: dict[int, pd.DataFrame],
        simulation_config: SimulationConfig,
        enable_validation: bool,
    ) -> tuple[list, list, bool]:
        """Build training (and optional validation) payloads with global weights attached."""
        validation_split = (
            self._build_payload_split(
                track_data,
                self.turn_batches,
                self.file_turn_map,
                self.start_bpms,
                self.end_bpms,
                simulation_config,
                self.machine_deltaps,
            )
            if enable_validation
            else None
        )
        if validation_split is not None:
            return (
                validation_split.training_payloads,
                validation_split.validation_payloads,
                validation_split.duplicated_validation_payload,
            )

        training_payloads = self.create_worker_payloads(
            track_data,
            self.turn_batches,
            self.file_turn_map,
            self.start_bpms,
            self.end_bpms,
            simulation_config,
            self.machine_deltaps,
        )
        training_payloads = self.payload_builder.attach_global_weights(
            training_payloads,
            simulation_config.num_batches,
            optimise_momenta=simulation_config.optimise_momenta,
        )
        return training_payloads, [], False

    def _spawn_training_workers(
        self,
        training_payloads: list,
        simulation_config: SimulationConfig,
        worker_mode: str,
        n_run_turns: int,
        initial_knobs: dict[str, float],
    ) -> None:
        """Spawn one process per training payload and record its runtime metadata."""
        for worker_id, (data, config, file_idx) in enumerate(training_payloads):
            parent, child = mp.Pipe()
            worker_class = self._select_worker_class(
                config.kick_plane,
                simulation_config.optimise_momenta,
                validation=False,
            )
            worker = worker_class(
                child,
                worker_id,
                data,
                config,
                simulation_config,
                mode=worker_mode,
            )
            worker.start()
            self.parent_conns.append(parent)
            self.workers.append(worker)
            self._worker_particle_counts.append(len(data.init_coords))
            parent.send((initial_knobs, -1))
            bpm_names = self.setup_helper.get_worker_bpm_names(
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.kick_plane,
                config.bad_bpms,
            )
            self.worker_metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=worker_id,
                    file_idx=file_idx,
                    config=config,
                    bpm_names=bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            LOGGER.debug(
                "Trn worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d",
                worker_id,
                file_idx,
                config.tracking_start_bpm,
                config.tracking_end_bpm,
                config.sdir,
                config.kick_plane,
                len(bpm_names),
            )

    def _spawn_validation_workers(
        self,
        validation_payloads: list,
        training_worker_count: int,
        simulation_config: SimulationConfig,
        worker_mode: str,
        n_run_turns: int,
        initial_knobs: dict[str, float],
    ) -> None:
        """Spawn one validation process per validation payload and record its metadata."""
        covered_ranges: set[tuple[int, str, str]] = set()
        for val_offset, validation_payload in enumerate(validation_payloads):
            val_worker_id = training_worker_count + val_offset
            val_parent, val_child = mp.Pipe()
            val_data, val_config, val_file_idx = validation_payload
            validation_class = self._select_worker_class(
                val_config.kick_plane,
                simulation_config.optimise_momenta,
                validation=True,
            )
            val_worker = validation_class(
                val_child,
                val_worker_id,
                [validation_payload],
                simulation_config,
                mode=worker_mode,
            )
            val_worker.start()
            val_parent.send((initial_knobs, -1))
            self.validation_parent_conns.append(val_parent)
            self.validation_workers.append(val_worker)

            val_bpm_names = self.setup_helper.get_worker_bpm_names(
                val_config.tracking_start_bpm,
                val_config.tracking_end_bpm,
                val_config.sdir,
                val_config.kick_plane,
                val_config.bad_bpms,
            )
            self.validation_metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=val_worker_id,
                    file_idx=val_file_idx,
                    config=val_config,
                    bpm_names=val_bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            val_tracks = payload_track_count(validation_payload)
            self.validation_loss_weights.append(float(val_tracks))
            covered_ranges.add(
                (
                    val_file_idx,
                    val_config.tracking_start_bpm,
                    val_config.tracking_end_bpm,
                )
            )
            LOGGER.debug(
                "Val worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                val_worker_id,
                val_file_idx,
                val_config.tracking_start_bpm,
                val_config.tracking_end_bpm,
                val_config.sdir,
                val_config.kick_plane,
                len(val_bpm_names),
                val_tracks,
            )

        LOGGER.info(
            "Validation setup: payloads=%d, covered_ranges=%d, tracks=%d",
            len(validation_payloads),
            len(covered_ranges),
            int(sum(self.validation_loss_weights)),
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
        diagnostics: list[dict[str, object]] = []
        channels = self._channels()
        channels.send_all({"cmd": "diagnostics", "knobs": initial_knobs})
        for result in channels.recv_all():
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected diagnostics payload from worker: {type(result)}")
            diagnostics.append(result)  # ty:ignore[invalid-argument-type]
        return diagnostics

    def _build_bpm_masks_from_diagnostics(
        self,
        diagnostics: list[dict[str, object]],
        bpm_sigma_threshold: float,
    ) -> list[np.ndarray]:
        """Build keep-masks from per-BPM losses."""
        bpm_masks: list[np.ndarray] = []

        for meta, diag in zip(self.worker_metadata, diagnostics, strict=True):
            worker_id = int(diag["worker_id"])
            loss_per_point = np.asarray(diag["loss_per_bpm"], dtype=np.float64)
            loss_per_bpm = self.payload_builder.diagnostic_loss_per_bpm(
                loss_per_point=loss_per_point,
                bpm_names=meta.bpm_names,
                n_run_turns=meta.n_run_turns,
                worker_id=worker_id,
            )
            bpm_z = self._compute_positive_z_scores(loss_per_bpm)
            keep_mask = np.ones(len(meta.bpm_names), dtype=bool)
            outlier_indices = np.where(bpm_z > bpm_sigma_threshold)[0]

            for bpm_idx in outlier_indices:
                keep_mask[bpm_idx] = False
                LOGGER.warning(
                    "Worker %d: loss at BPM %s is %.2f standard deviations away from the mean, ignoring for optimisation.",
                    worker_id,
                    meta.bpm_names[bpm_idx],
                    bpm_z[bpm_idx],
                )

            bpm_masks.append(keep_mask)

        return bpm_masks

    def _classify_worker_outliers(
        self,
        worker_losses: np.ndarray,
        worker_sigma_threshold: float,
    ) -> list[bool]:
        """Identify high-loss worker outliers from adjusted worker losses."""
        worker_z = self._compute_positive_z_scores(worker_losses)
        worker_disabled: list[bool] = []
        n_disabled = 0

        for idx, meta in enumerate(self.worker_metadata):
            z_score = float(worker_z[idx])
            disable = z_score > worker_sigma_threshold
            worker_disabled.append(disable)
            if disable:
                n_disabled += 1
                LOGGER.warning(
                    "Worker %d with starting BPM %s is %.2f standard deviations away from the mean, ignoring.",
                    meta.worker_id,
                    meta.start_bpm,
                    z_score,
                )

        if n_disabled == 0:
            max_z = float(np.max(worker_z)) if worker_z.size else 0.0
            LOGGER.warning(
                "Worker outlier screening: no workers exceeded threshold %.2fσ (max z-score %.2f).",
                worker_sigma_threshold,
                max_z,
            )

        return worker_disabled

    def _summarise_screening_losses(
        self,
        diagnostics: list[dict[str, object]],
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
    ) -> None:
        """Log loss before masking and projected loss after masking/disabling."""
        raw_worker_losses: list[float] = []
        projected_worker_losses: list[float] = []

        for idx, (diag, mask, disable, meta) in enumerate(
            zip(diagnostics, bpm_masks, worker_disabled, self.worker_metadata, strict=True)
        ):
            loss_per_point = np.asarray(diag["loss_per_bpm"], dtype=np.float64)
            expanded_mask = self.payload_builder.expand_bpm_mask(mask, meta.n_run_turns)
            if loss_per_point.size != expanded_mask.size:
                raise RuntimeError(
                    f"Worker diagnostics at index {idx} has incompatible mask/point lengths "
                    f"({expanded_mask.size} mask points vs {loss_per_point.size} losses)"
                )

            raw_loss = float(np.nansum(loss_per_point))
            kept_loss = float(np.nansum(loss_per_point[expanded_mask])) if not disable else 0.0
            raw_worker_losses.append(raw_loss)
            projected_worker_losses.append(kept_loss)

        raw_total = float(np.sum(raw_worker_losses))
        projected_total = float(np.sum(projected_worker_losses))
        n_workers = max(1, len(raw_worker_losses))
        raw_mean = raw_total / n_workers
        projected_mean = projected_total / n_workers
        reduction = 100.0 * (1.0 - projected_total / raw_total) if raw_total > 0.0 else 0.0

        LOGGER.info(
            "Pre-screening loss summary: total=%.6e, mean/worker=%.6e",
            raw_total,
            raw_mean,
        )
        LOGGER.info(
            "Projected post-screening loss summary: total=%.6e, mean/worker=%.6e (reduction=%.2f%%)",
            projected_total,
            projected_mean,
            reduction,
        )

    def _apply_screening_actions(
        self,
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
    ) -> None:
        """Send mask/disable settings to workers and verify acknowledgements."""
        for conn, keep_mask, disable, meta in zip(
            self.parent_conns, bpm_masks, worker_disabled, self.worker_metadata, strict=True
        ):
            expanded_mask = self.payload_builder.expand_bpm_mask(keep_mask, meta.n_run_turns)
            conn.send(
                {
                    "cmd": "apply_mask",
                    "keep_bpm_mask": expanded_mask.tolist(),
                    "disable_worker": disable,
                }
            )

        acknowledgements = (
            [conn.recv() for conn in self.parent_conns]
            if self.channels is None
            else self._channels().recv_all()
        )
        for ack in acknowledgements:
            ack_dict = cast("dict[object, object]", ack) if isinstance(ack, dict) else None
            if ack_dict is None or ack_dict.get("status") != "ok":
                raise RuntimeError(f"Failed to apply worker mask settings: {ack}")

    @staticmethod
    def _assert_control_ack(response: object, *, command: str) -> None:
        response_dict = cast("dict[object, object]", response) if isinstance(response, dict) else None
        if response_dict is None or response_dict.get("status") != "ok":
            raise RuntimeError(f"Unexpected worker ack for {command} command: {response!r}")

    @staticmethod
    def _plane_value(kick_plane: object) -> str:
        """Return the string value for enum or string kick-plane fields."""
        return str(getattr(kick_plane, "value", kick_plane))

    @classmethod
    def _payload_key(
        cls,
        payload: tuple[TrackingData, WorkerConfig, int],
    ) -> tuple[int, str, str, int, str]:
        """Return stable identity fields for matching refreshed payloads."""
        _data, config, file_idx = payload
        return (
            file_idx,
            config.tracking_start_bpm,
            config.tracking_end_bpm,
            config.sdir,
            cls._plane_value(config.kick_plane),
        )

    def _assert_payload_keys_match(
        self,
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        metadata: list[WorkerRuntimeMetadata],
        *,
        label: str,
    ) -> None:
        """Ensure reconstructed payloads still correspond to the live workers."""
        if len(payloads) != len(metadata):
            raise RuntimeError(
                f"Cannot replace {label} tracking data: payload count changed "
                f"from {len(metadata)} to {len(payloads)}"
            )
        payload_keys = [self._payload_key(payload) for payload in payloads]
        metadata_keys = [
            (
                meta.file_idx,
                meta.start_bpm,
                meta.end_bpm,
                meta.sdir,
                self._plane_value(meta.kick_plane),
            )
            for meta in metadata
        ]
        if payload_keys != metadata_keys:
            raise RuntimeError(f"Cannot replace {label} tracking data: worker layout changed")

    def screen_initial_outliers(
        self,
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> None:
        """Screen and mask outliers before optimisation starts."""
        if not self.parent_conns:
            LOGGER.warning("No workers available for pre-optimisation outlier screening")
            return

        LOGGER.info(
            "Running pre-optimisation outlier screening (BPM=%.1fσ, worker=%.1fσ)",
            bpm_sigma_threshold,
            worker_sigma_threshold,
        )

        diagnostics = self._request_worker_diagnostics(initial_knobs)
        worker_losses: list[float] = []
        for idx, diag in enumerate(diagnostics):
            total_loss_raw = diag.get("total_loss")
            if not isinstance(total_loss_raw, int | float | np.floating):
                raise RuntimeError(
                    f"Worker diagnostics at index {idx} missing numeric total_loss: {diag}"
                )
            worker_losses.append(float(total_loss_raw))

        worker_disabled = self._classify_worker_outliers(
            np.asarray(worker_losses, dtype=np.float64),
            worker_sigma_threshold,
        )
        bpm_masks = self._build_bpm_masks_from_diagnostics(diagnostics, bpm_sigma_threshold)
        self._summarise_screening_losses(diagnostics, bpm_masks, worker_disabled)
        self._apply_screening_actions(bpm_masks, worker_disabled)

        LOGGER.warning(
            "Pre-optimisation screening complete: masked %d BPM entries across workers, disabled %d/%d workers",
            sum(int((~mask).sum()) for mask in bpm_masks),
            sum(worker_disabled),
            len(self.parent_conns),
        )

    def collect_worker_results(self, total_turns: int) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch."""
        total_loss = 0.0
        agg_grad = np.zeros_like(self.optimise_knobs, dtype=np.float64)
        if not self.parent_conns:
            raise RuntimeError("No workers to collect results from!")

        for i, result in enumerate(self._channels().recv_all()):
            if not isinstance(result, tuple) or len(result) != 3:
                raise_for_worker_error_payload(result)
            _, grad, loss = result  # ty:ignore[not-iterable]
            grad_flat = grad.flatten()
            if i == 0:
                agg_grad = grad_flat.copy()
            else:
                agg_grad += grad_flat
            total_loss += loss

        return total_loss / total_turns, agg_grad

    def send_init_condition_updates(self, new_px_py: np.ndarray) -> None:
        """Push updated initial px/py to every training worker.

        ``new_px_py`` must be a float64 array of shape ``(n_total_particles, 2)``
        where the rows are ordered to match the training workers in creation order
        and, within each worker, in particle order.  The total number of rows must
        equal ``sum(self._worker_particle_counts)``.

        Workers handle the update before processing the next gradient batch, so
        this method is safe to call between epochs (from the epoch_end_hook).
        """
        expected = sum(self._worker_particle_counts)
        if new_px_py.shape != (expected, 2):
            raise ValueError(
                f"new_px_py must have shape ({expected}, 2), got {new_px_py.shape}"
            )
        channels = self._channels()
        offset = 0
        for conn, n in zip(channels.parent_conns, self._worker_particle_counts):
            chunk = new_px_py[offset : offset + n]
            conn.send(
                {
                    "cmd": "update_init_coords",
                    "px": chunk[:, [0]],
                    "py": chunk[:, [1]],
                }
            )
            offset += n
        # Collect acknowledgements
        for conn, worker in zip(channels.parent_conns, channels.workers):
            WorkerChannels._recv(conn, worker)

    def compute_validation_loss(self, current_knobs: dict[str, float]) -> float | None:
        """Evaluate the held-out validation worker at the current knobs."""
        if self.validation_channels is None:
            return None

        self._validation_channels().send_all({"cmd": "validate", "knobs": current_knobs})
        results = self._validation_channels().recv_all()
        losses: list[float] = []
        for result in results:
            result_dict = cast("dict[object, object]", result) if isinstance(result, dict) else None
            if result_dict is None:
                raise RuntimeError(f"Unexpected validation payload from worker: {result!r}")

            loss_value = result_dict.get("loss")
            if not isinstance(loss_value, int | float | np.floating):
                raise RuntimeError(f"Validation worker payload missing numeric loss: {result!r}")
            losses.append(float(loss_value))

        if not losses:
            return None

        weights = np.asarray(self.validation_loss_weights, dtype=np.float64)
        if weights.size != len(losses):
            LOGGER.warning(
                "Validation weighting mismatch (weights=%d, losses=%d), using unweighted mean",
                weights.size,
                len(losses),
            )
            return float(np.mean(np.asarray(losses, dtype=np.float64)))

        if np.sum(weights) <= 0.0:
            return float(np.mean(np.asarray(losses, dtype=np.float64)))
        return float(np.average(np.asarray(losses, dtype=np.float64), weights=weights))

    def _stop_validation_workers(self) -> None:
        """Send termination signal to validation workers and wait for them to finish."""
        if self.validation_channels is not None:
            with contextlib.suppress(BrokenPipeError, EOFError):
                self.validation_channels.send_all((None, None))
        else:
            for conn in self.validation_parent_conns:
                with contextlib.suppress(BrokenPipeError, EOFError):
                    conn.send((None, None))
        for worker in self.validation_workers:
            worker.join()

    def terminate_workers(self) -> None:
        """Terminate training and validation workers and clean up processes."""
        LOGGER.info("Terminating workers...")
        if self.channels is not None:
            with contextlib.suppress(BrokenPipeError, EOFError):
                self.channels.send_all((None, None))
        else:
            for conn in self.parent_conns:
                with contextlib.suppress(BrokenPipeError, EOFError):
                    conn.send((None, None))

        for worker in self.workers:
            worker.join()
        self._stop_validation_workers()

    def termination_and_hessian(
        self,
        n_knobs: int,
        estimate_hessian: bool = True,
        parallelism: bool | int = True,
    ) -> np.ndarray:
        """Terminate training workers, collect Hessians, then stop validation."""
        LOGGER.info("Terminating workers...")
        hessians = self._collect_hessians(estimate_hessian, parallelism)
        self._stop_validation_workers()
        return np.add.reduce(hessians) if hessians else np.zeros((n_knobs, n_knobs))

    def _collect_hessians(self, estimate_hessian: bool, parallelism: bool | int) -> list[np.ndarray]:
        """Collect worker Hessians and shut workers down with bounded concurrency.

        When max_parallel >= n_workers (parallelism=True), all workers are shut down in a
        single broadcast using the pre-built self.channels. Otherwise workers are stopped in
        consecutive chunks of max_parallel; parallelism=False gives chunk size 1 (serial).
        """
        max_parallel = self._normalise_hessian_parallelism(parallelism)

        def _drain(channels: WorkerChannels, workers: list) -> list[np.ndarray]:
            if not estimate_hessian:
                channels.send_all({"cmd": "set_hessian_mode", "enabled": False})
                for response in channels.recv_all():
                    self._assert_control_ack(response, command="set_hessian_mode")
            channels.send_all((None, None))
            hessians = []
            for hessian in channels.recv_all():
                if not isinstance(hessian, np.ndarray):
                    raise RuntimeError(f"Unexpected Hessian payload from worker: {hessian!r}")
                hessians.append(hessian)
            for worker in workers:
                worker.join()
            return hessians

        if max_parallel >= len(self.workers):
            channels = self.channels if self.channels is not None else WorkerChannels(self.parent_conns, self.workers)
            return _drain(channels, self.workers)

        hessians: list[np.ndarray] = []
        for start in range(0, len(self.workers), max_parallel):
            chunk_conns = self.parent_conns[start : start + max_parallel]
            chunk_workers = self.workers[start : start + max_parallel]
            hessians.extend(_drain(WorkerChannels(chunk_conns, chunk_workers), chunk_workers))
        return hessians

    def _normalise_hessian_parallelism(self, parallelism: bool | int) -> int:
        """Convert config input into an explicit concurrency cap."""
        if isinstance(parallelism, bool):
            return len(self.workers) if parallelism else 1
        if parallelism < 1:
            raise ValueError("Hessian parallelism must be at least 1")
        return min(parallelism, len(self.workers))
