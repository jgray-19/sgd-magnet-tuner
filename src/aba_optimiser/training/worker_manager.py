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
        self.beam_energy = accelerator.beam_energy
        self.flattop_turns = flattop_turns
        self.num_tracks = num_tracks
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.python_logfile = python_logfile
        self.optimise_knobs = optimise_knobs
        self.worker_metadata: list[WorkerRuntimeMetadata] = []
        self.validation_parent_conns: list[Connection] = []
        self.validation_workers: list[mp.Process] = []
        self.validation_channels: WorkerChannels | None = None
        self.validation_metadata: list[WorkerRuntimeMetadata] = []
        self.validation_loss_weights: list[float] = []
        self.channels: WorkerChannels | None = None

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
        )
        self.payload_builder = WorkerPayloadBuilder(
            accelerator=accelerator,
            all_bpms=all_bpms,
            beam_energy=self.beam_energy,
        )

    def _sync_helpers(self) -> None:
        """Keep helper objects aligned with mutable manager attributes."""
        self.setup_helper.bad_bpms = self.bad_bpms
        self.setup_helper.file_kick_planes = self.file_kick_planes
        self.setup_helper.corrector_strengths_files = self.corrector_strengths_files
        self.setup_helper.tune_knobs_files = self.tune_knobs_files
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
        if validation:
            return (
                ValidationTrackingWorker
                if optimise_momenta
                else PositionOnlyValidationTrackingWorker
            )
        return TrackingWorker if optimise_momenta else PositionOnlyTrackingWorker

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
    ) -> None:
        """Start training workers plus one separate validation worker."""
        payloads = self.create_worker_payloads(
            track_data,
            turn_batches,
            file_turn_map,
            start_bpms,
            end_bpms,
            simulation_config,
            machine_deltaps,
        )
        n_run_turns = 1 if simulation_config.run_arc_by_arc else simulation_config.n_run_turns
        payloads = self.payload_builder.attach_global_weights(payloads, simulation_config.num_batches)
        validation_split = split_validation_payloads(payloads, LOGGER)
        training_payloads = validation_split.training_payloads
        validation_payloads = validation_split.validation_payloads
        duplicated_validation_payload = validation_split.duplicated_validation_payload

        worker_mode = "arc-by-arc" if simulation_config.run_arc_by_arc else "multi-turn"
        LOGGER.info(
            "Worker tracking mode: %s (n_run_turns=%d)",
            worker_mode,
            simulation_config.n_run_turns,
        )
        LOGGER.info(
            "Starting %d trn worker(s) + %d val worker(s)",
            len(training_payloads),
            1 if validation_payloads else 0,
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
            parent.send((initial_knobs, -1))
            bpm_names = self.setup_helper.get_worker_bpm_names(
                config.start_bpm,
                config.end_bpm,
                config.sdir,
                config.kick_plane,
                config.bad_bpms,
            )
            self.worker_metadata.append(
                self.setup_helper.make_runtime_metadata(
                    worker_id=worker_id,
                    config=config,
                    bpm_names=bpm_names,
                    n_run_turns=n_run_turns,
                )
            )
            LOGGER.debug(
                "Trn worker %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d",
                worker_id,
                file_idx,
                config.start_bpm,
                config.end_bpm,
                config.sdir,
                config.kick_plane,
                len(bpm_names),
            )

        if validation_payloads:
            val_worker_id = len(training_payloads)
            val_parent, val_child = mp.Pipe()
            first_val_config = validation_payloads[0][1]
            validation_class = self._select_worker_class(
                first_val_config.kick_plane,
                simulation_config.optimise_momenta,
                validation=True,
            )
            val_worker = validation_class(
                val_child,
                val_worker_id,
                validation_payloads,
                simulation_config,
                mode=worker_mode,
            )
            val_worker.start()
            val_parent.send((initial_knobs, -1))

            total_val_tracks = 0
            covered_ranges: set[tuple[int, str, str]] = set()
            self.validation_parent_conns.append(val_parent)
            self.validation_workers.append(val_worker)

            for val_data, val_config, val_file_idx in validation_payloads:
                val_bpm_names = self.setup_helper.get_worker_bpm_names(
                    val_config.start_bpm,
                    val_config.end_bpm,
                    val_config.sdir,
                    val_config.kick_plane,
                    val_config.bad_bpms,
                )
                self.validation_metadata.append(
                    self.setup_helper.make_runtime_metadata(
                        worker_id=val_worker_id,
                        config=val_config,
                        bpm_names=val_bpm_names,
                        n_run_turns=n_run_turns,
                    )
                )
                val_tracks = payload_track_count((val_data, val_config, val_file_idx))
                total_val_tracks += val_tracks
                covered_ranges.add((val_file_idx, val_config.start_bpm, val_config.end_bpm))
                LOGGER.debug(
                    "Val payload: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, observed_bpms=%d, tracks=%d",
                    val_file_idx,
                    val_config.start_bpm,
                    val_config.end_bpm,
                    val_config.sdir,
                    val_config.kick_plane,
                    len(val_bpm_names),
                    val_tracks,
                )

            self.validation_loss_weights.append(float(total_val_tracks))
            LOGGER.info(
                "Val worker %d: payloads=%d, covered_ranges=%d, tracks=%d",
                val_worker_id,
                len(validation_payloads),
                len(covered_ranges),
                total_val_tracks,
            )

        self.channels = WorkerChannels(self.parent_conns, self.workers)
        self.validation_channels = (
            WorkerChannels(self.validation_parent_conns, self.validation_workers)
            if self.validation_workers
            else None
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

        if self.validation_channels is not None:
            with contextlib.suppress(BrokenPipeError, EOFError):
                self.validation_channels.send_all((None, None))
        else:
            for conn in self.validation_parent_conns:
                with contextlib.suppress(BrokenPipeError, EOFError):
                    conn.send((None, None))

        for worker in self.workers:
            worker.join()
        for worker in self.validation_workers:
            worker.join()

    def termination_and_hessian(self, n_knobs: int, estimate_hessian: bool = True) -> np.ndarray:
        """Terminate training workers, collect Hessians, then stop validation."""
        LOGGER.info("Terminating workers...")
        del n_knobs
        channels = self._channels()
        if not estimate_hessian:
            channels.send_all({"cmd": "set_hessian_mode", "enabled": False})
            for response in channels.recv_all():
                response_dict = (
                    cast("dict[object, object]", response) if isinstance(response, dict) else None
                )
                if response_dict is None or response_dict.get("status") != "ok":
                    raise RuntimeError(
                        f"Unexpected worker ack for hessian mode command: {response!r}"
                    )
        channels.send_all((None, None))
        hessians = []
        for hessian in channels.recv_all():
            if not isinstance(hessian, np.ndarray):
                raise RuntimeError(f"Unexpected Hessian payload from worker: {hessian!r}")
            hessians.append(hessian)
        for worker in self.workers:
            worker.join()

        if self.validation_channels is not None:
            with contextlib.suppress(BrokenPipeError, EOFError):
                self.validation_channels.send_all((None, None))
        else:
            for conn in self.validation_parent_conns:
                with contextlib.suppress(BrokenPipeError, EOFError):
                    conn.send((None, None))
        for worker in self.validation_workers:
            worker.join()
        return sum(hessians)
