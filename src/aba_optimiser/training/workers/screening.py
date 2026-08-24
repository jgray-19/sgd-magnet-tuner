"""Pre-optimisation outlier screening for tracking workers.

:class:`OutlierScreener` is the cohesive algorithm that the
:class:`~aba_optimiser.training.workers.manager.WorkerManager` runs once before
optimisation begins: it asks each worker for per-BPM diagnostics, flags BPMs and
whole workers whose loss is an outlier (positive-side z-score above a sigma
threshold), and pushes the resulting keep-masks back to the workers. It operates
purely on the worker metadata, connections, and channels passed in, so it holds
no worker-process state of its own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.training.workers.payloads import WorkerPayloadBuilder
    from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata
    from aba_optimiser.workers.protocol import WorkerChannels

LOGGER = logging.getLogger(__name__)


class OutlierScreener:
    """Detect and mask high-loss BPMs and workers before optimisation starts."""

    def __init__(self, payload_builder: WorkerPayloadBuilder) -> None:
        self.payload_builder = payload_builder

    def screen(
        self,
        *,
        channels: WorkerChannels,
        parent_conns: list[Connection],
        worker_metadata: list[WorkerRuntimeMetadata],
        initial_knobs: dict[str, float],
        bpm_sigma_threshold: float = 2.0,
        worker_sigma_threshold: float = 2.0,
    ) -> None:
        """Screen and mask outliers before optimisation starts."""
        if not parent_conns:
            LOGGER.warning("No workers available for pre-optimisation outlier screening")
            return

        LOGGER.info(
            "Running pre-optimisation outlier screening (BPM=%.1fσ, worker=%.1fσ)",
            bpm_sigma_threshold,
            worker_sigma_threshold,
        )

        diagnostics = self.request_worker_diagnostics(channels, initial_knobs)
        worker_losses: list[float] = []
        for idx, diag in enumerate(diagnostics):
            total_loss_raw = diag.get("total_loss")
            if not isinstance(total_loss_raw, int | float | np.floating):
                raise RuntimeError(
                    f"Worker diagnostics at index {idx} missing numeric total_loss: {diag}"
                )
            worker_losses.append(float(total_loss_raw))

        worker_disabled = self.classify_worker_outliers(
            np.asarray(worker_losses, dtype=np.float64),
            worker_metadata,
            worker_sigma_threshold,
        )
        bpm_masks = self.build_bpm_masks_from_diagnostics(
            diagnostics, worker_metadata, bpm_sigma_threshold
        )
        self.summarise_screening_losses(diagnostics, bpm_masks, worker_disabled, worker_metadata)
        self.apply_screening_actions(parent_conns, worker_metadata, bpm_masks, worker_disabled, channels)

        LOGGER.warning(
            "Pre-optimisation screening complete: masked %d BPM entries across workers, disabled %d/%d workers",
            sum(int((~mask).sum()) for mask in bpm_masks),
            sum(worker_disabled),
            len(parent_conns),
        )

    @staticmethod
    def compute_positive_z_scores(values: np.ndarray) -> np.ndarray:
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

    def request_worker_diagnostics(
        self,
        channels: WorkerChannels,
        initial_knobs: dict[str, float],
    ) -> list[dict[str, object]]:
        """Request diagnostics from all workers and return validated payloads."""
        diagnostics: list[dict[str, object]] = []
        channels.send_all({"cmd": "diagnostics", "knobs": initial_knobs})
        for result in channels.recv_all():
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected diagnostics payload from worker: {type(result)}")
            diagnostics.append(result)  # ty:ignore[invalid-argument-type]
        return diagnostics

    def build_bpm_masks_from_diagnostics(
        self,
        diagnostics: list[dict[str, object]],
        worker_metadata: list[WorkerRuntimeMetadata],
        bpm_sigma_threshold: float,
    ) -> list[np.ndarray]:
        """Build keep-masks from per-BPM losses."""
        bpm_masks: list[np.ndarray] = []

        for meta, diag in zip(worker_metadata, diagnostics, strict=True):
            worker_id = int(diag["worker_id"])
            loss_per_point = np.asarray(diag["loss_per_bpm"], dtype=np.float64)
            loss_per_bpm = self.payload_builder.diagnostic_loss_per_bpm(
                loss_per_point=loss_per_point,
                bpm_names=meta.bpm_names,
                n_run_turns=meta.n_run_turns,
                worker_id=worker_id,
            )
            bpm_z = self.compute_positive_z_scores(loss_per_bpm)
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

    def classify_worker_outliers(
        self,
        worker_losses: np.ndarray,
        worker_metadata: list[WorkerRuntimeMetadata],
        worker_sigma_threshold: float,
    ) -> list[bool]:
        """Identify high-loss worker outliers from adjusted worker losses."""
        worker_z = self.compute_positive_z_scores(worker_losses)
        worker_disabled: list[bool] = []
        n_disabled = 0

        for idx, meta in enumerate(worker_metadata):
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

    def summarise_screening_losses(
        self,
        diagnostics: list[dict[str, object]],
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
        worker_metadata: list[WorkerRuntimeMetadata],
    ) -> None:
        """Log loss before masking and projected loss after masking/disabling."""
        raw_worker_losses: list[float] = []
        projected_worker_losses: list[float] = []

        for idx, (diag, mask, disable, meta) in enumerate(
            zip(diagnostics, bpm_masks, worker_disabled, worker_metadata, strict=True)
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

    def apply_screening_actions(
        self,
        parent_conns: list[Connection],
        worker_metadata: list[WorkerRuntimeMetadata],
        bpm_masks: list[np.ndarray],
        worker_disabled: list[bool],
        channels: WorkerChannels | None = None,
    ) -> None:
        """Send mask/disable settings to workers and verify acknowledgements."""
        for conn, keep_mask, disable, meta in zip(
            parent_conns, bpm_masks, worker_disabled, worker_metadata, strict=True
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
            [conn.recv() for conn in parent_conns]
            if channels is None
            else channels.recv_all()
        )
        for ack in acknowledgements:
            ack_dict = cast("dict[object, object]", ack) if isinstance(ack, dict) else None
            if ack_dict is None or ack_dict.get("status") != "ok":
                raise RuntimeError(f"Failed to apply worker mask settings: {ack}")
