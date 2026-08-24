"""Worker orchestration for tracking optimisation.

`WorkerManager` intentionally focuses on process orchestration, screening, and
result collection. Worker-range selection lives in :mod:`worker_setup`, and
payload construction lives in :mod:`worker_payloads`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from aba_optimiser.training.config.tracking import ArcByArcTrackingPlan, TrackingPlan
from aba_optimiser.training.workers.payloads import WorkerPayloadBuilder
from aba_optimiser.training.workers.screening import OutlierScreener
from aba_optimiser.training.workers.setup import WorkerRuntimeMetadata, WorkerSetupHelper
from aba_optimiser.training.workers.spawning import WorkerSpawner
from aba_optimiser.training.workers.validation import (
    ValidationSplitResult,
    split_validation_payloads,
)
from aba_optimiser.workers.protocol import WorkerChannels, raise_for_worker_error_payload

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.connection import Connection
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers import TrackingData, WorkerConfig


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
        interface_options_per_file: list[dict],
        all_bpms: list[str],
        file_kick_planes: dict[int, str] | None = None,
        bad_bpms: list[str] | None = None,
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
        self.interface_options_per_file = interface_options_per_file
        self.bad_bpms = bad_bpms
        self.all_bpms = all_bpms
        self.file_kick_planes = file_kick_planes or {}
        self.use_fixed_bpm = use_fixed_bpm
        self.kinetic_energy = accelerator.kinetic_energy
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
        self._validation_worker_particle_counts: list[int] = []
        self.channels: WorkerChannels | None = None
        self.track_data: dict[int, pd.DataFrame] = {}
        self.turn_batches: list[list[int]] = []
        self.validation_turn_batches: list[list[int]] = []
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
            interface_options_per_file=interface_options_per_file,
            debug=debug,
            mad_logfile=mad_logfile,
            python_logfile=python_logfile,
            tracking_plan=self.tracking_plan,
        )
        self.payload_builder = WorkerPayloadBuilder(
            accelerator=accelerator,
            all_bpms=all_bpms,
            tracking_anchor_markers=self.tracking_plan.tracking_anchor_markers,
        )

    def _sync_helpers(self) -> None:
        """Keep helper objects aligned with mutable manager attributes."""
        self.setup_helper.bad_bpms = self.bad_bpms
        self.setup_helper.file_kick_planes = self.file_kick_planes
        self.setup_helper.interface_options_per_file = self.interface_options_per_file
        self.setup_helper.tracking_plan = self.tracking_plan
        self.setup_helper.tracking_anchor_markers = set(
            self.tracking_plan.tracking_anchor_markers
        )
        self.payload_builder.all_bpms = self.all_bpms
        self.payload_builder.tracking_anchor_markers = set(
            self.tracking_plan.tracking_anchor_markers
        )

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

        self._summarise_file_usage(payloads, len(self.interface_options_per_file))
        return payloads

    def _build_payload_split(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
    ) -> ValidationSplitResult:
        """Build weighted training/validation payloads from current track data.

        Training and validation payloads are built from disjoint turn sets (the
        held-out validation turns were removed from ``turn_batches`` upstream in
        ``DataManager``). Weights are normalised across the *combined* set so that
        training and validation losses live on the same scale and are comparable.
        """
        training_payloads = self.create_worker_payloads(
            track_data,
            turn_batches,
            file_turn_map,
            start_bpms,
            end_bpms,
            simulation_config,
            machine_deltaps,
        )
        if not validation_turn_batches:
            self.payload_builder.attach_global_weights(
                training_payloads,
                simulation_config.num_batches,
                optimise_momenta=simulation_config.optimise_momenta,
            )
            return ValidationSplitResult(training_payloads, [])

        validation_candidates = self.create_worker_payloads(
            track_data,
            validation_turn_batches,
            file_turn_map,
            start_bpms,
            end_bpms,
            simulation_config,
            machine_deltaps,
        )
        # Normalise weights over both sets at once (attach_global_weights mutates in
        # place) so validation loss is directly comparable to training loss.
        self.payload_builder.attach_global_weights(
            training_payloads + validation_candidates,
            simulation_config.num_batches,
            optimise_momenta=simulation_config.optimise_momenta,
        )
        return split_validation_payloads(training_payloads, validation_candidates, LOGGER)

    def start_workers(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        validation_turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
        machine_deltaps: list[float],
        initial_knobs: dict[str, float],
        enable_validation: bool = True,
    ) -> None:
        """Start training workers plus held-out validation workers."""
        self.track_data = {}
        self.turn_batches = turn_batches
        self.validation_turn_batches = validation_turn_batches
        self.file_turn_map = file_turn_map
        self.start_bpms = start_bpms
        self.end_bpms = end_bpms
        self.simulation_config = simulation_config
        self.machine_deltaps = machine_deltaps

        n_run_turns = 1 if simulation_config.run_arc_by_arc else simulation_config.n_run_turns
        worker_mode = "arc-by-arc" if simulation_config.run_arc_by_arc else "multi-turn"

        training_payloads, validation_payloads = self._build_worker_payloads(
            track_data, simulation_config, enable_validation
        )

        LOGGER.info(
            "Worker tracking mode: %s (n_run_turns=%d)",
            worker_mode,
            simulation_config.n_run_turns,
        )
        LOGGER.info(
            "Starting %d trn worker(s) + %d held-out val worker(s)",
            len(training_payloads),
            len(validation_payloads),
        )

        spawner = WorkerSpawner(self.setup_helper)
        training = spawner.spawn_training(
            training_payloads, simulation_config, worker_mode, n_run_turns, initial_knobs
        )
        self.parent_conns = training.parent_conns
        self.workers = training.workers
        self.worker_metadata = training.worker_metadata
        self._worker_particle_counts: list[int] = training.particle_counts

        self.validation_parent_conns = []
        self.validation_workers = []
        self.validation_metadata = []
        if validation_payloads:
            validation = spawner.spawn_validation(
                validation_payloads,
                len(training_payloads),
                simulation_config,
                worker_mode,
                n_run_turns,
                initial_knobs,
            )
            self.validation_parent_conns = validation.parent_conns
            self.validation_workers = validation.workers
            self.validation_metadata = validation.metadata
            self._validation_worker_particle_counts = validation.particle_counts

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
    ) -> tuple[list, list]:
        """Build training and held-out validation payloads with global weights attached."""
        validation_batches = self.validation_turn_batches if enable_validation else []
        split = self._build_payload_split(
            track_data,
            self.turn_batches,
            validation_batches,
            self.file_turn_map,
            self.start_bpms,
            self.end_bpms,
            simulation_config,
            self.machine_deltaps,
        )
        return split.training_payloads, split.validation_payloads

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
        OutlierScreener(self.payload_builder).screen(
            channels=self._channels(),
            parent_conns=self.parent_conns,
            worker_metadata=self.worker_metadata,
            initial_knobs=initial_knobs,
            bpm_sigma_threshold=bpm_sigma_threshold,
            worker_sigma_threshold=worker_sigma_threshold,
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

    def send_init_condition_updates(self, new_coords: np.ndarray) -> None:
        """Push updated initial ``x, px, y, py`` to every training and validation worker.

        ``new_coords`` must be a float64 array of shape ``(n_total_particles, 4)``
        whose columns are ``x, px, y, py`` and whose rows are ordered: training
        workers first (in creation order), then validation workers (in creation
        order), and within each worker in particle order.  The total number of
        rows must equal ``sum(self._worker_particle_counts) +
        sum(self._validation_worker_particle_counts)``.

        Positions travel with the momenta because the launch point can sit on a
        closed orbit that the fitted magnets themselves shape; see
        ``TrackingWorker._send_init_condition_update``. A caller with nothing new
        to say about position passes the current x/y straight back.

        Workers handle the update before processing the next gradient batch, so
        this method is safe to call between epochs (from the epoch_end_hook).
        """
        expected = sum(self._worker_particle_counts) + sum(self._validation_worker_particle_counts)
        if new_coords.shape != (expected, 4):
            raise ValueError(
                f"new_coords must have shape ({expected}, 4) of x, px, y, py; "
                f"got {new_coords.shape}"
            )

        def _send_to_channels(channels: WorkerChannels, counts: list[int], offset: int) -> int:
            for conn, n in zip(channels.parent_conns, counts):
                chunk = new_coords[offset : offset + n]
                conn.send({
                    "cmd": "update_init_coords",
                    **{
                        name: chunk[:, [column]]
                        for column, name in enumerate(("x", "px", "y", "py"))
                    },
                })
                offset += n
            for conn, worker in zip(channels.parent_conns, channels.workers):
                WorkerChannels._recv(conn, worker)
            return offset

        offset = _send_to_channels(self._channels(), self._worker_particle_counts, 0)
        if self.validation_channels is not None:
            _send_to_channels(
                self._validation_channels(), self._validation_worker_particle_counts, offset
            )

    def build_update_coords(self, updated_track_data: dict[int, pd.DataFrame]) -> np.ndarray:
        """Build the combined ``x, px, y, py`` array for training and validation workers.

        Returns a float64 array of shape ``(n_total_particles, 4)`` suitable for
        passing directly to :meth:`send_init_condition_updates`.  Training worker
        rows come first, followed by validation worker rows.

        When there are no validation workers the result is identical to extracting
        ``init_coords[:, :4]`` from :meth:`create_worker_payloads`.
        """
        has_validation = bool(self._validation_worker_particle_counts)
        if has_validation:
            split = self._build_payload_split(
                updated_track_data,
                self.turn_batches,
                self.validation_turn_batches,
                self.file_turn_map,
                self.start_bpms,
                self.end_bpms,
                self.simulation_config,
                self.machine_deltaps,
            )
            all_payloads = split.training_payloads + split.validation_payloads
        else:
            all_payloads = self.create_worker_payloads(
                updated_track_data,
                self.turn_batches,
                self.file_turn_map,
                self.start_bpms,
                self.end_bpms,
                self.simulation_config,
                self.machine_deltaps,
            )
        rows = [
            [float(value) for value in data.init_coords[i, :4]]
            for data, _config, _file_idx in all_payloads
            for i in range(len(data.init_coords))
        ]
        return np.asarray(rows, dtype=np.float64)

    def compute_validation_loss(self, current_knobs: dict[str, float]) -> float | None:
        """Evaluate the held-out validation workers at the current knobs.

        The validation workers track turns that were removed from training, so this
        is a genuine out-of-sample loss. Returns ``None`` when no validation workers
        exist (validation disabled or too little data), in which case the caller
        falls back to training loss.
        """
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

        # Each validation worker already reports a per-turn, per-BPM-point loss, so
        # combine them with an unweighted mean over workers -- the same reduction the
        # training loop uses (loop.py: total_loss / n_workers). This keeps the
        # validation number on the same scale as the reported training loss.
        return float(np.mean(np.asarray(losses, dtype=np.float64)))

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
        """Kill all workers immediately, for aborting after an error or interrupt.

        Unlike the clean shutdown in ``termination_and_hessian``, this makes no
        attempt to drain payloads or join gracefully: the workers may be wedged in
        a failed simulation and would never respond to a termination sentinel, so we
        send SIGTERM and reap them rather than waiting.
        """
        LOGGER.info("Terminating workers...")
        for worker in (*self.workers, *self.validation_workers):
            worker.terminate()
            worker.join()

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
