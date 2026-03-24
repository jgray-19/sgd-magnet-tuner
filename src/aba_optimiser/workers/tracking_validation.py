"""Validation-only tracking workers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.mad.scripts import (
    build_validation_init_script,
    build_validation_script,
    dump_debug_script,
)
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import TrackingData, WorkerConfig, split_array_to_batches
from aba_optimiser.workers.tracking import OBSERVABLE_SPECS, TrackingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class _ValidationTask:
    """Prepared validation workload for one file/range/direction payload."""

    task_id: int
    file_idx: int
    config: WorkerConfig
    observables: tuple[str, ...]
    comparisons: dict[str, list[np.ndarray]]
    weights: dict[str, list[np.ndarray]]
    init_coords: list[list[list[float]]]
    init_pts: list[list[float]]
    batch_size: int
    num_batches: int
    normalisation_points: int
    keep_bpm_mask: np.ndarray
    run_track_init_text: str
    run_track_script: str
    track_count: int
    mad: MAD | None = None
    nbpms: int | None = None


class ValidationTrackingWorker(TrackingWorker):
    """Single-process validation worker that loops through multiple payloads."""

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        payloads: list[tuple[TrackingData, WorkerConfig, int]],
        simulation_config: SimulationConfig,
        mode: str = "multi-turn",
    ) -> None:
        if mode not in ("multi-turn", "arc-by-arc"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'multi-turn' or 'arc-by-arc'")
        if not payloads:
            raise ValueError("ValidationTrackingWorker requires at least one payload")

        self.mode = mode
        self.validation_payloads = payloads
        placeholder_config = payloads[0][1]
        AbstractWorker.__init__(self, conn, worker_id, payloads, placeholder_config, simulation_config)

    @staticmethod
    def _resolve_observables_for_config(
        config: WorkerConfig,
        include_momentum: bool,
    ) -> tuple[str, ...]:
        kick_plane = config.kick_plane
        if kick_plane == "xy":
            return ("x", "y", "px", "py") if include_momentum else ("x", "y")
        if kick_plane == "x":
            return ("x", "px") if include_momentum else ("x",)
        if kick_plane == "y":
            return ("y", "py") if include_momentum else ("y",)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    @staticmethod
    def _batch_observables(
        data: TrackingData,
        observables: tuple[str, ...],
        n_init: int,
        num_batches: int,
    ) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
        """Return batched comparison and weight arrays for the active observables."""
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for ValidationTrackingWorker")

        comparison_arrays: dict[str, np.ndarray] = {}
        weight_arrays: dict[str, np.ndarray] = {}
        for observable in observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            source = getattr(data, source_attr)[:n_init]
            comparison_arrays[observable] = source[:, :, plane_idx]
            weight_arrays[observable] = getattr(data.precomputed_weights, observable)[:n_init]

        comparisons = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in comparison_arrays.items()
        }
        weights = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in weight_arrays.items()
        }
        return comparisons, weights

    @staticmethod
    def _task_script_suffix(task: _ValidationTask) -> str:
        start = task.config.start_bpm.replace('.', '_')
        end = task.config.end_bpm.replace('.', '_')
        return f"task_{task.task_id}_{start}_{end}_sdir_{task.config.sdir}"

    def _dump_debug_scripts_for_task(self, task: _ValidationTask) -> None:
        """Write generated MAD scripts to disk when debugging is enabled."""
        suffix = self._task_script_suffix(task)
        dump_debug_script(
            f"run_val_init_{suffix}",
            task.run_track_init_text,
            debug=task.config.debug,
            mad_logfile=task.config.mad_logfile,
            worker_id=self.worker_id,
        )
        dump_debug_script(
            f"run_val_{suffix}",
            task.run_track_script,
            debug=task.config.debug,
            mad_logfile=task.config.mad_logfile,
            worker_id=self.worker_id,
        )

    def prepare_data(self, payloads: list[tuple[TrackingData, WorkerConfig, int]]) -> None:
        """Prepare all validation payloads for sequential evaluation in one process."""
        tasks: list[_ValidationTask] = []
        for task_id, (data, config, file_idx) in enumerate(payloads):
            observables = self._resolve_observables_for_config(config, self.include_momentum)
            num_batches = min(self.simulation_config.num_batches, len(data.init_coords))
            if num_batches <= 0:
                raise ValueError(f"Worker {self.worker_id}: No initial coordinates available")

            n_init = len(data.init_coords)
            init_coords = data.init_coords
            if np.isnan(init_coords).any():
                raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")

            comparisons, weights = self._batch_observables(data, observables, n_init, num_batches)
            init_coords_batches = split_array_to_batches(init_coords, num_batches)
            init_pts_batches = split_array_to_batches(data.init_pts[:n_init], num_batches)

            task = _ValidationTask(
                task_id=task_id,
                file_idx=file_idx,
                config=config,
                observables=observables,
                comparisons=comparisons,
                weights=weights,
                init_coords=[batch.tolist() for batch in init_coords_batches],
                init_pts=[batch.tolist() for batch in init_pts_batches],
                batch_size=len(init_coords_batches[0]),
                num_batches=num_batches,
                normalisation_points=comparisons[observables[0]][0].shape[1],
                keep_bpm_mask=np.ones(comparisons[observables[0]][0].shape[1], dtype=bool),
                run_track_init_text=build_validation_init_script(observables),
                run_track_script=build_validation_script(observables),
                track_count=int(n_init),
            )
            self._dump_debug_scripts_for_task(task)
            tasks.append(task)

        self.validation_tasks = tasks
        self._activate_task(tasks[0])

    def _activate_task(self, task: _ValidationTask) -> None:
        """Load one prepared task into the inherited TrackingWorker helpers."""
        self.config = task.config
        self.bpm_range = f"{task.config.start_bpm}/{task.config.end_bpm}"
        self.tracking_range = self.bpm_range
        if task.config.sdir < 0:
            self.tracking_range = f"{task.config.end_bpm}/{task.config.start_bpm}"

        self.observables = task.observables
        self.comparisons = task.comparisons
        self.weights = task.weights
        self.init_coords = task.init_coords
        self.init_pts = task.init_pts
        self.batch_size = task.batch_size
        self.num_batches = task.num_batches
        self.normalisation_points = task.normalisation_points
        self.keep_bpm_mask = task.keep_bpm_mask
        self.run_track_init_text = task.run_track_init_text
        self.run_track_script = task.run_track_script

    def _setup_da_maps(self, mad: MAD) -> None:
        """Create only the coordinate DA state needed for numeric tracking."""
        mad.send("coord_names = {'x', 'px', 'y', 'py', 't', 'pt'}")
        mad.send("da_x0_base = damap{nv=#coord_names, np=0, mo=1, po=1, vn=coord_names}")

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """Set up a non-gradient MAD interface using the active validation task."""
        del init_knobs
        LOGGER.debug("Worker %s: Setting up validation MAD interface", self.worker_id)
        LOGGER.debug("Worker %s: Using BPM range %s", self.worker_id, self.bpm_range)

        worker_logfile = self._resolve_per_worker_logfile(self.config.mad_logfile)
        init_bpm = self.config.start_bpm if self.config.sdir > 0 else self.config.end_bpm
        mad_iface = GradientDescentMadInterface(
            accelerator=self.config.accelerator,
            magnet_range=self.config.magnet_range,
            bpm_range=self.bpm_range,
            corrector_strengths=self.config.corrector_strengths,
            tune_knobs_file=self.config.tune_knobs_file,
            bad_bpms=self.config.bad_bpms,
            debug=self.config.debug,
            mad_logfile=worker_logfile,
            py_name="python",
            start_bpm=init_bpm,
        )

        mad = mad_iface.mad
        mad["nbpms"] = mad_iface.nbpms
        mad["sdir"] = self.config.sdir
        mad.load("MAD", "damap", "matrix", "vector")

        self.setup_mad_sequence(mad)
        self._setup_da_maps(mad)
        return mad, mad_iface.nbpms

    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for validation computation."""
        mad.send(self.run_track_init_text)

    def _receive_tracking_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive only observable arrays from MAD-NG."""
        results: dict[str, np.ndarray] = {}
        for observable in self.observables:
            results[observable] = np.asarray(mad.recv()).squeeze(-1)
        return results

    def _run_tracking_batch(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> dict[str, np.ndarray]:
        """Run MAD-NG tracking for one validation batch."""
        machine_pt = knob_updates.get("pt", 0.0)

        update_commands = [
            f"loaded_sequence['{name}'] = {val:.15e}" for name, val in knob_updates.items() if name != "pt"
        ]
        if update_commands:
            mad.send("\n".join(update_commands))

        mad.send(f"batch = {batch + 1}")
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")
        mad.send(self.run_track_script)
        return self._receive_tracking_results(mad)

    def compute_gradients_and_loss(self, mad: MAD, knob_updates: dict[str, float], batch: int):
        """Validation workers do not compute gradients."""
        del mad, knob_updates, batch
        raise NotImplementedError("ValidationTrackingWorker does not compute gradients")

    def compute_validation_loss(self, mad: MAD, knob_updates: dict[str, float]) -> float:
        """Return validation loss with the same per-payload normalization as training."""
        total_loss = 0.0
        for batch in range(self.num_batches):
            results = self._run_tracking_batch(mad, knob_updates, batch)
            batch_loss, _ = self._compute_loss_and_bpm_contributions(results, batch)
            total_loss += batch_loss / self.normalisation_points
        return total_loss / max(1, self.num_batches)

    def _compute_weighted_validation_loss(self, knob_updates: dict[str, float]) -> float:
        """Aggregate all validation payloads inside this single process."""
        weighted_loss = 0.0
        total_tracks = 0.0
        for task in self.validation_tasks:
            if task.mad is None:
                raise RuntimeError(
                    f"Worker {self.worker_id}: validation task {task.task_id} was not initialised"
                )
            self._activate_task(task)
            task_loss = self.compute_validation_loss(task.mad, knob_updates)
            weighted_loss += task.track_count * task_loss
            total_tracks += task.track_count
        if total_tracks <= 0.0:
            return 0.0
        return weighted_loss / total_tracks

    @staticmethod
    def _parse_knobs(command: dict[str, object], worker_id: int) -> dict[str, float]:
        raw_knobs = command.get("knobs", {})
        if not isinstance(raw_knobs, dict):
            raise ValueError(f"Worker {worker_id}: validation command missing knob dictionary")

        parsed: dict[str, float] = {}
        for knob_name, knob_value in raw_knobs.items():
            if not isinstance(knob_name, str):
                raise ValueError(f"Worker {worker_id}: knob name {knob_name!r} is not a string")
            if not isinstance(knob_value, int | float | np.floating):
                raise ValueError(
                    f"Worker {worker_id}: knob {knob_name!r} has non-numeric value {knob_value!r}"
                )
            parsed[knob_name] = float(knob_value)
        return parsed

    def run(self) -> None:
        """Main validation-worker run loop."""
        try:
            self.configure_python_worker_logging()
            knob_values, _batch = self.conn.recv()
            if knob_values is None:
                return

            total_tracks = 0
            for task in self.validation_tasks:
                self._activate_task(task)
                task.mad, task.nbpms = self.setup_mad_interface(knob_values)
                self.send_initial_conditions(task.mad)
                self._initialise_mad_computation(task.mad)
                total_tracks += task.track_count
                LOGGER.debug(
                    "Worker %s validation task %d: file=%d, range=%s/%s, sdir=%d, kick_plane=%s, tracks=%d, bpms=%d",
                    self.worker_id,
                    task.task_id,
                    task.file_idx,
                    task.config.start_bpm,
                    task.config.end_bpm,
                    task.config.sdir,
                    task.config.kick_plane,
                    task.track_count,
                    task.nbpms,
                )

            LOGGER.debug(
                "Worker %s: Ready for validation with %d payloads and %d tracks",
                self.worker_id,
                len(self.validation_tasks),
                total_tracks,
            )

            while True:
                message = self.conn.recv()
                if isinstance(message, tuple):
                    knob_values, batch = message
                    if knob_values is None or batch is None:
                        LOGGER.debug("Worker %s: Received termination signal", self.worker_id)
                        break
                    raise ValueError(
                        f"Worker {self.worker_id}: unexpected batch command for validation worker"
                    )

                if not isinstance(message, dict):
                    raise ValueError(
                        f"Worker {self.worker_id}: unexpected validation payload {type(message)}"
                    )

                cmd = message.get("cmd")
                if cmd != "validate":
                    raise ValueError(f"Worker {self.worker_id}: unknown command {cmd}")

                parsed_knobs = self._parse_knobs(message, self.worker_id)
                loss = self._compute_weighted_validation_loss(parsed_knobs)
                self.conn.send(
                    {
                        "worker_id": self.worker_id,
                        "loss": loss,
                        "payloads": len(self.validation_tasks),
                        "tracks": total_tracks,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="validation")
        finally:
            LOGGER.debug("Worker %s: Terminating", self.worker_id)
            for task in getattr(self, "validation_tasks", []):
                if task.mad is not None:
                    task.mad.send("shush()")
                    task.mad = None


class PositionOnlyValidationTrackingWorker(ValidationTrackingWorker):
    """Validation worker that compares only x/y position observables."""

    observables = ("x", "y")
    include_momentum = False
    hessian_weight_order = ("x", "y")
