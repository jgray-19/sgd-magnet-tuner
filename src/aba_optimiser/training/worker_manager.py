"""Worker management for the optimisation controller."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.workers.base_worker import BaseWorker

from aba_optimiser.config import BEAM_ENERGY, DELTAP, PARTICLE_MASS
from aba_optimiser.physics.deltap import dp2pt

LOGGER = logging.getLogger(__name__)


class WorkerManager:
    """Manages worker processes for parallel optimisation."""

    def __init__(self, worker_class: type[BaseWorker], n_data_points: dict[str, int]):
        self.Worker = worker_class
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []

    def create_worker_payloads(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_start_points: list[str],
    ) -> list[dict]:
        """Create payloads for all workers."""
        payloads = []
        for batch_idx, turn_batch in enumerate(turn_batches):
            start_bpm = bpm_start_points[batch_idx % len(bpm_start_points)]
            assert turn_batch, (
                f"Turn batch {batch_idx} for BPM {start_bpm} is empty. "
                f"Check TRACKS_PER_WORKER and NUM_WORKERS."
            )
            x_comp, y_comp, init_coords = self._make_worker_payload(
                track_data,
                turn_batch,
                energy_turn_map,
                start_bpm,
                self.n_data_points[start_bpm],
            )
            x_comp.setflags(write=False)
            y_comp.setflags(write=False)
            payloads.append(
                {
                    "wid": batch_idx,
                    "x_comp": x_comp,
                    "y_comp": y_comp,
                    "init_coords": init_coords,
                    "start_bpm": start_bpm,
                }
            )
        return payloads

    def _make_worker_payload(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batch: list[int],
        energy_turn_map: dict[int, str],
        start_bpm: str,
        n_data_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create x/y comparison arrays and init_coords for a worker."""
        x_list, y_list, init_coord_list = [], [], []
        for t in turn_batch:
            energy_type = energy_turn_map[t]
            df = track_data[energy_type]
            if energy_type == "plus":
                pt = dp2pt(DELTAP, PARTICLE_MASS, BEAM_ENERGY)
            elif energy_type == "minus":
                pt = dp2pt(-DELTAP, PARTICLE_MASS, BEAM_ENERGY)
            else:
                pt = 0.0

            starting_row = df.loc[(t, start_bpm)]

            # Initial 4-element phase-space vector: x, px, y, py
            init_coords = starting_row[["x", "px", "y", "py"]].to_numpy(
                dtype="float64", copy=False
            )

            # Extend initcoords to include t=0 and pt
            init_coords = np.concatenate((init_coords, [0, pt]))
            init_coord_list.append(init_coords)

            obs_turns = self.Worker.get_observation_turns(t)

            blocks = []
            for ot in obs_turns:
                pos = df.index.get_loc((ot, start_bpm))
                blocks.append(df.iloc[pos : pos + n_data_points])

            filtered = pd.concat(blocks, axis=0)
            if filtered.shape[0] == 0:
                raise ValueError(f"No data available for turn {t}")

            x_list.append(filtered["x"].to_numpy(dtype="float64", copy=False))
            y_list.append(filtered["y"].to_numpy(dtype="float64", copy=False))

        x_comp = np.stack(x_list, axis=0)
        y_comp = np.stack(y_list, axis=0)
        init_coords = np.stack(init_coord_list, axis=0)
        return x_comp, y_comp, init_coords

    def start_workers(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_start_points: list[str],
        var_x: float,
        var_y: float,
        optimise_sextupoles: bool,
    ):
        """Start worker processes and return their parent connections.

        Note: process objects are stored internally on the manager and are
        joined/terminated in `terminate_workers()`; callers only need the
        connections to communicate with workers.
        """
        payloads = self.create_worker_payloads(
            track_data, turn_batches, energy_turn_map, bpm_start_points
        )

        for payload in payloads:
            parent, child = mp.Pipe()
            w = self.Worker(
                child,
                worker_id=payload["wid"],
                x_comparisons=payload["x_comp"],
                y_comparisons=payload["y_comp"],
                init_coords=payload["init_coords"],
                start_bpm=payload["start_bpm"],
                optimise_sextupoles=optimise_sextupoles,
            )
            w.start()
            self.parent_conns.append(parent)
            self.workers.append(w)
            parent.send((var_x, var_y))

    def send_knobs_to_workers(self, current_knobs: dict[str, float]) -> None:
        """Send current knob values to all workers."""
        for conn in self.parent_conns:
            conn.send(current_knobs)

    def collect_worker_results(self, total_turns: int) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch."""
        total_loss = 0.0
        agg_grad: None | np.ndarray = None
        for conn in self.parent_conns:
            _, grad, loss = conn.recv()
            agg_grad = grad if agg_grad is None else agg_grad + grad
            total_loss += loss
        total_loss /= total_turns
        agg_grad = agg_grad.flatten() / total_turns
        return total_loss, agg_grad

    def terminate_workers(self) -> np.ndarray:
        """Terminate all workers and collect final Hessian information."""
        LOGGER.info("Terminating workers...")

        # Signal workers to stop
        for conn in self.parent_conns:
            conn.send(None)

        # Collect Hessian information from workers
        hessians = []
        for conn in self.parent_conns:
            h_local = conn.recv()
            hessians.append(h_local)

        # Sum all Hessians
        h_global = sum(hessians)

        # Wait for all workers to finish
        for w in self.workers:
            w.join()

        return h_global
