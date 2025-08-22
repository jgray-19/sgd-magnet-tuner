"""Worker management for the optimisation controller."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.workers.base import BaseWorker

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
        comp: pd.DataFrame,
        turn_batches: list[list[int]],
        bpm_start_points: list[str],
    ) -> list[dict]:
        """Create payloads for all workers."""
        payloads = []
        wid = 0
        for start_bpm in bpm_start_points:
            for batch_idx, turn_batch in enumerate(turn_batches):
                assert turn_batch, (
                    f"Turn batch {batch_idx} for BPM {start_bpm} is empty. "
                    f"Check TRACKS_PER_WORKER and NUM_WORKERS."
                )
                x_comp, y_comp, init_coords = self._make_worker_payload(
                    comp, turn_batch, start_bpm, self.n_data_points[start_bpm]
                )
                payloads.append(
                    {
                        "wid": wid,
                        "x_comp": x_comp,
                        "y_comp": y_comp,
                        "init_coords": init_coords,
                        "start_bpm": start_bpm,
                    }
                )
                wid += 1
        return payloads

    def _make_worker_payload(
        self,
        comparison_dataframe: pd.DataFrame,
        turn_batch: list[int],
        start_bpm: str,
        n_data_points: int,
    ):
        """Create x/y comparison arrays and init_coords for a worker."""
        x_list, y_list, init_coords = [], [], []
        for t in turn_batch:
            starting_row = comparison_dataframe.loc[(t, start_bpm)]
            init_coords.append(
                [
                    starting_row["x"],
                    starting_row["px"],
                    starting_row["y"],
                    starting_row["py"],
                    0.0,
                    0.0,
                ]
            )
            obs_turns = self.Worker.get_observation_turns(t)

            blocks = []
            for ot in obs_turns:
                pos = comparison_dataframe.index.get_loc((ot, start_bpm))
                blocks.append(comparison_dataframe.iloc[pos : pos + n_data_points])

            filtered = pd.concat(blocks, axis=0)
            if filtered.shape[0] == 0:
                raise ValueError(f"No data available for turn {t}")

            x_list.append(filtered["x"].to_numpy(dtype="float64", copy=False))
            y_list.append(filtered["y"].to_numpy(dtype="float64", copy=False))

        x_comp = np.stack(x_list, axis=0)
        y_comp = np.stack(y_list, axis=0)
        return x_comp, y_comp, init_coords

    def start_workers(
        self,
        comp: pd.DataFrame,
        turn_batches: list[list[int]],
        bpm_start_points: list[str],
        var_x: float,
        var_y: float,
    ) -> list[Connection]:
        """Start worker processes and return their parent connections.

        Note: process objects are stored internally on the manager and are
        joined/terminated in `terminate_workers()`; callers only need the
        connections to communicate with workers.
        """
        payloads = self.create_worker_payloads(comp, turn_batches, bpm_start_points)

        for payload in payloads:
            parent, child = mp.Pipe()
            w = self.Worker(
                child,
                payload["wid"],
                payload["x_comp"],
                payload["y_comp"],
                payload["init_coords"],
                payload["start_bpm"],
            )
            w.start()
            self.parent_conns.append(parent)
            self.workers.append(w)
            parent.send((var_x, var_y))

        return self.parent_conns

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
