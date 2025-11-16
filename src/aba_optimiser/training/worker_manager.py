"""Worker management for the optimisation controller.

This module provides the WorkerManager class which handles the creation,
management, and coordination of worker processes for parallel optimisation
tasks in the ABA optimiser framework. It facilitates distributed computation
of loss functions and gradients across multiple worker processes.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    PARTICLE_MASS,
)
from aba_optimiser.physics.deltap import dp2pt
from aba_optimiser.workers.base_worker import WorkerConfig, WorkerData
from aba_optimiser.workers.arc_by_arc import ArcByArcWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.config import OptSettings


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
        n_data_points: dict[str, int],
        ybpm: str,
        magnet_range: str,
        sequence_file_path: Path,
        corrector_strengths_files: list[Path],
        tune_knobs_files: list[Path],
        bad_bpms: list[str] | None = None,
        seq_name: str | None = None,
        beam_energy: float = 6800.0,
        flattop_turns: int = 1000,
        num_tracks: int = 1,
    ):
        """Initialise the WorkerManager."""
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []
        self.y_bpm = ybpm
        self.magnet_range = magnet_range
        self.sequence_file_path = sequence_file_path
        self.corrector_strengths_files = corrector_strengths_files
        self.tune_knobs_files = tune_knobs_files
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}
        self.bad_bpms = bad_bpms
        self.seq_name = seq_name
        self.beam_energy = beam_energy
        self.flattop_turns = flattop_turns
        self.num_tracks = num_tracks

    def _compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Compute transverse momentum based on file index."""
        return dp2pt(machine_deltaps[file_idx], PARTICLE_MASS, self.beam_energy)

    def create_worker_payloads(
        self,
        track_data: dict[int, pd.DataFrame],
        turn_batches: list[list[int]],
        file_turn_map: dict[int, int],
        bpm_ranges: list[str],
        machine_deltaps: list[float],
        opt_settings: OptSettings,
    ) -> list[tuple[WorkerData, WorkerConfig, int]]:
        """Create payloads for all workers.

        Args:
            track_data (dict[int, pd.DataFrame]): Dictionary mapping file indices to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches, each containing turn numbers for a worker
            file_turn_map (dict[int, int]): Mapping from turn numbers to file indices
            bpm_ranges (list[str]): List of BPM ranges
            machine_deltaps (list[float]): List of machine deltaps corresponding to each file
            opt_settings (OptSettings): Optimisation settings containing configuration flags

        Returns:
            list[tuple[WorkerData, WorkerConfig, int]]: List of tuples containing:
                - WorkerData: Data arrays for the worker
                - WorkerConfig: Configuration parameters for the worker
                - int: Worker ID

        Raises:
            AssertionError: If a turn batch is empty
        """
        payloads = []
        # Precompute arrays once per file for memory efficiency
        arrays_cache = {idx: self._extract_arrays(df) for idx, df in track_data.items()}
        
        LOGGER.info(f"Creating worker payloads: {len(bpm_ranges)} BPM ranges x {len(turn_batches)} batches x 2 directions")
        for sdir in [1, -1]:
            for bpm_i, bpm_range in enumerate(bpm_ranges):
                start_bpm, end_bpm = bpm_range.split("/")
                for batch_idx, turn_batch in enumerate(turn_batches):
                    if opt_settings.different_turns_per_range and batch_idx % len(bpm_ranges) != bpm_i:
                        continue

                    if not turn_batch:
                        raise ValueError(f"Empty batch {batch_idx} for {bpm_range}")
                    
                    # All turns must be from same file
                    primary_file_idx = file_turn_map[turn_batch[0]]
                    if any(file_turn_map[t] != primary_file_idx for t in turn_batch):
                        raise ValueError(f"Batch {batch_idx} has turns from multiple files")

                    # Create payload arrays
                    pos, mom, pos_var, mom_var, init_coords, pts = self._make_worker_payload(
                        turn_batch, file_turn_map, start_bpm, end_bpm,
                        self.n_data_points[bpm_range], sdir, machine_deltaps, arrays_cache
                    )

                    for arr in (pos, mom, pos_var, mom_var):
                        arr.setflags(write=False)

                    primary_file_idx = file_turn_map[turn_batch[0]]
                    if any(file_turn_map[t] != primary_file_idx for t in turn_batch):
                        raise ValueError(f"Batch {batch_idx} has turns from multiple files")
                    
                    LOGGER.debug(
                        f"Worker {len(payloads)}: file={primary_file_idx}, "
                        f"range={bpm_range}, sdir={sdir}, turns={len(turn_batch)}"
                    )
                    
                    payloads.append((
                        WorkerData(
                            position_comparisons=pos,
                            momentum_comparisons=mom,
                            position_variances=pos_var,
                            momentum_variances=mom_var,
                            init_coords=init_coords,
                            init_pts=pts
                        ),
                        WorkerConfig(
                            start_bpm=start_bpm,
                            end_bpm=end_bpm,
                            magnet_range=self.magnet_range,
                            sequence_file_path=self.sequence_file_path,
                            corrector_strengths=self.corrector_strengths_files[primary_file_idx],
                            tune_knobs_file=self.tune_knobs_files[primary_file_idx],
                            beam_energy=self.beam_energy,
                            sdir=sdir,
                            bad_bpms=self.bad_bpms,
                            seq_name=self.seq_name
                        ),
                        batch_idx
                    ))
                    
        # Log summary of worker file assignments
        file_usage = {}
        for payload_data, payload_config, batch_idx in payloads:
            file_idx = self.corrector_strengths_files.index(payload_config.corrector_strengths)
            file_usage[file_idx] = file_usage.get(file_idx, 0) + 1
        
        LOGGER.info(
            f"Created {len(payloads)} workers using files: " +
            ", ".join(f"file_{idx}={count} workers" for idx, count in sorted(file_usage.items()))
        )
        
        # Verify all files are used if we have enough batches
        num_files = len(self.corrector_strengths_files)
        if len(file_usage) < num_files:
            LOGGER.warning(
                f"Only {len(file_usage)}/{num_files} measurement files are being used by workers! "
                f"This may lead to poor optimization if different files have different deltap values."
            )
        
        return payloads

    def _extract_arrays(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Extract numpy arrays from DataFrame once for memory efficiency."""
        return {
            "df": df,
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
            df = cache["df"]
            arr_x, arr_y, arr_px, arr_py = cache["x"], cache["y"], cache["px"], cache["py"]
            arr_vx, arr_vy, arr_vpx, arr_vpy = cache["var_x"], cache["var_y"], cache["var_px"], cache["var_py"]

            pts[i] = self._compute_pt(file_idx, machine_deltaps)

            # Determine initial turn for backward tracking
            init_turn = turn
            if sdir == -1:
                start_pos = self._get_pos(df, turn, start_bpm)
                end_pos = start_pos + n_data_points - 1
                init_turn = self.get_turn(df, end_pos)
                if init_turn != turn:
                    LOGGER.warning(f"Reversed init turn {turn} -> {init_turn}")

            init_pos = self._get_pos(df, init_turn, init_bpm)
            init_coords[i, :] = [arr_x[init_pos], arr_px[init_pos], arr_y[init_pos], arr_py[init_pos], 0.0, pts[i]]

            # Extract data slice
            sl = slice(init_pos, init_pos - n_data_points, -1) if sdir == -1 else slice(init_pos, init_pos + n_data_points)
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
        bpm_ranges: list[str],
        opt_settings: OptSettings,
        machine_deltaps: list[float],
    ):
        """Start worker processes and return their parent connections.

        Args:
            track_data (dict[int, pd.DataFrame]): Dictionary mapping file indices to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches for worker distribution
            file_turn_map (dict[int, int]): Mapping from turn numbers to file indices
            bpm_ranges (list[str]): List of BPM ranges
            opt_settings (OptSettings): Optimisation settings
            machine_deltaps (list[float]): List of machine deltaps corresponding to each file

        Note:
            Process objects are stored internally on the manager and are
            joined/terminated in `terminate_workers()`; callers only need the
            connections to communicate with workers.
        """
        payloads = self.create_worker_payloads(
            track_data, turn_batches, file_turn_map, bpm_ranges, machine_deltaps, opt_settings
        )
        LOGGER.info(f"Starting {len(payloads)} workers...")

        for data, config, worker_id in payloads:
            parent, child = mp.Pipe()
            w = ArcByArcWorker(child, worker_id, data, config, opt_settings)
            w.start()
            self.parent_conns.append(parent)
            self.workers.append(w)
            parent.send(None)

    def collect_worker_results(self, total_turns: int) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch.

        Args:
            total_turns (int): Total number of turns across all workers for normalisation

        Returns:
            tuple[float, np.ndarray]: A tuple containing:
                - total_loss: average loss across all turns
                - agg_grad: aggregated and normalised gradient array
        """
        total_loss = 0.0
        agg_grad = None

        for conn in self.parent_conns:
            _, grad, loss = conn.recv()
            agg_grad = grad if agg_grad is None else agg_grad + grad
            total_loss += loss

        return total_loss / total_turns, agg_grad.flatten() / total_turns

    def terminate_workers(self) -> np.ndarray:
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
        
        hessians = [conn.recv() for conn in self.parent_conns]
        for w in self.workers:
            w.join()
        
        return sum(hessians)
