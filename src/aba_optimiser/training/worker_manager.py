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

from aba_optimiser.config import PARTICLE_MASS
from aba_optimiser.physics.deltap import dp2pt
from aba_optimiser.training.utils import create_bpm_range_specs
from aba_optimiser.workers import TrackingData, TrackingWorker, WorkerConfig
from aba_optimiser.workers.tracking_position_only import PositionOnlyTrackingWorker

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    import pandas as pd

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
        sequence_file_path: Path,
        corrector_strengths_files: list[Path],
        tune_knobs_files: list[Path],
        bad_bpms: list[str] | None = None,
        seq_name: str | None = None,
        beam_energy: float = 6800.0,
        flattop_turns: int = 1000,
        num_tracks: int = 1,
        use_fixed_bpm: bool = True,
        debug: bool = False,
        mad_logfile: Path | None = None,
        optimise_knobs: list[str] | None = None,
    ):
        """Initialise the WorkerManager."""
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []
        self.y_bpm = ybpm
        self.magnet_range = magnet_range
        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self.sequence_file_path = sequence_file_path
        self.corrector_strengths_files = corrector_strengths_files
        self.tune_knobs_files = tune_knobs_files
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}
        self.bad_bpms = bad_bpms
        self.seq_name = seq_name
        self.use_fixed_bpm = use_fixed_bpm
        self.beam_energy = beam_energy
        self.flattop_turns = flattop_turns
        self.num_tracks = num_tracks
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.optimise_knobs = optimise_knobs

    def _compute_pt(self, file_idx: int, machine_deltaps: list[float]) -> float:
        """Compute transverse momentum based on file index."""
        return dp2pt(machine_deltaps[file_idx], PARTICLE_MASS, self.beam_energy)

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
                )
                config = WorkerConfig(
                    start_bpm=start_bpm,
                    end_bpm=end_bpm,
                    magnet_range=self.magnet_range,
                    sequence_file_path=self.sequence_file_path,
                    corrector_strengths=self.corrector_strengths_files[primary_file_idx],
                    tune_knobs_file=self.tune_knobs_files[primary_file_idx],
                    beam_energy=self.beam_energy,
                    sdir=sdir,
                    bad_bpms=self.bad_bpms,
                    seq_name=self.seq_name,
                    debug=self.debug,
                    mad_logfile=self.mad_logfile,
                    optimise_knobs=self.optimise_knobs,
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
                if init_turn != turn:
                    LOGGER.warning(f"Reversed init turn {turn} -> {init_turn}")

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
        LOGGER.info(f"Starting {len(payloads)} workers...")

        # Select worker class based on whether momenta optimization is enabled
        if simulation_config.optimise_momenta:
            worker_class = TrackingWorker
            LOGGER.info("Using TrackingWorker (position + momentum)")
        else:
            worker_class = PositionOnlyTrackingWorker
            LOGGER.info("Using PositionOnlyTrackingWorker (position only)")

        for worker_id, (data, config, _file_idx) in enumerate(payloads):
            parent, child = mp.Pipe()
            w = worker_class(child, worker_id, data, config, simulation_config, mode="arc-by-arc")
            w.start()
            self.parent_conns.append(parent)
            self.workers.append(w)
            parent.send(None)

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
        agg_grad = None
        contrib_count = None  # Count of non-zero contributions per knob

        for conn in self.parent_conns:
            _, grad, loss = conn.recv()
            grad_flat = grad.flatten()

            if agg_grad is None:
                agg_grad = grad_flat.copy()
                contrib_count = (grad_flat != 0).astype(np.float64)
            else:
                agg_grad += grad_flat
                contrib_count += (grad_flat != 0).astype(np.float64)

            total_loss += loss

        # Average each knob's gradient by the number of workers that contributed
        # to it (avoiding division by zero for knobs with no contributions)
        contrib_count = np.maximum(contrib_count, 1.0)
        avg_grad = agg_grad / contrib_count

        return total_loss / total_turns, avg_grad

    def terminate_workers(self) -> None:
        """Terminate all workers and clean up processes."""
        LOGGER.info("Terminating workers...")
        for conn in self.parent_conns:
            conn.send((None, None))

        for w in self.workers:
            w.join()

    def termination_and_hessian(self) -> np.ndarray:
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
