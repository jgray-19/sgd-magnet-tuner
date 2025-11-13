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
    BEAM_ENERGY,
    DELTAP,
    DIFFERENT_TURNS_PER_RANGE,
    PARTICLE_MASS,
)
from aba_optimiser.physics.deltap import dp2pt
from aba_optimiser.workers.base_worker import BaseWorker, WorkerConfig, WorkerData

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

    # Mapping from energy type to momentum deviation
    # Used to convert energy type strings ("plus", "minus", "zero") to numerical dpp values
    ENERGY_DPP_MAP = {"plus": DELTAP, "minus": -DELTAP, "zero": 0.0}

    def __init__(
        self,
        worker_class: type[BaseWorker],
        n_data_points: dict[str, int],
        ybpm: str,
        magnet_range: str,
        sequence_file_path: Path,
        corrector_strengths_file: Path,
        tune_knobs_file: Path,
        bad_bpms: list[str] | None = None,
        seq_name: str | None = None,
        beam_energy: float = BEAM_ENERGY,
    ):
        """Initialise the WorkerManager.

                    position_variances,
                    momentum_variances,
        n_data_points (dict[str, int]): Dictionary mapping BPM names to number of data points to collect
        """
        self.Worker = worker_class
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []
        self.y_bpm = (
            ybpm  # Name at which the vertical kick is highest for the start points
        )
        self.magnet_range = magnet_range
        self.sequence_file_path = sequence_file_path
        self.corrector_strengths_file = corrector_strengths_file
        self.tune_knobs_file = tune_knobs_file
        # Per-DataFrame cache mapping (turn, bpm) -> iloc position for faster slicing
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}
        self.bad_bpms = bad_bpms
        self.seq_name = seq_name
        self.beam_energy = beam_energy

    def _compute_pt(self, energy_type: str) -> float:
        """Compute transverse momentum based on energy type.

        Args:
            energy_type (str): The type of energy deviation. Must be 'plus', 'minus', or 'zero'

        Returns:
            float: The transverse momentum calculated from energy deviation
        """
        dpp = self.ENERGY_DPP_MAP[energy_type]
        return dp2pt(dpp, PARTICLE_MASS, self.beam_energy)

    def _create_init_coords(self, starting_row: pd.Series, pt: float) -> np.ndarray:
        """Create initial coordinates array for particle tracking.

        Args:
            starting_row (pd.Series): DataFrame row containing initial particle coordinates
            pt (float): Transverse momentum value

        Returns:
            np.ndarray: 6-element array containing [x, px, y, py, t=0, pt]
                representing the initial 6D phase space coordinates
        """
        # Initial 4-element phase-space vector: x, px, y, py
        coords = starting_row[["x", "px", "y", "py"]].to_numpy(
            dtype="float64", copy=False
        )
        # Extend to include t=0 and pt to create 6D phase space coordinates
        # Final format: [x, px, y, py, t, pt] where t=0 (time) and pt is transverse momentum
        return np.concatenate((coords, [0, pt]))

    def create_worker_payloads(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_ranges: list[str],
    ) -> list[tuple[WorkerData, WorkerConfig, int]]:
        """Create payloads for all workers.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches, each containing turn numbers for a worker
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            bpm_start_points (list[str]): List of starting BPM names for each batch

        Returns:
            list[tuple[WorkerData, WorkerConfig, int]]: List of tuples containing:
                - WorkerData: Data arrays for the worker
                - WorkerConfig: Configuration parameters for the worker
                - int: Worker ID

        Raises:
            AssertionError: If a turn batch is empty
        """
        payloads = []
        # Precompute numpy views per energy type to avoid repeated pandas -> numpy conversions
        precomputed_arrays: dict[str, dict[str, object]] = {}
        for etype, df in track_data.items():
            precomputed_arrays[etype] = {
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
        # sdir = 1  # Start with positive direction
        num_ranges = len(bpm_ranges)
        for sdir in [1, -1]:  # Alternate tracking direction for each set of workers
            for bpm_i, bpm_range in enumerate(bpm_ranges):
                start_bpm, end_bpm = bpm_range.split("/")
                for batch_idx, turn_batch in enumerate(turn_batches):
                    # Cycle through BPM start points using modulo to distribute work evenly, if required
                    if DIFFERENT_TURNS_PER_RANGE and batch_idx % num_ranges != bpm_i:
                        continue

                    # Ensure the turn batch is not empty
                    assert turn_batch, (
                        f"Turn batch {batch_idx} for range {bpm_range} is empty. "
                        f"Check TRACKS_PER_WORKER and NUM_WORKERS."
                    )

                    # Create the payload data for this worker (position/momentum comparisons, initial coords, etc.)
                    (
                        position_comp,
                        momentum_comp,
                        position_variances,
                        momentum_variances,
                        init_coords,
                        pt_array,
                    ) = self._make_worker_payload(
                        track_data,
                        turn_batch,
                        energy_turn_map,
                        start_bpm,
                        end_bpm,
                        self.n_data_points[bpm_range],
                        sdir,
                        precomputed_arrays,
                    )

                    # Make arrays read-only to prevent accidental modification in worker processes
                    position_comp.setflags(write=False)
                    momentum_comp.setflags(write=False)
                    position_variances.setflags(write=False)
                    momentum_variances.setflags(write=False)

                    # Create WorkerData and WorkerConfig instances
                    data = WorkerData(
                        position_comparisons=position_comp,  # Shape: (n_turns, n_data_points, 2)
                        momentum_comparisons=momentum_comp,  # Shape: (n_turns, n_data_points, 2)
                        position_variances=position_variances,
                        momentum_variances=momentum_variances,
                        init_coords=init_coords,
                        init_pts=pt_array,
                    )
                    config = WorkerConfig(
                        start_bpm=start_bpm,
                        end_bpm=end_bpm,
                        magnet_range=self.magnet_range,
                        sequence_file_path=self.sequence_file_path,
                        sdir=sdir,
                        bad_bpms=self.bad_bpms,
                        seq_name=self.seq_name,
                        corrector_strengths=self.corrector_strengths_file,
                        tune_knobs_file=self.tune_knobs_file,
                        beam_energy=self.beam_energy,
                    )

                    # Package data and config for the worker
                    payloads.append((data, config, batch_idx))
                # sdir *= -1  # Alternate the sign of the tracking direction
        return payloads

    def _make_worker_payload(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batch: list[int],
        energy_turn_map: dict[int, str],
        start_bpm: str,
        end_bpm: str,
        n_data_points: int,
        sdir: int,
        precomputed_arrays: dict[str, dict[str, object]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create 2D comparison arrays and initial coordinates for a worker.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batch (list[int]): List of turn numbers to process for this worker
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            start_bpm (str): Starting BPM name for data collection
            n_data_points (int): Number of data points to collect per observation turn

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the position comparisons, momentum comparisons,
            positional variances, momentum variances, initial coordinates per turn,
            and pt array
        """
        # Preallocate output arrays (n_turns, n_points, 2) for 2D structure
        n_turns = len(turn_batch)
        position_comp = np.empty((n_turns, n_data_points, 2), dtype="float64")  # [x, y]
        momentum_comp = np.empty(
            (n_turns, n_data_points, 2), dtype="float64"
        )  # [px, py]
        position_variances = np.empty(
            (n_turns, n_data_points, 2), dtype="float64"
        )  # [var_x, var_y]
        momentum_variances = np.empty(
            (n_turns, n_data_points, 2), dtype="float64"
        )  # [var_px, var_py]

        init_coords_array = np.empty((n_turns, 6), dtype="float64")
        pt_array = np.empty((n_turns,), dtype="float64")
        init_bpm = start_bpm if sdir == 1 else end_bpm

        # Process each turn in the batch
        for i, turn in enumerate(turn_batch):
            # Get the energy type for this turn (determines which DataFrame to use)
            energy_type = energy_turn_map[turn]
            if precomputed_arrays is not None and energy_type in precomputed_arrays:
                entry = precomputed_arrays[energy_type]
                df: pd.DataFrame = entry["df"]
                arr_x: np.ndarray = entry["x"]
                arr_y: np.ndarray = entry["y"]
                arr_px: np.ndarray = entry["px"]
                arr_py: np.ndarray = entry["py"]
                arr_vx: np.ndarray = entry["var_x"]
                arr_vy: np.ndarray = entry["var_y"]
                arr_vpx: np.ndarray = entry["var_px"]
                arr_vpy: np.ndarray = entry["var_py"]
            else:
                df = track_data[energy_type]
                arr_x = df["x"].to_numpy(dtype="float64", copy=False)
                arr_y = df["y"].to_numpy(dtype="float64", copy=False)
                arr_px = df["px"].to_numpy(dtype="float64", copy=False)
                arr_py = df["py"].to_numpy(dtype="float64", copy=False)
                arr_vx = df["var_x"].to_numpy(dtype="float64", copy=False)
                arr_vy = df["var_y"].to_numpy(dtype="float64", copy=False)
                arr_vpx = df["var_px"].to_numpy(dtype="float64", copy=False)
                arr_vpy = df["var_py"].to_numpy(dtype="float64", copy=False)

            # Calculate transverse momentum for this energy type
            pt = self._compute_pt(energy_type)
            pt_array[i] = pt

            # Initial coords selection (sdir-dependent)
            init_turn = turn
            if sdir == -1:
                # identify the turn of the initial bpm
                start_pos = self._get_pos(df, turn, start_bpm)
                end_pos = start_pos + n_data_points - 1
                if (init_turn := self.get_turn(df, end_pos)) != turn:
                    assert (
                        # start_bpm == df.iloc[start_pos].index[1]
                        # and end_bpm == df.iloc[end_pos].index[1]
                        start_bpm == df.index[start_pos][1]
                        and end_bpm == df.index[end_pos][1]
                    )
                    LOGGER.warning(f"Reversed init turn from {turn} to {init_turn}")

            init_pos = self._get_pos(df, init_turn, init_bpm)
            assert init_bpm == df.index[init_pos][1], (
                f"Initial BPM mismatch at pos {init_pos}: expected {init_bpm}, "
                f"found {df.index[init_pos][1]}"
            )
            x0 = arr_x[init_pos]
            px0 = arr_px[init_pos]
            y0 = arr_y[init_pos]
            py0 = arr_py[init_pos]
            init_coords = np.array([x0, px0, y0, py0, 0.0, pt], dtype="float64")
            init_coords_array[i, :] = init_coords

            # Collect observation data for this turn
            if sdir == -1:
                sl = slice(init_pos, init_pos - n_data_points, -1)
                position_comp[i, :, 0] = arr_x[sl]  # x positions
                position_comp[i, :, 1] = arr_y[sl]  # y positions
                momentum_comp[i, :, 0] = arr_px[sl]  # px momenta
                momentum_comp[i, :, 1] = arr_py[sl]  # py momenta
                position_variances[i, :, 0] = arr_vx[sl]  # x variances
                position_variances[i, :, 1] = arr_vy[sl]  # y variances
                momentum_variances[i, :, 0] = arr_vpx[sl]  # px variances
                momentum_variances[i, :, 1] = arr_vpy[sl]  # py variances
                # weights[i, :, :] = 1.0  # Uncomment to set uniform weights
            else:
                sl = slice(init_pos, init_pos + n_data_points)
                position_comp[i, :, 0] = arr_x[sl]  # x positions
                position_comp[i, :, 1] = arr_y[sl]  # y positions
                momentum_comp[i, :, 0] = arr_px[sl]  # px momenta
                momentum_comp[i, :, 1] = arr_py[sl]  # py momenta
                position_variances[i, :, 0] = arr_vx[sl]  # x variances
                position_variances[i, :, 1] = arr_vy[sl]  # y variances
                # weights[i, :, :] = 1.0  # Uncomment to set uniform weights
                momentum_variances[i, :, 0] = arr_vpx[sl]  # px variances
                momentum_variances[i, :, 1] = arr_vpy[sl]  # py variances

        return (
            position_comp,
            momentum_comp,
            position_variances,
            momentum_variances,
            init_coords_array,
            pt_array,
        )

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
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_ranges: list[str],
        opt_settings: OptSettings,
    ):
        """Start worker processes and return their parent connections.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches for worker distribution
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            bpm_start_points (list[str]): List of starting BPM names
            opt_settings (OptSettings): Optimisation settings

        Note:
            Process objects are stored internally on the manager and are
            joined/terminated in `terminate_workers()`; callers only need the
            connections to communicate with workers.
        """
        # Create payloads containing all data needed for each worker
        payloads = self.create_worker_payloads(
            track_data, turn_batches, energy_turn_map, bpm_ranges
        )

        # Start a worker process for each payload
        for data, config, worker_id in payloads:
            # Create a Pipe for bidirectional communication between parent and child
            # parent_conn is used by the manager, child_conn is passed to the worker
            parent, child = mp.Pipe()

            # Create and start the worker process
            # The worker receives: connection, worker_id, data, config, opt_settings
            w = self.Worker(
                child,  # Connection for communication
                worker_id,  # Worker ID
                data,  # WorkerData instance
                config,  # WorkerConfig instance
                opt_settings,  # Optimisation settings
            )
            w.start()  # Start the worker process

            # Store connections and process references for later use
            self.parent_conns.append(parent)
            self.workers.append(w)

            # Send an initial handshake to signal the worker to start listening
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
        agg_grad: None | np.ndarray = None

        # Collect results from each worker process
        for conn in self.parent_conns:
            # Receive data from worker: (worker_id, gradient_array, loss_value)
            _, grad, loss = conn.recv()

            # Aggregate gradients: initialise with first worker's gradient, then add subsequent ones
            agg_grad = grad if agg_grad is None else agg_grad + grad

            # Sum up losses from all workers
            total_loss += loss

        # Normalise total loss by total number of turns across all workers
        total_loss /= total_turns

        # Normalise aggregated gradient by total number of turns
        # This gives the average gradient across all turns and workers
        agg_grad = agg_grad.flatten() / total_turns

        return total_loss, agg_grad

    def terminate_workers(self) -> np.ndarray:
        """Terminate all workers and collect final Hessian information.

        Returns:
            np.ndarray: Global Hessian matrix computed as the sum of all worker Hessians

        Note:
            This method signals all workers to stop, collects their final Hessian
            contributions, and waits for all processes to finish before returning.
        """
        LOGGER.info("Terminating workers...")

        # Signal all workers to stop by sending None through their connections
        for conn in self.parent_conns:
            conn.send((None, None))

        # Collect Hessian matrices from each worker
        # Each worker computes a local Hessian based on its subset of data
        hessians = []
        for conn in self.parent_conns:
            # Receive the local Hessian matrix from this worker
            h_local = conn.recv()
            hessians.append(h_local)

        # Sum all local Hessians to get the global Hessian
        # This combines information from all workers into a single Hessian matrix
        h_global = sum(hessians)

        # Wait for all worker processes to finish (cleanup)
        for w in self.workers:
            w.join()

        return h_global
