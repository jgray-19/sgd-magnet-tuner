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
import pandas as pd

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from aba_optimiser.config import OptSettings
    from aba_optimiser.workers.base_worker import BaseWorker

from aba_optimiser.config import (
    BEAM_ENERGY,
    DELTAP,
    DIFFERENT_TURNS_PER_RANGE,
    PARTICLE_MASS,
)
from aba_optimiser.physics.deltap import dp2pt

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
        bad_bpms: list[str] | None = None,
    ):
        """Initialize the WorkerManager.

        Args:
            worker_class (type[BaseWorker]): The class to use for creating worker processes
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
        # Per-DataFrame cache mapping (turn, bpm) -> iloc position for faster slicing
        self._pos_cache: dict[int, dict[tuple[int, str], int]] = {}
        self.bad_bpms = bad_bpms

    def _compute_pt(self, energy_type: str) -> float:
        """Compute transverse momentum based on energy type.

        Args:
            energy_type (str): The type of energy deviation. Must be 'plus', 'minus', or 'zero'

        Returns:
            float: The transverse momentum calculated from energy deviation
        """
        dpp = self.ENERGY_DPP_MAP[energy_type]
        return dp2pt(dpp, PARTICLE_MASS, BEAM_ENERGY)

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

    def _collect_observation_data(
        self,
        df: pd.DataFrame,
        obs_turns: list[int],
        start_bpm: str,
        n_data_points: int,
        turn: int,
    ) -> pd.DataFrame:
        """Collect observation data for a specific turn from track data.

        Args:
            df (pd.DataFrame): Track data DataFrame with MultiIndex (turn, bpm)
            obs_turns (list[int]): List of observation turn numbers to collect data for
            start_bpm (str): Starting BPM name for data collection
            n_data_points (int): Number of data points to collect per observation turn
            turn (int): The current turn number (for error reporting)

        Returns:
            pd.DataFrame: Concatenated DataFrame containing observation data

        Raises:
            ValueError: If no data is available for the specified turn
        """
        blocks = []
        for ot in obs_turns:
            # Find the position of this (observation_turn, start_bpm) in the DataFrame
            pos = df.index.get_loc((ot, start_bpm))
            # Extract n_data_points rows starting from this position
            # This gives us data for this observation turn at the starting BPM
            blocks.append(df.iloc[pos : pos + n_data_points])
        # Concatenate all blocks into a single DataFrame
        filtered = pd.concat(blocks, axis=0)

        if filtered.shape[0] == 0:
            raise ValueError(f"No data available for turn {turn}")
        return filtered

    def create_worker_payloads(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_ranges: list[str],
    ) -> list[dict]:
        """Create payloads for all workers.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches, each containing turn numbers for a worker
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            bpm_start_points (list[str]): List of starting BPM names for each batch

        Returns:
            list[dict]: List of payload dictionaries, each containing:
                - wid: worker ID
                - x_comp: x-coordinate comparison arrays
                - y_comp: y-coordinate comparison arrays
                - init_coords: initial coordinates arrays
                - start_bpm: starting BPM name

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
                "x_weight": df["x_weight"].to_numpy(dtype="float64", copy=False),
                "y_weight": df["y_weight"].to_numpy(dtype="float64", copy=False),
            }
        sdir = 1  # Start with positive direction
        num_ranges = len(bpm_ranges)
        # for sdir in [1, -1]:  # Alternate tracking direction for each set of workers
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
                print(self.n_data_points)

                # Create the payload data for this worker (x/y comparisons, initial coords, etc.)
                (
                    x_comp,
                    y_comp,
                    px_comp,
                    py_comp,
                    init_coords,
                    pt_array,
                    # x_weights,
                    # y_weights,
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
                x_comp.setflags(write=False)
                y_comp.setflags(write=False)

                # Package all data needed by the worker into a payload dictionary
                payloads.append(
                    {
                        "worker_id": batch_idx,  # Worker ID
                        "x_comparisons": x_comp,  # Expected x coordinates for loss calculation
                        "y_comparisons": y_comp,  # Expected y coordinates for loss calculation
                        "px_comparisons": px_comp,  # Expected px coordinates for loss calculation
                        "py_comparisons": py_comp,  # Expected py coordinates for loss calculation
                        # "x_weights": x_weights,  # Weights for x coordinates
                        # "y_weights": y_weights,  # Weights for y coordinates
                        "init_coords": init_coords,  # Initial particle coordinates
                        "start_bpm": start_bpm,
                        "end_bpm": end_bpm,
                        "init_pts": pt_array,
                        "sdir": sdir,
                        "bad_bpms": self.bad_bpms,  # List of bad BPMs
                    }
                )
                sdir *= -1  # Alternate the sign of the tracking direction
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
        """Create x/y comparison arrays and initial coordinates for a worker.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batch (list[int]): List of turn numbers to process for this worker
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            start_bpm (str): Starting BPM name for data collection
            n_data_points (int): Number of data points to collect per observation turn

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the x/y/px/py comparisons, initial coordinates per turn, and pt array
        """
        # Preallocate output arrays (n_turns, n_points)
        n_turns = len(turn_batch)
        x_comp = np.empty((n_turns, n_data_points), dtype="float64")
        y_comp = np.empty((n_turns, n_data_points), dtype="float64")
        px_comp = np.empty((n_turns, n_data_points), dtype="float64")
        py_comp = np.empty((n_turns, n_data_points), dtype="float64")
        # x_weights = np.empty((n_turns, n_data_points), dtype="float64")
        # y_weights = np.empty((n_turns, n_data_points), dtype="float64")

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
                # arr_wx: np.ndarray = entry["x_weight"]
                # arr_wy: np.ndarray = entry["y_weight"]
            else:
                df = track_data[energy_type]
                arr_x = df["x"].to_numpy(dtype="float64", copy=False)
                arr_y = df["y"].to_numpy(dtype="float64", copy=False)
                arr_px = df["px"].to_numpy(dtype="float64", copy=False)
                arr_py = df["py"].to_numpy(dtype="float64", copy=False)
                # arr_wx = df["x_weight"].to_numpy(dtype="float64", copy=False)
                # arr_wy = df["y_weight"].to_numpy(dtype="float64", copy=False)

            # Calculate transverse momentum for this energy type
            pt = self._compute_pt(energy_type)
            pt_array[i] = pt

            # Observation turns for this turn
            obs_turns = self.Worker.get_observation_turns(turn)

            # Initial coords selection (sdir-dependent)
            init_turn = turn
            if sdir == -1:
                # identify the turn of the initial bpm
                start_pos = self._get_pos(df, turn, start_bpm)
                if (init_turn := self.get_turn(df, start_pos + n_data_points)) != turn:
                    LOGGER.warning(f"Reversed init turn from {turn} to {init_turn}")

            init_pos = self._get_pos(df, init_turn, init_bpm)
            x0 = arr_x[init_pos]
            px0 = arr_px[init_pos]
            y0 = arr_y[init_pos]
            py0 = arr_py[init_pos]
            init_coords = np.array([x0, px0, y0, py0, 0.0, pt], dtype="float64")
            init_coords_array[i, :] = init_coords

            # Collect observation data for this turn
            # if len(obs_turns) == 1:
            if sdir == -1:
                sl = slice(init_pos, init_pos - n_data_points, -1)
                x_comp[i, :] = arr_x[sl]
                y_comp[i, :] = arr_y[sl]
                px_comp[i, :] = arr_px[sl]
                py_comp[i, :] = arr_py[sl]
                # x_weights[i, :] = arr_wx[sl]
                # y_weights[i, :] = arr_wy[sl]
                # x_weights[i, :] = 1.0
                # y_weights[i, :] = 1.0
            else:
                sl = slice(init_pos, init_pos + n_data_points)
                x_comp[i, :] = arr_x[sl]
                y_comp[i, :] = arr_y[sl]
                px_comp[i, :] = arr_px[sl]
                py_comp[i, :] = arr_py[sl]
                # x_weights[i, :] = arr_wx[sl]
                # y_weights[i, :] = arr_wy[sl]
                # x_weights[i, :] = 1.0
                # y_weights[i, :] = 1.0
            # Get the turn that the final BPM corresponds to and check it matches
            # the turn of the first BPM +/- 1
            first_bpm_turn = self.get_turn(df, init_pos)
            last_bpm_turn = self.get_turn(df, sl.stop - 1 if sdir == 1 else sl.stop + 1)
            if not (
                last_bpm_turn == first_bpm_turn
                or abs(last_bpm_turn - first_bpm_turn) == 1
            ):
                raise ValueError(
                    f"Unexpected turn mismatch: first BPM turn {first_bpm_turn}, "
                    f"last BPM turn {last_bpm_turn}, slice {sl}, sdir {sdir}"
                )

            if len(obs_turns) > 1:
                raise NotImplementedError(
                    "Multiple observation turns not implemented in payload creation"
                )

        return (
            x_comp,
            y_comp,
            px_comp,
            py_comp,
            init_coords_array,
            pt_array,
            # x_weights,
            # y_weights,
        )
        # return x_comp, y_comp, init_coords_array, pt_array

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
        var_x: float,
        var_y: float,
        opt_settings: OptSettings,
    ):
        """Start worker processes and return their parent connections.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batches (list[list[int]]): List of turn batches for worker distribution
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            bpm_start_points (list[str]): List of starting BPM names
            var_x (float): Variance for x-coordinate measurements
            var_y (float): Variance for y-coordinate measurements
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
        for payload in payloads:
            # Create a Pipe for bidirectional communication between parent and child
            # parent_conn is used by the manager, child_conn is passed to the worker
            parent, child = mp.Pipe()

            # Create and start the worker process
            # The worker receives: connection, worker_id, comparison data, initial coords, BPM, sextupole flag
            w = self.Worker(
                child,  # Connection for communication
                **payload,  # Unpack the payload dictionary into keyword arguments
                opt_settings=opt_settings,  # Pass the optimisation settings
                magnet_range=self.magnet_range,
            )
            w.start()  # Start the worker process

            # Store connections and process references for later use
            self.parent_conns.append(parent)
            self.workers.append(w)

            # Send initial variance values to the worker for noise modeling
            parent.send((var_x, var_y))

    def send_knobs_to_workers(self, current_knobs: dict[str, float]) -> None:
        """Send current knob values to all workers.

        Args:
            current_knobs (dict[str, float]): Dictionary mapping knob names to their current values
        """
        # Send the current magnet knob settings to each worker
        # Workers use these values to compute particle trajectories and losses
        for conn in self.parent_conns:
            conn.send(current_knobs)

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

            # Aggregate gradients: initialize with first worker's gradient, then add subsequent ones
            agg_grad = grad if agg_grad is None else agg_grad + grad

            # Sum up losses from all workers
            total_loss += loss

        # Normalize total loss by total number of turns across all workers
        total_loss /= total_turns

        # Normalize aggregated gradient by total number of turns
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
