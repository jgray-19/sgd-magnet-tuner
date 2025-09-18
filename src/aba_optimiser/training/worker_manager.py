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
    DIFFERENT_TURNS_FOR_START_BPM,
    MAGNET_RANGE,
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

    def __init__(self, worker_class: type[BaseWorker], n_data_points: dict[str, int]):
        """Initialize the WorkerManager.

        Args:
            worker_class (type[BaseWorker]): The class to use for creating worker processes
            n_data_points (dict[str, int]): Dictionary mapping BPM names to number of data points to collect
        """
        self.Worker = worker_class
        self.n_data_points = n_data_points
        self.parent_conns: list[Connection] = []
        self.workers: list[mp.Process] = []

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
        bpm_start_points: list[str],
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
        num_start_points = len(bpm_start_points)
        for bpm_i, start_bpm in enumerate(bpm_start_points):
            for batch_idx, turn_batch in enumerate(turn_batches):
                # Cycle through BPM start points using modulo to distribute work evenly, if required
                if (
                    DIFFERENT_TURNS_FOR_START_BPM
                    and batch_idx % num_start_points != bpm_i
                ):
                    continue

                # Ensure the turn batch is not empty
                assert turn_batch, (
                    f"Turn batch {batch_idx} for BPM {start_bpm} is empty. "
                    f"Check TRACKS_PER_WORKER and NUM_WORKERS."
                )

                # Create the payload data for this worker (x/y comparisons, initial coords, etc.)
                x_comp, y_comp, x_weights, y_weights, init_coords, pt_array = (
                    self._make_worker_payload(
                        track_data,
                        turn_batch,
                        energy_turn_map,
                        start_bpm,
                        self.n_data_points[start_bpm],
                    )
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
                        "x_weights": x_weights,  # Weights for x-coordinate differences
                        "y_weights": y_weights,  # Weights for y-coordinate differences
                        "init_coords": init_coords,  # Initial particle coordinates
                        "start_bpm": start_bpm,  # Starting BPM for this worker
                        "init_pts": pt_array,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create x/y comparison arrays and initial coordinates for a worker.

        Args:
            track_data (dict[str, pd.DataFrame]): Dictionary mapping energy types to track data DataFrames
            turn_batch (list[int]): List of turn numbers to process for this worker
            energy_turn_map (dict[int, str]): Mapping from turn numbers to energy types
            start_bpm (str): Starting BPM name for data collection
            n_data_points (int): Number of data points to collect per observation turn

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - x_comp: stacked x-coordinate comparison arrays
                - y_comp: stacked y-coordinate comparison arrays
                - init_coords_array: stacked initial coordinate arrays
                - dpp_array: array of momentum deviations
        """
        # Initialize lists to collect data for each turn in the batch
        x_arrays = []
        y_arrays = []
        x_weight_arrays = []
        y_weight_arrays = []
        init_coord_arrays = []
        pt_values = []

        y_bpm = MAGNET_RANGE.split("/")[0]  # Get the second BPM in the
        bpm_kick_plane = "y" if start_bpm == y_bpm else "x"
        # Process each turn in the batch
        for turn in turn_batch:
            # Get the energy type for this turn (determines which DataFrame to use)
            energy_type = energy_turn_map[turn]
            df = track_data[energy_type]

            # Calculate transverse momentum and momentum deviation for this energy type
            pt = self._compute_pt(energy_type)

            # Extract the starting coordinates for this turn and starting BPM
            starting_row = df.loc[(turn, start_bpm)]
            kicking_plane = starting_row["kick_plane"]
            if bpm_kick_plane not in kicking_plane:
                continue  # Skip if the kick plane does not match

            init_coords = self._create_init_coords(starting_row, pt)

            # Set the other plane coordinates to zero if not xy kick
            if kicking_plane == "x":
                init_coords[2:4] = 0.0
            elif kicking_plane == "y":
                init_coords[0:2] = 0.0

            init_coord_arrays.append(init_coords)
            pt_values.append(pt)

            # Get the observation turns for this turn (turns where we collect data)
            obs_turns = self.Worker.get_observation_turns(turn)

            # Collect all observation data for this turn across multiple observation turns
            filtered_data = self._collect_observation_data(
                df, obs_turns, start_bpm, n_data_points, turn
            )

            # Extract x and y coordinate arrays from the filtered data
            x_arrays.append(filtered_data["x"].to_numpy(dtype="float64", copy=False))
            y_arrays.append(filtered_data["y"].to_numpy(dtype="float64", copy=False))
            x_weight_arrays.append(
                filtered_data["x_weight"].to_numpy(dtype="float64", copy=False)
            )
            y_weight_arrays.append(
                filtered_data["y_weight"].to_numpy(dtype="float64", copy=False)
            )

        # Stack all arrays into 3D arrays (turn, observation_turn, data_point)
        x_comp = np.stack(x_arrays, axis=0)
        y_comp = np.stack(y_arrays, axis=0)
        x_weights = np.stack(x_weight_arrays, axis=0)
        y_weights = np.stack(y_weight_arrays, axis=0)
        init_coords_array = np.stack(init_coord_arrays, axis=0)
        pt_array = np.array(pt_values, dtype="float64")

        return x_comp, y_comp, x_weights, y_weights, init_coords_array, pt_array

    def start_workers(
        self,
        track_data: dict[str, pd.DataFrame],
        turn_batches: list[list[int]],
        energy_turn_map: dict[int, str],
        bpm_start_points: list[str],
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
            track_data, turn_batches, energy_turn_map, bpm_start_points
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
            conn.send(None)

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
