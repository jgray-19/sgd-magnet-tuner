from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    HESSIAN_SCRIPT,
    MAGNET_RANGE,
    SEQUENCE_FILE,
    TRACK_INIT,
    TRACK_NO_KNOBS_INIT,
    TRACK_SCRIPT,
    USE_NOISY_DATA,
)
from aba_optimiser.mad.mad_interface import MadInterface

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import OptSettings

LOGGER = logging.getLogger(__name__)


class BaseWorker(Process, ABC):
    """
    Base worker process that provides common functionality for running
    tracking simulations in parallel and communicating gradients and loss
    back to the master process.
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        x_comparisons: np.ndarray,
        y_comparisons: np.ndarray,
        x_weights: np.ndarray,
        y_weights: np.ndarray,
        init_coords: np.ndarray,
        start_bpm: str,
        init_pts: np.ndarray,
        opt_settings: OptSettings,
    ) -> None:
        """
        New constructor that accepts compact numpy payloads built by the parent
        process to avoid sending the full DataFrame to every worker.
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn

        LOGGER.debug(
            f"Initialising worker {worker_id} for BPM {start_bpm} with {len(init_coords)} particles"
        )

        num_batches = opt_settings.num_batches

        # reduce data to be divisible by num_batches
        n_init = len(init_coords) - (len(init_coords) % num_batches)
        init_coords = init_coords[:n_init]
        self.x_comparisons = x_comparisons[:n_init]
        self.y_comparisons = y_comparisons[:n_init]
        self.x_weights = x_weights[:n_init]
        self.y_weights = y_weights[:n_init]

        # Split init_coords into num_batches batches
        init_coords = np.array_split(init_coords, num_batches)

        # Convert init_coords from list of matrices into list of list of lists
        self.init_coords = [ic.tolist() for ic in init_coords]
        self.batch_size = len(self.init_coords[0])

        self.start_bpm = start_bpm
        self.num_batches = num_batches
        self.opt_settings = opt_settings

        self.init_pts = init_pts
        self._split_data_to_batches()

        # Load common MAD scripts
        self.run_track_script = TRACK_SCRIPT.read_text()
        self.run_track_init_path = (
            TRACK_NO_KNOBS_INIT if opt_settings.only_energy else TRACK_INIT
        )

    @staticmethod
    @abstractmethod
    def get_bpm_range(start_bpm) -> str:
        """Get the magnet range specific to the worker type."""
        pass

    def get_n_data_points(self, nbpms: int) -> int:
        """Get the number of data points for comparison."""
        pass

    @abstractmethod
    def setup_mad_sequence(self, mad: MAD) -> None:
        """Setup MAD sequence specific to the worker type."""
        pass

    @staticmethod
    @abstractmethod
    def get_observation_turns(turn: int) -> list[int]:
        """Get the list of observation turns for a given starting turn."""
        pass

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def _split_data_to_batches(self):
        """
        Split the comparison data, which will be of dimensions (n_tracks, n_data_points),
        into self.num_batches batches, producing arrays of shape (batch_size, n_data_points).
        Also split the initial conditions into N batches.
        """
        self.x_comparisons = np.array_split(
            self.x_comparisons, self.num_batches, axis=0
        )

        self.y_comparisons = np.array_split(
            self.y_comparisons, self.num_batches, axis=0
        )
        self.x_weights = np.array_split(self.x_weights, self.num_batches, axis=0)
        self.y_weights = np.array_split(self.y_weights, self.num_batches, axis=0)

        self.init_pts = [
            arr.tolist()
            for arr in np.array_split(self.init_pts, self.num_batches, axis=0)
        ]

    def send_initial_conditions(self, mad: MAD) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send("""
init_coords = py:recv()
init_pts = py:recv()
""")
        mad.send(self.init_coords).send(self.init_pts)

        mad.send("""
da_x0_c = table.new(num_batches, 0)
for i=1,num_batches do
    da_x0_c[i] = table.new(batch_size, 0)
    for j=1,batch_size do
        da_x0_c[i][j] = da_x0_base:copy()
        da_x0_c[i][j]:set0(init_coords[i][j])
    end
end
            """)

    def setup_mad_interface(self) -> tuple[MAD, int]:
        """
        Initialize MAD interface and setup common MAD configuration.

        Returns:
            mad_iface: The MAD interface object
            mad: The MAD object
            nbpms: Number of BPMs
        """
        LOGGER.debug(f"Worker {self.worker_id}: Setting up MAD interface")

        # Get magnet range specific to worker type
        bpm_range = self.get_bpm_range(self.start_bpm)
        LOGGER.debug(f"Worker {self.worker_id}: Using BPM range {bpm_range}")

        mad_iface = MadInterface(
            SEQUENCE_FILE,
            magnet_range=MAGNET_RANGE,
            bpm_range=bpm_range,
            opt_settings=self.opt_settings,
            use_real_strengths="noisy" if USE_NOISY_DATA else True,
            # discard_mad_output=False,
        )

        mad = mad_iface.mad
        mad["knob_names"] = mad_iface.knob_names[:-1]  # ignore pt
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["nbpms"] = mad_iface.nbpms

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix", "vector")
        mad.load("MAD.utility", "tblcat")

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad)
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    MADX[param] = MADX[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

        # Setup sequence specific to worker type
        self.setup_mad_sequence(mad)

        return mad, mad_iface.nbpms

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients and loss for a given batch of particle tracking data.

        This method processes simulation results from MAD-NG (a particle accelerator
        simulation tool) to calculate gradients with respect to optimisation knobs
        (including energy deviation 'deltap') and the associated loss function. It
        uses vectorised NumPy operations for efficiency, computing gradients via
        vector-Jacobian products and loss as the sum of squared differences.

        Parameters:
            mad (MAD): The MAD-NG interface object for sending commands and receiving
                simulation results.
            knob_updates (dict[str, float]): Dictionary of knob names (e.g., magnet
                strengths) to their updated values. The 'deltap' key (energy deviation)
                is handled separately and removed from this dict.
            batch (int): The current batch index (used to select comparison data and
                adjust for 1-based Lua indexing in MAD).

        Returns:
            tuple[np.ndarray, float]:
                - grad (np.ndarray): Gradient array of shape (n_knobs + 1,), where
                  n_knobs is the number of knobs (excluding 'deltap'). Includes
                  derivatives w.r.t. knobs and 'deltap'.
                - loss (float): Scalar loss value, computed as the sum of squared
                  differences between simulated and reference positions in x and y.

        Notes:
            - Assumes MAD has been set up with initial conditions and scripts loaded.
            - Gradients are computed using the chain rule and Jacobian-vector products
              for efficiency in optimisation loops.
            - Loss is minimised in optimisation (e.g., via SGD), aiming to match
              reference trajectories.

        Example:
            >>> grad, loss = worker.compute_gradients_and_loss(mad, {"k1": 0.01, "deltap": 1e-3}, 0)
            >>> print(grad.shape)  # (n_knobs + 1,)
            >>> print(loss)  # e.g., 0.123
        """
        # Extract energy deviation ('pt') from knob updates
        machine_pt = knob_updates.pop("pt")

        # Prepare MAD commands to update knob values in the sequence
        update_commands = [
            f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items()
        ]

        # Send updates, batch index (adjusted for Lua's 1-based indexing), and energy deviation to MAD
        mad.send("\n".join(update_commands))
        mad.send(f"batch = {batch + 1}")  # Lua uses 1-based indexing
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")
        mad.send(self.run_track_script)

        # Receive simulation results from MAD: positions and derivatives
        x_results = mad.recv()  # Simulated x-positions: list of arrays
        y_results = mad.recv()  # Simulated y-positions: list of arrays
        dx_dk_results = mad.recv()  # Derivatives of x w.r.t. knobs: list of arrays
        dy_dk_results = mad.recv()  # Derivatives of y w.r.t. knobs: list of arrays

        # Convert results to NumPy arrays for efficient vectorised processing
        # Squeeze to remove singleton dimensions (e.g., from MAD's output format)
        # Shape: (n_particles, n_data_points)
        particle_positions_x = np.asarray(x_results).squeeze(-1)
        particle_positions_y = np.asarray(y_results).squeeze(-1)

        # Shape: (n_particles, n_knobs + 1, n_data_points)
        dx_dk = np.stack(dx_dk_results, axis=0)
        dy_dk = np.stack(dy_dk_results, axis=0)

        # Compute differences between simulated and reference positions for loss and gradients
        # Shape: (n_particles, n_data_points)
        position_diffs_x = (
            particle_positions_x - self.x_comparisons[batch]
        )  # * self.x_weights[batch]
        position_diffs_y = (
            particle_positions_y - self.y_comparisons[batch]
        )  # * self.y_weights[batch]

        # Compute gradients using vector-Jacobian products (efficient via einsum)
        # gx: Sum over particles and data points of (derivatives * diffs). Shape: (n_knobs + 1,)
        gx = np.einsum("pkm,pm->k", dx_dk, position_diffs_x)
        gy = np.einsum("pkm,pm->k", dy_dk, position_diffs_y)

        # Combine gradients and compute loss (factor of 2 for derivative of squared error)
        grad = 2.0 * (gx + gy)  # Shape: (n_knobs + 1,)
        loss = np.sum(position_diffs_x**2) + np.sum(position_diffs_y**2)  # Scalar

        return grad, loss

    def run(self) -> None:
        """Main worker run loop."""
        # Setup MAD interface
        mad, nbpms = self.setup_mad_interface()

        # Send initial conditions
        self.send_initial_conditions(mad)

        # Initialise the MAD environment ready for tracking
        mad.send(self.run_track_init_path.read_text())

        # Main tracking loop
        hessian_var_x, hessian_var_y = self.conn.recv()
        LOGGER.debug(
            f"Worker {self.worker_id}: Received Hessian vars x={hessian_var_x}, y={hessian_var_y}"
        )
        knob_updates = self.conn.recv()  # shape (n_knobs,)

        batch = 0
        while knob_updates is not None:
            # Process tracking and compute gradients
            grad, loss = self.compute_gradients_and_loss(mad, knob_updates, batch)

            # Send results back - normalised with nbpms
            self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))

            # Receive next knob updates
            knob_updates = self.conn.recv()

            # Move to the next batch
            batch = (batch + 1) % self.num_batches

        # Final hessian estimation
        mad.send("""
var_x = py:recv()
var_y = py:recv()
""")
        mad.send(1 / hessian_var_x)
        mad.send(1 / hessian_var_y)
        mad.send(HESSIAN_SCRIPT.read_text())
        h_part = mad.recv()
        self.conn.send(h_part)  # shape (n_knobs, n_knobs)
