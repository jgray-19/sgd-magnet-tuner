from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    MAD_SCRIPTS_DIR,
    MAGNET_RANGE,
    QUAD_OPT_SETTINGS,
    SEQ_NAME,
    SEQUENCE_FILE,
    SEXT_OPT_SETTINGS,
)
from aba_optimiser.mad.mad_interface import MadInterface

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

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
        init_coords: np.ndarray,
        start_bpm: str,
        optimise_sextupoles: bool = False,
    ) -> None:
        """
        New constructor that accepts compact numpy payloads built by the parent
        process to avoid sending the full DataFrame to every worker.
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn

        LOGGER.debug(
            f"Initializing worker {worker_id} for BPM {start_bpm} with {len(init_coords)} particles"
        )

        self.x_comparisons = x_comparisons
        self.y_comparisons = y_comparisons

        num_batches = QUAD_OPT_SETTINGS.num_batches
        if optimise_sextupoles:
            num_batches = SEXT_OPT_SETTINGS.num_batches

        init_coords = np.array_split(init_coords, num_batches)

        # Convert init_coords from list of matrices into list of list of lists
        self.init_coords = [ic.tolist() for ic in init_coords]
        self.batch_size = len(self.init_coords[0])

        self.start_bpm = start_bpm
        self.num_batches = num_batches
        self.optimise_sextupoles = optimise_sextupoles

        self._split_data_to_batches()

        # Load common MAD scripts
        self.run_track_script = (MAD_SCRIPTS_DIR / "run_track.mad").read_text()

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

    def send_initial_conditions(self, mad: MAD) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send("""
da_x0_c = table.new(num_batches, 0)
init_coords = py:recv()
for i=1,num_batches do
    da_x0_c[i] = table.new(batch_size, 0)
    for j=1,batch_size do
        da_x0_c[i][j] = da_x0_base:copy()
        da_x0_c[i][j]:set0(init_coords[i][j])
    end
end
            """)
        mad.send(self.init_coords)

    def setup_mad_interface(self) -> MAD:
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
            optimise_sextupoles=self.optimise_sextupoles,
        )

        mad = mad_iface.mad
        mad["knob_names"] = mad_iface.knob_names
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix", "vector")
        mad.load("MAD.utility", "tblcat")

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad)
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    loaded_sequence[param] = loaded_sequence[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

        # Setup sequence specific to worker type
        self.setup_mad_sequence(mad)

        return mad

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """
        Process tracking results and compute gradients and loss.

        Returns:
            grad: Gradient array
            loss: Loss value
        """
        # Update knob values
        update_string = [
            f"MADX.{SEQ_NAME}['{name}']:set0({val:.15e})"
            for name, val in knob_updates.items()
        ]

        # Run tracking
        mad.send("\n".join(update_string))
        mad.send(f"batch = {batch + 1}")  # + 1 for python to lua list index
        mad.send(self.run_track_script)

        # Get results
        x_res = mad.recv()
        y_res = mad.recv()
        dx_dk = mad.recv()
        dy_dk = mad.recv()

        # Process results into arrays
        x_stack = np.asarray(x_res).squeeze(-1)  # (batch_size, n_data_points)
        y_stack = np.asarray(y_res).squeeze(-1)  # (batch_size, n_data_points)
        dx_stack = np.stack(dx_dk, axis=0)
        dy_stack = np.stack(dy_dk, axis=0)

        x_diffs = x_stack - self.x_comparisons[batch]
        y_diffs = y_stack - self.y_comparisons[batch]

        # Compute gradients using vector-Jacobian products
        gx = np.einsum("pkm,pm->k", dx_stack, x_diffs)  # (K,)
        gy = np.einsum("pkm,pm->k", dy_stack, y_diffs)  # (K,)

        # Build gradient and loss
        grad = 2.0 * (gx + gy)  # (K,)
        loss = (x_diffs**2).sum() + (y_diffs**2).sum()  # scalar

        return grad, loss

    def run(self) -> None:
        """Main worker run loop."""
        # Setup MAD interface
        mad = self.setup_mad_interface()

        # Send initial conditions
        self.send_initial_conditions(mad)

        # Initialise the MAD environment ready for tracking
        mad.send((MAD_SCRIPTS_DIR / "run_track_init.mad").read_text())

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

            # Send results back
            self.conn.send((self.worker_id, grad, loss))

            # Receive next knob updates
            knob_updates = self.conn.recv()

            batch = (batch + 1) % self.num_batches

        # Final hessian estimation
        mad.send("""
var_x = py:recv()
var_y = py:recv()
""")
        mad.send(1 / hessian_var_x)
        mad.send(1 / hessian_var_y)
        mad.send((MAD_SCRIPTS_DIR / "estimate_hessian.mad").read_text())
        h_part = mad.recv()
        self.conn.send(h_part)  # shape (n_knobs, n_knobs)
