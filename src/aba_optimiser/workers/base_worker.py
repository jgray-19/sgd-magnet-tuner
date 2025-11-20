from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Process
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.mad.scripts import HESSIAN_SCRIPT, TRACK_INIT, TRACK_SCRIPT

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerData:
    """Container for worker tracking data arrays."""

    position_comparisons: np.ndarray  # Shape: (n_turns, n_data_points, 2) - [x, y]
    momentum_comparisons: np.ndarray  # Shape: (n_turns, n_data_points, 2) - [px, py]
    position_variances: np.ndarray  # Shape: (n_turns, n_data_points, 2) - [var_x, var_y]
    momentum_variances: np.ndarray  # Shape: (n_turns, n_data_points, 2) - [var_px, var_py]
    init_coords: np.ndarray  # Non-kicked planes are zeroed out
    init_pts: np.ndarray


@dataclass
class WorkerConfig:
    """Container for worker configuration parameters."""

    start_bpm: str
    end_bpm: str
    magnet_range: str
    sequence_file_path: Path
    corrector_strengths: Path
    tune_knobs_file: Path
    beam_energy: float
    sdir: int = 1
    bad_bpms: list[str] | None = None
    seq_name: str | None = None


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
        data: WorkerData,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        """
        Constructor that accepts grouped data and config parameters for cleaner code.
        """
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn

        LOGGER.debug(
            f"Initialising worker {worker_id} for BPM {config.start_bpm} with {len(data.init_coords)} particles"
        )

        num_batches = simulation_config.num_batches

        # reduce data to be divisible by num_batches
        n_init = len(data.init_coords) - (len(data.init_coords) % num_batches)
        init_coords = data.init_coords[:n_init]

        LOGGER.debug(
            f"Worker {worker_id}: Using {n_init} initial coordinates divided into {num_batches} batches"
        )

        # Unpack 2D arrays into separate x/y arrays
        # Shape: (n_tracks, n_data_points)
        self.x_comparisons = data.position_comparisons[:n_init, :, 0]  # x positions
        self.y_comparisons = data.position_comparisons[:n_init, :, 1]  # y positions
        self.px_comparisons = data.momentum_comparisons[:n_init, :, 0]  # px momenta
        self.py_comparisons = data.momentum_comparisons[:n_init, :, 1]  # py momenta
        positional_variances = data.position_variances[:n_init]
        self.x_variances = positional_variances[:, :, 0]
        self.y_variances = positional_variances[:, :, 1]

        momentum_variances = data.momentum_variances[:n_init]
        self.px_variances = momentum_variances[:, :, 0]
        self.py_variances = momentum_variances[:, :, 1]

        x_weights = self._variance_to_weight(self.x_variances)
        y_weights = self._variance_to_weight(self.y_variances)
        px_weights = self._variance_to_weight(self.px_variances)
        py_weights = self._variance_to_weight(self.py_variances)

        self.hessian_weight_x = self._aggregate_hessian_weights(x_weights)
        self.hessian_weight_y = self._aggregate_hessian_weights(y_weights)
        self.hessian_weight_px = self._aggregate_hessian_weights(px_weights)
        self.hessian_weight_py = self._aggregate_hessian_weights(py_weights)

        self.x_weights = self._normalise_weights(x_weights)
        self.y_weights = self._normalise_weights(y_weights)
        self.px_weights = self._normalise_weights(px_weights)
        self.py_weights = self._normalise_weights(py_weights)

        # Split init_coords into num_batches batches
        init_coords = np.array_split(init_coords, num_batches)

        # Check if there are nans in the initial coordinates
        if np.isnan(init_coords).any():
            raise ValueError("NaNs found in initial coordinates")

        # Convert init_coords from list of matrices into list of list of lists
        self.init_coords = [ic.tolist() for ic in init_coords]
        self.batch_size = len(self.init_coords[0])

        self.config = config
        self.num_batches = num_batches
        self.simulation_config = simulation_config

        self.init_pts = data.init_pts
        self._split_data_to_batches()

        # Load common MAD scripts
        self.run_track_script = TRACK_SCRIPT.read_text()
        self.run_track_init_path = TRACK_INIT

    @abstractmethod
    def get_bpm_range(self, sdir: int) -> str:
        """Get the magnet range specific to the worker type."""
        pass

    @staticmethod
    @abstractmethod
    def get_n_data_points(nbpms: int) -> int:
        """Get the number of data points for comparison."""
        pass

    @abstractmethod
    def setup_mad_sequence(self, mad: MAD) -> None:
        """Setup MAD sequence specific to the worker type."""
        pass

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def _adjust_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Multiply weights by the first BPM weight.
        """
        first_bpm_weight = weights[:, 0]
        return weights * first_bpm_weight[:, np.newaxis]

    @staticmethod
    def _variance_to_weight(variances: np.ndarray) -> np.ndarray:
        """Convert variances to inverse-variance weights, zeroing invalid entries."""
        weights = np.zeros_like(variances, dtype=np.float64)
        valid = np.isfinite(variances) & (variances > 0.0)
        np.divide(1.0, variances, out=weights, where=valid)
        return weights

    @staticmethod
    def _normalise_weights(weights: np.ndarray) -> np.ndarray:
        """Normalise weights so that the maximum weight is 1."""
        max_weight = np.max(weights)
        if max_weight > 0:
            return weights / max_weight
        return weights

    @staticmethod
    def _aggregate_hessian_weights(weights: np.ndarray) -> np.ndarray:
        """Aggregate per-track weights into a single weight per BPM for Hessian use."""
        if weights.size == 0:
            return np.array([], dtype=np.float64)

        sums = np.sum(weights, axis=0)
        counts = np.count_nonzero(weights, axis=0)
        aggregated = np.zeros_like(sums, dtype=np.float64)
        np.divide(sums, counts, out=aggregated, where=counts > 0)
        return aggregated

    def _split_data_to_batches(self):
        """
        Split the comparison data, which will be of dimensions (n_tracks, n_data_points),
        into self.num_batches batches, producing arrays of shape (batch_size, n_data_points).
        Also split the initial conditions into N batches.
        """
        self.x_comparisons = np.array_split(self.x_comparisons, self.num_batches, axis=0)
        self.y_comparisons = np.array_split(self.y_comparisons, self.num_batches, axis=0)
        self.px_comparisons = np.array_split(self.px_comparisons, self.num_batches, axis=0)
        self.py_comparisons = np.array_split(self.py_comparisons, self.num_batches, axis=0)
        self.x_weights = np.array_split(self.x_weights, self.num_batches, axis=0)
        self.y_weights = np.array_split(self.y_weights, self.num_batches, axis=0)
        self.px_weights = np.array_split(self.px_weights, self.num_batches, axis=0)
        self.py_weights = np.array_split(self.py_weights, self.num_batches, axis=0)

        self.init_pts = [
            arr.tolist() for arr in np.array_split(self.init_pts, self.num_batches, axis=0)
        ]

    def send_initial_conditions(self, mad: MAD) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send("""
init_coords = python:recv()
init_pts = python:recv()
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

    def setup_mad_interface(self, init_knobs: dict[str, float]) -> tuple[MAD, int]:
        """
        Initialise MAD interface and setup common MAD configuration.

        Returns:
            mad_iface: The MAD interface object
            mad: The MAD object
            nbpms: Number of BPMs
        """
        LOGGER.debug(f"Worker {self.worker_id}: Setting up MAD interface")

        # Get magnet range specific to worker type
        bpm_range = self.get_bpm_range(sdir=1)
        LOGGER.debug(f"Worker {self.worker_id}: Using BPM range {bpm_range}")

        mad_iface = OptimisationMadInterface(
            self.config.sequence_file_path,
            py_name="python",
            seq_name=self.config.seq_name,
            magnet_range=self.config.magnet_range,
            bpm_range=bpm_range,
            simulation_config=self.simulation_config,
            bad_bpms=self.config.bad_bpms,
            corrector_strengths=self.config.corrector_strengths,
            tune_knobs_file=self.config.tune_knobs_file,
            beam_energy=self.config.beam_energy,
        )
        knob_names = mad_iface.knob_names
        if knob_names != list(init_knobs.keys()):
            raise ValueError(
                f"Worker {self.worker_id}: Knob names from MAD {knob_names} do not match initial knobs {list(init_knobs.keys())}"
            )

        # Remove "pt" from knob names if present
        if "pt" in knob_names:
            knob_names.remove("pt")

        mad = mad_iface.mad
        mad["knob_names"] = knob_names
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["nbpms"] = mad_iface.nbpms
        mad["sdir"] = self.config.sdir
        mad["optimise_energy"] = self.simulation_config.optimise_energy

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
        machine_pt = knob_updates.pop("pt", 0.0)

        # Prepare MAD commands to update knob values in the sequence
        update_commands = [f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]

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
        px_results = mad.recv()  # Simulated px-momenta: list of arrays
        py_results = mad.recv()  # Simulated py-momenta: list of arrays

        dx_dk_results = mad.recv()  # Derivatives of x w.r.t. knobs: list of arrays
        dy_dk_results = mad.recv()  # Derivatives of y w.r.t. knobs: list of arrays
        dpx_dk_results = mad.recv()  # Derivatives of px w.r.t. knobs: list of arrays
        dpy_dk_results = mad.recv()  # Derivatives of py w.r.t. knobs: list of arrays
        # del px_results, py_results, dpx_dk_results, dpy_dk_results  # Unused for now

        # Convert results to NumPy arrays for efficient vectorised processing
        # Squeeze to remove singleton dimensions (e.g., from MAD's output format)
        # Shape: (n_particles, n_data_points)
        particle_positions_x = np.asarray(x_results).squeeze(-1)
        particle_positions_y = np.asarray(y_results).squeeze(-1)
        particle_momenta_px = np.asarray(px_results).squeeze(-1)
        particle_momenta_py = np.asarray(py_results).squeeze(-1)

        # Shape: (n_particles, n_knobs + 1, n_data_points)
        dx_dk = np.stack(dx_dk_results, axis=0)
        dy_dk = np.stack(dy_dk_results, axis=0)
        dpx_dk = np.stack(dpx_dk_results, axis=0)
        dpy_dk = np.stack(dpy_dk_results, axis=0)

        wx = self.x_weights[batch]
        wy = self.y_weights[batch]
        wpx = self.px_weights[batch]
        wpy = self.py_weights[batch]

        residual_x = particle_positions_x - self.x_comparisons[batch]
        residual_y = particle_positions_y - self.y_comparisons[batch]
        residual_px = particle_momenta_px - self.px_comparisons[batch]
        residual_py = particle_momenta_py - self.py_comparisons[batch]

        gx = np.einsum("pkm,pm->k", dx_dk, wx * residual_x)
        gpx = np.einsum("pkm,pm->k", dpx_dk, wpx * residual_px)
        gy = np.einsum("pkm,pm->k", dy_dk, wy * residual_y)
        gpy = np.einsum("pkm,pm->k", dpy_dk, wpy * residual_py)
        loss = (
            np.sum(wx * residual_x**2)
            + np.sum(wpx * residual_px**2)
            + np.sum(wy * residual_y**2)
            + np.sum(wpy * residual_py**2)
        )

        grad = 2.0 * (gx + gy + gpx + gpy)
        # grad = 2.0 * (gy + gpy)
        return grad, loss

    def run(self) -> None:
        """Main worker run loop."""
        self.conn.recv()  # Initial handshake from the manager
        knob_values, batch = self.conn.recv()  # shape (n_knobs,)

        # Setup MAD interface
        mad, nbpms = self.setup_mad_interface(knob_values)

        # Send initial conditions
        self.send_initial_conditions(mad)

        # Main tracking loop
        LOGGER.debug(
            "Worker %s: Prepared Hessian weights for %d BPMs",
            self.worker_id,
            self.hessian_weight_x.size,
        )

        # Initialise the MAD environment ready for tracking
        mad.send(self.run_track_init_path.read_text())

        while knob_values is not None:
            # Process tracking and compute gradients
            grad, loss = self.compute_gradients_and_loss(mad, knob_values, batch)

            # Send results back - normalised with nbpms
            self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))

            # Receive next knob updates
            knob_values, batch = self.conn.recv()

        # Final hessian estimation
        mad.send("""
weights_x = python:recv()
weights_px = python:recv()
weights_y = python:recv()
weights_py = python:recv()
""")
        mad.send(self.hessian_weight_x.tolist())
        mad.send(self.hessian_weight_px.tolist())
        mad.send(self.hessian_weight_y.tolist())
        mad.send(self.hessian_weight_py.tolist())
        mad.send(HESSIAN_SCRIPT.read_text())
        h_part = mad.recv()
        self.conn.send(h_part)  # shape (n_knobs, n_knobs)
        del mad
