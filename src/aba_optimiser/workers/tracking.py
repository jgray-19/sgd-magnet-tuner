"""Particle tracking worker for multi-turn beam dynamics simulations.

This module implements the TrackingWorker class which performs particle
tracking simulations and computes gradients for optimization. It handles
both position and momentum observables with full symmetry between x/y planes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.scripts import HESSIAN_SCRIPT, TRACK_INIT, TRACK_SCRIPT
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import TrackingData, WeightProcessor, split_array_to_batches

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class PhaseSpaceWeights:
    """Container for weights in all phase space dimensions.

    Attributes:
        x: Weights for horizontal position
        y: Weights for vertical position
        px: Weights for horizontal momentum
        py: Weights for vertical momentum
    """

    x: list[np.ndarray]  # List of batches, each (batch_size, n_data_points)
    y: list[np.ndarray]
    px: list[np.ndarray]
    py: list[np.ndarray]


@dataclass
class PhaseSpaceComparisons:
    """Container for reference data in all phase space dimensions.

    Attributes:
        x: Reference horizontal positions
        y: Reference vertical positions
        px: Reference horizontal momenta
        py: Reference vertical momenta
    """

    x: list[np.ndarray]  # List of batches, each (batch_size, n_data_points)
    y: list[np.ndarray]
    px: list[np.ndarray]
    py: list[np.ndarray]


class TrackingWorker(AbstractWorker[TrackingData]):
    """Worker for particle tracking simulations.

    This worker performs particle tracking through accelerator lattices,
    computing positions and momenta at each BPM. It calculates gradients
    of the loss function with respect to optimization knobs using
    differential algebra techniques.

    Supports two modes:
    - 'multi-turn': Track particles for multiple turns (default)
    - 'arc-by-arc': Single-turn tracking through one arc/section

    The implementation treats x/y and position/momentum symmetrically,
    ensuring consistent handling of all phase space dimensions.
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        data: TrackingData,
        config: WorkerConfig,
        simulation_config: SimulationConfig,
        mode: str = "multi-turn",
    ) -> None:
        """Initialize the tracking worker.

        Args:
            conn: Pipe connection for communicating with main process
            worker_id: Unique identifier for this worker
            data: TrackingData container with reference measurements
            config: Configuration parameters
            simulation_config: Simulation configuration settings
            mode: Tracking mode - 'multi-turn' or 'arc-by-arc'

        Raises:
            ValueError: If mode is not 'multi-turn' or 'arc-by-arc'
        """
        if mode not in ("multi-turn", "arc-by-arc"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'multi-turn' or 'arc-by-arc'")
        self.mode = mode
        super().__init__(conn, worker_id, data, config, simulation_config)

    def prepare_data(self, data: TrackingData) -> None:
        """Process and prepare tracking data for computation.

        Extracts position and momentum data for both planes, computes
        weights from variances, splits data into batches, and prepares
        initial conditions.

        Args:
            data: TrackingData container with reference measurements
        """
        num_batches = self.simulation_config.num_batches

        # Ensure data is divisible by number of batches
        n_init = len(data.init_coords) - (len(data.init_coords) % num_batches)
        init_coords = data.init_coords[:n_init]

        LOGGER.debug(
            f"Worker {self.worker_id}: Processing {n_init} particles in {num_batches} batches"
        )

        # Validate initial conditions
        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")

        # Extract and process comparison data symmetrically for all dimensions
        self._extract_comparisons(data, n_init)
        self._compute_weights(data, n_init)
        self._prepare_batches(init_coords, data.init_pts, num_batches)

        # Load MAD-NG tracking scripts
        self.run_track_script = TRACK_SCRIPT.read_text()
        self.run_track_init_path = TRACK_INIT

    def _extract_comparisons(self, data: TrackingData, n_init: int) -> None:
        """Extract reference data for all phase space dimensions.

        Args:
            data: TrackingData container
            n_init: Number of particles to use
        """
        # Extract position data (x, y)
        positions = data.position_comparisons[:n_init]
        self.x_comparisons_full = positions[:, :, 0]
        self.y_comparisons_full = positions[:, :, 1]

        # Extract momentum data (px, py)
        momenta = data.momentum_comparisons[:n_init]
        self.px_comparisons_full = momenta[:, :, 0]
        self.py_comparisons_full = momenta[:, :, 1]

    def _compute_weights(self, data: TrackingData, n_init: int) -> None:
        """Compute weights from variances for all phase space dimensions.

        Converts measurement variances to inverse-variance weights,
        normalizes them, and prepares Hessian weights.

        Args:
            data: TrackingData container
            n_init: Number of particles to use
        """
        # Extract variances symmetrically
        pos_variances = data.position_variances[:n_init]
        mom_variances = data.momentum_variances[:n_init]

        x_vars = pos_variances[:, :, 0]
        y_vars = pos_variances[:, :, 1]
        px_vars = mom_variances[:, :, 0]
        py_vars = mom_variances[:, :, 1]

        # Convert variances to weights symmetrically
        x_weights = WeightProcessor.variance_to_weight(x_vars)
        y_weights = WeightProcessor.variance_to_weight(y_vars)
        px_weights = WeightProcessor.variance_to_weight(px_vars)
        py_weights = WeightProcessor.variance_to_weight(py_vars)

        # Prepare Hessian weights (aggregated across particles)
        self.hessian_weight_x = WeightProcessor.aggregate_hessian_weights(x_weights)
        self.hessian_weight_y = WeightProcessor.aggregate_hessian_weights(y_weights)
        self.hessian_weight_px = WeightProcessor.aggregate_hessian_weights(px_weights)
        self.hessian_weight_py = WeightProcessor.aggregate_hessian_weights(py_weights)

        # Normalize weights globally across all dimensions for gradient computation
        self.x_weights_full, self.y_weights_full, self.px_weights_full, self.py_weights_full = (
            WeightProcessor.normalise_weights_globally(x_weights, y_weights, px_weights, py_weights)
        )

    def _prepare_batches(
        self, init_coords: np.ndarray, init_pts: np.ndarray, num_batches: int
    ) -> None:
        """Split data and initial conditions into batches.

        Args:
            init_coords: Initial particle coordinates
            init_pts: Initial transverse momentum values
            num_batches: Number of batches to create
        """
        # Split initial conditions
        init_coords_batches = split_array_to_batches(init_coords, num_batches)
        init_pts_batches = split_array_to_batches(init_pts, num_batches)

        # Convert to nested lists for MAD-NG
        self.init_coords = [batch.tolist() for batch in init_coords_batches]
        self.init_pts = [batch.tolist() for batch in init_pts_batches]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches

        # Split comparison data into batches (symmetrically)
        self.comparisons = PhaseSpaceComparisons(
            x=split_array_to_batches(self.x_comparisons_full, num_batches),
            y=split_array_to_batches(self.y_comparisons_full, num_batches),
            px=split_array_to_batches(self.px_comparisons_full, num_batches),
            py=split_array_to_batches(self.py_comparisons_full, num_batches),
        )

        # Split weights into batches (symmetrically)
        self.weights = PhaseSpaceWeights(
            x=split_array_to_batches(self.x_weights_full, num_batches),
            y=split_array_to_batches(self.y_weights_full, num_batches),
            px=split_array_to_batches(self.px_weights_full, num_batches),
            py=split_array_to_batches(self.py_weights_full, num_batches),
        )

    def get_bpm_range(self, sdir: int) -> str:
        """Get BPM range string for MAD-NG tracking.

        Args:
            sdir: Direction of propagation (+1 forward, -1 backward)

        Returns:
            BPM range in format "start/end" or "end/start" for backward
        """
        if sdir == -1:
            return f"{self.config.end_bpm}/{self.config.start_bpm}"
        return f"{self.config.start_bpm}/{self.config.end_bpm}"

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for tracking.

        Sets batch size, number of batches, and optimization flags.
        For arc-by-arc mode, also sets single-turn tracking.

        Args:
            mad: MAD-NG interface object
        """
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["optimise_energy"] = self.simulation_config.optimise_energy

        if self.mode == "arc-by-arc":
            # Single-turn tracking for arc-by-arc mode
            mad["n_run_turns"] = 1
            # Set tracking range (starts at turn 0 for single turn)
            mad["tracking_range"] = self.get_bpm_range(self.config.sdir)

    def _setup_da_maps(self, mad: MAD) -> None:
        """Setup differential algebra maps for tracking.

        Creates base DAMAP and adds knob parameters for differentiation.

        Args:
            mad: MAD-NG interface object
        """
        # Remove "pt" from knob names if present (handled separately)
        knob_names = list(mad["knob_names"])
        if "pt" in knob_names:
            knob_names.remove("pt")
            mad["knob_names"] = knob_names

        # Create base DAMAP
        self.create_base_damap(mad, knob_order=1)

        # Add knobs as TPSA variables
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    MADX[param] = MADX[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

    def send_initial_conditions(self, mad: MAD) -> None:
        """Send initial particle coordinates to MAD-NG.

        Creates DAMAP objects for each particle in each batch.

        Args:
            mad: MAD-NG interface object
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

    def _initialize_mad_computation(self, mad: MAD) -> None:
        """Initialize MAD-NG environment for tracking computations.

        Args:
            mad: MAD-NG interface object
        """
        mad.send(self.run_track_init_path.read_text())

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute gradients and loss for a batch of particle tracking.

        Performs tracking simulation, receives position and momentum data
        along with their derivatives, and computes loss and gradients using
        weighted least-squares formulation.

        Args:
            mad: MAD-NG interface object
            knob_updates: Dictionary of knob names to values
            batch: Batch index to process

        Returns:
            Tuple of (gradient array, loss value)
        """
        # Extract and update energy deviation
        machine_pt = knob_updates.pop("pt", 0.0)

        # Send knob updates to MAD-NG
        update_commands = [f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]
        mad.send("\n".join(update_commands))

        # Set batch index (Lua uses 1-based indexing)
        mad.send(f"batch = {batch + 1}")

        # Update energy deviation for this batch
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")

        # Run tracking
        mad.send(self.run_track_script)

        # Receive results symmetrically for all phase space dimensions
        results = self._receive_tracking_results(mad)

        # Compute loss and gradients
        return self._compute_loss_and_gradients(results, batch)

    def _receive_tracking_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive tracking results from MAD-NG.

        Args:
            mad: MAD-NG interface object

        Returns:
            Dictionary with keys: x, y, px, py, dx_dk, dy_dk, dpx_dk, dpy_dk
        """
        # Receive position and momentum data
        x_results = mad.recv()
        y_results = mad.recv()
        px_results = mad.recv()
        py_results = mad.recv()

        # Receive derivatives
        dx_dk_results = mad.recv()
        dy_dk_results = mad.recv()
        dpx_dk_results = mad.recv()
        dpy_dk_results = mad.recv()

        # Convert to numpy arrays
        # Shape: (n_particles, n_data_points)
        x = np.asarray(x_results).squeeze(-1)
        y = np.asarray(y_results).squeeze(-1)
        px = np.asarray(px_results).squeeze(-1)
        py = np.asarray(py_results).squeeze(-1)

        # Shape: (n_particles, n_knobs, n_data_points)
        dx_dk = np.stack(dx_dk_results, axis=0)
        dy_dk = np.stack(dy_dk_results, axis=0)
        dpx_dk = np.stack(dpx_dk_results, axis=0)
        dpy_dk = np.stack(dpy_dk_results, axis=0)
        dpy_dk = np.stack(dpy_dk_results, axis=0)

        return {
            "x": x,
            "y": y,
            "px": px,
            "py": py,
            "dx_dk": dx_dk,
            "dy_dk": dy_dk,
            "dpx_dk": dpx_dk,
            "dpy_dk": dpy_dk,
        }

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute weighted loss and gradients from tracking results.

        Uses symmetric treatment of all phase space dimensions.

        Args:
            results: Dictionary of tracking results and derivatives
            batch: Batch index

        Returns:
            Tuple of (gradient array, loss value)
        """
        # Get weights and comparisons for this batch
        wx = self.weights.x[batch]
        wy = self.weights.y[batch]
        wpx = self.weights.px[batch]
        wpy = self.weights.py[batch]

        # Compute residuals symmetrically
        residual_x = results["x"] - self.comparisons.x[batch]
        residual_y = results["y"] - self.comparisons.y[batch]
        residual_px = results["px"] - self.comparisons.px[batch]
        residual_py = results["py"] - self.comparisons.py[batch]

        # Compute gradients using Einstein summation (symmetric for all dimensions)
        # einsum("pkm,pm->k") computes: sum over particles and data points
        gx = np.einsum("pkm,pm->k", results["dx_dk"], wx * residual_x)
        gy = np.einsum("pkm,pm->k", results["dy_dk"], wy * residual_y)
        gpx = np.einsum("pkm,pm->k", results["dpx_dk"], wpx * residual_px)
        gpy = np.einsum("pkm,pm->k", results["dpy_dk"], wpy * residual_py)

        # Compute loss (weighted sum of squared residuals)
        loss_x = np.sum(wx * residual_x**2)
        loss_y = np.sum(wy * residual_y**2)
        loss_px = np.sum(wpx * residual_px**2)
        loss_py = np.sum(wpy * residual_py**2)

        # Total gradient and loss (factor of 2 from derivative of squared residuals)
        grad = 2.0 * (gx + gy + gpx + gpy)
        loss = loss_x + loss_y + loss_px + loss_py

        return grad, loss

    def run(self) -> None:
        """Main worker run loop with Hessian calculation.

        Extends the base run method to compute approximate Hessian
        after the main optimization loop completes.
        """
        # Initial handshake
        self.conn.recv()
        knob_values, batch = self.conn.recv()

        # Setup MAD interface
        mad, nbpms = self.setup_mad_interface(knob_values)

        # Send initial conditions
        self.send_initial_conditions(mad)

        # Initialize MAD environment for computation
        self._initialize_mad_computation(mad)

        LOGGER.debug(f"Worker {self.worker_id}: Ready for computation with {nbpms} BPMs")

        # Main computation loop
        while knob_values is not None:
            # Compute gradients and loss
            grad, loss = self.compute_gradients_and_loss(mad, knob_values, batch)

            # Normalize and send results
            self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))

            # Wait for next knob values
            knob_values, batch = self.conn.recv()

        # Compute Hessian approximation
        LOGGER.debug(f"Worker {self.worker_id}: Computing Hessian approximation")
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

        # Cleanup
        LOGGER.debug(f"Worker {self.worker_id}: Terminating")
        del mad

    @staticmethod
    def get_n_data_points(nbpms: int, n_turns: int = 1) -> int:
        """Get number of data points for tracking.

        Args:
            nbpms: Number of BPMs in the range
            n_turns: Number of tracking turns (default 1 for arc-by-arc)

        Returns:
            Total number of data points (nbpms * n_turns)
        """
        return nbpms * n_turns
