"""Position-only particle tracking worker for beam dynamics simulations.

This module implements the PositionOnlyTrackingWorker class which performs particle
tracking simulations and computes gradients for optimisation using only position
observables (x, y). This is useful when momentum data (px, py) is unreliable or
not needed for optimisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.scripts import (
    HESSIAN_SCRIPT_POS_ONLY,
    TRACK_INIT_POS_ONLY,
    TRACK_SCRIPT_POS_ONLY,
)
from aba_optimiser.workers.common import TrackingData, split_array_to_batches
from aba_optimiser.workers.tracking import TrackingWorker

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)


@dataclass
class PositionWeights:
    """Container for weights in position dimensions only.

    Attributes:
        x: Weights for horizontal position
        y: Weights for vertical position
    """

    x: list[np.ndarray]  # List of batches, each (batch_size, n_data_points)
    y: list[np.ndarray]


@dataclass
class PositionComparisons:
    """Container for reference data in position dimensions only.

    Attributes:
        x: Reference horizontal positions
        y: Reference vertical positions
    """

    x: list[np.ndarray]  # List of batches, each (batch_size, n_data_points)
    y: list[np.ndarray]


class PositionOnlyTrackingWorker(TrackingWorker):
    """Worker for position-only particle tracking simulations.

    Inherits from TrackingWorker but only tracks positions (x, y),
    ignoring momenta (px, py). This reduces memory usage and computation
    time when momentum data is not needed for optimisation.
    """

    def prepare_data(self, data: TrackingData) -> None:
        """Process and prepare tracking data for computation (position-only).

        Overrides parent to use position-only MAD scripts and skip momentum processing.
        """
        num_batches = self.simulation_config.num_batches
        n_init = len(data.init_coords) - (len(data.init_coords) % num_batches)
        init_coords = data.init_coords[:n_init]

        LOGGER.debug(
            f"Worker {self.worker_id}: Processing {n_init} particles in {num_batches} batches (position-only)"
        )

        if np.isnan(init_coords).any():
            raise ValueError(f"Worker {self.worker_id}: NaNs found in initial coordinates")

        self._extract_comparisons(data, n_init)
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for TrackingWorker")
        self._load_precomputed_weights(data.precomputed_weights, n_init)
        self._prepare_batches(init_coords, data.init_pts, num_batches)

        # Use position-only MAD scripts
        self.run_track_script = TRACK_SCRIPT_POS_ONLY.read_text()
        self.run_track_init_path = TRACK_INIT_POS_ONLY

    def _extract_comparisons(self, data: TrackingData, n_init: int) -> None:
        """Extract reference data for position dimensions only."""
        positions = data.position_comparisons[:n_init]
        self.x_comparisons_full = positions[:, :, 0]
        self.y_comparisons_full = positions[:, :, 1]

    def _prepare_batches(
        self, init_coords: np.ndarray, init_pts: np.ndarray, num_batches: int
    ) -> None:
        """Split data and initial conditions into batches (position-only)."""
        init_coords_batches = split_array_to_batches(init_coords, num_batches)
        init_pts_batches = split_array_to_batches(init_pts, num_batches)

        self.init_coords = [batch.tolist() for batch in init_coords_batches]
        self.init_pts = [batch.tolist() for batch in init_pts_batches]
        self.batch_size = len(self.init_coords[0])
        self.num_batches = num_batches

        self.comparisons = PositionComparisons(
            x=split_array_to_batches(self.x_comparisons_full, num_batches),
            y=split_array_to_batches(self.y_comparisons_full, num_batches),
        )
        self.weights = PositionWeights(
            x=split_array_to_batches(self.x_weights_full, num_batches),
            y=split_array_to_batches(self.y_weights_full, num_batches),
        )

    def _receive_tracking_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive tracking results from MAD-NG (position-only, 4 items)."""
        x_results = mad.recv()
        y_results = mad.recv()
        dx_dk_results = mad.recv()
        dy_dk_results = mad.recv()

        return {
            "x": np.asarray(x_results).squeeze(-1),
            "y": np.asarray(y_results).squeeze(-1),
            "dx_dk": np.stack(dx_dk_results, axis=0),
            "dy_dk": np.stack(dy_dk_results, axis=0),
        }

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute weighted loss and gradients using position data only."""
        wx = self.weights.x[batch]
        wy = self.weights.y[batch]

        residual_x = results["x"] - self.comparisons.x[batch]
        residual_y = results["y"] - self.comparisons.y[batch]

        gx = np.einsum("pkm,pm->k", results["dx_dk"], wx * residual_x)
        gy = np.einsum("pkm,pm->k", results["dy_dk"], wy * residual_y)

        loss_x = np.sum(wx * residual_x**2)
        loss_y = np.sum(wy * residual_y**2)

        return 2.0 * (gx + gy), loss_x + loss_y

    def _send_hessian_weights(self, mad: MAD) -> None:
        """Send Hessian weights to MAD (position only, no momentum weights)."""
        mad.send("""
weights_x = python:recv()
weights_y = python:recv()
""")
        mad.send(self.hessian_weight_x.tolist())
        mad.send(self.hessian_weight_y.tolist())

    def run(self) -> None:
        """Main worker run loop with Hessian calculation."""
        self.conn.recv()
        knob_values, batch = self.conn.recv()

        mad, nbpms = self.setup_mad_interface(knob_values)
        self.send_initial_conditions(mad)
        self._initialise_mad_computation(mad)

        LOGGER.debug(f"Worker {self.worker_id}: Ready for position-only computation with {nbpms} BPMs")

        while knob_values is not None:
            grad, loss = self.compute_gradients_and_loss(mad, knob_values, batch)
            self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))
            knob_values, batch = self.conn.recv()

        LOGGER.debug(f"Worker {self.worker_id}: Computing Hessian approximation (position-only)")
        self._send_hessian_weights(mad)
        mad.send(HESSIAN_SCRIPT_POS_ONLY.read_text())
        self.conn.send(mad.recv())

        LOGGER.debug(f"Worker {self.worker_id}: Terminating")
        del mad
