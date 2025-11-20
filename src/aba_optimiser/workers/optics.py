"""Optics function worker for beta function optimization.

This module implements the OpticsWorker class which computes beta functions
and their gradients for optics matching. It treats beta_x and beta_y
symmetrically throughout the computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.scripts import TRACK_OPTICS_INIT, TRACK_OPTICS_SCRIPT
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import OpticsData, WeightProcessor

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)


@dataclass
class BetaWeights:
    """Container for beta function weights.

    Attributes:
        x: Weights for horizontal beta function
        y: Weights for vertical beta function
    """

    x: np.ndarray  # Shape: (n_bpms,)
    y: np.ndarray  # Shape: (n_bpms,)


@dataclass
class BetaComparisons:
    """Container for reference beta function data.

    Attributes:
        x: Reference horizontal beta functions
        y: Reference vertical beta functions
    """

    x: np.ndarray  # Shape: (n_bpms,)
    y: np.ndarray  # Shape: (n_bpms,)


class OpticsWorker(AbstractWorker[OpticsData]):
    """Worker for optics function computations.

    This worker computes beta functions (and potentially other Twiss parameters)
    at BPM locations and their gradients with respect to optimization knobs.
    It uses differential algebra for automatic differentiation.

    The implementation treats beta_x and beta_y symmetrically to ensure
    consistent handling of both transverse planes.
    """

    def prepare_data(self, data: OpticsData) -> None:
        """Process and prepare optics data for computation.

        Extracts beta function measurements, computes weights from variances,
        and prepares initial Twiss parameters.

        Args:
            data: OpticsData container with reference measurements
        """
        LOGGER.debug(
            f"Worker {self.worker_id}: Processing optics data "
            f"with {len(data.beta_comparisons)} BPMs"
        )

        # Extract beta comparisons symmetrically
        self.comparisons = BetaComparisons(
            x=data.beta_comparisons[:, 0],  # beta_x
            y=data.beta_comparisons[:, 1],  # beta_y
        )

        # Compute weights from variances symmetrically
        betx_vars = data.beta_variances[:, 0]
        bety_vars = data.beta_variances[:, 1]

        betx_weights = WeightProcessor.variance_to_weight(betx_vars)
        bety_weights = WeightProcessor.variance_to_weight(bety_vars)

        # Store Hessian weights (used for approximate Hessian calculation)
        self.hessian_weights = BetaWeights(x=betx_weights, y=bety_weights)

        # Normalize weights for gradient computation
        self.weights = BetaWeights(
            x=WeightProcessor.normalize_weights(betx_weights),
            y=WeightProcessor.normalize_weights(bety_weights),
        )

        # Store initial conditions
        self.init_coords = data.init_coords

        # Load MAD-NG optics tracking scripts
        self.run_track_script = TRACK_OPTICS_SCRIPT.read_text()
        self.run_track_init_path = TRACK_OPTICS_INIT

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
        """Configure MAD-NG sequence for optics computation.

        Sets tracking range for optics propagation.

        Args:
            mad: MAD-NG interface object
        """
        mad["tracking_range"] = self.get_bpm_range(self.config.sdir)

    def _setup_da_maps(self, mad: MAD) -> None:
        """Setup differential algebra maps for optics computation.

        Creates base DAMAP with second-order expansion (required for beta
        function computation) and adds knob parameters.

        Args:
            mad: MAD-NG interface object
        """
        # Load additional modules for optics (gphys, monomial)
        mad.load("MAD", "gphys", "monomial")

        # Create base DAMAP with order 2 (needed for beta functions)
        self.create_base_damap(mad, knob_order=2)

        # Add knobs as TPSA variables
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    MADX[param] = MADX[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

    def send_initial_conditions(self, mad: MAD) -> None:
        """Send initial Twiss parameters to MAD-NG.

        Converts initial Twiss parameters (beta, alpha, dispersion) into
        a DAMAP representation using MAD-NG's beta0 and bet2map functions.

        Args:
            mad: MAD-NG interface object
        """
        mad.send("""
init_coords = python:recv()
local B0 = MAD.beta0 {
    beta11=init_coords['beta11'],
    beta22=init_coords['beta22'],
    alfa11=init_coords['alfa11'],
    alfa22=init_coords['alfa22'],
    dx=init_coords['dx'],
    dpx=init_coords['dpx'],
    dy=init_coords['dy'],
    dpy=init_coords['dpy'],
    sdir=sdir,
    rank=4,
}
da_x0_c = gphys.bet2map(B0, da_x0_base:copy())
""")
        mad.send(self.init_coords)

        # Verify initial conditions were set correctly
        if not mad.send("python:send(true)").recv():
            raise RuntimeError(f"Worker {self.worker_id}: Failed to send initial conditions to MAD")

    def _initialize_mad_computation(self, mad: MAD) -> None:
        """Initialize MAD-NG environment for optics computations.

        Args:
            mad: MAD-NG interface object
        """
        mad.send(self.run_track_init_path.read_text())

        # Verify initialization
        if not mad.send("python:send(true)").recv():
            raise RuntimeError(f"Worker {self.worker_id}: Failed to initialize MAD computation")

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float]:
        """Compute gradients and loss for optics functions.

        Performs optics propagation, receives beta function data and
        derivatives, and computes loss and gradients using weighted
        least-squares.

        Args:
            mad: MAD-NG interface object
            knob_updates: Dictionary of knob names to values
            batch: Batch index (unused for optics, kept for interface consistency)

        Returns:
            Tuple of (gradient array, loss value)
        """
        # Send knob updates to MAD-NG
        update_commands = [f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]
        mad.send("\n".join(update_commands))

        # Run optics computation
        mad.send(self.run_track_script)

        # Receive results symmetrically
        results = self._receive_optics_results(mad)

        # Compute loss and gradients
        return self._compute_loss_and_gradients(results)

    def _receive_optics_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive optics computation results from MAD-NG.

        Args:
            mad: MAD-NG interface object

        Returns:
            Dictionary with keys: betx, bety, dbetx_dk, dbety_dk
        """
        # Receive beta functions
        betx_results = mad.recv()
        bety_results = mad.recv()

        # Receive derivatives
        dbetx_dk_results = mad.recv()
        dbety_dk_results = mad.recv()

        # Convert to numpy arrays
        # Shape: (n_bpms,)
        betx = np.asarray(betx_results).squeeze(-1)
        bety = np.asarray(bety_results).squeeze(-1)

        # Shape: (n_bpms, n_knobs) - note: MAD sends as (n_knobs, n_bpms) transposed
        dbetx_dk = np.asarray(dbetx_dk_results)
        dbety_dk = np.asarray(dbety_dk_results)

        return {
            "betx": betx,
            "bety": bety,
            "dbetx_dk": dbetx_dk,
            "dbety_dk": dbety_dk,
        }

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float]:
        """Compute weighted loss and gradients from optics results.

        Uses symmetric treatment of beta_x and beta_y.

        Args:
            results: Dictionary of optics results and derivatives

        Returns:
            Tuple of (gradient array, loss value)
        """
        # Compute residuals symmetrically
        residual_x = results["betx"] - self.comparisons.x
        residual_y = results["bety"] - self.comparisons.y

        # Compute gradients (weighted residual @ jacobian)
        # Shape: (n_bpms,) @ (n_bpms, n_knobs) -> (n_knobs,)
        gx = (self.weights.x * residual_x) @ results["dbetx_dk"]
        gy = (self.weights.y * residual_y) @ results["dbety_dk"]

        # Compute loss (weighted sum of squared residuals)
        loss_x = np.sum(self.weights.x * residual_x**2)
        loss_y = np.sum(self.weights.y * residual_y**2)

        # Total gradient and loss (factor of 2 from derivative of squared residuals)
        grad = 2.0 * (gx + gy)
        loss = loss_x + loss_y

        return grad, loss

    @staticmethod
    def get_n_data_points(nbpms: int) -> int:
        """Get number of data points for optics computation.

        Args:
            nbpms: Number of BPMs in the range

        Returns:
            Number of data points (equal to number of BPMs)
        """
        return nbpms
