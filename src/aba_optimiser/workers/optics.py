"""Optics function worker for phase optimisation.

This module implements the OpticsWorker class which computes phase advances
between consecutive BPMs and their gradients for optics matching. Phase advances
are measured between pairs of adjacent BPMs, resulting in (n_bpms - 1) measurements.
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
class OpticsWeights:
    """Container for optics function weights.

    Attributes:
        phase_adv_x: Weights for horizontal phase advances
        phase_adv_y: Weights for vertical phase advances
        beta_x: Weights for horizontal beta functions
        beta_y: Weights for vertical beta functions
    """

    phase_adv_x: np.ndarray  # Shape: (n_bpms-1,)
    phase_adv_y: np.ndarray  # Shape: (n_bpms-1,)
    beta_x: np.ndarray  # Shape: (n_bpms,)
    beta_y: np.ndarray  # Shape: (n_bpms,)


@dataclass
class OpticsComparisons:
    """Container for reference optics data.

    Attributes:
        phase_adv_x: Reference horizontal phase advances between consecutive BPMs
        phase_adv_y: Reference vertical phase advances between consecutive BPMs
        beta_x: Reference horizontal beta functions at each BPM
        beta_y: Reference vertical beta functions at each BPM
    """

    phase_adv_x: np.ndarray  # Shape: (n_bpms-1,)
    phase_adv_y: np.ndarray  # Shape: (n_bpms-1,)
    beta_x: np.ndarray  # Shape: (n_bpms,)
    beta_y: np.ndarray  # Shape: (n_bpms,)


class OpticsWorker(AbstractWorker[OpticsData]):
    """Worker for optics function computations.

    This worker computes phase advances between consecutive BPMs and
    beta functions at each BPM, along with their gradients with respect
    to optimisation knobs. It uses differential algebra for automatic
    differentiation.

    Since phase advances are measured between BPM pairs, there are
    (n_bpms - 1) measurement points for n_bpms BPMs. Beta functions
    are measured at each BPM, giving n_bpms measurements.
    """

    def prepare_data(self, data: OpticsData) -> None:
        """Process and prepare optics data for computation.

        Extracts phase advance measurements between consecutive BPMs,
        beta function measurements at each BPM, computes weights from
        variances, and prepares initial Twiss parameters.

        Args:
            data: OpticsData container with reference measurements
        """
        LOGGER.debug(
            f"Worker {self.worker_id}: Processing optics data with {len(data.comparisons)}"
            f" phase advance measurements and {len(data.beta_comparisons)} beta function measurements"
        )

        # Extract phase advance comparisons
        self.comparisons = OpticsComparisons(
            phase_adv_x=data.comparisons[:, 0],  # phase_adv_x
            phase_adv_y=data.comparisons[:, 1],  # phase_adv_y
            beta_x=data.beta_comparisons[:, 0],  # beta_x
            beta_y=data.beta_comparisons[:, 1],  # beta_y
        )

        # Compute weights from variances
        phase_adv_x_vars = data.variances[:, 0]
        phase_adv_y_vars = data.variances[:, 1]
        beta_x_vars = data.beta_variances[:, 0]
        beta_y_vars = data.beta_variances[:, 1]

        phase_adv_x_weights = WeightProcessor.variance_to_weight(phase_adv_x_vars)
        phase_adv_y_weights = WeightProcessor.variance_to_weight(phase_adv_y_vars)
        beta_x_weights = WeightProcessor.variance_to_weight(beta_x_vars)
        beta_y_weights = WeightProcessor.variance_to_weight(beta_y_vars)

        # Store Hessian weights (used for approximate Hessian calculation)
        self.hessian_weights = OpticsWeights(
            phase_adv_x=phase_adv_x_weights,
            phase_adv_y=phase_adv_y_weights,
            beta_x=beta_x_weights,
            beta_y=beta_y_weights,
        )

        # Normalize weights for gradient computation (normalize globally across all measurements)
        norm_phase_adv_x_weight, norm_phase_adv_y_weight, norm_beta_x_weight, norm_beta_y_weight = (
            WeightProcessor.normalise_weights_globally(
                phase_adv_x_weights, phase_adv_y_weights, beta_x_weights, beta_y_weights
            )
        )
        self.weights = OpticsWeights(
            phase_adv_x=norm_phase_adv_x_weight,
            phase_adv_y=norm_phase_adv_y_weight,
            beta_x=norm_beta_x_weight
            * 0.1,  # Scale beta weights by 0.1 to prioritise phase advances
            beta_y=norm_beta_y_weight
            * 0.1,  # Scale beta weights by 0.1 to prioritise phase advances
        )

        # Store initial conditions
        self.init_coords = data.init_coords

        # Load MAD-NG optics tracking scripts
        self.run_track_script = TRACK_OPTICS_SCRIPT.read_text()
        self.run_track_init_path = TRACK_OPTICS_INIT

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for optics computation.

        Sets tracking range for optics propagation.

        Args:
            mad: MAD-NG interface object
        """
        mad["tracking_range"] = self.tracking_range

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
        text = self.run_track_init_path.read_text()
        # Remove all comments from the initialization script and embed worker ID
        filtered_lines = [
            line for line in text.splitlines() if not line.strip().startswith("--") and line.strip()
        ]
        filtered_lines[0] += "!Worker ID: " + str(self.worker_id)
        mad.send("\n".join(filtered_lines))

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
            Dictionary with keys: betx, bety, phase_adv_x, phase_adv_y,
            dbeta_x_dk, dbeta_y_dk,
            dphase_adv_x_dk, dphase_adv_y_dk
        """
        # Receive beta functions at each BPM
        betx_results = mad.recv()
        bety_results = mad.recv()

        # Receive phase advance functions
        phase_adv_x_results = mad.recv()
        phase_adv_y_results = mad.recv()

        # Receive beta derivatives
        dbeta_x_dk_results = mad.recv()
        dbeta_y_dk_results = mad.recv()

        # Receive phase advance derivatives
        dphase_adv_x_dk_results = mad.recv()
        dphase_adv_y_dk_results = mad.recv()

        # Convert to numpy arrays
        # Shape: (n_bpms,)
        betx = np.asarray(betx_results).squeeze(-1)
        bety = np.asarray(bety_results).squeeze(-1)

        # Shape: (n_bpms-1,)
        phase_adv_x = np.asarray(phase_adv_x_results).squeeze(-1)
        phase_adv_y = np.asarray(phase_adv_y_results).squeeze(-1)

        # Shape: (n_bpms, n_knobs) - note: MAD sends as (n_knobs, n_bpms) transposed
        dbeta_x_dk = np.asarray(dbeta_x_dk_results)
        dbeta_y_dk = np.asarray(dbeta_y_dk_results)

        # Shape: (n_bpms-1, n_knobs) - note: MAD sends as (n_knobs, n_bpms-1) transposed
        dphase_adv_x_dk = np.asarray(dphase_adv_x_dk_results)
        dphase_adv_y_dk = np.asarray(dphase_adv_y_dk_results)

        return {
            "betx": betx,
            "bety": bety,
            "phase_adv_x": phase_adv_x,
            "phase_adv_y": phase_adv_y,
            "dbeta_x_dk": dbeta_x_dk,
            "dbeta_y_dk": dbeta_y_dk,
            "dphase_adv_x_dk": dphase_adv_x_dk,
            "dphase_adv_y_dk": dphase_adv_y_dk,
        }

    def _compute_loss_and_gradients(
        self, results: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float]:
        """Compute weighted loss and gradients from optics results.

        Uses symmetric treatment of phase_adv_x, phase_adv_y, beta_x, beta_y.
        Computes cyclical residuals for phase advances (modulo 2π).

        Args:
            results: Dictionary of optics results and derivatives

        Returns:
            Tuple of (gradient array, loss value)
        """
        # Compute cyclical residuals for phase advances (phase advances are modulo 1)
        diff_phase_adv_x = results["phase_adv_x"] - self.comparisons.phase_adv_x
        diff_phase_adv_y = results["phase_adv_y"] - self.comparisons.phase_adv_y

        # Wrap differences to [-0.5, 0.5] range for cyclical phase advances (modulo 1)
        residual_phase_adv_x = (diff_phase_adv_x + 0.5) % 1 - 0.5
        residual_phase_adv_y = (diff_phase_adv_y + 0.5) % 1 - 0.5

        # Compute beta function residuals (non-cyclical)
        residual_beta_x = results["betx"] - self.comparisons.beta_x
        residual_beta_y = results["bety"] - self.comparisons.beta_y

        # Compute gradients (weighted residual @ jacobian)
        # Phase advance contributions: Shape: (n_bpms-1,) @ (n_bpms-1, n_knobs) -> (n_knobs,)
        gpx = (self.weights.phase_adv_x * residual_phase_adv_x) @ results["dphase_adv_x_dk"]
        gpy = (self.weights.phase_adv_y * residual_phase_adv_y) @ results["dphase_adv_y_dk"]

        # Beta function contributions: Shape: (n_bpms,) @ (n_bpms, n_knobs) -> (n_knobs,)
        gbx = (self.weights.beta_x * residual_beta_x) @ results["dbeta_x_dk"]
        gby = (self.weights.beta_y * residual_beta_y) @ results["dbeta_y_dk"]

        # Compute loss (weighted sum of squared residuals)
        loss_px = np.sum(self.weights.phase_adv_x * residual_phase_adv_x**2)
        loss_py = np.sum(self.weights.phase_adv_y * residual_phase_adv_y**2)
        loss_bx = np.sum(self.weights.beta_x * residual_beta_x**2)
        loss_by = np.sum(self.weights.beta_y * residual_beta_y**2)

        # Total gradient and loss (factor of 2 from derivative of squared residuals)
        grad = 2.0 * (gpx + gpy + gbx + gby)
        loss = loss_px + loss_py + loss_bx + loss_by

        return grad, loss

    @staticmethod
    def get_n_data_points(nbpms: int) -> int:
        """Get number of data points for optics computation.

        Args:
            nbpms: Number of BPMs in the range

        Returns:
            Number of phase advance measurements (nbpms - 1, between consecutive BPMs)
        """
        return nbpms - 1
