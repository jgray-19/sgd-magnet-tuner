from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from multiprocessing import Process
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.mad.scripts import HESSIAN_SCRIPT, TRACK_OPTICS_INIT, TRACK_OPTICS_SCRIPT

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkerData:
    """Container for worker tracking data arrays."""

    beta_comparisons: np.ndarray  # Shape: (bpms, 2) - [x, y]
    beta_variances: np.ndarray  # Shape: (bpms, 2) - [x, y]
    init_coords: dict[str, np.ndarray]  # betx, bety, alfx, alfy, dx, dpx, dy, dpy


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
    simulation_config: SimulationConfig
    sdir: int = 1
    bad_bpms: list[str] | None = None
    seq_name: str | None = None


class OpticsWorker(Process, ABC):
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

        # Shape: (n_tracks, n_data_points)
        self.betx_comparisons = data.beta_comparisons[:, 0]  # x positions
        self.bety_comparisons = data.beta_comparisons[:, 1]  # y positions
        self.betx_variances = data.beta_variances[:, 0]
        self.bety_variances = data.beta_variances[:, 1]

        betx_weights = self._variance_to_weight(self.betx_variances)
        bety_weights = self._variance_to_weight(self.bety_variances)

        self.hessian_weight_x = betx_weights
        self.hessian_weight_y = bety_weights

        self.betx_weights = self._normalise_weights(betx_weights)
        self.bety_weights = self._normalise_weights(bety_weights)

        # Check if there are nans in the initial coordinates
        # if any(np.isnan(val) for val in data.init_coords.values()):
        #     raise ValueError("NaNs found in initial coordinates")

        # Convert init_coords from list of matrices into list of list of lists
        self.init_coords = data.init_coords
        self.config = config

        # Load common MAD scripts
        self.run_track_script = TRACK_OPTICS_SCRIPT.read_text()
        self.run_track_init_path = TRACK_OPTICS_INIT

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

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

    def send_initial_conditions(self, mad: MAD) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
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
    !x=init_coords['x'],
    !y=init_coords['y'],
    sdir=sdir,
    rank=4,
}
da_x0_c = gphys.bet2map(B0, da_x0_base:copy())
""")
        mad.send(self.init_coords)

        if not mad.send("python:send(true)").recv():
            raise RuntimeError(f"Worker {self.worker_id}: Failed to send initial conditions to MAD")

    def get_bpm_range(self, sdir: int) -> str:
        """Get the magnet range for arc-by-arc mode."""
        if sdir == -1:
            return self.config.end_bpm + "/" + self.config.start_bpm
        return self.config.start_bpm + "/" + self.config.end_bpm

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
            bad_bpms=self.config.bad_bpms,
            corrector_strengths=self.config.corrector_strengths,
            simulation_config=self.config.simulation_config,
            tune_knobs_file=self.config.tune_knobs_file,
            beam_energy=self.config.beam_energy,
        )
        knob_names = mad_iface.knob_names
        if knob_names != list(init_knobs.keys()):
            raise ValueError(
                f"Worker {self.worker_id}: Knob names from MAD {knob_names} do not match initial knobs {list(init_knobs.keys())}"
            )

        mad = mad_iface.mad
        mad["knob_names"] = knob_names
        mad["nbpms"] = mad_iface.nbpms
        mad["sdir"] = self.config.sdir

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix", "vector", "gphys", "monomial")
        mad.load("MAD.utility", "tblcat")

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad, 2)
        mad.send("""
knob_monomials = {}
for i,param in ipairs(knob_names) do
    MADX[param] = MADX[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")
        mad["tracking_range"] = self.get_bpm_range(self.config.sdir)

        if not mad.send("python:send(true)").recv():
            raise RuntimeError(f"Worker {self.worker_id}: Failed to setup MAD interface")
        return mad, mad_iface.nbpms

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], _: int
    ) -> tuple[np.ndarray, float]:
        # Prepare MAD commands to update knob values in the sequence
        update_commands = [f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]

        # Send updates, batch index, and energy deviation to MAD
        mad.send("\n".join(update_commands))
        mad.send(self.run_track_script)

        # Receive simulation results from MAD: positions and derivatives
        betx_results = mad.recv()  # Simulated x-positions: array of points
        bety_results = mad.recv()  # Simulated y-positions: array of points

        dbetx_dk_results = mad.recv()  # Derivatives of x w.r.t. knobs: matrix
        dbety_dk_results = mad.recv()  # Derivatives of y w.r.t. knobs: matrix

        # Convert results to NumPy arrays for efficient vectorised processing
        # Squeeze to remove singleton dimensions (e.g., from MAD's output format)
        # Shape: (n_data_points,)
        betx = np.asarray(betx_results).squeeze(-1)
        bety = np.asarray(bety_results).squeeze(-1)

        residual_x = betx - self.betx_comparisons
        residual_y = bety - self.bety_comparisons

        gx = (self.betx_weights * residual_x) @ dbetx_dk_results
        gy = (self.bety_weights * residual_y) @ dbety_dk_results
        loss = np.sum(self.betx_weights * residual_x**2) + np.sum(self.bety_weights * residual_y**2)

        grad = 2.0 * (gx + gy)
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
        # Still to be implemented
        LOGGER.debug(f"Worker {self.worker_id}: Terminating MAD interface")
        del mad
