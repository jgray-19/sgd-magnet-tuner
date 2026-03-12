"""Particle tracking worker for multi-turn beam dynamics simulations.

This module implements the TrackingWorker class which performs particle
tracking simulations and computes gradients for optimisation. It handles
both position and momentum observables with full symmetry between x/y planes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad.scripts import (
    build_tracking_hessian_script,
    build_tracking_init_script,
    build_tracking_script,
    dump_debug_script,
)
from aba_optimiser.workers.abstract_worker import AbstractWorker
from aba_optimiser.workers.common import (
    PrecomputedTrackingWeights,
    TrackingData,
    split_array_to_batches,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from pymadng import MAD

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.workers.common import WorkerConfig

LOGGER = logging.getLogger(__name__)

OBSERVABLE_SPECS: dict[str, tuple[str, int]] = {
    "x": ("position_comparisons", 0),
    "y": ("position_comparisons", 1),
    "px": ("momentum_comparisons", 0),
    "py": ("momentum_comparisons", 1),
}


class TrackingWorker(AbstractWorker[TrackingData]):
    """Worker for particle tracking simulations.

    This worker performs particle tracking through accelerator lattices,
    computing positions and momenta at each BPM. It calculates gradients
    of the loss function with respect to optimisation knobs using
    differential algebra techniques.

    Supports two modes:
    - 'multi-turn': Track particles for multiple turns (default)
    - 'arc-by-arc': Single-turn tracking through one arc/section

    The implementation treats x/y and position/momentum symmetrically,
    ensuring consistent handling of all phase space dimensions.
    """

    observables: tuple[str, ...] = ("x", "y", "px", "py")
    include_momentum = True
    hessian_weight_order: tuple[str, ...] = ("x", "y", "px", "py")

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

        Extracts the active observables, loads precomputed weights, splits
        data into batches, and prepares initial conditions.

        Args:
            data: TrackingData container with reference measurements
        """
        self.observables = self._resolve_observables()
        self.hessian_weight_order = self.observables
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

        self.comparison_arrays = self._extract_observable_arrays(data, n_init)
        if data.precomputed_weights is None:
            raise ValueError("Precomputed weights must be provided for TrackingWorker")
        self.weight_arrays, self.hessian_weights = self._load_precomputed_weights(
            data.precomputed_weights,
            n_init,
        )
        self._prepare_batches(init_coords, data.init_pts, num_batches)

        self.worker_disabled = False
        n_points = self.comparisons[self.observables[0]][0].shape[1]
        self.keep_bpm_mask = np.ones(n_points, dtype=bool)

        self.run_track_init_text = build_tracking_init_script(self.observables)
        self.run_track_script = build_tracking_script(self.observables)
        self.hessian_script_text = build_tracking_hessian_script(self.observables)
        self._dump_debug_scripts()

    def _dump_debug_scripts(self) -> None:
        """Write generated MAD scripts to disk when debugging is enabled."""
        dump_debug_script(
            "run_track_init",
            self.run_track_init_text,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )
        dump_debug_script(
            "run_track",
            self.run_track_script,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )
        dump_debug_script(
            "estimate_hessian",
            self.hessian_script_text,
            debug=self.config.debug,
            mad_logfile=self.config.mad_logfile,
            worker_id=self.worker_id,
        )

    def _resolve_observables(self) -> tuple[str, ...]:
        """Return the observables active for this worker configuration."""
        kick_plane = self.config.kick_plane
        if kick_plane == "xy":
            return ("x", "y", "px", "py") if self.include_momentum else ("x", "y")
        if kick_plane == "x":
            return ("x", "px") if self.include_momentum else ("x",)
        if kick_plane == "y":
            return ("y", "py") if self.include_momentum else ("y",)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    def _extract_observable_arrays(self, data: TrackingData, n_init: int) -> dict[str, np.ndarray]:
        """Return comparison arrays for the observables used by this worker."""
        arrays: dict[str, np.ndarray] = {}
        for observable in self.observables:
            source_attr, plane_idx = OBSERVABLE_SPECS[observable]
            source = getattr(data, source_attr)[:n_init]
            arrays[observable] = source[:, :, plane_idx]
        return arrays

    def _load_precomputed_weights(
        self,
        weights: PrecomputedTrackingWeights,
        n_init: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return per-particle and Hessian weights for the active observables."""
        batch_weights = {
            observable: getattr(weights, observable)[:n_init] for observable in self.observables
        }
        hessian_weights = {
            observable: getattr(weights, f"hessian_{observable}") for observable in self.observables
        }
        return batch_weights, hessian_weights

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

        self.comparisons = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in self.comparison_arrays.items()
        }
        self.weights = {
            observable: split_array_to_batches(values, num_batches)
            for observable, values in self.weight_arrays.items()
        }

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Configure MAD-NG sequence for tracking.

        Sets batch size, number of batches, and optimisation flags.
        For arc-by-arc mode, also sets single-turn tracking.

        Args:
            mad: MAD-NG interface object
        """
        mad["batch_size"] = self.batch_size
        mad["num_batches"] = self.num_batches
        mad["optimise_energy"] = self.config.accelerator.optimise_energy

        mad["tracking_range"] = self.tracking_range
        if self.mode == "arc-by-arc":
            # Single-turn tracking for arc-by-arc mode
            mad["n_run_turns"] = 1
        else:
            # Multi-turn tracking over the full ring, each worker gets its own init BPM.
            mad["n_run_turns"] = self.simulation_config.n_run_turns

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

    def _initialise_mad_computation(self, mad: MAD) -> None:
        """Initialise MAD-NG environment for tracking computations.

        Args:
            mad: MAD-NG interface object
        """
        mad.send(self.run_track_init_text)

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
        results = self._run_tracking_batch(mad, knob_updates, batch)

        # Compute loss and gradients
        return self._compute_loss_and_gradients(results, batch)

    def _receive_tracking_results(self, mad: MAD) -> dict[str, np.ndarray]:
        """Receive tracking results from MAD-NG.

        Args:
            mad: MAD-NG interface object

        Returns:
            Dictionary with one result array and one derivative array per active
            observable.
        """
        results: dict[str, np.ndarray] = {}
        for observable in self.observables:
            results[observable] = np.asarray(mad.recv()).squeeze(-1)
        for observable in self.observables:
            results[self._gradient_key(observable)] = np.stack(mad.recv(), axis=0)
        return results

    def _run_tracking_batch(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> dict[str, np.ndarray]:
        """Run MAD-NG tracking for a single batch and return all outputs."""
        machine_pt = knob_updates.get("pt", 0.0)

        update_commands = [
            f"MADX['{name}']:set0({val:.15e})" for name, val in knob_updates.items() if name != "pt"
        ]
        if update_commands:
            mad.send("\n".join(update_commands))

        mad.send(f"batch = {batch + 1}")
        mad.send(f"""
for i = 1, batch_size do
    da_x0_c[batch][i].pt:set0({machine_pt:.15e} + init_pts[batch][i])
end
""")
        mad.send(self.run_track_script)
        return self._receive_tracking_results(mad)

    @staticmethod
    def _gradient_key(observable: str) -> str:
        """Return the gradient result key for an observable."""
        return f"d{observable}_dk"

    def _masked_weights(self, batch: int) -> dict[str, np.ndarray]:
        """Return batch weights with runtime BPM masking applied."""
        bpm_mask = self.keep_bpm_mask.reshape(1, -1)
        return {
            observable: self.weights[observable][batch] * bpm_mask
            for observable in self.observables
        }

    def _residuals(self, results: dict[str, np.ndarray], batch: int) -> dict[str, np.ndarray]:
        """Return residual arrays for the current batch."""
        return {
            observable: results[observable] - self.comparisons[observable][batch]
            for observable in self.observables
        }

    def _compute_loss_and_bpm_contributions(
        self, results: dict[str, np.ndarray], batch: int
    ) -> tuple[float, np.ndarray]:
        """Compute total loss and per-BPM contributions for one batch."""
        weights = self._masked_weights(batch)
        residuals = self._residuals(results, batch)
        loss_bpm = np.zeros(self.keep_bpm_mask.size, dtype=np.float64)
        for observable in self.observables:
            loss_bpm += np.sum(weights[observable] * residuals[observable] ** 2, axis=0)
        return float(np.sum(loss_bpm)), loss_bpm

    def compute_diagnostics(
        self, mad: MAD, knob_updates: dict[str, float]
    ) -> tuple[float, np.ndarray]:
        """Compute total and per-BPM losses across all batches at current knobs."""
        total_loss = 0.0
        loss_per_bpm = np.zeros_like(self.keep_bpm_mask, dtype=np.float64)

        for batch in range(self.num_batches):
            results = self._run_tracking_batch(mad, knob_updates, batch)
            batch_loss, batch_loss_bpm = self._compute_loss_and_bpm_contributions(results, batch)
            total_loss += batch_loss
            loss_per_bpm += batch_loss_bpm

        return total_loss, loss_per_bpm

    def _apply_runtime_mask(self, keep_bpm_mask: np.ndarray) -> None:
        """Apply BPM keep-mask for subsequent optimisation and Hessian steps."""
        if keep_bpm_mask.ndim != 1:
            raise ValueError("keep_bpm_mask must be a 1D array")
        if keep_bpm_mask.size != self.keep_bpm_mask.size:
            raise ValueError(
                f"Mask size mismatch for worker {self.worker_id}: "
                f"expected {self.keep_bpm_mask.size}, got {keep_bpm_mask.size}"
            )
        self.keep_bpm_mask = keep_bpm_mask.astype(bool, copy=True)

    def _handle_control_command(self, mad: MAD, command: dict[str, object], nbpms: int) -> None:
        """Handle control-plane commands from parent process."""
        cmd = command.get("cmd")
        if cmd == "diagnostics":
            raw_knobs = command.get("knobs", {})
            if not isinstance(raw_knobs, dict):
                raise ValueError(
                    f"Worker {self.worker_id}: diagnostics command missing knob dictionary"
                )
            diagnostic_knobs: dict[str, float] = {}
            for knob_name, knob_value in raw_knobs.items():
                if not isinstance(knob_name, str):
                    raise ValueError(
                        f"Worker {self.worker_id}: knob name {knob_name!r} is not a string"
                    )
                if not isinstance(knob_value, int | float | np.floating):
                    raise ValueError(
                        f"Worker {self.worker_id}: knob {knob_name!r} has non-numeric value {knob_value!r}"
                    )
                diagnostic_knobs[knob_name] = float(knob_value)
            total_loss, loss_per_bpm = self.compute_diagnostics(mad, diagnostic_knobs)
            self.conn.send(
                {
                    "worker_id": self.worker_id,
                    "total_loss": total_loss / nbpms,
                    "loss_per_bpm": (loss_per_bpm / nbpms).tolist(),
                }
            )
            return

        if cmd == "apply_mask":
            keep_bpm_mask = np.asarray(command.get("keep_bpm_mask", []), dtype=bool)
            disable_worker = bool(command.get("disable_worker", False))
            if keep_bpm_mask.size:
                self._apply_runtime_mask(keep_bpm_mask)
            self.worker_disabled = disable_worker
            self.conn.send({"worker_id": self.worker_id, "status": "ok"})
            return

        raise ValueError(f"Worker {self.worker_id}: Unknown command {cmd}")

    def _send_hessian_weights(self, mad: MAD) -> None:
        """Send Hessian weights to MAD for the active observables."""
        hmask = self.keep_bpm_mask.astype(np.float64)
        bindings = "\n".join(
            f"weights_{observable} = python:recv()" for observable in self.hessian_weight_order
        )
        mad.send(f"{bindings}\n")
        for observable in self.hessian_weight_order:
            mad.send((self.hessian_weights[observable] * hmask).tolist())

    def _get_hessian_script(self) -> str:
        """Get the MAD script used for Hessian approximation."""
        return self.hessian_script_text

    def _compute_hessian_part(self, mad: MAD, n_knobs: int) -> np.ndarray:
        """Compute this worker's Hessian contribution."""
        self._send_hessian_weights(mad)
        mad.send(self._get_hessian_script())
        return np.asarray(mad.recv())

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
        weights = self._masked_weights(batch)
        residuals = self._residuals(results, batch)
        gradient_shape = results[self._gradient_key(self.observables[0])].shape[1]
        grad = np.zeros(gradient_shape, dtype=np.float64)
        for observable in self.observables:
            grad += np.einsum(
                "pkm,pm->k",
                results[self._gradient_key(observable)],
                weights[observable] * residuals[observable],
            )
        loss, _ = self._compute_loss_and_bpm_contributions(results, batch)
        return 2.0 * grad, loss

    def run(self) -> None:
        """Main worker run loop with Hessian calculation.

        Extends the base run method to compute approximate Hessian
        after the main optimisation loop completes.
        """
        mad: MAD | None = None
        n_knobs = 0
        computation_success = True

        try:
            knob_values, batch = self.conn.recv()
            if knob_values is None:
                return
            n_knobs = len(knob_values)

            mad, nbpms = self.setup_mad_interface(knob_values)
            self.send_initial_conditions(mad)
            self._initialise_mad_computation(mad)

            LOGGER.debug(f"Worker {self.worker_id}: Ready for computation with {nbpms} BPMs")

            message: tuple[dict[str, float] | None, int | None] | dict[str, object] = (
                self.conn.recv()
            )
            while isinstance(message, dict):
                self._handle_control_command(mad, message, nbpms)
                message = self.conn.recv()

            while True:
                knob_values, batch = message
                if knob_values is None or batch is None:
                    LOGGER.debug(f"Worker {self.worker_id}: Received termination signal")
                    break
                try:
                    if self.worker_disabled:
                        self.conn.send((self.worker_id, np.zeros(n_knobs), 0.0))
                    else:
                        grad, loss = self.compute_gradients_and_loss(mad, knob_values, int(batch))
                        self.conn.send((self.worker_id, grad / nbpms, loss / nbpms))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="computation")
                    computation_success = False
                    break

                message = self.conn.recv()

            if computation_success and not self.worker_disabled:
                LOGGER.debug(f"Worker {self.worker_id}: Computing Hessian approximation")
                try:
                    self.conn.send(self._compute_hessian_part(mad, n_knobs))
                except Exception as exc:  # noqa: BLE001
                    self.send_error_payload(exc, phase="hessian")
                    computation_success = False
            else:
                self.conn.send(np.zeros((n_knobs, n_knobs)))
        except Exception as exc:  # noqa: BLE001
            self.send_error_payload(exc, phase="startup")
        finally:
            LOGGER.debug(f"Worker {self.worker_id}: Terminating")
            if mad is not None:
                mad.send("shush()")
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
