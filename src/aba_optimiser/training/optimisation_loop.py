"""Optimisation loop management for the controller."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser
from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
from aba_optimiser.training.scheduler import LRScheduler

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from tensorboardX import SummaryWriter

    from aba_optimiser.config import OptimiserConfig, SimulationConfig

LOGGER = logging.getLogger(__name__)


class OptimisationLoop:
    """Manages the optimisation loop and statistics tracking."""

    def __init__(
        self,
        initial_strengths: np.ndarray,
        knob_names: list[str],
        true_strengths: dict[str, float],
        optimiser_config: OptimiserConfig,
        simulation_config: SimulationConfig,
        optimiser_type: str | None = None,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.use_true_strengths = len(true_strengths) > 0
        self.smoothed_grad_norm: float = 0.0
        self.grad_norm_alpha = optimiser_config.grad_norm_alpha
        self.expected_rel_error = optimiser_config.expected_rel_error
        self.max_clipping_ratio: float = 0.0  # Track max clipping ratio per epoch

        self.max_epochs = optimiser_config.max_epochs
        self.gradient_converged_value = optimiser_config.gradient_converged_value

        # Initialise optimiser
        opt_type = optimiser_type if optimiser_type is not None else optimiser_config.optimiser_type
        self._init_optimiser(initial_strengths.shape, opt_type)

        # Initialise scheduler
        self.scheduler = LRScheduler(
            warmup_epochs=optimiser_config.warmup_epochs,
            decay_epochs=optimiser_config.decay_epochs,
            start_lr=optimiser_config.warmup_lr_start,
            max_lr=optimiser_config.max_lr,
            min_lr=optimiser_config.min_lr,
        )

        self.some_magnets = (
            simulation_config.optimise_quadrupoles or simulation_config.optimise_bends
        )
        self.num_batches = simulation_config.num_batches

        # Convert total expected relative error into a per-step trust-region bound
        self.trust_region_safety = 3.0  # allow errors a few times worse than typical
        self.param_floor = 1e-6  # same floor used in trust region

        self.total_steps = max(1, self.max_epochs * self.num_batches)
        if self.expected_rel_error > 0:
            self.rel_sigma_step = (
                self.trust_region_safety * self.expected_rel_error / np.sqrt(self.total_steps)
            )
            LOGGER.info(
                "Per-parameter trust region enabled: "
                f"rel_sigma_total={self.expected_rel_error:.3e}, "
                f"rel_sigma_step={self.rel_sigma_step:.3e}, "
                f"steps={self.total_steps}, safety={self.trust_region_safety:g}"
            )
        else:
            LOGGER.info("Per-parameter trust region disabled")
            self.rel_sigma_step = 0.0

    def _init_optimiser(self, shape: tuple[int, ...], optimiser_type: str) -> None:
        """Initialise the optimiser based on type."""
        optimiser_kwargs = {
            "shape": shape,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
        }
        if optimiser_type == "amsgrad":
            self.optimiser = AMSGradOptimiser(**optimiser_kwargs)  # ty:ignore[invalid-argument-type]
        elif optimiser_type == "adam":
            self.optimiser = AdamOptimiser(**optimiser_kwargs)  # ty:ignore[invalid-argument-type]
        elif optimiser_type == "lbfgs":
            self.optimiser = LBFGSOptimiser(
                history_size=20,
                eps=1e-12,
                weight_decay=optimiser_kwargs["weight_decay"],  # ty:ignore[invalid-argument-type]
            )
        else:
            raise ValueError(f"Unknown optimiser type: {optimiser_type}")
        LOGGER.info(f"Using optimiser: {self.optimiser.__class__.__name__}")

    def run_optimisation(
        self,
        current_knobs: dict[str, float],
        parent_conns: list[Connection],
        writer: SummaryWriter,
        run_start: float,
        total_turns: int,
    ) -> dict[str, float]:
        """Run the main optimisation loop."""
        if "pt" in current_knobs and current_knobs["pt"] == 0.0:
            current_knobs["pt"] = 1e-6  # Initialise pt to non-zero

        prev_loss = None

        for epoch in range(self.max_epochs):
            self.max_clipping_ratio = 0.0  # Reset once per epoch
            epoch_start = time.time()

            epoch_loss = 0.0
            epoch_grad = np.zeros(len(self.knob_names))
            lr = self.scheduler(epoch)
            prev_knobs = current_knobs.copy()

            for batch in range(self.num_batches):
                # Send knobs to workers
                for conn in parent_conns:
                    conn.send((current_knobs, batch))

                batch_loss, batch_grad = self._collect_batch_results(parent_conns)
                epoch_loss += batch_loss
                epoch_grad += batch_grad

                # Update knobs after each batch
                current_knobs = self._update_knobs(current_knobs, batch_grad, lr)

            # Average over batches for logging
            epoch_loss /= total_turns
            epoch_grad /= total_turns

            grad_norm = float(np.linalg.norm(epoch_grad[epoch_grad != 0.0]))
            self._update_smoothed_grad_norm(grad_norm)

            self._log_epoch_stats(
                writer,
                epoch,
                epoch_loss,
                grad_norm,
                lr,
                epoch_start,
                run_start,
                current_knobs,
                self.max_clipping_ratio,
            )

            if prev_loss is not None:
                rel_loss_change = (
                    abs(epoch_loss - prev_loss) / abs(prev_loss) if prev_loss != 0 else 0
                )
                if rel_loss_change < 1e-6 and epoch > 10:
                    LOGGER.info(f"\nLoss change below threshold. Stopping early at epoch {epoch}.")
                    break
            prev_loss = epoch_loss

            if self.smoothed_grad_norm < self.gradient_converged_value:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. Stopping early at epoch {epoch}."
                )
                break
            max_rel_knob_change = max(
                abs(current_knobs[k] - prev_knobs[k]) / abs(prev_knobs[k])
                if prev_knobs[k] != 0
                else 0
                for k in self.knob_names
            )
            if max_rel_knob_change < 1e-8 and epoch > 10:
                LOGGER.info(f"\nKnob updates below threshold. Stopping early at epoch {epoch}.")
                break

        return current_knobs

    def _collect_batch_results(self, parent_conns: list[Connection]) -> tuple[float, np.ndarray]:
        """Collect results from all workers for a batch.

        Aggregates gradients using per-knob averaging: each knob's gradient is
        averaged only over the workers that contributed a non-zero gradient for
        that knob. This prevents magnets at the edges of the BPM range (which
        are only visible to fewer workers) from being under-weighted compared
        to magnets in the middle (which contribute gradients from all workers).
        """
        total_loss = 0.0
        agg_grad = np.zeros(len(self.knob_names), dtype=float)

        for conn in parent_conns:
            _, grad, loss = conn.recv()
            if loss == float("inf"):
                LOGGER.error("Worker error detected, stopping optimisation immediately.")
                raise RuntimeError("Worker error detected during optimisation")

            grad_flat = grad.flatten()

            agg_grad += grad_flat
            total_loss += loss

        return total_loss, agg_grad

    def _update_knobs(
        self, current_knobs: dict[str, float], agg_grad: np.ndarray, lr: float
    ) -> dict[str, float]:
        """Update knob values using the optimiser with per-parameter trust region.

        First applies the optimizer step, then constrains the update magnitude
        using a per-parameter box trust region based on expected_rel_error.
        """
        param_vec = np.array([current_knobs[k] for k in self.knob_names])

        # Let the optimizer propose the step (no gradient scaling)
        new_vec = self.optimiser.step(param_vec, agg_grad, lr)

        # Apply per-parameter trust region to constrain the update
        if self.rel_sigma_step > 0:
            new_vec = self._apply_trust_region_box(
                params=param_vec,
                proposed=new_vec,
                rel_sigma=self.rel_sigma_step,
                param_floor=self.param_floor,
                k=1.0,
            )

        return dict(zip(self.knob_names, new_vec))

    def _apply_trust_region_box(
        self,
        params: np.ndarray,
        proposed: np.ndarray,
        *,
        sigma_abs: np.ndarray | None = None,
        rel_sigma: float | None = None,
        param_floor: float = 0.0,
        k: float = 1.0,
    ) -> np.ndarray:
        """Per-parameter (box) trust region: |Δp_i| <= k * sigma_i

        Constrains the update step for each parameter independently.
        Use either sigma_abs (absolute std per parameter) OR rel_sigma (relative std).

        Args:
            params: Current parameter values
            proposed: Proposed new parameter values from optimizer
            sigma_abs: Per-parameter absolute std (if using absolute mode)
            rel_sigma: Relative std (if using relative mode)
            param_floor: Minimum magnitude for relative calculation
            k: Trust region scale factor (default 1.0 = 1-sigma)

        Returns:
            Clipped parameter values respecting the trust region
        """
        delta = proposed - params

        if sigma_abs is not None:
            if sigma_abs.shape != params.shape:
                raise ValueError("sigma_abs must have same shape as params")
            limit = k * sigma_abs
        elif rel_sigma is not None:
            limit = k * rel_sigma * np.maximum(np.abs(params), param_floor)
        else:
            raise ValueError("Provide sigma_abs or rel_sigma")

        # Track clipping ratio: max(|delta_i| / limit_i) before clipping
        with np.errstate(divide="ignore", invalid="ignore"):
            clipping_ratio = np.abs(delta) / (limit + 1e-30)
            clipping_ratio = np.nan_to_num(clipping_ratio, nan=0.0, posinf=1e30)
        self.max_clipping_ratio = float(np.max(clipping_ratio)) if clipping_ratio.size > 0 else 0.0

        # Clip the delta to the trust region
        delta = np.clip(delta, -limit, +limit)
        return params + delta

    def _update_smoothed_grad_norm(self, grad_norm: float) -> None:
        """Update the exponential moving average of the gradient norm."""
        if self.smoothed_grad_norm == 0.0:  # Exact 0 case for first update
            self.smoothed_grad_norm = grad_norm
        else:
            self.smoothed_grad_norm = (
                self.grad_norm_alpha * self.smoothed_grad_norm
                + (1.0 - self.grad_norm_alpha) * grad_norm
            )

    def _log_epoch_stats(
        self,
        writer: SummaryWriter,
        epoch: int,
        loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
        current_knobs: dict[str, float],
        clipping_ratio: float = 0.0,
    ) -> None:
        """Log statistics for the current epoch."""
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("grad_norm", grad_norm, epoch)
        if self.rel_sigma_step > 0:
            writer.add_scalar("trust_region_clipping_ratio", clipping_ratio, epoch)
            writer.add_scalar("trust_region_rel_sigma_step", self.rel_sigma_step, epoch)

        if self.use_true_strengths:
            true_diff = [abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names]
            rel_diff = [
                diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
                for k, diff in zip(self.knob_names, true_diff)
            ]

            sum_true_diff = np.sum(true_diff)
            sum_rel_diff = np.sum(rel_diff)

            writer.add_scalar("true_diff", sum_true_diff, epoch)
            writer.add_scalar("rel_diff", sum_rel_diff, epoch)
        else:
            writer.add_scalar("avg_knob_value", np.mean(list(current_knobs.values())), epoch)

        if self.some_magnets and self.use_true_strengths:
            true_middle_diff = [
                abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names[5:-5]
            ]
            rel_middle_diff = [
                diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
                for k, diff in zip(self.knob_names[5:-5], true_middle_diff)
            ]

            sum_true_middle_diff = np.sum(true_middle_diff)
            sum_rel_middle_diff = np.sum(rel_middle_diff)

            writer.add_scalar("true_middle_diff", sum_true_middle_diff, epoch)
            writer.add_scalar("rel_middle_diff", sum_rel_middle_diff, epoch)

        writer.add_scalar("learning_rate", lr, epoch)
        writer.flush()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

        clipping_str = f"clipping_ratio={clipping_ratio:.3e}, " if self.rel_sigma_step > 0 else ""
        middle = (
            f"true_diff={sum_true_diff:.3e}, rel_diff={sum_rel_diff:.3e}, "
            if self.use_true_strengths
            else ""
        ) + (
            f"true_middle_diff={sum_true_middle_diff:.3e}, rel_middle_diff={sum_rel_middle_diff:.3e}, "
            if self.some_magnets and self.use_true_strengths
            else ""
        )
        message = (
            f"\rEpoch {epoch}: "
            f"loss={loss:.3e}, "
            f"grad_norm={grad_norm:.3e}, "
            f"{clipping_str}"
            f"{middle}"
            f"lr={lr:.3e}, "
            f"epoch_time={epoch_time:.3f}s, "
            f"total_time={total_time:.3f}s"
        )
        LOGGER.info(message)
