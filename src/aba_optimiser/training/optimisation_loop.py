"""Optimisation loop management for the controller."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    DECAY_EPOCHS,
    GRAD_NORM_ALPHA,
    GRADIENT_CONVERGED_VALUE,
    MAX_EPOCHS,
    MAX_LR,
    MIN_LR,
    WARMUP_EPOCHS,
    WARMUP_LR_START,
)
from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser
from aba_optimiser.training.scheduler import LRScheduler

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from tensorboardX import SummaryWriter

LOGGER = logging.getLogger(__name__)


class OptimisationLoop:
    """Manages the optimisation loop and statistics tracking."""

    def __init__(
        self,
        initial_strengths: np.ndarray,
        knob_names: list[str],
        true_strengths: dict[str, float],
        optimiser_type: str = "adam",
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.smoothed_grad_norm: float | None = None

        # Initialise optimiser
        self._init_optimiser(initial_strengths.shape, optimiser_type)

        # Initialise scheduler
        # Build scheduler using config values (avoid calling LRScheduler without args)
        self.scheduler = LRScheduler(
            warmup_epochs=WARMUP_EPOCHS,
            decay_epochs=DECAY_EPOCHS,
            start_lr=WARMUP_LR_START,
            max_lr=MAX_LR,
            min_lr=MIN_LR,
        )

    def _init_optimiser(self, shape: tuple, optimiser_type: str) -> None:
        """Initialise the optimiser based on type."""
        optimiser_kwargs = {
            "shape": shape,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0,
        }
        if optimiser_type == "amsgrad":
            self.optimiser = AMSGradOptimiser(**optimiser_kwargs)
        else:
            self.optimiser = AdamOptimiser(**optimiser_kwargs)

    def run_optimisation(
        self,
        current_knobs: dict[str, float],
        parent_conns: list[Connection],
        writer: SummaryWriter,
        run_start: float,
        total_turns: int,
    ) -> dict[str, float]:
        """Run the main optimisation loop."""
        for epoch in range(MAX_EPOCHS):
            epoch_start = time.time()

            # Send knobs to workers
            for conn in parent_conns:
                conn.send(current_knobs)

            lr = self.scheduler(epoch)
            total_loss, agg_grad = self._collect_worker_results(
                parent_conns, total_turns
            )

            current_knobs = self._update_knobs(current_knobs, agg_grad, lr)
            grad_norm = np.linalg.norm(agg_grad)
            self._update_smoothed_grad_norm(grad_norm)

            self._log_epoch_stats(
                writer,
                epoch,
                total_loss,
                grad_norm,
                lr,
                epoch_start,
                run_start,
                current_knobs,
            )

            if self.smoothed_grad_norm < GRADIENT_CONVERGED_VALUE:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. "
                    f"Stopping early at epoch {epoch}."
                )
                break

        return current_knobs

    def _collect_worker_results(
        self, parent_conns: list[Connection], total_turns: int
    ) -> tuple[float, np.ndarray]:
        """Collect results from all workers for an epoch."""
        total_loss = 0.0
        agg_grad: None | np.ndarray = None
        for conn in parent_conns:
            _, grad, loss = conn.recv()
            agg_grad = grad if agg_grad is None else agg_grad + grad
            total_loss += loss
        total_loss /= total_turns
        agg_grad = agg_grad.flatten() / total_turns
        return total_loss, agg_grad

    def _update_knobs(
        self, current_knobs: dict[str, float], agg_grad: np.ndarray, lr: float
    ) -> dict[str, float]:
        """Update knob values using the optimiser."""
        param_vec = np.array([current_knobs[k] for k in self.knob_names])
        new_vec = self.optimiser.step(param_vec, agg_grad, lr)
        return dict(zip(self.knob_names, new_vec))

    def _update_smoothed_grad_norm(self, grad_norm: float) -> None:
        """Update the exponential moving average of the gradient norm."""
        if self.smoothed_grad_norm is None:
            self.smoothed_grad_norm = grad_norm
        else:
            self.smoothed_grad_norm = (
                GRAD_NORM_ALPHA * self.smoothed_grad_norm
                + (1.0 - GRAD_NORM_ALPHA) * grad_norm
            )

    def _log_epoch_stats(
        self,
        writer: SummaryWriter,
        epoch: int,
        total_loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
        current_knobs: dict[str, float],
    ) -> None:
        """Log statistics for the current epoch."""
        true_diff = [
            abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names
        ]
        rel_diff = [
            diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
            for k, diff in zip(self.knob_names, true_diff)
        ]

        sum_true_diff = np.sum(true_diff)
        sum_rel_diff = np.sum(rel_diff)

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("grad_norm", grad_norm, epoch)
        writer.add_scalar("true_diff", sum_true_diff, epoch)
        writer.add_scalar("rel_diff", sum_rel_diff, epoch)
        writer.add_scalar("learning_rate", lr, epoch)
        writer.flush()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

        LOGGER.info(
            f"\rEpoch {epoch}: "
            f"loss={total_loss:.3e}, "
            f"grad_norm={grad_norm:.3e}, "
            f"true_diff={sum_true_diff:.3e}, "
            f"rel_diff={sum_rel_diff:.3e}, "
            f"lr={lr:.3e}, "
            f"epoch_time={epoch_time:.3f}s, "
            f"total_time={total_time:.3f}s",
        )
