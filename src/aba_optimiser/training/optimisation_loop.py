"""Optimisation loop management for the controller."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import (
    GRAD_NORM_ALPHA,
    QUAD_OPT_SETTINGS,
    SEXT_OPT_SETTINGS,
)
from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser
from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
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
        optimise_sextupoles: bool = False,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.smoothed_grad_norm: float | None = None

        config = SEXT_OPT_SETTINGS if optimise_sextupoles else QUAD_OPT_SETTINGS
        self.max_epochs = config.max_epochs
        self.gradient_converged_value = config.gradient_converged_value

        # Initialise optimiser
        self._init_optimiser(initial_strengths.shape, optimiser_type)

        # Initialise scheduler
        # Build scheduler using config values (avoid calling LRScheduler without args)
        self.scheduler = LRScheduler(
            warmup_epochs=config.warmup_epochs,
            decay_epochs=config.decay_epochs,
            start_lr=config.warmup_lr_start,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
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
        elif optimiser_type == "adam":
            self.optimiser = AdamOptimiser(**optimiser_kwargs)
        elif optimiser_type == "lbfgs":
            self.optimiser = LBFGSOptimiser(
                history_size=20,
                eps=1e-12,
                weight_decay=optimiser_kwargs["weight_decay"],
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
        for epoch in range(self.max_epochs):
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

            if self.smoothed_grad_norm < self.gradient_converged_value:
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

        true_middle_diff = [
            abs(current_knobs[k] - self.true_strengths[k])
            for k in self.knob_names[5:-5]
        ]
        rel_middle_diff = [
            diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
            for k, diff in zip(self.knob_names[5:-5], true_middle_diff)
        ]

        sum_true_middle_diff = np.sum(true_middle_diff)
        sum_rel_middle_diff = np.sum(rel_middle_diff)

        writer.add_scalar("loss", total_loss, epoch)
        writer.add_scalar("grad_norm", grad_norm, epoch)
        writer.add_scalar("true_diff", sum_true_diff, epoch)
        writer.add_scalar("rel_diff", sum_rel_diff, epoch)
        writer.add_scalar("true_middle_diff", sum_true_middle_diff, epoch)
        writer.add_scalar("rel_middle_diff", sum_rel_middle_diff, epoch)
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
            f"true_middle_diff={sum_true_middle_diff:.3e}, "
            f"rel_middle_diff={sum_rel_middle_diff:.3e}, "
            f"lr={lr:.3e}, "
            f"epoch_time={epoch_time:.3f}s, "
            f"total_time={total_time:.3f}s",
        )
