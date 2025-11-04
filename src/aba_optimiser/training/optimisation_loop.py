"""Optimisation loop management for the controller."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import GRAD_NORM_ALPHA
from aba_optimiser.optimisers.adam import AdamOptimiser
from aba_optimiser.optimisers.amsgrad import AMSGradOptimiser
from aba_optimiser.optimisers.lbfgs import LBFGSOptimiser
from aba_optimiser.training.scheduler import LRScheduler

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from tensorboardX import SummaryWriter

    from aba_optimiser.config import OptSettings

LOGGER = logging.getLogger(__name__)


class OptimisationLoop:
    """Manages the optimisation loop and statistics tracking."""

    def __init__(
        self,
        initial_strengths: np.ndarray,
        knob_names: list[str],
        true_strengths: dict[str, float],
        opt_settings: OptSettings,
        optimiser_type: str = "adam",
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.use_true_strengths = len(true_strengths) > 0
        self.smoothed_grad_norm: float | None = None

        self.max_epochs = opt_settings.max_epochs
        self.gradient_converged_value = opt_settings.gradient_converged_value

        # Initialise optimiser
        self._init_optimiser(initial_strengths.shape, optimiser_type)

        # Initialise scheduler
        # Build scheduler using opt_settings values (avoid calling LRScheduler without args)
        self.scheduler = LRScheduler(
            warmup_epochs=opt_settings.warmup_epochs,
            decay_epochs=opt_settings.decay_epochs,
            start_lr=opt_settings.warmup_lr_start,
            max_lr=opt_settings.max_lr,
            min_lr=opt_settings.min_lr,
        )

        self.some_magnets = not opt_settings.only_energy
        self.num_batches = opt_settings.num_batches

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
        if current_knobs["pt"] == 0 and not self.some_magnets:
            current_knobs["pt"] = 1e-6  # Initialise pt to non-zero

        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            epoch_loss = 0.0
            epoch_grad = None
            lr = self.scheduler(epoch)
            prev_knobs = current_knobs.copy()

            for batch in range(self.num_batches):
                # Send knobs to workers
                for conn in parent_conns:
                    conn.send((current_knobs, batch))

                batch_loss, batch_grad = self._collect_batch_results(parent_conns)
                epoch_loss += batch_loss
                if epoch_grad is None:
                    epoch_grad = batch_grad
                else:
                    epoch_grad += batch_grad

                # Update knobs after each batch
                current_knobs = self._update_knobs(current_knobs, batch_grad, lr)

            # Average over batches for logging
            epoch_loss /= total_turns
            epoch_grad /= total_turns

            grad_norm = np.linalg.norm(epoch_grad)
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
            )

            if self.smoothed_grad_norm < self.gradient_converged_value:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. "
                    f"Stopping early at epoch {epoch}."
                )
                break
            if (
                sum(abs(current_knobs[k] - prev_knobs[k]) for k in self.knob_names)
                < 1e-11
                and epoch > 10
            ):
                LOGGER.info(
                    f"\nKnob updates below threshold. Stopping early at epoch {epoch}."
                )
                break

        return current_knobs

    def _collect_batch_results(
        self, parent_conns: list[Connection]
    ) -> tuple[float, np.ndarray]:
        """Collect results from all workers for a batch."""
        total_loss = 0.0
        agg_grad: None | np.ndarray = None
        for conn in parent_conns:
            _, grad, loss = conn.recv()
            agg_grad = grad if agg_grad is None else agg_grad + grad
            total_loss += loss
        return total_loss, agg_grad.flatten()

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
        loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
        current_knobs: dict[str, float],
    ) -> None:
        """Log statistics for the current epoch."""
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("grad_norm", grad_norm, epoch)

        if self.use_true_strengths:
            true_diff = [
                abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names
            ]
            rel_diff = [
                diff / abs(self.true_strengths[k]) if self.true_strengths[k] != 0 else 0
                for k, diff in zip(self.knob_names, true_diff)
            ]

            sum_true_diff = np.sum(true_diff)
            sum_rel_diff = np.sum(rel_diff)

            writer.add_scalar("true_diff", sum_true_diff, epoch)
            writer.add_scalar("rel_diff", sum_rel_diff, epoch)
        else:
            writer.add_scalar(
                "avg_knob_value", np.mean(list(current_knobs.values())), epoch
            )

        if self.some_magnets and self.use_true_strengths:
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

            writer.add_scalar("true_middle_diff", sum_true_middle_diff, epoch)
            writer.add_scalar("rel_middle_diff", sum_rel_middle_diff, epoch)

        writer.add_scalar("learning_rate", lr, epoch)
        writer.flush()

        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

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
            f"{middle}"
            f"lr={lr:.3e}, "
            f"epoch_time={epoch_time:.3f}s, "
            f"total_time={total_time:.3f}s"
        )
        LOGGER.info(message)
