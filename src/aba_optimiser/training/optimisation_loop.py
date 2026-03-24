"""Optimisation loop management for the controller."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from aba_optimiser.optimisers import adam as _adam  # noqa: F401
from aba_optimiser.optimisers import amsgrad as _amsgrad  # noqa: F401
from aba_optimiser.optimisers import lbfgs as _lbfgs  # noqa: F401
from aba_optimiser.optimisers.base import BaseOptimiser
from aba_optimiser.training.scheduler import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tensorboardX import SummaryWriter

    from aba_optimiser.config import OptimiserConfig, SimulationConfig
    from aba_optimiser.training.controller_config import CheckpointConfig
    from aba_optimiser.workers.protocol import WorkerChannels

LOGGER = logging.getLogger(__name__)


class _CheckpointState(TypedDict):
    saved_epoch: int
    next_epoch: int
    current_knobs: dict[str, float]
    prev_loss: float | None


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
        abs_offsets: np.ndarray | None = None,
        dabs_dopt: np.ndarray | None = None,
    ):
        self.knob_names = knob_names
        self.true_strengths = true_strengths
        self.use_true_strengths = len(true_strengths) > 0
        self.smoothed_grad_norm: float = 0.0
        self.smoothed_loss_change: float = 0.0
        self.grad_norm_alpha = optimiser_config.grad_norm_alpha
        self.expected_rel_error_abs = optimiser_config.expected_rel_error
        self.max_clipping_ratio: float = 0.0  # Track max clipping ratio per epoch

        # Track best knobs and loss for rejection logic
        self.best_loss: float = float("inf")
        self.best_knobs: dict[str, float] = {}
        self.loss_improvement_threshold = 1e-4  # Minimum relative improvement to accept new best

        self.max_epochs = optimiser_config.max_epochs
        self.gradient_converged_value = optimiser_config.gradient_converged_value
        self.optimiser: BaseOptimiser

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
        self.num_batches = simulation_config.num_batches

        # Convert total expected relative error into a per-step trust-region bound
        self.trust_region_safety = 3.0  # allow errors a few times worse than typical
        self.absolute_param_floor = 1e-6

        if abs_offsets is None:
            self.abs_offsets = np.zeros_like(initial_strengths, dtype=np.float64)
        else:
            self.abs_offsets = np.asarray(abs_offsets, dtype=np.float64)

        if dabs_dopt is None:
            self.dabs_dopt = np.ones_like(initial_strengths, dtype=np.float64)
        else:
            self.dabs_dopt = np.asarray(dabs_dopt, dtype=np.float64)

        if self.abs_offsets.shape != initial_strengths.shape:
            raise ValueError("abs_offsets must have same shape as initial_strengths")
        if self.dabs_dopt.shape != initial_strengths.shape:
            raise ValueError("dabs_dopt must have same shape as initial_strengths")
        if np.any(self.dabs_dopt == 0.0):
            raise ValueError("dabs_dopt contains zero entries, cannot map trust region")

        self.dopt_dabs = 1.0 / self.dabs_dopt

        # Convert user-space relative tolerance (absolute space) into an internal
        # optimisation-space relative tolerance scalar.
        self.expected_rel_error = self._transform_expected_rel_error_to_optimisation_space(
            initial_strengths
        )

        # Per-parameter floor in optimisation space to avoid vanishing limits near zero.
        # Use a stable baseline mapped from absolute-space, not only current delta values.
        baseline_opt_scale = np.maximum(
            np.abs(self.abs_offsets * self.dopt_dabs),
            np.abs(self.dopt_dabs) * self.absolute_param_floor,
        )
        self.optimisation_floor_vec = np.maximum(
            np.abs(initial_strengths),
            baseline_opt_scale,
        )

        self.total_steps = max(1, self.max_epochs * self.num_batches)
        if self.expected_rel_error > 0:
            self.rel_sigma_step = (
                self.trust_region_safety * self.expected_rel_error / np.sqrt(self.total_steps)
            )
            LOGGER.info(
                "Per-parameter trust region enabled: "
                f"rel_sigma_total_abs={self.expected_rel_error_abs:.3e}, "
                f"rel_sigma_total_opt={self.expected_rel_error:.3e}, "
                f"rel_sigma_step={self.rel_sigma_step:.3e}, "
                f"steps={self.total_steps}, safety={self.trust_region_safety:g}"
            )
        else:
            LOGGER.info("Per-parameter trust region disabled")
            self.rel_sigma_step = 0.0

    def _transform_expected_rel_error_to_optimisation_space(
        self,
        initial_strengths: np.ndarray,
    ) -> float:
        """Map user absolute-space relative tolerance to optimisation-space scalar.

        The user config is defined in absolute strength space. Internally we optimise
        in optimisation space (e.g. dknl), so we convert to an equivalent scalar by
        evaluating per-parameter local scale factors at the initial point and taking
        a robust median across knobs.
        """
        if self.expected_rel_error_abs <= 0:
            return 0.0

        # Reference scales are defined around the absolute baseline (offset), not
        # the current delta values, to avoid exploding conversions near dk~=0.
        abs_scale = np.maximum(np.abs(self.abs_offsets), self.absolute_param_floor)
        opt_scale = np.maximum(
            np.abs(self.abs_offsets * self.dopt_dabs),
            np.abs(self.dopt_dabs) * self.absolute_param_floor,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            conversion = abs_scale * np.abs(self.dopt_dabs) / opt_scale
        conversion = np.nan_to_num(conversion, nan=1.0, posinf=1.0, neginf=1.0)

        # Keep conversion robust to outlier knobs with unusual local scaling.
        conversion = np.clip(conversion, 1e-2, 1e2)

        return float(self.expected_rel_error_abs * np.median(conversion))

    def _init_optimiser(self, shape: tuple[int, ...], optimiser_type: str) -> None:
        """Initialise the optimiser based on type."""
        if optimiser_type in {"adam", "amsgrad"}:
            self.optimiser = BaseOptimiser.create(
                optimiser_type,
                shape=shape,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0,
            )
        elif optimiser_type == "lbfgs":
            self.optimiser = BaseOptimiser.create(
                optimiser_type,
                history_size=20,
                eps=1e-12,
                weight_decay=0,
            )
        else:
            raise ValueError(f"Unknown optimiser type: {optimiser_type}")
        LOGGER.info(f"Using optimiser: {self.optimiser.__class__.__name__}")

    @staticmethod
    def _checkpoint_options(
        checkpoint_config: CheckpointConfig | None,
    ) -> tuple[Path | None, int, bool]:
        """Unpack checkpoint options with sensible defaults when disabled."""
        if checkpoint_config is None:
            return None, 0, False
        return (
            checkpoint_config.checkpoint_path,
            checkpoint_config.checkpoint_every_n_epochs,
            checkpoint_config.restore_from_checkpoint,
        )

    @staticmethod
    def _should_save_periodic_checkpoint(
        checkpoint_path: Path | None,
        checkpoint_every_n_epochs: int,
        epoch: int,
    ) -> bool:
        """Return True when this epoch should trigger periodic checkpointing."""
        return (
            checkpoint_path is not None
            and checkpoint_every_n_epochs > 0
            and (epoch + 1) % checkpoint_every_n_epochs == 0
        )

    @staticmethod
    def _should_save_final_checkpoint(
        checkpoint_path: Path | None,
        checkpoint_every_n_epochs: int,
        last_completed_epoch: int,
    ) -> bool:
        """Return True when a final checkpoint should be written on loop exit."""
        return (
            checkpoint_path is not None
            and checkpoint_every_n_epochs > 0
            and last_completed_epoch >= 0
        )

    def _is_new_best(
        self,
        epoch_loss: float,
        prev_loss: float | None,
        sum_diff: float,
    ) -> bool:
        """Decide whether the current epoch should replace the best known state."""
        should_save_as_best = True
        if self.best_loss != float("inf") and prev_loss is not None:
            loss_improvement = (
                (self.best_loss - epoch_loss) / abs(prev_loss) if prev_loss != 0 else 0
            )
            if loss_improvement < self.loss_improvement_threshold:
                best_sum_diff = self._calculate_diff(self.best_knobs)
                if sum_diff > best_sum_diff:
                    should_save_as_best = False
                    LOGGER.debug(
                        f"Not saving as best: loss improvement {loss_improvement:.3e} < {self.loss_improvement_threshold:.3e} "
                        f"and rel_diff {sum_diff:.3e} > {best_sum_diff:.3e}."
                    )
        return should_save_as_best and epoch_loss < self.best_loss

    def _should_stop_for_loss_change(
        self,
        epoch: int,
        epoch_loss: float,
        prev_loss: float | None,
    ) -> bool:
        """Update smoothed loss-change metric and decide if loss-based early stop triggers."""
        if prev_loss is None:
            return False

        rel_loss_change = abs(epoch_loss - prev_loss) / abs(prev_loss) if prev_loss != 0 else 0
        if self.smoothed_loss_change == 0.0:  # Exact 0 case for first update
            self.smoothed_loss_change = rel_loss_change
        else:
            self.smoothed_loss_change = (
                self.grad_norm_alpha * self.smoothed_loss_change
                + (1.0 - self.grad_norm_alpha) * rel_loss_change
            )
        return self.smoothed_loss_change < 1e-6 and epoch > 0.2 * self.max_epochs

    def run_optimisation(
        self,
        current_knobs: dict[str, float],
        channels: WorkerChannels,
        writer: SummaryWriter | None,
        run_start: float,
        total_turns: int,
        checkpoint_config: CheckpointConfig | None = None,
        validation_loss_fn: Callable[[dict[str, float]], float | None] | None = None,
    ) -> dict[str, float]:
        """Run the main optimisation loop."""
        checkpoint_path, checkpoint_every_n_epochs, restore_from_checkpoint = (
            self._checkpoint_options(checkpoint_config)
        )

        if "pt" in current_knobs and current_knobs["pt"] == 0.0:
            current_knobs["pt"] = 1e-6  # Initialise pt to non-zero

        prev_loss = None
        start_epoch = 0

        if restore_from_checkpoint:
            if checkpoint_path is None:
                raise ValueError("restore_from_checkpoint=True requires checkpoint_path to be set")
            checkpoint_state = self._load_checkpoint(checkpoint_path)
            current_knobs = checkpoint_state["current_knobs"]
            prev_loss = checkpoint_state["prev_loss"]
            start_epoch = checkpoint_state["next_epoch"]
            LOGGER.info(
                "Restored optimisation checkpoint from %s at epoch %d",
                checkpoint_path,
                checkpoint_state["saved_epoch"],
            )

        last_completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.max_epochs):
            self.max_clipping_ratio = 0.0  # Reset once per epoch
            epoch_start = time.time()

            epoch_loss = 0.0
            epoch_grad = np.zeros(len(self.knob_names))
            lr = self.scheduler(epoch)

            for batch in range(self.num_batches):
                channels.send_all((current_knobs, batch))

                batch_loss, batch_grad = self._collect_batch_results(channels)
                epoch_loss += batch_loss
                epoch_grad += batch_grad

                # Update knobs after each batch
                current_knobs = self._update_knobs(current_knobs, batch_grad, lr)

            # Keep training loss on a single-worker scale by averaging over batches.
            epoch_loss /= max(1, self.num_batches)
            epoch_grad /= total_turns

            grad_norm = float(np.linalg.norm(epoch_grad[epoch_grad != 0.0]))
            self._update_smoothed_grad_norm(grad_norm)

            # Calculate relative differences for rejection logic
            sum_true_diff = self._calculate_diff(current_knobs)

            validation_loss = (
                validation_loss_fn(current_knobs) if validation_loss_fn is not None else None
            )
            stop_loss = validation_loss if validation_loss is not None else epoch_loss

            new_best = False
            if self._is_new_best(stop_loss, prev_loss, sum_true_diff):
                self.best_loss = stop_loss
                self.best_knobs = current_knobs.copy()
                new_best = True

            stop_for_loss_change = self._should_stop_for_loss_change(epoch, stop_loss, prev_loss)
            if not stop_for_loss_change:
                prev_loss = stop_loss
                last_completed_epoch = epoch

            stop_for_grad_norm = self.smoothed_grad_norm < self.gradient_converged_value
            saved_checkpoint = False
            if (
                not stop_for_loss_change
                and not stop_for_grad_norm
                and self._should_save_periodic_checkpoint(
                    checkpoint_path, checkpoint_every_n_epochs, epoch
                )
            ):
                assert checkpoint_path is not None
                self._save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    current_knobs=current_knobs,
                    prev_loss=prev_loss,
                )
                saved_checkpoint = True

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
                sum_true_diff,
                new_best,
                saved_checkpoint,
                validation_loss,
            )

            if stop_for_loss_change:
                LOGGER.info(f"\nLoss change below threshold. Stopping early at epoch {epoch}.")
                break

            if stop_for_grad_norm:
                LOGGER.info(
                    f"\nGradient norm below threshold: {self.smoothed_grad_norm:.3e}. Stopping early at epoch {epoch}."
                )
                break
        if self._should_save_final_checkpoint(
            checkpoint_path, checkpoint_every_n_epochs, last_completed_epoch
        ):
            assert checkpoint_path is not None
            self._save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=last_completed_epoch,
                current_knobs=current_knobs,
                prev_loss=prev_loss,
            )

        return self.best_knobs

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        current_knobs: dict[str, float],
        prev_loss: float | None,
    ) -> None:
        """Save optimisation state so the run can be resumed later."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_epoch": int(epoch),
            "next_epoch": int(epoch + 1),
            "knob_names": self.knob_names,
            "current_knobs": {k: float(v) for k, v in current_knobs.items()},
            "best_knobs": {k: float(v) for k, v in self.best_knobs.items()},
            "best_loss": float(self.best_loss),
            "prev_loss": None if prev_loss is None else float(prev_loss),
            "smoothed_grad_norm": float(self.smoothed_grad_norm),
            "smoothed_loss_change": float(self.smoothed_loss_change),
            "max_clipping_ratio": float(self.max_clipping_ratio),
            "optimiser_class": self.optimiser.__class__.__name__,
            "optimiser_state": self.optimiser.state_to_dict(),
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2))

    def _load_checkpoint(self, checkpoint_path: Path) -> _CheckpointState:
        """Load optimisation state from checkpoint and apply it to this loop."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        payload = cast("dict[str, Any]", json.loads(checkpoint_path.read_text()))

        saved_knob_names = payload.get("knob_names", [])
        if saved_knob_names != self.knob_names:
            raise ValueError(
                "Checkpoint knob names do not match current optimisation setup. "
                "Ensure you are resuming the same optimisation problem."
            )

        self.best_knobs = {str(k): float(v) for k, v in payload.get("best_knobs", {}).items()}
        self.best_loss = float(payload.get("best_loss", float("inf")))
        self.smoothed_grad_norm = float(payload.get("smoothed_grad_norm", 0.0))
        self.smoothed_loss_change = float(payload.get("smoothed_loss_change", 0.0))
        self.max_clipping_ratio = float(payload.get("max_clipping_ratio", 0.0))

        optimiser_state = payload.get("optimiser_state", {})
        if optimiser_state:
            self.optimiser.load_state_dict(cast("dict[str, Any]", optimiser_state))

        current_knobs = {str(k): float(v) for k, v in payload.get("current_knobs", {}).items()}
        if set(current_knobs.keys()) != set(self.knob_names):
            raise ValueError(
                "Checkpoint current knobs do not match expected knob set for this optimisation."
            )

        return {
            "saved_epoch": int(payload.get("saved_epoch", 0)),
            "next_epoch": int(payload.get("next_epoch", 0)),
            "current_knobs": current_knobs,
            "prev_loss": (
                float(payload["prev_loss"]) if payload.get("prev_loss") is not None else None
            ),
        }

    def _collect_batch_results(self, channels: WorkerChannels) -> tuple[float, np.ndarray]:
        """Collect results from all workers for a batch.

        Aggregates gradients using per-knob averaging: each knob's gradient is
        averaged only over the workers that contributed a non-zero gradient for
        that knob. This prevents magnets at the edges of the BPM range (which
        are only visible to fewer workers) from being under-weighted compared
        to magnets in the middle (which contribute gradients from all workers).
        """
        total_loss = 0.0
        agg_grad = np.zeros(len(self.knob_names), dtype=float)
        results = channels.recv_all()
        n_workers = len(results)
        if n_workers == 0:
            raise RuntimeError("No training workers returned batch results")

        for result in results:
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError(f"Unexpected worker result payload: {result!r}")

            _, grad, loss = cast("tuple[object, np.ndarray, float]", result)
            if loss == float("inf"):
                LOGGER.error("Worker error detected, stopping optimisation immediately.")
                raise RuntimeError("Worker error detected during optimisation")

            grad_flat = grad.flatten()

            agg_grad += grad_flat
            total_loss += float(loss)

        return total_loss / n_workers, agg_grad

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

        # Apply per-parameter trust region in optimisation space using the
        # transformed internal tolerance.
        if self.rel_sigma_step > 0:
            sigma_abs_opt = self.rel_sigma_step * np.maximum(
                np.abs(param_vec),
                self.optimisation_floor_vec,
            )
            new_vec = self._apply_trust_region_box(
                params=param_vec,
                proposed=new_vec,
                sigma_abs=sigma_abs_opt,
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

    def _calculate_diff(self, current_knobs: dict[str, float]) -> float:
        """Calculate sum of absolute and relative differences from true strengths.

        Returns:
            Tuple of (sum_true_diff, sum_rel_diff)
        """
        if not self.use_true_strengths:
            return sum(current_knobs.values())

        true_diff = [abs(current_knobs[k] - self.true_strengths[k]) for k in self.knob_names]

        return np.sum(true_diff)

    def _log_epoch_stats(
        self,
        writer: SummaryWriter | None,
        epoch: int,
        loss: float,
        grad_norm: float,
        lr: float,
        epoch_start: float,
        run_start: float,
        current_knobs: dict[str, float],
        clipping_ratio: float = 0.0,
        sum_true_diff: float = 0.0,
        new_best: bool = False,
        saved_checkpoint: bool = False,
        validation_loss: float | None = None,
    ) -> None:
        """Log statistics for the current epoch."""
        # Log scalars to TensorBoard
        if writer is not None:
            loss_scalars = {"train": loss}
            if validation_loss is not None:
                loss_scalars["validation"] = validation_loss
            writer.add_scalars("loss", loss_scalars, epoch)

            scalars = {
                "grad_norm": grad_norm,
                "learning_rate": lr,
                "sum_true_diff": sum_true_diff,
            }
            if self.rel_sigma_step > 0:
                scalars.update(
                    {
                        "trust_region_clipping_ratio": clipping_ratio,
                        "trust_region_rel_sigma_step": self.rel_sigma_step,
                    }
                )

            for key, value in scalars.items():
                writer.add_scalar(key, value, epoch)
            writer.flush()

        # Calculate times
        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start

        # Build log message
        parts = [f"Ep {epoch}: loss={loss:.3e}"]
        if validation_loss is not None:
            parts.append(f"val={validation_loss:.3e}")
        parts.append(f"g={grad_norm:.3e}")
        if self.rel_sigma_step > 0:
            parts.append(f"clip={clipping_ratio:.2e}")
        parts.append(f"td={sum_true_diff:.3e}")
        parts.append(f"lr={lr:.2e}, et={epoch_time:.1f}s, tt={total_time:.1f}s")
        message = ", ".join(parts)

        if new_best:
            message += " [b]"
        if saved_checkpoint:
            message += " [s]"
        LOGGER.info(f"\r{message}")
