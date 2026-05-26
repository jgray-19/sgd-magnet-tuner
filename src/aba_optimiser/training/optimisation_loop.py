"""Optimisation loop management for the controller."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
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

        # Track best knobs and loss for rejection logic
        self.best_loss: float = float("inf")
        self.best_knobs: dict[str, float] = {}
        self.loss_improvement_threshold = 1e-4  # Minimum relative improvement to accept new best

        self.max_epochs = optimiser_config.max_epochs
        self.gradient_converged_value = optimiser_config.gradient_converged_value
        self.optimiser: BaseOptimiser

        # Initialise optimiser
        if optimiser_type is not None:
            optimiser_config = replace(optimiser_config, optimiser_type=optimiser_type)
        self._init_optimiser(initial_strengths.shape, optimiser_config)

        # Initialise scheduler
        self.scheduler = LRScheduler(
            warmup_epochs=optimiser_config.warmup_epochs,
            decay_epochs=optimiser_config.decay_epochs,
            start_lr=optimiser_config.warmup_lr_start,
            max_lr=optimiser_config.max_lr,
            min_lr=optimiser_config.min_lr,
        )
        self.num_batches = simulation_config.num_batches

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

    def _init_optimiser(self, shape: tuple[int, ...], optimiser_config: OptimiserConfig) -> None:
        """Initialise the optimiser based on type."""
        optimiser_type = optimiser_config.optimiser_type
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
                history_size=optimiser_config.lbfgs_history_size,
                eps=1e-12,
                weight_decay=0,
                max_grad_norm=optimiser_config.lbfgs_max_grad_norm,
                max_step_norm=optimiser_config.lbfgs_max_step_norm,
                powell_damping=optimiser_config.lbfgs_powell_damping,
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
        epoch_end_hook: Callable[[dict[str, float]], None] | None = None,
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
            checkpoint_state = self._load_checkpoint(
                checkpoint_path,
                base_current_knobs=current_knobs,
            )
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
            epoch_start = time.time()

            epoch_loss = 0.0
            epoch_grad = np.zeros(len(self.knob_names))
            lr = self.scheduler(epoch)
            pre_epoch_knobs = current_knobs
            epoch_had_particle_loss = False

            for batch in range(self.num_batches):
                channels.send_all((current_knobs, batch))

                batch_loss, batch_grad, had_particle_loss = self._collect_batch_results(channels)
                epoch_loss += batch_loss
                epoch_grad += batch_grad
                epoch_had_particle_loss = epoch_had_particle_loss or had_particle_loss

                # Update knobs after each batch (only when no particle loss this epoch)
                if not epoch_had_particle_loss:
                    current_knobs = self._update_knobs(current_knobs, batch_grad, lr)

            if epoch_had_particle_loss:
                LOGGER.warning(
                    "Epoch %d: particle loss detected — rejecting knob updates and restoring "
                    "pre-epoch parameters",
                    epoch,
                )
                current_knobs = pre_epoch_knobs

            # Keep training loss on a single-worker scale by averaging over batches.
            epoch_loss /= max(1, self.num_batches)
            epoch_grad /= total_turns

            grad_norm = float(np.linalg.norm(epoch_grad[epoch_grad != 0.0]))
            self._update_smoothed_grad_norm(grad_norm)

            # Calculate relative differences for rejection logic
            sum_true_diff = self._calculate_diff(current_knobs)

            if epoch_end_hook is not None:
                epoch_end_hook(current_knobs)

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
            "optimiser_class": self.optimiser.__class__.__name__,
            "optimiser_state": self.optimiser.state_to_dict(),
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _clip_name_list_for_error(names: set[str]) -> str:
        """Format long knob-name sets for concise error messages."""
        return ", ".join(sorted(names)[:10]) + ("..." if len(names) > 10 else "")

    @staticmethod
    def _nonfinite_knob_names(knobs: dict[str, float]) -> list[str]:
        """Return knob names whose values are not finite."""
        return [name for name, value in knobs.items() if not np.isfinite(value)]

    def _validate_finite_knob_values(self, knobs: dict[str, float], *, label: str) -> None:
        """Reject knob maps that contain NaN or infinite values."""
        nonfinite = self._nonfinite_knob_names(knobs)
        if nonfinite:
            raise ValueError(
                f"{label} contains non-finite knob values: "
                + self._clip_name_list_for_error(set(nonfinite))
            )

    def _initialise_merged_current_knobs(
        self,
        base_current_knobs: dict[str, float] | None,
    ) -> dict[str, float]:
        """Build the starting current-knob map used when restoring checkpoints."""
        current_knob_set = set(self.knob_names)
        if base_current_knobs is None:
            return {k: (1e-6 if k == "pt" else 0.0) for k in self.knob_names}

        if set(base_current_knobs.keys()) != current_knob_set:
            raise ValueError(
                "base_current_knobs must contain exactly the current optimisation knob set."
            )
        merged = {str(k): float(v) for k, v in base_current_knobs.items()}
        self._validate_finite_knob_values(merged, label="base_current_knobs")
        return merged

    def _validate_checkpoint_knob_compatibility(
        self,
        saved_knob_names: list[str],
    ) -> set[str]:
        """Ensure checkpoint knobs are a subset of the current optimisation knobs."""
        saved_knob_set = set(saved_knob_names)
        current_knob_set = set(self.knob_names)

        missing_in_current = saved_knob_set.difference(current_knob_set)
        if missing_in_current:
            raise ValueError(
                "Checkpoint knob names are not compatible with current optimisation setup. "
                "Current setup is missing checkpoint knobs: "
                + self._clip_name_list_for_error(missing_in_current)
            )
        return saved_knob_set

    def _parse_checkpoint_knob_values(
        self,
        payload: dict[str, Any],
        field_name: str,
    ) -> dict[str, float]:
        """Parse and validate a knob-value mapping from checkpoint payload."""
        current_knob_set = set(self.knob_names)
        checkpoint_values = {
            str(k): float(v) for k, v in cast("dict[str, Any]", payload.get(field_name, {})).items()
        }
        unknown_names = set(checkpoint_values.keys()).difference(current_knob_set)
        if unknown_names:
            raise ValueError(
                f"Checkpoint {field_name} contain names that are not in the current optimisation setup: "
                + self._clip_name_list_for_error(unknown_names)
            )
        self._validate_finite_knob_values(checkpoint_values, label=f"Checkpoint {field_name}")
        return checkpoint_values

    @staticmethod
    def _expand_vector_to_current_knobs(
        vector: list[float],
        saved_knob_names: list[str],
        current_knob_names: list[str],
        fill_value: float,
    ) -> list[float]:
        """Remap a saved optimizer vector onto the current knob layout with padding."""
        if len(vector) != len(saved_knob_names):
            raise ValueError(
                "Optimiser state vector length does not match checkpoint knob_names length."
            )
        saved_index = {name: i for i, name in enumerate(saved_knob_names)}
        return [
            float(vector[saved_index[name]]) if name in saved_index else fill_value
            for name in current_knob_names
        ]

    def _resize_optimiser_state_for_current_knobs(
        self,
        optimiser_state: dict[str, Any],
        saved_knob_names: list[str],
        merged_current_knobs: dict[str, float],
    ) -> dict[str, Any]:
        """Resize/remap optimiser state vectors to match current knob layout."""
        if saved_knob_names == self.knob_names:
            return optimiser_state

        state = dict(optimiser_state)
        state_type = str(state.get("type", "")).lower()

        if state_type in {"adam", "amsgrad"}:
            state["m"] = self._expand_vector_to_current_knobs(
                cast("list[float]", state["m"]),
                saved_knob_names,
                self.knob_names,
                fill_value=0.0,
            )
            state["v"] = self._expand_vector_to_current_knobs(
                cast("list[float]", state["v"]),
                saved_knob_names,
                self.knob_names,
                fill_value=0.0,
            )
            if state_type == "amsgrad" and "v_hat_max" in state:
                state["v_hat_max"] = self._expand_vector_to_current_knobs(
                    cast("list[float]", state["v_hat_max"]),
                    saved_knob_names,
                    self.knob_names,
                    fill_value=0.0,
                )
            return state

        if state_type == "lbfgs":
            state["S"] = [
                self._expand_vector_to_current_knobs(
                    cast("list[float]", vec),
                    saved_knob_names,
                    self.knob_names,
                    fill_value=0.0,
                )
                for vec in cast("list[list[float]]", state.get("S", []))
            ]
            state["Y"] = [
                self._expand_vector_to_current_knobs(
                    cast("list[float]", vec),
                    saved_knob_names,
                    self.knob_names,
                    fill_value=0.0,
                )
                for vec in cast("list[list[float]]", state.get("Y", []))
            ]
            prev_params = state.get("prev_params")
            if prev_params is not None:
                state["prev_params"] = self._expand_vector_to_current_knobs(
                    cast("list[float]", prev_params),
                    saved_knob_names,
                    self.knob_names,
                    fill_value=0.0,
                )
            prev_grads = state.get("prev_grads")
            if prev_grads is not None:
                state["prev_grads"] = self._expand_vector_to_current_knobs(
                    cast("list[float]", prev_grads),
                    saved_knob_names,
                    self.knob_names,
                    fill_value=0.0,
                )
            return state

        # Unknown optimiser states are passed through unchanged and may still fail
        # in load_state_dict with a clearer optimiser-specific message.
        _ = merged_current_knobs
        return state

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        base_current_knobs: dict[str, float] | None = None,
    ) -> _CheckpointState:
        """Load optimisation state from checkpoint and apply it to this loop."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        payload = cast("dict[str, Any]", json.loads(checkpoint_path.read_text()))

        saved_knob_names = [str(k) for k in payload.get("knob_names", [])]
        saved_knob_set = self._validate_checkpoint_knob_compatibility(saved_knob_names)

        merged_current_knobs = self._initialise_merged_current_knobs(base_current_knobs)

        checkpoint_current = self._parse_checkpoint_knob_values(payload, "current_knobs")

        missing_checkpoint_current = saved_knob_set.difference(set(checkpoint_current.keys()))
        if missing_checkpoint_current:
            raise ValueError(
                "Checkpoint current knobs are missing saved checkpoint knob values: "
                + self._clip_name_list_for_error(missing_checkpoint_current)
            )

        merged_current_knobs.update(checkpoint_current)
        self._validate_finite_knob_values(merged_current_knobs, label="Restored current knobs")

        checkpoint_best = self._parse_checkpoint_knob_values(payload, "best_knobs")
        self._validate_finite_knob_values(checkpoint_best, label="Checkpoint best_knobs")

        self.best_knobs = merged_current_knobs.copy()
        self.best_knobs.update(checkpoint_best)
        self.best_loss = float(payload.get("best_loss", float("inf")))
        self.smoothed_grad_norm = float(payload.get("smoothed_grad_norm", 0.0))
        self.smoothed_loss_change = float(payload.get("smoothed_loss_change", 0.0))

        optimiser_state = payload.get("optimiser_state", {})
        if optimiser_state:
            try:
                resized_state = self._resize_optimiser_state_for_current_knobs(
                    cast("dict[str, Any]", optimiser_state),
                    saved_knob_names,
                    merged_current_knobs,
                )
                self.optimiser.load_state_dict(resized_state)
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.info("Skipping optimiser state restore: %s", exc)

        return {
            "saved_epoch": int(payload.get("saved_epoch", 0)),
            "next_epoch": int(payload.get("next_epoch", 0)),
            "current_knobs": merged_current_knobs,
            "prev_loss": (
                float(payload["prev_loss"]) if payload.get("prev_loss") is not None else None
            ),
        }

    def _collect_batch_results(
        self, channels: WorkerChannels
    ) -> tuple[float, np.ndarray, bool]:
        """Collect results from all workers for a batch.

        Aggregates gradients using per-knob averaging: each knob's gradient is
        averaged only over the workers that contributed a non-zero gradient for
        that knob. This prevents magnets at the edges of the BPM range (which
        are only visible to fewer workers) from being under-weighted compared
        to magnets in the middle (which contribute gradients from all workers).

        Returns a third element `had_particle_loss` — True if any worker detected
        particle loss this batch. The caller should reject the knob update for
        the enclosing epoch in that case.
        """
        import math

        total_loss = 0.0
        agg_grad = np.zeros(len(self.knob_names), dtype=float)
        had_particle_loss = False
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

            if math.isnan(float(loss)):
                had_particle_loss = True
                continue

            grad_flat = grad.flatten()
            agg_grad += grad_flat
            total_loss += float(loss)

        return total_loss / n_workers, agg_grad, had_particle_loss

    def _update_knobs(
        self, current_knobs: dict[str, float], agg_grad: np.ndarray, lr: float
    ) -> dict[str, float]:
        """Update knob values using the optimiser."""
        param_vec = np.array([current_knobs[k] for k in self.knob_names])
        new_vec = self.optimiser.step(param_vec, agg_grad, lr)
        return dict(zip(self.knob_names, new_vec))

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
        parts.append(f"td={sum_true_diff:.3e}")
        parts.append(f"lr={lr:.2e}, et={epoch_time:.1f}s, tt={total_time:.1f}s")
        message = ", ".join(parts)

        if new_best:
            message += " [b]"
        if saved_checkpoint:
            message += " [s]"
        LOGGER.info(f"\r{message}")
