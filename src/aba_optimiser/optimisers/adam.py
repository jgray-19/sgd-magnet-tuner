from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class AdamOptimiser:
    """
    Implements the Adam optimisation algorithm as a stateful optimiser.

    Usage:
        optim = AdamOptimiser(shape=params.shape,
                              beta1=0.9,
                              beta2=0.999,
                              eps=1e-8,
                              weight_decay=0.0)
        new_params = optim.step(params, grads, lr)
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """
        Initialise the Adam optimiser state.

        Args:
            shape: Shape of the parameter vector to be optimised.
            beta1: Exponential decay rate for the first moment estimates.
            beta2: Exponential decay rate for the second moment estimates.
            eps: Small term to avoid division by zero.
            weight_decay: Coefficient for L2 weight decay (adds to gradient).
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        LOGGER.debug(
            f"Initialising Adam optimiser with shape={shape}, beta1={beta1}, beta2={beta2}, eps={eps}, weight_decay={weight_decay}"
        )

        # Initialise first and second moments and timestep
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)
        self.t = 0

    def step(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        lr: float,
        # diag_hessian: np.ndarray,
    ) -> np.ndarray:
        """
        Perform a single optimisation step.

        Args:
            params: Current parameters as a NumPy array.
            grads: Current gradients of loss w.r.t. params.
            lr: Learning rate for this step.

        Returns:
            Updated parameters after applying the Adam update.
        """
        self.t += 1

        LOGGER.debug(f"Adam step {self.t}: lr={lr}, weight_decay={self.weight_decay}")

        # Apply weight decay directly to gradients if specified
        if self.weight_decay != 0:
            grads = grads + self.weight_decay * params
            LOGGER.debug(f"Applied weight decay: {self.weight_decay}")

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Parameter update
        update = lr * (
            m_hat / (np.sqrt(v_hat) + self.eps)
        )  # / (np.sqrt(diag_hessian) + self.eps)

        new_params = params - update
        update_norm = np.linalg.norm(update)
        LOGGER.debug(f"Update norm: {update_norm:.6e}")

        return new_params
