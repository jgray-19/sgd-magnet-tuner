from __future__ import annotations

import numpy as np

from aba_optimiser.adam import AdamOptimiser


class AMSGradOptimiser(AdamOptimiser):
    """
    AMSGrad optimizer: like Adam but keeps the max of all past v̂,
    ensuring the denominator never decreases.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            shape, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay
        )
        # track the running maximum of the bias-corrected second moment
        self.v_hat_max = np.zeros(shape, dtype=float)

    def step(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        lr: float,
        # diag_hessian: np.ndarray,
    ) -> np.ndarray:
        """
        Perform one AMSGrad update.

        Args:
            params: Current parameters array.
            grads:  Gradients array (same shape as params).
            lr:     Learning rate for this step.
        Returns:
            Updated parameters array.
        """
        # increment time step
        self.t += 1

        # apply weight decay if any
        if self.weight_decay != 0:
            grads = grads + self.weight_decay * params

        # update biased first and second moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        # bias-corrected estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # keep the element-wise maximum of v̂ over time
        self.v_hat_max = np.maximum(self.v_hat_max, v_hat)

        # parameter update
        # update = lr * (m_hat / (np.sqrt(self.v_hat_max) + self.eps)) / (np.sqrt(diag_hessian) + self.eps)
        update = lr * (m_hat / (np.sqrt(self.v_hat_max) + self.eps))
        return params - update
