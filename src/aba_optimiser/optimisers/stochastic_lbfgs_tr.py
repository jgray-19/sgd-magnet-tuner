from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np


class StochasticLBFGSTROptimiser:
    """
    Stochastic L-BFGS with a trust region and optional box projection.

    This implementation is designed for noisy objectives with mini-batch
    evaluations. It uses a trust-region acceptance test with the gain ratio
    rho = (f(x) - f(x+p)) / predicted_reduction, where predicted_reduction is
    computed from a linear model -g^T p (robust under noise). L-BFGS history
    updates are skipped on rejected/tiny steps to avoid polluting curvature.

        API
        ---
        step(params, grads, loss, eval_trial_loss_fn, seed=None)
                -> (new_params, accepted, rho, tr_radius, p)
            - params: x_k (np.ndarray)
            - grads:  g_k evaluated at x_k on the same batch used for `loss`
            - loss:   f(x_k) (float)
            - eval_trial_loss_fn: callable that returns f(theta, seed) for the same batch
            - seed:   optional seed to enforce common random numbers (use same for loss, grads)

    Notes
    -----
    - SVRG is intentionally not included; correct SVRG needs mini-batch
      handles/indices to form g_i(x) - g_i(x_tilde) + g_bar(x_tilde).
    - Predicted reduction uses just the linear term. You can extend with a
      quadratic term if you maintain a Hessian-vector product B p.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        memory_size: int = 20,
        initial_tr_radius: float = 1e-3,
        min_tr_radius: float = 1e-8,
        max_tr_radius: float = 1e-1,
        accept_ratio: float = 0.1,
        success_ratio: float = 0.7,
        tr_shrink: float = 0.5,
        tr_expand: float = 2.0,
        damping: float = 1e-4,
        box_constraints: tuple[np.ndarray, np.ndarray] | None = None,
        eps_s: float = 1e-12,
    ) -> None:
        # Trust-region parameters
        self.tr_radius = float(initial_tr_radius)
        self.min_tr_radius = float(min_tr_radius)
        self.max_tr_radius = float(max_tr_radius)
        self.accept_ratio = float(accept_ratio)
        self.success_ratio = float(success_ratio)
        self.tr_shrink = float(tr_shrink)
        self.tr_expand = float(tr_expand)

        # L-BFGS parameters and history
        self.damping = float(damping)
        self.eps_s = float(eps_s)
        self.s_history = deque(maxlen=memory_size)
        self.y_history = deque(maxlen=memory_size)
        self.rho_history = deque(maxlen=memory_size)

        # Constraints
        self.box_constraints = box_constraints

        # Previous state
        self.prev_params = np.zeros(shape, dtype=float)
        self.prev_grads = np.zeros(shape, dtype=float)
        self.prev_loss: float | None = None
        self.have_prev = False
        self.last_step_accepted = True  # nothing to reject at start

    # ------------------------- projections -------------------------
    def project_to_box(self, params: np.ndarray) -> np.ndarray:
        if self.box_constraints is None:
            return params
        lo, hi = self.box_constraints
        return np.clip(params, lo, hi)

    def project_to_tr(self, step: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(step))
        if nrm == 0.0 or nrm <= self.tr_radius:
            return step
        return step * (self.tr_radius / nrm)

    # ------------------------- L-BFGS core -------------------------
    def two_loop_recursion(self, grad: np.ndarray) -> np.ndarray:
        if not self.s_history:
            return -grad
        q = grad.copy()
        n = len(self.s_history)
        alpha = np.zeros(n, dtype=float)

        # backward loop
        for i in reversed(range(n)):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            alpha[i] = rho_i * float(np.dot(s_i, q))
            q -= alpha[i] * y_i

        # initial H0 scaling
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        gamma = float(np.dot(s_last, y_last)) / (
            float(np.dot(y_last, y_last)) + self.damping
        )
        r = gamma * q

        # forward loop
        for i in range(n):
            s_i = self.s_history[i]
            y_i = self.y_history[i]
            rho_i = self.rho_history[i]
            beta = rho_i * float(np.dot(y_i, r))
            r += s_i * (alpha[i] - beta)

        return -r

    def _try_push_history(self, s: np.ndarray, y: np.ndarray) -> bool:
        s_norm2 = float(np.dot(s, s))
        if s_norm2 < self.eps_s:
            return False
        sty = float(np.dot(s, y))
        if sty <= self.damping * s_norm2:
            return False
        self.s_history.append(s.copy())
        self.y_history.append(y.copy())
        self.rho_history.append(1.0 / (sty + self.damping))
        return True

    # ------------------------- trust region -------------------------
    def update_tr_radius(self, rho: float) -> None:
        if rho < self.accept_ratio:
            self.tr_radius = max(self.tr_radius * self.tr_shrink, self.min_tr_radius)
        elif rho > self.success_ratio:
            self.tr_radius = min(self.tr_radius * self.tr_expand, self.max_tr_radius)

    # ------------------------- main step -------------------------
    def step(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        loss: float,
        eval_trial_loss_fn: Callable[[np.ndarray, int | None], float],
        seed: int | None = None,
    ) -> tuple[np.ndarray, bool, float, float, np.ndarray]:
        """
        One trust-region iteration with trial loss evaluated inside for consistency.

        Args:
            params: x_k
            grads:  g_k = grad f(x_k)
            loss:   f(x_k)
            eval_trial_loss_fn: callback to compute f(x_k + p_k) on the same batch/seed
            seed:   optional seed to enforce common random numbers

        Returns:
            (new_params, accepted, rho, tr_radius, proposed_step)
        """
        # L-BFGS history update uses the last accepted step
        if self.have_prev and self.last_step_accepted:
            s = params - self.prev_params
            y = grads - self.prev_grads
            self._try_push_history(s, y)

        # propose step using current gradient
        p = self.two_loop_recursion(grads)
        p = self.project_to_tr(p)
        trial_params = self.project_to_box(params + p)
        p = trial_params - params  # actual proposed step after projection(s)

        # gain ratio (linear predicted reduction)
        predicted = -float(np.dot(grads, p))
        # If projection nulls or reverses the step, auto-reject and shrink TR
        if predicted <= 0.0 or np.allclose(p, 0.0):
            self.update_tr_radius(0.0)
            self.prev_params = params.copy()
            self.prev_grads = grads.copy()
            self.prev_loss = float(loss)
            self.have_prev = True
            self.last_step_accepted = False
            return params.copy(), False, 0.0, float(self.tr_radius), np.zeros_like(p)

        trial_loss = float(eval_trial_loss_fn(trial_params, seed))
        actual = float(loss - trial_loss)
        rho = actual / predicted

        # trust-region radius update
        self.update_tr_radius(rho)

        # accept/reject
        if rho >= self.accept_ratio and trial_loss <= loss:
            new_params = trial_params
            accepted = True
        else:
            new_params = params.copy()
            accepted = False

        # state for next iteration: always store the point we are leaving (x_k)
        # so next call sees s = x_{k+1} - x_k and can update history
        self.prev_params = params.copy()
        self.prev_grads = grads.copy()
        self.prev_loss = float(loss)
        self.have_prev = True
        self.last_step_accepted = accepted

        return new_params, accepted, rho, float(self.tr_radius), p
