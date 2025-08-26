from __future__ import annotations

import numpy as np


class LBFGSOptimiser:
    """
    Limited-memory BFGS with an adaptive, line-search-free step length.

    Adaptive LR:
      lr_eff = base_lr * eta_k
      eta_k  = EMA( clip( (s^T s) / (s^T y), eta_min, eta_max ) )

    Notes
    -----
    * Two-loop recursion computes d_k = -H_k g_k; we only scale by lr_eff.
    * H0 scaling gamma = (s^T y)/(y^T y) kept (good conditioning).
    * BB1 multiplier uses the same (s,y) you already build; no extra evals.
    """

    def __init__(
        self,
        history_size: int = 20,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        # --- adaptive LR knobs ---
        use_adaptive_lr: bool = True,
        bb_clip: tuple[float, float] = (1e-2, 1e2),  # clip for raw BB1
        ema_beta: float = 0.8,  # smoothing of BB1
        eta_init: float = 1.0,  # initial multiplier
    ):
        self.history_size = int(history_size)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # State needed to build (s, y)
        self.prev_params: np.ndarray | None = None
        self.prev_grads: np.ndarray | None = None

        # L-BFGS memory
        self.S: list[np.ndarray] = []  # s_i = x_{i+1} - x_i
        self.Y: list[np.ndarray] = []  # y_i = g_{i+1} - g_i
        self.RHO: list[float] = []  # 1 / (y_i^T s_i)

        # time step
        self.t = 0

        # Adaptive LR state
        self.use_adaptive_lr = bool(use_adaptive_lr)
        self.bb_min, self.bb_max = bb_clip
        self.ema_beta = float(ema_beta)
        self.eta_ema = float(eta_init)

    def _push_pair(self, s: np.ndarray, y: np.ndarray) -> None:
        ys = float(np.dot(y, s))
        yy = float(np.dot(y, y))
        # Safeguards against near-singular updates
        if ys <= 1e-20 or yy <= 1e-20:
            return
        if len(self.S) == self.history_size:
            self.S.pop(0)
            self.Y.pop(0)
            self.RHO.pop(0)
        self.S.append(s)
        self.Y.append(y)
        self.RHO.append(1.0 / ys)

    def _two_loop(self, g: np.ndarray) -> np.ndarray:
        """Standard two-loop recursion to compute r ≈ H * g."""
        if not self.S:
            return g.copy()  # H ≈ I at start

        q = g.copy()
        alpha = [0.0] * len(self.S)

        # First loop: descending order
        for i in range(len(self.S) - 1, -1, -1):
            s_i, y_i, rho_i = self.S[i], self.Y[i], self.RHO[i]
            alpha[i] = rho_i * np.dot(s_i, q)
            q = q - alpha[i] * y_i

        # Initial H0 scaling using the most recent pair (gamma ≈ BB2)
        s_last, y_last = self.S[-1], self.Y[-1]
        yy = float(np.dot(y_last, y_last))
        ys = float(np.dot(y_last, s_last))
        gamma = ys / yy if yy > 0 else 1.0
        r = gamma * q

        # Second loop: ascending order
        for i in range(len(self.S)):
            s_i, y_i, rho_i = self.S[i], self.Y[i], self.RHO[i]
            beta = rho_i * np.dot(y_i, r)
            r = r + s_i * (alpha[i] - beta)
        return r

    def _bb1_multiplier(self, s: np.ndarray, y: np.ndarray) -> float:
        """
        BB1 spectral step (s^T s)/(s^T y); clipped + EMA-smoothed.
        Use only when we have a fresh pair.
        """
        denom = float(np.dot(s, y))
        if denom <= self.eps:
            return self.eta_ema  # keep previous multiplier if curvature poor
        raw = float(np.dot(s, s)) / denom
        # clip to avoid outliers, then EMA
        raw = min(max(raw, self.bb_min), self.bb_max)
        self.eta_ema = self.ema_beta * self.eta_ema + (1.0 - self.ema_beta) * raw
        return self.eta_ema

    def step(self, params: np.ndarray, grads: np.ndarray, lr: float) -> np.ndarray:
        """
        Perform one L-BFGS step with adaptive LR multiplier.

        Args
        ----
        params : np.ndarray
            Current parameter vector x_k.
        grads : np.ndarray
            Current gradient g_k = ∇f(x_k).
        lr : float
            Base learning rate (e.g. from your cosine scheduler).
        """
        self.t += 1

        # Weight decay (L2 regularisation)
        g = grads + (self.weight_decay * params if self.weight_decay != 0 else 0)

        # On k ≥ 1, build (s_{k-1}, y_{k-1}) from the PREVIOUS state
        eta_mult = self.eta_ema
        if self.prev_params is not None and self.prev_grads is not None:
            s = params - self.prev_params
            y = g - self.prev_grads
            self._push_pair(s, y)
            if self.use_adaptive_lr:
                eta_mult = self._bb1_multiplier(s, y)

        # Compute quasi-Newton direction: d = - H_k * g_k
        Hg = self._two_loop(g)  # noqa: N806
        d = -Hg

        # Take step with adaptive multiplier (no line search)
        lr_eff = lr * eta_mult
        new_params = params + lr_eff * d

        # Save state for the next call
        self.prev_params = params.copy()
        self.prev_grads = g.copy()

        return new_params
