from __future__ import annotations

# import os
import numpy as np
import pandas as pd

# import tfs
from numba import njit

# from tqdm import tqdm
from aba_optimiser.config import (
    # ACD_ON,
    MAGNET_RANGE,
    # FLATTOP_TURNS,
    MOMENTUM_STD_DEV,
    # NOISE_FILE,
    POSITION_STD_DEV,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from aba_optimiser.mad_interface import MadInterface

# from aba_optimiser.phase_space import PhaseSpaceDiagnostics


class BPMKalmanFilter:
    """
    Self-contained Kalman filter for a 4D transverse state (x, px, y, py).
    On initialization, it:
      - Connects to MAD-X via MadInterface
      - Retrieves the BPM list from SEQ_NAME
      - Computes BPM-to-BPM 4x4 transfer matrices
      - Computes process noise Q from quadrupole perturbations
      - Sets up measurement noise R, identity observation H, and special BPM
    Call `run(meas_df)` to execute filtering; returns a pandas DataFrame of filtered states
    (one row per turn and BPM) with columns ["turn", "name", "x", "px", "y", "py",
    "var_x", "var_px", "var_y", "var_py", "weight_x", "weight_y"].
    """

    def __init__(self):
        # Initialize MAD-X interface
        self.mad_iface = MadInterface(
            SEQUENCE_FILE, MAGNET_RANGE, discard_mad_output=True
        )
        # Retrieve BPM names
        self.bpm_list = self._get_bpm_list()
        self.n_bpm = len(self.bpm_list)
        # Observation and special-case settings
        self.H = np.eye(4)
        self.special_bpm = "BPMYA.5L4.B1"
        # Compute transfer matrices
        self.bpm_mats = self._compute_bpm_mats()
        # — REPLACE fixed-Q with state-dependent Q_t —
        # 1) precompute per-BPM Jacobians dM/dk (4×4) at nominal k
        self.J_stack = self._compute_sensitivity_matrices()
        # 2) store quad-error variance: per-quad variances summed into a scalar
        knob_vars = (REL_K1_STD_DEV * self.mad_iface.receive_knob_values()) ** 2
        # combine independent quad variances
        self.sigma_k2 = float(np.sum(knob_vars))
        # TODO: for full process-noise model, store per-quad variances and build
        #      J_tensor of shape (n_bpm, n_quads, 4, 4) to compute Q_t = G^T diag(knob_vars) G
        # we'll compute Q_t inside the filter loop, so drop self.Q
        self.Q = None

        # Measurement noise (same for all BPMs)
        sigma = [POSITION_STD_DEV, MOMENTUM_STD_DEV, POSITION_STD_DEV, MOMENTUM_STD_DEV]
        R_mat = np.diag(np.square(np.array(sigma)))
        self.R = np.repeat(R_mat[np.newaxis, :, :], self.n_bpm, axis=0)

    def _get_bpm_list(self) -> list[str]:
        code = f"""
local {SEQ_NAME} in MADX
local bpm_list = {{}}
for _, elm in {SEQ_NAME}:iter() do
    if elm.name:match("BPM") then table.insert(bpm_list, elm.name) end
end
py:send(bpm_list, true)
"""
        self.mad_iface.mad.send(code)
        return self.mad_iface.mad.recv()

    def _compute_bpm_mats(self) -> list[np.ndarray]:
        mats = []
        for i, bpm in enumerate(self.bpm_list):
            prev_b = self.bpm_list[(i - 1) % self.n_bpm]
            code = f"""
local {SEQ_NAME} in MADX
local trk, mflw = track{{sequence={SEQ_NAME}, range=\"{prev_b}/{bpm}\", turn=1, mapdef=2}}
py:send(mflw[1]:get1())
"""
            self.mad_iface.mad.send(code)
            M_full = self.mad_iface.mad.recv()

            # Convert to x, px, y, py (remove t, pt)
            mats.append(np.array(M_full[:4, :4].copy()))
        return mats

    def _compute_sensitivity_matrices(self) -> list[np.ndarray]:
        """
        Compute Jacobian J_i = ∂M_i/∂k  for each BPM i
        """
        j_list = []
        rel_delta = 1e-4  # small perturbation on quad strength
        for i, bpm in enumerate(self.bpm_list):
            # get nominal one-turn map M0
            m0 = self.bpm_mats[i]
            # perturb all quads by +delta (you'll need to re-track in MAD-X)
            m_pert = self._track_with_quad_perturbation(bpm, +rel_delta)
            # finite-difference Jacobian
            j_list.append((m_pert[:4, :4] - m0) / rel_delta)
        return j_list

    def _track_with_quad_perturbation(self, bpm: str, rel_delta: float) -> np.ndarray:
        """
        Helper method to compute transfer matrix with perturbed quadrupoles
        """
        i = self.bpm_list.index(bpm)
        prev_b = self.bpm_list[(i - 1) % self.n_bpm]
        seq = f"MADX.{SEQ_NAME}"

        # Apply perturbation to all quadrupoles
        knob_names = self.mad_iface.knob_names
        knob_values = self.mad_iface.receive_knob_values()
        perturb_code = ""
        for knob in knob_names:
            perturb_code += (
                f"{seq}['{knob}'] = {seq}['{knob}'] + {seq}['{knob}']*{rel_delta}\n"
            )

        # Track with perturbation
        code = f"""
{perturb_code}
local trk, mflw = track{{sequence={seq}, range=\"{prev_b}/{bpm}\", turn=1, mapdef=2}}
py:send(mflw[1]:get1())
-- Restore original values
"""
        for i, knob in enumerate(knob_names):
            code += f"{seq}['{knob}'] = {knob_values[i]: .15e}\n"

        self.mad_iface.mad.send(code)
        M_pert = self.mad_iface.mad.recv()
        return np.array(M_pert)

    def run(
        self,
        meas_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the Kalman filter on a TFS-style measurement DataFrame.
        meas_df must have columns ["turn","name","x","px","y","py"].
        Returns a flat DataFrame with one row per (turn, BPM), including state estimates
        and variances: ["turn", "name", "x", "px", "y", "py",
        "var_x", "var_px", "var_y", "var_py", "weight_x", "weight_y"].
        """
        print("Running Kalman filter...")
        # Unique sorted turns
        turns = range(int(meas_df["turn"].min()), int(meas_df["turn"].max() + 1))
        n_turns = len(turns)
        # Build measurement array
        pivot = meas_df.pivot(
            index="turn", columns="name", values=["x", "px", "y", "py"]
        )
        pivot = pivot.reindex(
            columns=self.bpm_list, level=1
        )  # ensure correct BPM order
        meas = np.stack(
            [
                pivot[comp].loc[sorted(pivot.index)].values
                for comp in ["x", "px", "y", "py"]
            ],
            axis=-1,
        )
        # ------------------------------------------------------------------
        #  EM ITERATION LOOP
        # ------------------------------------------------------------------
        # working copies of Q/R in double precision for EM stability
        x_hat, P_hat, _ = self._filter_array(
            meas, do_smoothing=False, return_cross=True
        )

        # Build output DataFrame vectorized
        n_turns, n_bpm = x_hat.shape[0], len(self.bpm_list)
        turn_arr = np.repeat(np.array(list(turns)), n_bpm)
        name_arr = np.tile(np.array(self.bpm_list), n_turns)
        df_out = pd.DataFrame(
            {
                "turn": turn_arr,
                "name": name_arr,
                "x": x_hat[:, :, 0].reshape(-1),
                "px": x_hat[:, :, 1].reshape(-1),
                "y": x_hat[:, :, 2].reshape(-1),
                "py": x_hat[:, :, 3].reshape(-1),
                "var_x": P_hat[:, :, 0, 0].reshape(-1),
                "var_px": P_hat[:, :, 1, 1].reshape(-1),
                "var_y": P_hat[:, :, 2, 2].reshape(-1),
                "var_py": P_hat[:, :, 3, 3].reshape(-1),
                "weight_x": 1.0,  # For compatibility with old code
                "weight_y": 1.0,  # For compatibility with old code
            }
        )
        # df_out['id'] = meas_df['id']
        # df_out['eidx'] = meas_df['eidx']
        print("Kalman filter complete.")
        return df_out

    def _filter_array(
        self,
        measurements: np.ndarray,
        do_smoothing: bool = False,
        Q_in: list[np.ndarray] | None = None,
        R_in: np.ndarray | None = None,
        return_cross: bool = False,
    ) -> (
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]
    ):
        """
        Core Kalman filter (and optional RTS-smoother) over flattened turn x BPM data.
        measurements: (n_turns, n_bpm, 4)
        Q_in:         optional list of per-BPM process-noise matrices
        R_in:         optional array of per-BPM measurement-noise matrices
        return_cross: if True, also return (x_pr, P_pr) from the forward pass
        do_smoothing: apply Rauch-Tung-Striebel smoothing if True.
        """
        print("Filtering data...")
        # 1) flatten & stack
        n_turns, n_bpm, _ = measurements.shape
        N = n_turns * n_bpm
        bpm_of = np.tile(np.arange(n_bpm), n_turns)
        meas_flat = measurements.reshape(N, 4)
        M_stack = np.stack(self.bpm_mats)
        J_stack = np.stack(self.J_stack)
        R_stack = np.stack(R_in if R_in is not None else self.R)

        special_idx = self.bpm_list.index(self.special_bpm)
        reset_idx = np.where(bpm_of == special_idx)[0]
        boundaries = np.r_[[-1], reset_idx, [N]]
        # 2) call the JIT-compiled core
        x_fwd, P_fwd, x_pr, P_pr = _filter_flat_jit(
            meas_flat,
            bpm_of,
            M_stack,
            J_stack,
            self.sigma_k2,
            R_stack,
            boundaries,
            special_idx,
        )
        # 3) reshape outputs
        x_filt = x_fwd.reshape(n_turns, n_bpm, 4)
        P_filt = P_fwd.reshape(n_turns, n_bpm, 4, 4)

        if not do_smoothing:
            if return_cross:
                return x_filt, P_filt, (x_pr, P_pr)
            else:
                return x_filt, P_filt, None

        # otherwise run your Python-RTS smoother
        print("Running RTS smoother...")
        # after smoothing we get a flat list of length N-1
        x_s_flat, P_s_flat, P_cross_flat = _rts_smooth_flat_jit(
            x_fwd, P_fwd, x_pr, P_pr, bpm_of, M_stack
        )
        if return_cross:
            # Build a (T, B, 4, 4) array by padding then reshaping,
            # then drop the first “dummy” turn to get (T-1, B, 4, 4).
            # 1) pad one zero-matrix at the front so length becomes N
            pad = np.zeros((1, 4, 4))
            P_pad = np.concatenate([pad, P_cross_flat], axis=0)  # shape (N,4,4)

            # 2) reshape into (T, B, 4, 4)
            P_all = P_pad.reshape(n_turns, n_bpm, 4, 4)

            # 3) drop the turn=0 “dummy” (there is no cross from turn -1→0)
            P_cross = P_all[1:]  # now shape is (T-1, B, 4, 4)

            return (
                x_s_flat.reshape(n_turns, n_bpm, 4),
                P_s_flat.reshape(n_turns, n_bpm, 4, 4),
                P_cross,
            )
        else:
            return x_s_flat.reshape(n_turns, n_bpm, 4), P_s_flat.reshape(
                n_turns, n_bpm, 4, 4
            )


@njit(cache=True)
def _rts_smooth_flat_jit(
    x_fwd: np.ndarray,  # (N,4)
    P_fwd: np.ndarray,  # (N,4,4)
    x_pr: np.ndarray,  # (N,4)
    P_pr: np.ndarray,  # (N,4,4)
    bpm_of: np.ndarray,  # (N,)
    M_stack: np.ndarray,  # (n_bpm,4,4)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-compiled Rauch-Tung-Striebel smoother on a flattened sequence.

    Args:
        x_fwd: filtered states x_{k|k} (Nx4)
        P_fwd: filtered covariances P_{k|k} (Nx4x4)
        x_pr:  predicted states x_{k|k-1} (Nx4)
        P_pr:  predicted covariances P_{k|k-1} (Nx4x4)
        bpm_of: BPM index for each flat step (N,)
        M_stack: transport matrices stacked per BPM (n_bpmx4x4)

    Returns:
        x_s: smoothed states x_{k|N} (Nx4)
        P_s: smoothed covariances P_{k|N} (Nx4x4)
    """
    N = x_fwd.shape[0]
    x_s = np.empty(x_fwd.shape)
    P_s = np.empty(P_fwd.shape)
    # allocate cross-covariances for N-1 transitions:
    P_cross_flat = np.empty((N - 1, 4, 4))

    # small regularization constant
    eps = 1e-16

    # initialize at last time step
    x_s[N - 1] = x_fwd[N - 1]
    P_s[N - 1] = P_fwd[N - 1]

    # backward recursion
    for k in range(N - 2, -1, -1):
        # state-transition for step k→k+1
        F = M_stack[bpm_of[k + 1]]  # (4x4)

        # smoothing gain: Ck = P_fwd[k] F^T (P_pr[k+1] + eps I)^{-1}
        PFt = P_fwd[k] @ F.T  # (4x4)
        Ppr_next = P_pr[k + 1] + np.eye(4) * eps
        Ck = np.linalg.solve(Ppr_next.T, PFt.T).T  # (4x4)

        # *** record cross-covariance for transition k→k+1: ***
        #    P_cross[k] = Ck @ P_fwd[k]
        for i in range(4):
            for j in range(4):
                s = 0.0
                for m in range(4):
                    s += Ck[i, m] * P_fwd[k, m, j]
                P_cross_flat[k, i, j] = s

        # state update: x_s[k] = x_fwd[k] + Ck @ (x_s[k+1] - x_pr[k+1])
        for i in range(4):
            acc = x_fwd[k, i]
            for j in range(4):
                acc += Ck[i, j] * (x_s[k + 1, j] - x_pr[k + 1, j])
            x_s[k, i] = acc

        # covariance update: P_s[k] = P_fwd[k] + Ck @ (P_s[k+1] - (P_pr[k+1]+eps I)) @ Ck^T
        diff = P_s[k + 1] - Ppr_next
        temp = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                s = 0.0
                for ell in range(4):
                    s += Ck[i, ell] * diff[ell, j]
                temp[i, j] = s
        for i in range(4):
            for j in range(4):
                s = P_fwd[k, i, j]
                for ell in range(4):
                    s += temp[i, ell] * Ck[j, ell]
                P_s[k, i, j] = s

    # Return all three:
    return x_s, P_s, P_cross_flat


@njit(cache=True, nogil=True)
def _filter_flat_jit(
    meas_flat, bpm_of, M_stack, J_stack, sigma_k2, R_stack, boundaries, special_idx
):
    N = meas_flat.shape[0]
    # allocate outputs
    x_fwd = np.empty((N, 4))
    P_fwd = np.empty((N, 4, 4))
    x_pr = np.empty((N, 4))
    P_pr = np.empty((N, 4, 4))

    # forward pass
    for seg in range(len(boundaries) - 1):
        start = boundaries[seg]
        end = boundaries[seg + 1]

        # --- initialize first point of segment ---
        bpm_idx = 0 if start < 0 else special_idx

        # special BPM reset
        z0 = meas_flat[start]
        x_up0 = z0.copy()
        # diag(diag(R_special))
        P_up0 = np.zeros((4, 4))
        for i in range(4):
            P_up0[i, i] = R_stack[bpm_idx, i, i]

        x_fwd[start] = x_up0
        P_fwd[start] = P_up0
        x_pr[start] = x_up0
        P_pr[start] = P_up0
        x_prev, P_prev = x_up0, P_up0

        for k in range(start + 1, end):
            bpm_idx = bpm_of[k]
            # load 4x4 blocks for this BPM
            M = M_stack[bpm_idx]  # shape (4,4)
            # — Extended-KF: compute state-dependent Q_t —
            # TODO: replace the following rank-1 outer product Q with full Q built across all quadrupole knobs:
            #   1) compute G[q,a] = sum_k J_tensor[i,q,a,b] * x_prev[b]
            #   2) Q[a,b] = sum_q knob_vars[q] * G[q,a] * G[q,b]
            # previous filtered state x_prev
            # (you already have x_prev, P_prev from the loop)
            J = J_stack[bpm_idx]  # shape (4,4)
            Bx = np.zeros(4)
            for i in range(4):
                acc = 0.0
                for j in range(4):
                    acc += J[i, j] * x_prev[j]
                Bx[i] = acc
            # Q = sigma_k2 * np.outer(Bx, Bx)
            Q = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    Q[i, j] = sigma_k2 * Bx[i] * Bx[j]
            R = R_stack[bpm_idx]  # shape (4,4)
            z = meas_flat[k]  # shape (4,)

            # — predict step —
            # x_pred = M @ x_prev
            x_pred = np.zeros(4)
            for i in range(4):
                acc = 0.0
                for j in range(4):
                    acc += M[i, j] * x_prev[j]
                x_pred[i] = acc

            # P_pred = M @ P_prev @ M.T + Q
            # temp = M @ P_prev
            temp = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    s = 0.0
                    for ell in range(4):
                        s += M[i, ell] * P_prev[ell, j]
                    temp[i, j] = s
            # then P_pred = temp @ M.T + Q
            P_pred = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    s = 0.0
                    for ell in range(4):
                        s += temp[i, ell] * M[j, ell]
                    P_pred[i, j] = s + Q[i, j]

            # store for RTS smoother
            x_pr[k] = x_pred
            P_pr[k] = P_pred

            # — update step (measurement) —
            # scalar-measurement gating & sequential update
            x_up = x_pred.copy()
            P_up = P_pred.copy()

            # loop over each of the 4 components
            for j in range(4):
                if not np.isnan(z[j]):
                    # innovation variance S = P_up[j,j] + R[j,j]
                    S = P_up[j, j] + R[j, j]
                    # Kalman gain column K = P_up[:,j] / S
                    K_col = np.zeros(4)
                    for ii in range(4):
                        K_col[ii] = P_up[ii, j] / S

                    # innovation y = z[j] - x_up[j]
                    y_j = z[j] - x_up[j]

                    # state update x_up += K_col * y_j
                    for ii in range(4):
                        x_up[ii] += K_col[ii] * y_j

                    # covariance update P_up = P_up - K_col ⊗ P_up[j,:]
                    # save the j-th row of P_up before it changes
                    P_row = np.zeros(4)
                    for jj in range(4):
                        P_row[jj] = P_up[j, jj]

                    for ii in range(4):
                        for jj in range(4):
                            P_up[ii, jj] -= K_col[ii] * P_row[jj]

            # write back filtered state
            x_fwd[k] = x_up
            P_fwd[k] = P_up

            # seed next
            x_prev = x_up.copy()
            P_prev = P_up.copy()

    return x_fwd, P_fwd, x_pr, P_pr
