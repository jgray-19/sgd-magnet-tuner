from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from tqdm import tqdm

from aba_optimiser.config import (
    POSITION_STD_DEV,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from aba_optimiser.mad.mad_interface import OptimisationMadInterface
from aba_optimiser.physics.phase_space import PhaseSpaceDiagnostics

if TYPE_CHECKING:
    import tfs

LOGGER = logging.getLogger(__name__)


def _compute_q_for_bpm(
    i: int,
    bpm: str,
    bpm_list: list[str],
    n_bpm: int,
    tws: tfs.TfsDataFrame,
    meas_df: pd.DataFrame,
    num_coords_to_track: int,
    knob_names: list[str],
    prepare_derivatives: str,
    sigma_p: np.ndarray,
) -> np.ndarray:
    """Helper function to compute Q for a single BPM, suitable for parallel execution."""
    # Each parallel process needs its own MAD-X interface
    mad_iface = OptimisationMadInterface(
        SEQUENCE_FILE,
        bpm_pattern="BPM",
        discard_mad_output=True,
    )
    mad_iface.mad.load("MADX", SEQ_NAME)
    mad_iface.mad.load("MAD", "damap", "matrix")
    mad_iface.mad.load("MAD.utility", "tblcat")
    mad_iface.mad.load("MAD.element", "marker")
    mad_iface.mad.load("math", "sqrt")
    mad_iface.mad["knob_names"] = knob_names
    mad_iface.mad["num_coords"] = num_coords_to_track
    mad_iface.mad.send(
        """
local coords = {"x","px","y","py","t","pt"}
num_quads = #knob_names
x0_da_base = damap{nv=#coords, np=num_quads, mo=1, po=1, vn=tblcat(coords, knob_names)}
for i, knob in ipairs(knob_names) do
    MADX[knob] = MADX[knob] + x0_da_base[knob]
end
x0_da = table.new(num_coords, 0)
for i=1,num_coords do
    x0_da[i] = x0_da_base:copy()
end
"""
    )

    bpm_meas_df = meas_df[meas_df["name"] == bpm]
    psd = PhaseSpaceDiagnostics(
        bpm,
        x_data=bpm_meas_df["x"].to_numpy(),
        px_data=bpm_meas_df["px"].to_numpy(),
        y_data=bpm_meas_df["y"].to_numpy(),
        py_data=bpm_meas_df["py"].to_numpy(),
        tws=tws,
        num_points=num_coords_to_track,
    )
    x, px, y, py = psd.ellipse_points()
    init_coords = [[x[i], px[i], y[i], py[i], 0, 0] for i in range(len(x))]
    mad_iface.mad["X0"] = init_coords
    for i_coord in range(1, num_coords_to_track + 1):
        mad_iface.mad.send(f"x0_da[{i_coord}]:set0(X0[{i_coord}])")

    prev_b = bpm_list[(i - 1) % n_bpm]
    bpm_track = f"""
_, mflw = track{{sequence={SEQ_NAME}, X0=x0_da, nturn=1, range="{prev_b}/{bpm}"}}
"""
    mad_iface.mad.send(bpm_track + prepare_derivatives)
    jx = mad_iface.mad.recv()
    jpx = mad_iface.mad.recv()
    jy = mad_iface.mad.recv()
    jpy = mad_iface.mad.recv()
    g = np.vstack([jx[:, 0], jpx[:, 0], jy[:, 0], jpy[:, 0]]) / num_coords_to_track
    q_i = np.array(g @ sigma_p @ g.T)
    eps = 1e-16
    q_i += np.eye(4) * eps
    return q_i


class BPMKalmanFilter:
    """
    Self-contained Kalman filter for a 4D transverse state (x, px, y, py).
    On initialisation, it:
      - Connects to MAD-X via MadInterface
      - Retrieves the BPM list from SEQ_NAME
      - Computes BPM-to-BPM 4x4 transfer matrices
      - Computes process noise Q from quadrupole perturbations
      - Sets up measurement noise R, identity observation H, and special BPM
    Call `run(meas_df, x0, P0)` to execute filtering; returns a pandas DataFrame of filtered states.
    """

    def __init__(
        self,
    ):
        # Initialize MAD-X interface
        self.mad_iface = OptimisationMadInterface(
            SEQUENCE_FILE,
            bpm_pattern="BPM",
            # discard_mad_output=False,
            # debug=True,
            # stdout="dbg.out",
        )
        self.tws = self.mad_iface.run_twiss()
        # Retrieve BPM names
        self.bpm_list = self._get_bpm_list()
        self.n_bpm = len(self.bpm_list)
        # Observation and special-case settings
        self.H = np.eye(4)
        self.special_bpm = "BPMYA.5L4.B1"
        # Compute transfer matrices
        self.bpm_mats = self._compute_bpm_mats()
        # Measurement noise (same for all BPMs)
        sigma = [
            POSITION_STD_DEV,
            1e-4,  # px not measured
            POSITION_STD_DEV,
            1e-4,  # py not measured
        ]
        r_mat = np.diag(np.square(np.array(sigma)))
        self.R = np.repeat(r_mat[np.newaxis, :, :], self.n_bpm, axis=0)

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
            m_full = self.mad_iface.mad.recv()
            mats.append(np.array(m_full[:4, :4].copy()))
        return mats

    def _compute_q(
        self, meas_df: tfs.TfsDataFrame, rel_sigma_p: float
    ) -> list[np.ndarray]:
        num_coords_to_track = 10
        quad_strengths = self.mad_iface.receive_knob_values()

        # Define the code that will be executed for each BPM to get the derivatives
        # for each segment. This is constant for all parallel jobs.
        prepare_derivatives = """
local jx = matrix(num_quads,1):zeros()
local jpx = matrix(num_quads,1):zeros()
local jy = matrix(num_quads,1):zeros()
local jpy = matrix(num_quads,1):zeros()
for i, m in ipairs(mflw) do
    local jx_i = matrix(num_quads,1):zeros()
    local jpx_i = matrix(num_quads,1):zeros()
    local jy_i = matrix(num_quads,1):zeros()
    local jpy_i = matrix(num_quads,1):zeros()
    for k,name in ipairs(knob_names) do
        local mono = string.rep("0", 6 + k - 1) .. "1"
        jx_i:set(k,1, m.x:get(mono))
        jpx_i:set(k,1, m.px:get(mono))
        jy_i:set(k,1, m.y:get(mono))
        jpy_i:set(k,1, m.py:get(mono))
    end
    jx = jx + jx_i
    jpx = jpx + jpx_i
    jy = jy + jy_i
    jpy = jpy + jpy_i
end
py:send(jx); py:send(jpx); py:send(jy); py:send(jpy)
"""
        sigma_p = np.diag(np.square(quad_strengths * rel_sigma_p))

        # Parallel computation of Q matrices
        return Parallel(n_jobs=10, backend="loky")(
            delayed(_compute_q_for_bpm)(
                i,
                bpm,
                self.bpm_list,
                self.n_bpm,
                self.tws,
                meas_df,
                num_coords_to_track,
                self.mad_iface.knob_names,
                prepare_derivatives,
                sigma_p,
            )
            for i, bpm in tqdm(
                enumerate(self.bpm_list), total=self.n_bpm, desc="Computing Q"
            )
        )

    def run(
        self,
        meas_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the Kalman filter on a TFS-style measurement DataFrame.
        meas_df should have at least ["name","turn","x","y"]. If px/py are present,
        they are ignored as observations and will be estimated by the filter.
        """
        LOGGER.info(f"Running Kalman filter on {len(meas_df)} measurements")
        self.Q = self._compute_q(meas_df, REL_K1_STD_DEV)

        print("Running Kalman filter...")
        # Unique sorted turns
        turns = range(int(meas_df["turn"].min()), int(meas_df["turn"].max() + 1))
        n_turns = len(turns)
        LOGGER.debug(f"Processing {n_turns} turns for {len(self.bpm_list)} BPMs")

        # Build pivot tables for x and y only
        pivot_x = (
            meas_df.pivot(index="turn", columns="name", values="x")
            .reindex(index=sorted(set(turns)))
            .reindex(columns=self.bpm_list)
        )
        pivot_y = (
            meas_df.pivot(index="turn", columns="name", values="y")
            .reindex(index=sorted(set(turns)))
            .reindex(columns=self.bpm_list)
        )

        x_vals = pivot_x.values
        y_vals = pivot_y.values

        # Assemble measurement array (T, B, 4): fill px/py with NaN so they’re not used
        meas = np.full((n_turns, len(self.bpm_list), 4), np.nan, dtype=float)
        meas[:, :, 0] = x_vals  # x
        meas[:, :, 2] = y_vals  # y

        # Run filter (no smoothing here)
        x_hat, p_hat, _ = self._filter_array(
            meas, do_smoothing=False, return_cross=True
        )

        # Build output DataFrame vectorised
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
            }
        )
        print("Kalman filter complete.")
        return df_out

    def _filter_array(
        self,
        measurements: np.ndarray,
        do_smoothing: bool = False,
        q_in: list[np.ndarray] | None = None,
        r_in: np.ndarray | None = None,
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
        n_points = n_turns * n_bpm
        bpm_of = np.tile(np.arange(n_bpm), n_turns)
        meas_flat = measurements.reshape(n_points, 4)
        m_stack = np.stack(self.bpm_mats)
        q_stack = np.stack(q_in if q_in is not None else self.Q)
        r_stack = np.stack(r_in if r_in is not None else self.R)

        special_idx = self.bpm_list.index(self.special_bpm)
        reset_idx = np.where(bpm_of == special_idx)[0]
        boundaries = np.r_[[-1], reset_idx, [n_points]]

        # Only observe x (0) and y (2)
        obs_idx = np.array([0, 2], dtype=np.int64)

        # 2) call the JIT-compiled core
        x_fwd, p_fwd, x_pr, p_pr = _filter_flat_jit(
            meas_flat,
            bpm_of,
            m_stack,
            q_stack,
            r_stack,
            boundaries,
            special_idx,
            obs_idx,
        )
        # 3) reshape outputs
        x_filt = x_fwd.reshape(n_turns, n_bpm, 4)
        p_filt = p_fwd.reshape(n_turns, n_bpm, 4, 4)

        if not do_smoothing:
            if return_cross:
                return x_filt, p_filt, (x_pr, p_pr)
            return x_filt, p_filt, None

        # otherwise run your Python-RTS smoother
        print("Running RTS smoother...")
        # after smoothing we get a flat list of length N-1
        x_s_flat, p_s_flat, p_cross_flat = _rts_smooth_flat_jit(
            x_fwd, p_fwd, x_pr, p_pr, bpm_of, m_stack
        )
        if return_cross:
            # Build a (T, B, 4, 4) array by padding then reshaping,
            # then drop the first “dummy” turn to get (T-1, B, 4, 4).
            # 1) pad one zero-matrix at the front so length becomes N
            pad = np.zeros((1, 4, 4))
            p_pad = np.concatenate([pad, p_cross_flat], axis=0)  # shape (N,4,4)

            # 2) reshape into (T, B, 4, 4)
            p_all = p_pad.reshape(n_turns, n_bpm, 4, 4)

            # 3) drop the turn=0 “dummy” (there is no cross from turn -1→0)
            p_cross = p_all[1:]  # now shape is (T-1, B, 4, 4)

            return (
                x_s_flat.reshape(n_turns, n_bpm, 4),
                p_s_flat.reshape(n_turns, n_bpm, 4, 4),
                p_cross,
            )
        return x_s_flat.reshape(n_turns, n_bpm, 4), p_s_flat.reshape(
            n_turns, n_bpm, 4, 4
        )


@njit(cache=True)
def _rts_smooth_flat_jit(
    x_fwd: np.ndarray,  # (N,4)
    p_fwd: np.ndarray,  # (N,4,4)
    x_pr: np.ndarray,  # (N,4)
    p_pr: np.ndarray,  # (N,4,4)
    bpm_of: np.ndarray,  # (N,)
    m_stack: np.ndarray,  # (n_bpm,4,4)
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
    P_s = np.empty(p_fwd.shape)
    # allocate cross-covariances for N-1 transitions:
    P_cross_flat = np.empty((N - 1, 4, 4))

    # small regularisation constant
    eps = 1e-16

    # initialize at last time step
    x_s[N - 1] = x_fwd[N - 1]
    P_s[N - 1] = p_fwd[N - 1]

    # backward recursion
    for k in range(N - 2, -1, -1):
        # state-transition for step k→k+1
        F = m_stack[bpm_of[k + 1]]  # (4x4)

        # smoothing gain: Ck = P_fwd[k] F^T (P_pr[k+1] + eps I)^{-1}
        PFt = p_fwd[k] @ F.T  # (4x4)
        Ppr_next = p_pr[k + 1] + np.eye(4) * eps
        Ck = np.linalg.solve(Ppr_next.T, PFt.T).T  # (4x4)

        # *** record cross-covariance for transition k→k+1: ***
        #    P_cross[k] = Ck @ P_fwd[k]
        for i in range(4):
            for j in range(4):
                s = 0.0
                for m in range(4):
                    s += Ck[i, m] * P_s[k + 1, m, j]
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
                s = p_fwd[k, i, j]
                for ell in range(4):
                    s += temp[i, ell] * Ck[j, ell]
                P_s[k, i, j] = s

    # Return all three:
    return x_s, P_s, P_cross_flat


@njit(cache=True, nogil=True)
def _filter_flat_jit(
    meas_flat, bpm_of, M_stack, Q_stack, R_stack, boundaries, special_idx, obs_idx
):
    N = meas_flat.shape[0]
    x_fwd = np.empty((N, 4))
    P_fwd = np.empty((N, 4, 4))
    x_pr = np.empty((N, 4))
    P_pr = np.empty((N, 4, 4))

    for seg in range(len(boundaries) - 1):
        start = boundaries[seg]  # may be -1 for the first segment
        end = boundaries[seg + 1]

        # --- initialize at the FIRST real index of this segment ---
        k0 = start + 1  # first valid index in this segment
        bpm_idx0 = 0 if start < 0 else special_idx

        z0 = meas_flat[k0]
        # Initialize state with measured x,y; px,py set to 0 (or keep previous if you prefer)
        x_up0 = np.zeros(4)
        x_up0[0] = z0[0] if not np.isnan(z0[0]) else 0.0
        x_up0[2] = z0[2] if not np.isnan(z0[2]) else 0.0
        # Diagonal init from R of the init BPM
        P_up0 = np.zeros((4, 4))
        for i in range(4):
            P_up0[i, i] = R_stack[bpm_idx0, i, i]
        # Inflate unobserved momentum variances (px, py) to avoid
        # over-confident initialisation when R contains tiny values.
        # Use a loose prior for slopes (optics-based priors could be used
        # instead if available).
        large_prior = 1e2
        if P_up0[1, 1] < large_prior:
            P_up0[1, 1] = large_prior
        if P_up0[3, 3] < large_prior:
            P_up0[3, 3] = large_prior

        x_fwd[k0] = x_up0
        P_fwd[k0] = P_up0
        x_pr[k0] = x_up0
        P_pr[k0] = P_up0

        x_prev, P_prev = x_up0, P_up0

        # proceed through the rest of the segment
        for k in range(k0 + 1, end):
            bpm_idx = bpm_of[k]
            M = M_stack[bpm_idx]
            Q = Q_stack[bpm_idx]
            R = R_stack[bpm_idx]
            z = meas_flat[k]

            # Predict
            x_pred = np.zeros(4)
            for i in range(4):
                acc = 0.0
                for j in range(4):
                    acc += M[i, j] * x_prev[j]
                x_pred[i] = acc

            temp = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    s = 0.0
                    for ell in range(4):
                        s += M[i, ell] * P_prev[ell, j]
                    temp[i, j] = s
            P_pred = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    s = 0.0
                    for ell in range(4):
                        s += temp[i, ell] * M[j, ell]
                    P_pred[i, j] = s + Q[i, j]

            x_pr[k] = x_pred
            P_pr[k] = P_pred

            # Update — ONLY for observed indices (x=0, y=2), and only if not NaN
            x_up = x_pred.copy()
            P_up = P_pred.copy()

            for idx in range(obs_idx.size):
                j = obs_idx[idx]
                if not np.isnan(z[j]):
                    S = P_up[j, j] + R[j, j]
                    K_col = np.zeros(4)
                    for ii in range(4):
                        K_col[ii] = P_up[ii, j] / S

                    y_j = z[j] - x_up[j]

                    for ii in range(4):
                        x_up[ii] += K_col[ii] * y_j

                    P_row = np.zeros(4)
                    for jj in range(4):
                        P_row[jj] = P_up[j, jj]
                    for ii in range(4):
                        for jj in range(4):
                            P_up[ii, jj] -= K_col[ii] * P_row[jj]

            x_fwd[k] = x_up
            P_fwd[k] = P_up
            x_prev = x_up.copy()
            P_prev = P_up.copy()

    return x_fwd, P_fwd, x_pr, P_pr
