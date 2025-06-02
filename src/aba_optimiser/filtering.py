import os

import numpy as np
import pandas as pd
import tfs
from numba import njit
from tqdm import tqdm

from aba_optimiser.config import (
    ACD_ON,
    BPM_RANGE,
    FLATTOP_TURNS,
    MOMENTUM_STD_DEV,
    NOISE_FILE,
    POSITION_STD_DEV,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
    STD_CUT,
    WINDOWS,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.phase_space import PhaseSpaceDiagnostics


class BPMKalmanFilter:
    """
    Self-contained Kalman filter for a 4D transverse state (x, px, y, py).
    On initialization, it:
      - Connects to MAD-X via MadInterface
      - Retrieves the BPM list from SEQ_NAME
      - Computes BPM-to-BPM 4x4 transfer matrices
      - Computes process noise Q from quadrupole perturbations
      - Sets up measurement noise R, identity observation H, and special BPM
    Call `run(meas_df, x0, P0)` to execute filtering; returns a pandas DataFrame of filtered states.
    """

    def __init__(
            self,
            override_q: bool = True,
            q_file: str = "data/Q_matrix.npy",
        ):          
        # Initialize MAD-X interface
        self.mad_iface = MadInterface(
            SEQUENCE_FILE, BPM_RANGE, #discard_mad_output=False
        )
        # Retrieve BPM names
        self.bpm_list = self._get_bpm_list()
        self.n_bpm = len(self.bpm_list)
        # Observation and special-case settings
        self.H = np.eye(4)
        self.special_bpm = "BPMYA.5L4.B1"
        # Compute transfer matrices
        self.bpm_mats = self._compute_bpm_mats()
        # Compute or load process noise
        if not override_q and os.path.exists(q_file):
            print(f"Loading Q from {q_file}")
            loaded_q = np.load(q_file, allow_pickle=True)
            self.Q = [np.array(qi) for qi in loaded_q]
            print("Q loaded.")
        else:
            print("Calculating Q...")
            self.Q = self._compute_Q(rel_sigma_p=REL_K1_STD_DEV)
            np.save(q_file, np.array(self.Q, dtype=object))
            print(f"Q saved to {q_file}.")
        
        # Measurement noise (same for all BPMs)
        sigma = [POSITION_STD_DEV, 1e3*MOMENTUM_STD_DEV, POSITION_STD_DEV, 1e3*MOMENTUM_STD_DEV]
        R_mat = np.diag(np.square(np.array(sigma)))
        self.R = np.repeat(R_mat[np.newaxis, :, :], self.n_bpm, axis=0)

    def _get_bpm_list(self) -> list[str]:
        code = f"""
local {SEQ_NAME} in MADX
local bpm_list = {{}}
for _, elm in {SEQ_NAME}:iter() do
    if elm.name:match("BPM") then table.insert(bpm_list, elm.name) end
end
py:send(bpm_list)
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
            mats.append(np.array(M_full[:4, :4].copy()))
        return mats

    def _compute_Q(self, rel_sigma_p: float) -> list[np.ndarray]:
        # Compute Q for each BPM segment, similar to _compute_bpm_mats
        Q_list = []
        # Load MAD-NG modules and functions (not using local variables)
        self.mad_iface.mad.load("MADX", SEQ_NAME)
        self.mad_iface.mad.load("MAD", "damap", "matrix")
        self.mad_iface.mad.load("MAD.utility", "tblcat")
        self.mad_iface.mad.load("MAD.element", "marker")
        self.mad_iface.mad.load("math", "sqrt")
        # Send the setup code to MAD-NG
        quad_strengths = self.mad_iface.receive_knob_values()
        self.mad_iface.mad.send(f"""
local tws = twiss {{sequence = {SEQ_NAME}}}
local coords = {{"x","px","y","py","t","pt"}}
X0 = {{ 
    5e-3 / sqrt(tws["$start"].beta11),
    -1e-6,
    5e-3 / sqrt(tws["$start"].beta22),
    -1e-6,
    0,
    0
}}
num_quads = #knob_names
x0_da = damap{{nv=#coords, np=num_quads, mo=1, po=1, vn=tblcat(coords, knob_names)}}
for i, knob in ipairs(knob_names) do
    {SEQ_NAME}[knob] = {SEQ_NAME}[knob] + x0_da[knob]
end
x0_da:set0(X0)
""")
        # Define the code that will be executed for each BPM to get the derivatives
        # for each segment
        prepare_derivatives = """
local m = mflw[1]
local jx=matrix(num_quads,1):zeros(); local jpx=matrix(num_quads,1):zeros()
local jy=matrix(num_quads,1):zeros(); local jpy=matrix(num_quads,1):zeros()
for k,name in ipairs(knob_names) do
    local mono = string.rep("0", 6 + k - 1) .. "1"
    jx :set(k,1, m.x:get(mono))
    jpx:set(k,1, m.px:get(mono))
    jy :set(k,1, m.y:get(mono))
    jpy:set(k,1, m.py:get(mono))
end
py:send(jx); py:send(jpx); py:send(jy); py:send(jpy)
"""
        Sigma_p = np.diag(np.square(quad_strengths * rel_sigma_p))
        print("Sigma_p:", Sigma_p)
        for i, bpm in enumerate(self.bpm_list):
            prev_b = self.bpm_list[(i - 1) % self.n_bpm]
            bpm_track = f"""
_, mflw = track{{sequence={SEQ_NAME}, X0=x0_da, nturn=1, range=\"{prev_b}/{bpm}\"}}
"""
            self.mad_iface.mad.send(bpm_track + prepare_derivatives)
            jx = self.mad_iface.mad.recv()
            jpx = self.mad_iface.mad.recv()
            jy = self.mad_iface.mad.recv()
            jpy = self.mad_iface.mad.recv()
            G = np.vstack([jx[:, 0], jpx[:, 0], jy[:, 0], jpy[:, 0]])
            Q_i = np.array(G @ Sigma_p @ G.T)
            eps = 1e-16
            Q_i += np.eye(4) * eps
            Q_list.append(Q_i)
        return Q_list

    def run(
        self,
        meas_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute the Kalman filter on a TFS-style measurement DataFrame.
        meas_df must have columns ["name","turns","x","px","y","py"].
        Returns a DataFrame indexed by (turn, bpm) with columns ["x","px","y","py"].
        """
        print("Running Kalman filter...")
        # Unique sorted turns
        turns = range(int(meas_df["turn"].min()), int(meas_df["turn"].max() + 1))
        n_turns = len(turns)
        # Build measurement array
        pivot = meas_df.pivot(index="turn", columns="name", values=["x", "px", "y", "py"])
        pivot = pivot.reindex(columns=self.bpm_list, level=1)  # ensure correct BPM order
        meas = np.stack(
            [pivot[comp].loc[sorted(pivot.index)].values for comp in ["x", "px", "y", "py"]],
            axis=-1
        )
        # ------------------------------------------------------------------
        #  EM ITERATION LOOP
        # ------------------------------------------------------------------
        # working copies of Q/R in double precision for EM stability
        x_hat, P_hat, _ = self._filter_array(
            meas,
            do_smoothing=False,
            return_cross=True
        )

        # Build output DataFrame vectorized
        n_turns, n_bpm = x_hat.shape[0], len(self.bpm_list)
        turn_arr = np.repeat(np.array(list(turns)), n_bpm)
        name_arr = np.tile(np.array(self.bpm_list), n_turns)
        df_out = pd.DataFrame({
            "turn": turn_arr,
            "name": name_arr,
            "x": x_hat[:,:,0].reshape(-1),
            "px": x_hat[:,:,1].reshape(-1),
            "y": x_hat[:,:,2].reshape(-1),
            "py": x_hat[:,:,3].reshape(-1),
            "var_x": P_hat[:,:,0,0].reshape(-1),
            "var_px": P_hat[:,:,1,1].reshape(-1),
            "var_y": P_hat[:,:,2,2].reshape(-1),
            "var_py": P_hat[:,:,3,3].reshape(-1),
        })
        df_out['id'] = meas_df['id']
        df_out['eidx'] = meas_df['eidx']
        print("Kalman filter complete.")
        return df_out

    def _filter_array(
        self,
        measurements: np.ndarray,
        do_smoothing: bool      = False,
        Q_in: list[np.ndarray] | None = None,
        R_in: np.ndarray       | None = None,
        return_cross: bool     = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
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
        bpm_of    = np.tile(np.arange(n_bpm), n_turns)
        meas_flat = measurements.reshape(N, 4)
        M_stack   = np.stack(self.bpm_mats)
        Q_stack   = np.stack(Q_in if Q_in is not None else self.Q)
        R_stack   = np.stack(R_in if R_in is not None else self.R)
        
        special_idx = self.bpm_list.index(self.special_bpm)
        reset_idx   = np.where(bpm_of == special_idx)[0]
        boundaries  = np.r_[[-1], reset_idx, [N]]
        # 2) call the JIT-compiled core
        x_fwd, P_fwd, x_pr, P_pr = _filter_flat_jit(
            meas_flat, bpm_of, M_stack, Q_stack, R_stack,
            boundaries, special_idx
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
            P_cross = P_all[1:]   # now shape is (T-1, B, 4, 4)

            return (
                x_s_flat.reshape(n_turns, n_bpm, 4),
                P_s_flat.reshape(n_turns, n_bpm, 4, 4),
                P_cross
            )
        else:
            return x_s_flat.reshape(n_turns, n_bpm, 4), P_s_flat.reshape(n_turns, n_bpm, 4, 4)

@njit(cache=True)
def _rts_smooth_flat_jit(
    x_fwd: np.ndarray,   # (N,4)
    P_fwd: np.ndarray,   # (N,4,4)
    x_pr:  np.ndarray,   # (N,4)
    P_pr:  np.ndarray,   # (N,4,4)
    bpm_of: np.ndarray,  # (N,)
    M_stack: np.ndarray  # (n_bpm,4,4)
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
    P_cross_flat = np.empty((N-1, 4, 4))

    # small regularization constant
    eps = 1e-16

    # initialize at last time step
    x_s[N-1] = x_fwd[N-1]
    P_s[N-1] = P_fwd[N-1]
    
    # backward recursion
    for k in range(N-2, -1, -1):
        # state-transition for step k→k+1
        F = M_stack[bpm_of[k+1]]      # (4x4)
    
        # smoothing gain: Ck = P_fwd[k] F^T (P_pr[k+1] + eps I)^{-1}
        PFt = P_fwd[k] @ F.T           # (4x4)
        Ppr_next = P_pr[k+1] + np.eye(4) * eps
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
                acc += Ck[i, j] * (x_s[k+1, j] - x_pr[k+1, j])
            x_s[k, i] = acc

        # covariance update: P_s[k] = P_fwd[k] + Ck @ (P_s[k+1] - (P_pr[k+1]+eps I)) @ Ck^T
        diff = P_s[k+1] - Ppr_next
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
def _filter_flat_jit(meas_flat, bpm_of, M_stack, Q_stack, R_stack,
                    boundaries, special_idx):
    N = meas_flat.shape[0]
    # allocate outputs
    x_fwd = np.empty((N, 4))
    P_fwd = np.empty((N, 4, 4))
    x_pr  = np.empty((N, 4))
    P_pr  = np.empty((N, 4, 4))

    # forward pass
    for seg in range(len(boundaries)-1):
        start = boundaries[seg]
        end   = boundaries[seg+1]

        # --- initialize first point of segment ---
        if start < 0:
            bpm_idx = 0
        else:
            bpm_idx = special_idx
        
        # special BPM reset
        z0    = meas_flat[start]
        x_up0 = z0.copy()
        # diag(diag(R_special))
        P_up0 = np.zeros((4,4))
        for i in range(4):
            P_up0[i,i] = R_stack[bpm_idx,i,i]

        x_fwd[start] = x_up0
        P_fwd[start] = P_up0
        x_pr[start]  = x_up0
        P_pr[start]  = P_up0
        x_prev, P_prev = x_up0, P_up0

        for k in range(start + 1, end):
            bpm_idx = bpm_of[k]
            # load 4x4 blocks for this BPM
            M = M_stack[bpm_idx]     # shape (4,4)
            Q = Q_stack[bpm_idx]     # shape (4,4)
            R = R_stack[bpm_idx]     # shape (4,4)
            z = meas_flat[k]         # shape (4,)

            # — predict step —
            # x_pred = M @ x_prev
            x_pred = np.zeros(4)
            for i in range(4):
                acc = 0.0
                for j in range(4):
                    acc += M[i,j] * x_prev[j]
                x_pred[i] = acc

            # P_pred = M @ P_prev @ M.T + Q
            # temp = M @ P_prev
            temp = np.zeros((4,4))
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

def filter_noisy_data(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index(['turn', 'name'], inplace=True)
    
    # Get Twiss data
    bpm_names = []
    for start_bpm, _ in WINDOWS:
        bpm_names.append(start_bpm)
    bpm_names = list(set(bpm_names))

    # Pre-split dataframe for efficiency
    bpm_groups = {bpm: data.xs(bpm, level='name') for bpm in bpm_names if bpm in data.index.get_level_values('name')}

    failed_turns = set()
    # Process remaining BPMs
    for bpm in tqdm(bpm_names, desc="Filtering BPMs"):
        bpm_data = bpm_groups[bpm]
        if bpm_data.empty:
            print(f"Warning: No data available for BPM {bpm}")
            continue

        # Subset to existing rows after drop
        bpm_data = bpm_data[bpm_data.index.isin(data.index.get_level_values("turn"))]

        if bpm_data.empty:
            continue

        x, px, y, py = bpm_data["x"].values, bpm_data["px"].values, bpm_data["y"].values, bpm_data["py"].values
        ps_diag = PhaseSpaceDiagnostics(bpm, x, px, y, py)
        std_x, std_y = ps_diag.compute_residuals()

        gamma_x = (1 + ps_diag.alfax**2) / ps_diag.betax
        inv_x = gamma_x * x**2 + 2 * ps_diag.alfax * x * px + ps_diag.betax * px**2
        gamma_y = (1 + ps_diag.alfay**2) / ps_diag.betay
        inv_y = gamma_y * y**2 + 2 * ps_diag.alfay * y * py + ps_diag.betay * py**2

        residual_x = (inv_x/2 - ps_diag.emit_x) / ps_diag.emit_x
        residual_y = (inv_y/2 - ps_diag.emit_y) / ps_diag.emit_y

        full_idx = bpm_data.index
        fail_x = full_idx[np.abs(residual_x) > std_x * STD_CUT]
        fail_y = full_idx[np.abs(residual_y) > std_y * STD_CUT]
        failed_turns.update(fail_x)
        failed_turns.update(fail_y)

    data.reset_index(inplace=True)
    filtered = data[~data['turn'].isin(failed_turns)]
    return filtered

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tfs

    from aba_optimiser.config import (
        BPM_RANGE,
        NOISE_FILE,
        RAMP_UP_TURNS,
        SEQ_NAME,
        SEQUENCE_FILE,
        TRACK_DATA_FILE,
    )

    # Instantiate filter (auto-constructs MAD-X, Q, R, etc.)
    kf = BPMKalmanFilter()

    # Load measurement and true data from TFS files
    n_turns = FLATTOP_TURNS // 2
    meas_df = tfs.read(NOISE_FILE)
    true_df = tfs.read(TRACK_DATA_FILE)
    if ACD_ON:
        meas_df = meas_df[
            (meas_df["turn"] > RAMP_UP_TURNS)
            & (meas_df["turn"] <= n_turns + RAMP_UP_TURNS)
        ]
        true_df = true_df[
            (true_df["turn"] > RAMP_UP_TURNS)
            & (true_df["turn"] <= n_turns + RAMP_UP_TURNS)
        ]
    else:
        meas_df = meas_df[meas_df["turn"] <= n_turns]
        true_df = true_df[true_df["turn"] <= n_turns]
    # Run Kalman filter
    filtered_df = kf.run(meas_df)
    plot_bpm = "BPM.13R3.B1"
    filtered_df = filtered_df.set_index("name")
    meas_df = meas_df.set_index("name")
    true_df = true_df.set_index("name")

    # Print RMS errors per plane
    for idx, plane in enumerate(["x", "px", "y", "py"]):
        filtered_diff = filtered_df[plane] - true_df[plane]
        meas_diff = meas_df[plane] - true_df[plane]
        rms_f = filtered_diff.to_numpy().std()
        rms_raw = meas_diff.to_numpy().std()
        print(f"RMS filtered {plane}: {rms_f:.2e}, raw: {rms_raw:.2e}")

    # Plot phase-space for a sample BPM
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(
        meas_df.loc[plot_bpm]["x"], meas_df.loc[plot_bpm]["px"], s=1, label="Meas"
    )
    plt.scatter(
        filtered_df.loc[plot_bpm]["x"],
        filtered_df.loc[plot_bpm]["px"],
        s=1,
        label="Kalman",
    )
    plt.scatter(
        true_df.loc[plot_bpm]["x"], true_df.loc[plot_bpm]["px"], s=1, label="True"
    )
    plt.xlabel("x")
    plt.ylabel("px")
    plt.title("x-px")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(
        meas_df.loc[plot_bpm]["y"], meas_df.loc[plot_bpm]["py"], s=1, label="Meas"
    )
    plt.scatter(
        filtered_df.loc[plot_bpm]["y"],
        filtered_df.loc[plot_bpm]["py"],
        s=1,
        label="Kalman",
    )
    plt.scatter(
        true_df.loc[plot_bpm]["y"], true_df.loc[plot_bpm]["py"], s=1, label="True"
    )
    plt.xlabel("y")
    plt.ylabel("py")
    plt.title("y-py")
    plt.legend()
    plt.tight_layout()
    plt.show()
