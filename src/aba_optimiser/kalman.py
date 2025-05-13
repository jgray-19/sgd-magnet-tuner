import numpy as np
import pandas as pd
from typing import Tuple
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.config import SEQUENCE_FILE, BPM_RANGE, SEQ_NAME


class BPMKalmanFilter:
    """
    Self-contained Kalman filter for a 4D transverse state (x, px, y, py).
    On initialization, it:
      - Connects to MAD-X via MadInterface
      - Retrieves the BPM list from SEQ_NAME
      - Computes BPM-to-BPM 4×4 transfer matrices
      - Computes process noise Q from quadrupole perturbations
      - Sets up measurement noise R, identity observation H, and special BPM
    Call `run(meas_df, x0, P0)` to execute filtering; returns a pandas DataFrame of filtered states.
    """

    def __init__(self):
        # Initialize MAD-X interface
        self.mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
        # Retrieve BPM names
        self.bpm_list = self._get_bpm_list()
        self.n_bpm = len(self.bpm_list)
        # Observation and special-case settings
        self.H = np.eye(4)
        self.special_bpm = "BPMYA.5L4.B1"
        # Compute transfer matrices
        self.bpm_mats = self._compute_bpm_mats()
        # Compute process noise
        self.Q = self._compute_Q(sigma_p=1e-1)
        # Measurement noise (same for all BPMs)
        sigma = [1e-4, 3e-6, 1e-4, 3e-6]
        R_mat = np.diag(np.square(sigma))
        self.R = [R_mat.copy() for _ in range(self.n_bpm)]

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
            mats.append(M_full[:4, :4].copy())
        return mats

    def _compute_Q(self, sigma_p: float) -> list[np.ndarray]:
        # Extract jacobians from MAD-X
        N = self.n_bpm
        code = f"""
local {SEQ_NAME} in MADX
local damap, matrix in MAD
local tblcat in MAD.utility
local tws = twiss {{sequence = {SEQ_NAME}}}
local coords = {{"x","px","y","py","t","pt"}}
local init = {{1e-5/tws[1].beta11, 0, -1e-5/tws[1].beta22, 0, 0, 0}}
local elems = {{}}
for _, e in {SEQ_NAME}:iter() do
    if e.k1 and e.k1~=0 and e.name:match("MQ%.") then table.insert(elems, e.name) end
end
local nn = #elems
local da = damap{{nv=#coords, np=nn, mo=1, po=1, vn=tblcat(coords, elems)}}
da:set0(init)
local trk = track{{sequence={SEQ_NAME}, X0=da, nturn=1, savemap=true}}
trk:write("trk.tfs")
local jx=matrix(nn,{N}):zeros(); local jpx=matrix(nn,{N}):zeros()
local jy=matrix(nn,{N}):zeros(); local jpy=matrix(nn,{N}):zeros()
for b=1,{N} do local m=trk.__map[b]
  for k,name in ipairs(elems) do
    local mono = string.rep("0", 6 + k - 1) .. "1"
    jx :set(k,b, m.x:get(mono))
    jpx:set(k,b, m.px:get(mono))
    jy :set(k,b, m.y:get(mono))
    jpy:set(k,b, m.py:get(mono))
  end
end
py:send(jx); py:send(jpx); py:send(jy); py:send(jpy); py:send(nn)
"""
        self.mad_iface.mad.send(code)
        jx = self.mad_iface.mad.recv()
        jpx = self.mad_iface.mad.recv()
        jy = self.mad_iface.mad.recv()
        jpy = self.mad_iface.mad.recv()
        nn = self.mad_iface.mad.recv()
        # Build Q per BPM: G Σ_p G^T
        Sigma_p = np.eye(nn) * sigma_p**2
        Q_list = []
        for i in range(N):
            G = np.vstack([jx[:, i], jpx[:, i], jy[:, i], jpy[:, i]])
            Q_list.append(G @ Sigma_p @ G.T)
        return Q_list

    def run(
        self,
        meas_df: pd.DataFrame,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> pd.DataFrame:
        """
        Execute the Kalman filter on a TFS-style measurement DataFrame.
        meas_df must have columns ["name","turns","x","px","y","py"].
        Returns a DataFrame indexed by (turn, bpm) with columns ["x","px","y","py"].
        """
        print("Running Kalman filter...")
        # Unique sorted turns
        turns = range(
            int(meas_df["turn"].min()), int(meas_df["turn"].max() + 1)
        )
        n_turns = len(turns)
        # Build measurement array
        meas = np.zeros((n_turns, self.n_bpm, 4))
        for i, bpm in enumerate(self.bpm_list):
            dfb = meas_df[meas_df["name"] == bpm]
            for j, comp in enumerate(["x", "px", "y", "py"]):
                meas[:, i, j] = dfb[comp].values
        # Run standard filter
        x_hat, _ = self._filter_array(meas, x0, P0)
        # Build output DataFrame
        rows = []
        for ti, turn in enumerate(turns):
            for i, bpm in enumerate(self.bpm_list):
                row = {"turn": turn, "name": bpm}
                for j, comp in enumerate(["x", "px", "y", "py"]):
                    row[comp] = x_hat[ti, i, j]
                rows.append(row)
        df_out = pd.DataFrame(rows)
        print("Kalman filter complete.")
        return df_out

    def _filter_array(
        self,
        measurements: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal array-based filter. Returns x_hat, P_hat arrays.
        """
        n_turns = measurements.shape[0]
        x_hat = np.empty((n_turns, self.n_bpm, 4))
        P_hat = np.empty((n_turns, self.n_bpm, 4, 4))
        x_prev, P_prev = x0.copy(), P0.copy()
        for t in range(n_turns):
            for i in range(self.n_bpm):
                M = self.bpm_mats[i]
                x_pr = M @ x_prev
                P_pr = M @ P_prev @ M.T + self.Q[i]
                z = measurements[t, i]
                if self.bpm_list[i] == self.special_bpm:
                    x_up = z.copy()
                    P_up = np.diag(np.diag(self.R[i]))
                else:
                    valid = ~np.isnan(z)
                    if valid.any():
                        H_eff = self.H[valid]
                        R_eff = self.R[i][np.ix_(valid, valid)]
                        y = z[valid] - H_eff @ x_pr
                        S = H_eff @ P_pr @ H_eff.T + R_eff
                        K = P_pr @ H_eff.T @ np.linalg.inv(S)
                        x_up = x_pr + K @ y
                        P_up = (np.eye(4) - K @ H_eff) @ P_pr
                    else:
                        x_up, P_up = x_pr, P_pr
                x_hat[t, i] = x_up
                P_hat[t, i] = P_up
                x_prev, P_prev = x_up, P_up
        return x_hat, P_hat


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tfs
    from aba_optimiser.config import (
        NOISE_FILE,
        TRACK_DATA_FILE,
        RAMP_UP_TURNS,
        BPM_RANGE,
        SEQ_NAME,
        SEQUENCE_FILE,
    )

    # Instantiate filter (auto-constructs MAD-X, Q, R, etc.)
    kf = BPMKalmanFilter()

    # Load measurement and true data from TFS files
    n_turns = 1000
    meas_df = tfs.read(NOISE_FILE)
    meas_df = meas_df[
        (meas_df["turn"] > RAMP_UP_TURNS) & (meas_df["turn"] <= n_turns + RAMP_UP_TURNS)
    ]
    true_df = tfs.read(TRACK_DATA_FILE)
    true_df = true_df[
        (true_df["turn"] > RAMP_UP_TURNS) & (true_df["turn"] <= n_turns + RAMP_UP_TURNS)
    ]
    # Initial state and covariance
    x0 = np.zeros(4)
    P0 = np.diag([1e-6] * 4)

    # Run Kalman filter
    filtered_df = kf.run(meas_df, x0, P0)
    plot_bpm = "BPM.13R3.B1"
    filtered_df = filtered_df.set_index("name")
    meas_df = meas_df.set_index("name")
    true_df = true_df.set_index("name")


    # Print RMS errors per plane
    for idx, plane in enumerate(["x", "px", "y", "py"]):
        rms_f = np.sqrt(
            np.sum(
                (
                    filtered_df.loc[plot_bpm][plane].values
                    - true_df.loc[plot_bpm][plane].values
                )
                ** 2
            )
        )
        rms_raw = np.sqrt(
            np.sum(
                (
                    meas_df.loc[plot_bpm][plane].values
                    - true_df.loc[plot_bpm][plane].values
                )
                ** 2
            )
        )
        print(f"RMS filtered {plane}: {rms_f:.2e}, raw: {rms_raw:.2e}")

    # Plot phase-space for a sample BPM
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(
        filtered_df.loc[plot_bpm]["x"],
        filtered_df.loc[plot_bpm]["px"],
        s=1,
        label="Kalman",
    )
    plt.scatter(
        true_df.loc[plot_bpm]["x"], true_df.loc[plot_bpm]["px"], s=1, label="True"
    )
    plt.scatter(
        meas_df.loc[plot_bpm]["x"], meas_df.loc[plot_bpm]["px"], s=1, label="Meas"
    )
    plt.xlabel("x")
    plt.ylabel("px")
    plt.title("x-px")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(
        filtered_df.loc[plot_bpm]["y"], filtered_df.loc[plot_bpm]["py"], s=1, label="Kalman"
    )
    plt.scatter(true_df.loc[plot_bpm]["y"], true_df.loc[plot_bpm]["py"], s=1, label="True")
    plt.scatter(meas_df.loc[plot_bpm]["y"], meas_df.loc[plot_bpm]["py"], s=1, label="Meas")
    plt.xlabel("y")
    plt.ylabel("py")
    plt.title("y-py")
    plt.legend()
    plt.tight_layout()
    plt.show()
