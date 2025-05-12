"""
bpm_kalman.py
=============
Light-weight Kalman filter that marches *segment by segment* (BPM → BPM)
around a circular accelerator.  Designed for a 4-D transverse state
(x, px, y, py) with two position readings (x, y) at each BPM.

Any machine-specific information—how you build the transfer matrices or
load the measured data—should be provided by the calling script.
"""

import numpy as np
from typing import Sequence, Tuple, Optional
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.config import SEQUENCE_FILE, BPM_RANGE, SEQ_NAME, NOISE_FILE, TRACK_DATA_FILE, RAMP_UP_TURNS
import tfs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# ONE BPM in your lattice sits after the AC dipole ---------------
# (keep or remove the special-case block as you wish)
# ---------------------------------------------------------------
BPM_AFTER_ACD = "BPMYA.5L4.B1"


def bpm_kalman_filter(
    bpm_mats: Sequence[np.ndarray],     # 4*4 segment matrices
    # T_mats: np.ndarray,                     # 4*4*4 tensor of second-order terms
    measurements: np.ndarray,          # shape (n_turns, N_BPM, 4)
    Q: np.ndarray | Sequence[np.ndarray],
    R: np.ndarray | Sequence[np.ndarray],
    x0: np.ndarray,                    # 4-element state before BPM 0
    P0: np.ndarray,                    # 4*4 covariance
    bpm_list: Sequence[str],
    H: Optional[np.ndarray] = None,    # 4*4 observation matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear Kalman filter that runs BPM-to-BPM and handles *4-D* readings
    (x, px, y, py).  NaNs in the measurement array mean “component not
    available for this turn → skip that component in the update”.
    """
    bpm_mats = list(bpm_mats)
    n_bpm    = len(bpm_mats)
    n_turns  = measurements.shape[0]

    # Full observation matrix: BPM measures every state component
    if H is None:
        H = np.eye(4)

    # Expand Q and R if a single matrix was supplied
    if isinstance(Q, np.ndarray) and Q.ndim == 2:
        Q = [Q] * n_bpm
    if isinstance(R, np.ndarray) and R.ndim == 2:
        R = [R] * n_bpm

    # Output buffers
    x_hat = np.empty((n_turns, n_bpm, 4))
    P_hat = np.empty((n_turns, n_bpm, 4, 4))

    # x_prev = measurements[0, 0, :]       # (4,)
    x_prev = x0.copy()
    P_prev = P0.copy()

    # -----------------------------------------------------------
    # main loop:   turn  →  BPM 0 … BPM N-1 → next turn
    # -----------------------------------------------------------
    for k_turn in range(n_turns):
        for i_bpm in range(n_bpm):
            # 1. PREDICT one segment
            M_i  = bpm_mats[i_bpm]
            Q_i  = Q[i_bpm]
            x_pr = M_i @ x_prev
            P_pr = M_i @ P_prev @ M_i.T + Q_i

            # 2. UPDATE with measurement (may contain NaNs)
            z_full = measurements[k_turn, i_bpm]       # (4,)

            if bpm_list[i_bpm] == BPM_AFTER_ACD:
                # ***** special BPM after AC dipole *****
                x_up = z_full.copy()                   # trust the 4-D reading
                P_up = np.diag(np.diag(R[i_bpm]))      # covariance = BPM noise
            else:
                # Mask out NaNs so we only use the measured components
                valid = ~np.isnan(z_full)              # boolean mask length 4
                if valid.any():
                    H_eff  = H[valid]                  # rows that correspond
                    R_eff  = R[i_bpm][np.ix_(valid, valid)]
                    z      = z_full[valid]
                    y      = z - H_eff @ x_pr          # innovation
                    S      = H_eff @ P_pr @ H_eff.T + R_eff
                    K      = P_pr @ H_eff.T @ np.linalg.inv(S)
                    x_up   = x_pr + K @ y
                    P_up   = (np.eye(4) - K @ H_eff) @ P_pr
                else:
                    # nothing measured → keep prediction
                    x_up, P_up = x_pr, P_pr

            # save & feed forward
            x_hat[k_turn, i_bpm] = x_up
            P_hat[k_turn, i_bpm] = P_up
            x_prev, P_prev       = x_up, P_up

    return x_hat, P_hat

# ----------------------------------------------------------------------
# Example stub - remove or adapt for your testing framework
# ----------------------------------------------------------------------

if __name__ == "__main__":
    mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
    code = f"""
local {SEQ_NAME} in MADX
bpm_list = {{}}
for _, elm in {SEQ_NAME}:iter() do
    if elm.name:match("BPM") then 
        table.insert(bpm_list, elm.name)
    end
end
py:send(bpm_list)
"""
    mad_iface.mad.send(code)
    bpm_list = mad_iface.mad.recv()
    N_BPM   = len(bpm_list)
    
    # Retrieve the map between every BPM
    bpm_maps = []
    T_mats = []
    for i, bpm in enumerate(bpm_list):
        bpm_before = bpm_list[(i - 1) % N_BPM]  # previous BPM
        code = f"""
local {SEQ_NAME} in MADX
local trk, mflw = track{{
    sequence = {SEQ_NAME},
    range = "{bpm_before}/{bpm}",
    turn = 1,
    mapdef = 2
}}
py:send(mflw[1]:get1())
""" #+ """
# local matrix in MAD

# -- 1) set up the coordinate names and empty 4*4*4 tensor
# local coords = {"x","px","y","py"}   -- x_1,x_2,x_3,x_4
# local T = {}                         -- will hold T[comp][j][k]
# for _, comp in ipairs(coords) do
#   T[comp] = matrix(4,4):zeros()     -- zero 4*4 for each output comp
# end

# -- 2) loop over all second-order monomials j <= k
# for j = 1,4 do
#   for k = j,4 do

#     -- build the monomial string "abcd"
#     -- where a = exponent of x_1, b of x_2, c of x_3, d of x_4
#     local exps = {"0","0","0","0"}
#     if j == k then
#       exps[j] = "2"                  -- e.g. "2000"
#     else
#       exps[j] = "1"
#       exps[k] = "1"                  -- e.g. "1100"
#     end
#     local mono = table.concat(exps)  -- e.g. "1100", "0020"

#     -- 3) read out each component's coefficient
#     for i, comp in ipairs(coords) do
#       local C = mflw[1][comp]:get(mono)  -- T_{i,j,k}
#       T[comp]:set(j,k,C)
#       if j ~= k then
#         T[comp]:set(k,j,C)
#       end
#     end

#   end
# end
# for _, comp in ipairs(coords) do
#     py:send(T[comp])
# end
# """
        mad_iface.mad.send(code)
        M = mad_iface.mad.recv()
        # T = np.zeros((4, 4, 4))
        # for i, comp in enumerate(["x", "px", "y", "py"]):
        #     T[i, :, :] = mad_iface.mad.recv()

        
        # Convert the 6x6 matrix to 4x4, by extracting the top left 4x4
        # and ignoring the last two rows and columns (these are t and pt)
        M_4x4 = M[0:4, 0:4].copy()
        bpm_maps.append(M_4x4)
        # T_mats.append(T)

    # Lets calculate the covariance matrix Q for each BPM
    code = f"""
local {SEQ_NAME} in MADX
local damap, matrix in MAD
local tblcat in MAD.utility
""" + f"""
local tws = twiss {{sequence = {SEQ_NAME}}}
local init_coords = {{1e-5/tws[1].beta11, 0, -1e-5/tws[1].beta22, 0, 0, 0}}

local elem_names = {{}}
for i, elm, s, ds in {SEQ_NAME}:iter() do
    if elm.k1 and elm.k1 ~= 0 and elm.name:match("MQ%.") then
        -- Check if the element is a main quadrupole
        table.insert(elem_names, elm.name)
        
    end
end
local num_knobs = #elem_names
""" + f"""
knob_monomials = {{}}
da_x0 = damap{{nv=#coord_names, np=num_knobs, mo=1, po=1, vn=tblcat(coord_names, elem_names)}}
for i,param in ipairs(elem_names) do
    {SEQ_NAME}[param].k1 = {SEQ_NAME}[param].k1 + da_x0[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
da_x0:set0(init_coords)
""" + f"""
local trk = track{{sequence={SEQ_NAME}, X0=da_x0,
                    nturn=1, savemap=true}}
trk:write("trk.tfs")
assert(trk.lost == 0, "Lost particle in track")
""" + f"""
-- build jacobian matrices jx, jy -----------------------------
local jx  = matrix(num_knobs, {N_BPM}):zeros()
local jpx = matrix(num_knobs, {N_BPM}):zeros()
local jy  = matrix(num_knobs, {N_BPM}):zeros()
local jpy = matrix(num_knobs, {N_BPM}):zeros()
for b = 1, {N_BPM} do
    local map = trk.__map[b]
    for k, param in ipairs(elem_names) do
        local mono = knob_monomials[param]
        jx :set(k,b, map.x :get(mono))
        jpx:set(k,b, map.px:get(mono))
        jy :set(k,b, map.y :get(mono))
        jpy:set(k,b, map.py:get(mono))
    end
end
py:send(jx)
py:send(jpx)
py:send(jy)
py:send(jpy)
py:send(num_knobs)
"""
    mad_iface.mad.send(code)
    jx = mad_iface.mad.recv()
    jpx = mad_iface.mad.recv()
    jy = mad_iface.mad.recv()
    jpy = mad_iface.mad.recv()
    num_knobs = mad_iface.mad.recv()
    
    q_list = []
    sigma_p = 1e-3
    Sigma_p = np.eye(num_knobs) * sigma_p**2
    sum_q = 0
    for i in range(N_BPM):
        G = np.zeros((4, num_knobs))
        G[0, :] = jx[:, i]
        G[1, :] = jpx[:, i]
        G[2, :] = jy[:, i]
        G[3, :] = jpy[:, i]
        Q_i = G @ Sigma_p @ G.T
        
        q_list.append(Q_i)
        sum_q += np.sum(Q_i)    

    # 2) Get the number of BPMs in the sequence    # Get the number of BPMs in the sequence
    n_turns = 1000                   # how many turns to process

    # Synthetic 1 mm beta-matched beam size just for a demo
    sigma_x = sigma_y = 1e-3
    meas_df = tfs.read(NOISE_FILE, index="turn")
    meas_df = meas_df[(meas_df.index > RAMP_UP_TURNS) & (meas_df.index <= n_turns + RAMP_UP_TURNS)]

    true_data_df = tfs.read(TRACK_DATA_FILE, index="turn")
    true_data_df = true_data_df[(true_data_df.index > RAMP_UP_TURNS) & (true_data_df.index <= n_turns + RAMP_UP_TURNS)]

    meas = np.zeros((n_turns, N_BPM, 4))
    for i in range(N_BPM):
        bpm_df = meas_df[meas_df["name"] == bpm_list[i]]
        meas[:, i, 0] = bpm_df["x"].values
        meas[:, i, 1] = bpm_df["px"].values
        meas[:, i, 2] = bpm_df["y"].values
        meas[:, i, 3] = bpm_df["py"].values

    true_data = np.zeros((n_turns, N_BPM, 4))
    for i in range(N_BPM):
        bpm_df = true_data_df[true_data_df["name"] == bpm_list[i]]
        true_data[:, i, 0] = bpm_df["x"].values
        true_data[:, i, 1] = bpm_df["px"].values
        true_data[:, i, 2] = bpm_df["y"].values
        true_data[:, i, 3] = bpm_df["py"].values

    # 3) Covariances
    Q  = (1e-3)**2 * np.diag([sigma_x**2, (sigma_x/10)**2,
                              sigma_y**2, (sigma_y/10)**2])
    #      x         p_x         y         p_y
    sigma = [1e-4,   3e-6,      1e-4,     3e-6]   # metres - radians - metres - radians
    R = np.diag(np.square(sigma))                 # shape (4, 4)
    
    x0 = np.zeros(4)                     # initial state
    P0 = np.diag([1e-6]*4)

    # 4) Run filter
    x_hat, P_hat = bpm_kalman_filter(
        bpm_mats=bpm_maps,
        # T_mats=T_mats,
        measurements=meas,
        Q=q_list,
        R=R,
        x0=x0,
        P0=P0,
        bpm_list=bpm_list,
    )

    # 5) Check results
    for (plane, i) in zip(["x", "px", "y", "py"], range(4)):
        print(f"RMS filtered {plane} error:",
              np.sqrt(np.mean((x_hat[..., i] - true_data[..., i])**2)))
        print(f"RMS raw      {plane} error:",
              np.sqrt(np.mean((meas[..., i] - true_data[..., i])**2)))
    
    bpm_index = bpm_list.index("BPM.13R3.B1")
    # 6) Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(x_hat[:, bpm_index, 0],
                x_hat[:, bpm_index, 1],
                s=1, c="blue", label="Kalman")
    plt.scatter(true_data[:, bpm_index, 0],
                true_data[:, bpm_index, 1],
                s=1, c="red", label="True")
    plt.scatter(meas[:, bpm_index, 0],
                meas[:, bpm_index, 1],
                s=1, c="green", label="Measured")
    plt.title("x, px phase space")
    plt.xlabel("x")
    plt.ylabel("px")
    plt.legend(loc="upper right")
    plt.subplot(1, 2, 2)
    plt.scatter(x_hat[:, bpm_index, 2],
                x_hat[:, bpm_index, 3],
                s=1, c="blue", label="Kalman")
    plt.scatter(true_data[:, bpm_index, 2],
                true_data[:, bpm_index, 3],
                s=1, c="red", label="True")
    plt.scatter(meas[:, bpm_index, 2],
                meas[:, bpm_index, 3],
                s=1, c="green", label="Measured")
    plt.title("y, py phase space")
    plt.xlabel("y")
    plt.ylabel("py")
    plt.legend()
    plt.show()