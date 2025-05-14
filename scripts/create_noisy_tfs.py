import tfs
import numpy as np
import pandas as pd

from aba_optimiser.config import (
    BPM_RANGE,
    FILTERED_FILE,
    FILTER_DATA,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
    NOISE_FILE,
    TRUE_STRENGTHS,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import read_knobs
from aba_optimiser.kalman_numba import BPMKalmanFilter

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
BPM_NOISE = 1e-4      # BPM resolution (turn-by-turn)
TARGET     = 0.25     # desired phase separation in turns (π/2)

# -------------------------------------------------------------------------
# HELPER: find the previous BPM ~TARGET turns upstream
# -------------------------------------------------------------------------

def prev_bpm_to_pi_2(mu: pd.Series, Q1: float, target: float = TARGET) -> pd.DataFrame:
    """
    For each BPM_i find the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to `target` turns.
    Returns a DataFrame indexed by BPM_i with columns:
      - prev_bpm : name of BPM_j
      - delta    : (mu_i - mu_j - target) signed error in turns
    """
    v = mu.to_numpy(float)
    n = len(v)

    # backward differences mod Q1: (mu_i - mu_j) ∈ [0, Q1)
    backward = (v.reshape(n,1) - v.reshape(1,n) + Q1) % Q1
    np.fill_diagonal(backward, np.nan)

    # pick j minimizing |Δ_ij - target|
    idx   = np.nanargmin(np.abs(backward - target), axis=1)
    delta = backward[np.arange(n), idx] - target
    names = mu.index[idx]

    return pd.DataFrame({"prev_bpm": names, "delta": delta}, index=mu.index)

# -------------------------------------------------------------------------
# 1) READ TRACK DATA + ADD NOISE
# -------------------------------------------------------------------------

orig_data = tfs.read(TRACK_DATA_FILE)
data = orig_data.copy()
data["x"] += np.random.normal(0, BPM_NOISE, size=len(data))
data["y"] += np.random.normal(0, BPM_NOISE, size=len(data))

# -------------------------------------------------------------------------
# 2) INITIALISE MAD-X & GET TWISS
# -------------------------------------------------------------------------

mad = MadInterface(SEQUENCE_FILE, BPM_RANGE, stdout="/dev/null", redirect_sterr=True)
true_str = read_knobs(TRUE_STRENGTHS)
for knob, val in true_str.items():
    mad.mad.send(f"MADX.lhcb1['{knob}'] = {val}")

mad.mad.send(f"tws = twiss{{sequence=MADX.{SEQ_NAME}, observe=1}}")

tws: tfs.TfsDataFrame = mad.mad.tws.to_df().set_index("name")
print("Found tunes:", tws.q1, tws.q2)

# -------------------------------------------------------------------------
# 3) PRECOMPUTE LATTICE FUNCTIONS
# -------------------------------------------------------------------------

sqrt_betax = np.sqrt(tws["beta11"])
sqrt_betay = np.sqrt(tws["beta22"])
alfax = tws["alfa11"]
alfay = tws["alfa22"]

# Fast lookup maps
map_sqrt_betax = sqrt_betax.to_dict()
map_sqrt_betay = sqrt_betay.to_dict()
map_alfax = alfax.to_dict()
map_alfay = alfay.to_dict()

# find upstream BPMs & phase errors
prev_x_df = prev_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
    columns={"prev_bpm": "prev_bpm_x", "delta": "delta_x"}
)
prev_y_df = prev_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
    columns={"prev_bpm": "prev_bpm_y", "delta": "delta_y"}
)

# Assuming the BPMs are in the same order as the twiss data
# and the track data
bpm_list = tws.index.to_list()

# -------------------------------------------------------------------------
# 4) ANNOTATE TRACK DATA (vectorised prev_x/prev_y)
# -------------------------------------------------------------------------

# join upstream-BPM info & map downstream lattice as before
data = data.join(prev_x_df, on="name").join(prev_y_df, on="name")
for col, mapper in (
    ("sqrt_betax", map_sqrt_betax),
    ("sqrt_betay", map_sqrt_betay),
    ("alfax", map_alfax),
    ("alfay", map_alfay),
):
    data[col] = data["name"].map(mapper)

# build a BPM→ring‐index map
bpm_index = {b:i for i,b in enumerate(bpm_list)}

# 1) compute whether each “previous” BPM actually wrapped around
cur_i    = data["name"].map(bpm_index)
prev_ix  = data["prev_bpm_x"].map(bpm_index)
prev_iy  = data["prev_bpm_y"].map(bpm_index)

# if current BPM index < prev index, it really came from turn-1
turn_x = data["turn"] - (cur_i < prev_ix).astype(int)
turn_y = data["turn"] - (cur_i < prev_iy).astype(int)

# 2) flatten your coords table to prepare for merging:
coords_df = (
    data[["turn","name","x","y"]]
    .set_index(["turn","name"])
    .reset_index()
)

# 3) make small lookup tables for x and y
coords_x = (
    coords_df
    .rename(columns={"turn":"turn_x","name":"prev_bpm_x","x":"prev_x"})
    [["turn_x","prev_bpm_x","prev_x"]]
)
coords_y = (
    coords_df
    .rename(columns={"turn":"turn_y","name":"prev_bpm_y","y":"prev_y"})
    [["turn_y","prev_bpm_y","prev_y"]]
)

# 4) attach the “real” turn_x/turn_y to data
data = data.assign(turn_x=turn_x, turn_y=turn_y)

# 5) merge to get prev_x, prev_y; fill missing with 0
data = (
    data
    .merge(coords_x, on=["turn_x","prev_bpm_x"], how="left")
    .merge(coords_y, on=["turn_y","prev_bpm_y"], how="left")
)
data["prev_x"] = data["prev_x"].fillna(0)
data["prev_y"] = data["prev_y"].fillna(0)

# 6) clean up helper columns
data = data.drop(columns=["turn_x","turn_y"])

# 7) map upstream lattice functions exactly as before
data["sqrt_betax_p"] = data["prev_bpm_x"].map(map_sqrt_betax)
data["sqrt_betay_p"] = data["prev_bpm_y"].map(map_sqrt_betay)


# -------------------------------------------------------------------------
# 5) VECTORISED px/py COMPUTATION
# -------------------------------------------------------------------------

# normalised coords
d2 = data.copy()  # alias
x1 = d2["prev_x"] / d2["sqrt_betax_p"]   # z1 / √β1
x2 = d2["x"]      / d2["sqrt_betax"]     # z2 / √β2
y1 = d2["prev_y"] / d2["sqrt_betay_p"]
y2 = d2["y"]      / d2["sqrt_betay"]

# phase error in radians
phi_x = d2["delta_x"] * 2 * np.pi
phi_y = d2["delta_y"] * 2 * np.pi

# trig
c_x, s_x, t_x = np.cos(phi_x), np.sin(phi_x), np.tan(phi_x)
c_y, s_y, t_y = np.cos(phi_y), np.sin(phi_y), np.tan(phi_y)

# apply formula: pz2 = -(z1/√β1*(cos+sin tan) + z2/√β2*(tan+α2)) / √β2

data["px"] = -(
    x1 * (c_x + s_x * t_x)
  + x2 * (t_x + d2["alfax"])
) / d2["sqrt_betax"]


data["py"] = -(
    y1 * (c_y + s_y * t_y)
  + y2 * (t_y + d2["alfay"])
) / d2["sqrt_betay"]


# -------------------------------------------------------------------------
# 6) WRITE OUTPUT (only original cols + px/py)
# -------------------------------------------------------------------------

out_cols = ["name", "turn", "x", "px", "y", "py"]
tfs.write(NOISE_FILE, data[out_cols])
print("→ Saved:", NOISE_FILE)


# What is the difference between the original and noisy data?
x_diff  = data["x"] - orig_data["x"]
px_diff = data["px"] - orig_data["px"] 
y_diff  = data["y"] - orig_data["y"]
py_diff = data["py"] - orig_data["py"]

# print("x_diff", x_diff)
# print("px_diff", px_diff)
# print("y_diff", y_diff)
# print("py_diff", py_diff)

print("x_diff mean", x_diff.abs().mean(), "±", x_diff.std())
print("px_diff mean", px_diff.abs().mean(), "±", px_diff.std())
print("y_diff mean", y_diff.abs().mean(), "±", y_diff.std())
print("py_diff mean", py_diff.abs().mean(), "±", py_diff.std())

if FILTER_DATA:
    print("Filtering data with Kalman filter")
    kf = BPMKalmanFilter()
    filtered_df = kf.run(data[out_cols])
    tfs.write(
        FILTERED_FILE,
        filtered_df,
    )
    print("→ Saved:", FILTERED_FILE)
    x_diff  = filtered_df["x"] - orig_data["x"]
    px_diff = filtered_df["px"] - orig_data["px"]
    y_diff  = filtered_df["y"] - orig_data["y"]
    py_diff = filtered_df["py"] - orig_data["py"]
    print("x_diff mean", x_diff.abs().mean(), "±", x_diff.std())
    print("px_diff mean", px_diff.abs().mean(), "±", px_diff.std())
    print("y_diff mean", y_diff.abs().mean(), "±", y_diff.std())
    print("py_diff mean", py_diff.abs().mean(), "±", py_diff.std())