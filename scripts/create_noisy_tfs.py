import sys

import numpy as np
import tfs

from aba_optimiser.config import (
    BPM_RANGE,
    FILTERED_FILE,
    NOISE_FILE,
    POSITION_STD_DEV,
    # TRUE_STRENGTHS,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
)
from aba_optimiser.ellipse_filtering import filter_noisy_data
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import next_bpm_to_pi_2, prev_bpm_to_pi_2

out_cols = ["name", "turn", "id", "eidx", "x", "px", "y", "py"]
numeric_cols = ["x", "px", "y", "py"]
np.random.seed(42)

# -------------------------------------------------------------------------
# 1) READ TRACK DATA + ADD NOISE
# -------------------------------------------------------------------------

orig_data = tfs.read(TRACK_DATA_FILE)
data = orig_data.copy()
data["x"] += np.random.normal(0, POSITION_STD_DEV, size=len(data))
data["y"] += np.random.normal(0, POSITION_STD_DEV, size=len(data))

noise_diff_x = data["x"] - orig_data["x"]
noise_diff_y = data["y"] - orig_data["y"]
print("x_diff mean", noise_diff_x.abs().mean(), "±", noise_diff_x.std())
print("y_diff mean", noise_diff_y.abs().mean(), "±", noise_diff_y.std())

# -------------------------------------------------------------------------
# 2) INITIALISE MAD-X & GET TWISS
# -------------------------------------------------------------------------

mad = MadInterface(SEQUENCE_FILE, BPM_RANGE, stdout="/dev/null", redirect_sterr=True)

# For testing purposes.
# true_str = read_knobs(TRUE_STRENGTHS)
# for knob, val in true_str.items():
#     mad.mad.send(f"MADX.lhcb1['{knob}'] = {val}")

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

# find downstream BPMs & phase errors
next_x_df = next_bpm_to_pi_2(tws["mu1"], tws.q1).rename(
    columns={"next_bpm": "next_bpm_x", "delta": "delta_x"}
)
next_y_df = next_bpm_to_pi_2(tws["mu2"], tws.q2).rename(
    columns={"next_bpm": "next_bpm_y", "delta": "delta_y"}
)

# Assuming the BPMs are in the same order as the twiss data
# and the track data
bpm_list = tws.index.to_list()

# -------------------------------------------------------------------------
# 4) ANNOTATE TRACK DATA (vectorised prev_x/prev_y)
# -------------------------------------------------------------------------

# join upstream-BPM info & map downstream lattice as before
data_p = data.join(prev_x_df, on="name").join(prev_y_df, on="name")
data_n = data.join(next_x_df, on="name").join(next_y_df, on="name")
del data

for col, mapper in (
    ("sqrt_betax", map_sqrt_betax),
    ("sqrt_betay", map_sqrt_betay),
    ("alfax", map_alfax),
    ("alfay", map_alfay),
):
    data_p[col] = data_p["name"].map(mapper)
    data_n[col] = data_n["name"].map(mapper)

# build a BPM→ring‐index map
bpm_index = {b: i for i, b in enumerate(bpm_list)}

# 1) compute whether each “previous” BPM actually wrapped around
cur_i_p = data_p["name"].map(bpm_index)
prev_ix = data_p["prev_bpm_x"].map(bpm_index)
prev_iy = data_p["prev_bpm_y"].map(bpm_index)

cur_i_n = data_n["name"].map(bpm_index)
next_ix = data_n["next_bpm_x"].map(bpm_index)
next_iy = data_n["next_bpm_y"].map(bpm_index)

# if current BPM index < prev index, it really came from turn-1
turn_x_p = data_p["turn"] - (cur_i_p < prev_ix).astype(int)
turn_y_p = data_p["turn"] - (cur_i_p < prev_iy).astype(int)

# if next BPM index > current index, it wrapped into turn+1
turn_x_n = data_n["turn"] + (cur_i_n > next_ix).astype(int)
turn_y_n = data_n["turn"] + (cur_i_n > next_iy).astype(int)

# 2) flatten your coords table to prepare for merging:
coords_df_p = (
    data_p[["turn", "name", "x", "y"]].set_index(["turn", "name"]).reset_index()
)

coords_df_n = (
    data_n[["turn", "name", "x", "y"]].set_index(["turn", "name"]).reset_index()
)

# 3) make small lookup tables for x and y
coords_x_p = coords_df_p.rename(
    columns={"turn": "turn_x_p", "name": "prev_bpm_x", "x": "prev_x"}
)[["turn_x_p", "prev_bpm_x", "prev_x"]]
coords_y_p = coords_df_p.rename(
    columns={"turn": "turn_y_p", "name": "prev_bpm_y", "y": "prev_y"}
)[["turn_y_p", "prev_bpm_y", "prev_y"]]

coords_x_n = coords_df_n.rename(
    columns={"turn": "turn_x_n", "name": "next_bpm_x", "x": "next_x"}
)[["turn_x_n", "next_bpm_x", "next_x"]]

coords_y_n = coords_df_n.rename(
    columns={"turn": "turn_y_n", "name": "next_bpm_y", "y": "next_y"}
)[["turn_y_n", "next_bpm_y", "next_y"]]

# 4) attach the “real” turn_x_p/turn_y_p to data
data_p = data_p.assign(turn_x_p=turn_x_p, turn_y_p=turn_y_p)

# 5) merge to get prev_x, prev_y; fill missing with 0
data_p = data_p.merge(coords_x_p, on=["turn_x_p", "prev_bpm_x"], how="left").merge(
    coords_y_p, on=["turn_y_p", "prev_bpm_y"], how="left"
)

data_n = (
    data_n.assign(turn_x_n=turn_x_n, turn_y_n=turn_y_n)
    .merge(coords_x_n, on=["turn_x_n", "next_bpm_x"], how="left")
    .merge(coords_y_n, on=["turn_y_n", "next_bpm_y"], how="left")
)

data_p["prev_x"] = data_p["prev_x"].fillna(0)
data_p["prev_y"] = data_p["prev_y"].fillna(0)
data_n["next_x"] = data_n["next_x"].fillna(0)
data_n["next_y"] = data_n["next_y"].fillna(0)

# 6) clean up helper columns
data_p = data_p.drop(columns=["turn_x_p", "turn_y_p"])
data_n = data_n.drop(columns=["turn_x_n", "turn_y_n"])

# 7) map upstream lattice functions
data_p["sqrt_betax_p"] = data_p["prev_bpm_x"].map(map_sqrt_betax)
data_p["sqrt_betay_p"] = data_p["prev_bpm_y"].map(map_sqrt_betay)
data_n["sqrt_betax_n"] = data_n["next_bpm_x"].map(map_sqrt_betax)
data_n["sqrt_betay_n"] = data_n["next_bpm_y"].map(map_sqrt_betay)


# -------------------------------------------------------------------------
# 5) VECTORISED px/py COMPUTATION
# -------------------------------------------------------------------------

# normalised coords
d2_p = data_p.copy()  # alias
x1 = d2_p["prev_x"] / d2_p["sqrt_betax_p"]  # z1 / √β1
x2 = d2_p["x"] / d2_p["sqrt_betax"]  # z2 / √β2
y1 = d2_p["prev_y"] / d2_p["sqrt_betay_p"]
y2 = d2_p["y"] / d2_p["sqrt_betay"]

# phase error in radians
phi_x = d2_p["delta_x"] * 2 * np.pi
phi_y = d2_p["delta_y"] * 2 * np.pi

# trig
c_x, s_x, t_x = np.cos(phi_x), np.sin(phi_x), np.tan(phi_x)
c_y, s_y, t_y = np.cos(phi_y), np.sin(phi_y), np.tan(phi_y)

# apply formula: pz2 = -(z1/√β1*(cos+sin tan) + z2/√β2*(tan+α2)) / √β2

data_p["px"] = (
    -(x1 * (c_x + s_x * t_x) + x2 * (t_x + d2_p["alfax"])) / d2_p["sqrt_betax"]
)


data_p["py"] = (
    -(y1 * (c_y + s_y * t_y) + y2 * (t_y + d2_p["alfay"])) / d2_p["sqrt_betay"]
)


# Use new formula to compute px/py - pz1 = ((z2/√β2 - z1/√β1 * cos) / sin - α1* z1/√β1 )/√β1

d2_n = data_n.copy()  # alias
x1 = d2_n["x"] / d2_n["sqrt_betax"]  # z1 / √β1
x2 = d2_n["next_x"] / d2_n["sqrt_betax_n"]  # z2 / √β2
y1 = d2_n["y"] / d2_n["sqrt_betay"]
y2 = d2_n["next_y"] / d2_n["sqrt_betay_n"]

phi_x = (d2_n["delta_x"] + 0.25) * 2 * np.pi
phi_y = (d2_n["delta_y"] + 0.25) * 2 * np.pi

c_x, s_x = np.cos(phi_x), np.sin(phi_x)
c_y, s_y = np.cos(phi_y), np.sin(phi_y)

data_n["px"] = (
    (x2 - x1 * c_x) / s_x  # (z2/√β2 − z1/√β1⋅cosφ)/sinφ
    - d2_n["alfax"] * x1  # − α1⋅(z1/√β1)
) / d2_n["sqrt_betax"]  # ÷ √β1

data_n["py"] = ((y2 - y1 * c_y) / s_y - d2_n["alfay"] * y1) / d2_n["sqrt_betay"]

# -------------------------------------------------------------------------
# 6) WRITE OUTPUT (only original cols + px/py)
# -------------------------------------------------------------------------

# Synchronize boundary measurements (using multi-column assignment)
data_n.loc[data_n.index[-1], ["px", "py"]] = data_p.loc[data_p.index[-1], ["px", "py"]]
data_p.loc[data_p.index[0], ["px", "py"]] = data_n.loc[data_n.index[0], ["px", "py"]]

px_diff_p = data_p["px"] - orig_data["px"]
py_diff_p = data_p["py"] - orig_data["py"]
print("px_diff mean (prev, will write)", px_diff_p.abs().mean(), "±", px_diff_p.std())
print("py_diff mean (prev, will write)", py_diff_p.abs().mean(), "±", py_diff_p.std())

data_p[out_cols].to_feather(NOISE_FILE, compression="lz4")
print("→ Saved:", NOISE_FILE)
filtered_data = filter_noisy_data(data_p)

filtered_data[out_cols].to_feather(FILTERED_FILE, compression="lz4")
print("→ Saved filtered data:", FILTERED_FILE)

sys.exit(0)  # Exit early for testing purposes
# What is the difference between the original and noisy data?
x_diff_p = data_p["x"] - orig_data["x"]
px_diff_p = data_p["px"] - orig_data["px"]
y_diff_p = data_p["y"] - orig_data["y"]
py_diff_p = data_p["py"] - orig_data["py"]

x_diff_n = data_n["x"] - orig_data["x"]
px_diff_n = data_n["px"] - orig_data["px"]
y_diff_n = data_n["y"] - orig_data["y"]
py_diff_n = data_n["py"] - orig_data["py"]

# print("x_diff", x_diff)
# print("px_diff", px_diff)
# print("y_diff", y_diff)
# print("py_diff", py_diff)

print("x_diff mean (prev)", x_diff_p.abs().mean(), "±", x_diff_p.std())
print("y_diff mean (prev)", y_diff_p.abs().mean(), "±", y_diff_p.std())

print("Looking at the px and py differences -----------------------")
print("px_diff mean (prev)", px_diff_p.abs().mean(), "±", px_diff_p.std())
print("py_diff mean (prev)", py_diff_p.abs().mean(), "±", py_diff_p.std())

print("px_diff mean (next)", px_diff_n.abs().mean(), "±", px_diff_n.std())
print("py_diff mean (next)", py_diff_n.abs().mean(), "±", py_diff_n.std())


print("Filtering data with Kalman filter ------------------")
from aba_optimiser.kalman_filtering import BPMKalmanFilter

kf = BPMKalmanFilter()

# Run Kalman filter on both datasets
df_n, df_p = kf.run(data_n), kf.run(data_p)

x_diff_p = df_p["x"] - orig_data["x"]
px_diff_p = df_p["px"] - orig_data["px"]
y_diff_p = df_p["y"] - orig_data["y"]
py_diff_p = df_p["py"] - orig_data["py"]

x_diff_n = df_n["x"] - orig_data["x"]
px_diff_n = df_n["px"] - orig_data["px"]
y_diff_n = df_n["y"] - orig_data["y"]
py_diff_n = df_n["py"] - orig_data["py"]

print("x_diff mean (prev w/ k)", x_diff_p.abs().mean(), "±", x_diff_p.std())
print("y_diff mean (prev w/ k)", y_diff_p.abs().mean(), "±", y_diff_p.std())

print("Looking at the px and py differences -----------------------")
print("px_diff mean (prev w/ k)", px_diff_p.abs().mean(), "±", px_diff_p.std())
print("py_diff mean (prev w/ k)", py_diff_p.abs().mean(), "±", py_diff_p.std())

print("x_diff mean (next w/ k)", x_diff_n.abs().mean(), "±", x_diff_n.std())
print("y_diff mean (next w/ k)", y_diff_n.abs().mean(), "±", y_diff_n.std())

print("px_diff mean (next w/ k)", px_diff_n.abs().mean(), "±", px_diff_n.std())
print("py_diff mean (next w/ k)", py_diff_n.abs().mean(), "±", py_diff_n.std())

df_p.to_feather(FILTERED_FILE, compression="lz4")
print("→ Saved previous:", FILTERED_FILE)

# Fuse results via weighted averaging
# cols = [col for col in out_cols if col not in ["x", "px", "y", "py"]]
# filtered_df = data_n[cols].copy()
# for comp in ["x", "px", "y", "py"]:
#     var = f"var_{comp}"
#     weight_n = 1/df_n[var]
#     weight_p = 1/df_p[var]
#     filtered_df[comp] = (df_n[comp] * weight_n + df_p[comp] * weight_p) / (weight_n + weight_p)
#     filtered_df[var] = 1 / (weight_n + weight_p)

# filtered_df = kf.run(filtered_df)
# # -------------------------------------------------------------------------
# # 4) Write filtered data and log differences
# # -------------------------------------------------------------------------
# numeric_cols = [col for col in filtered_df.columns if col not in ["name", "turn", 'id', 'eidx']]
# filtered_df[numeric_cols] = filtered_df[numeric_cols]

# x_diff_n = filtered_df["x"] - orig_data["x"]
# px_diff  = filtered_df["px"] - orig_data["px"]
# y_diff   = filtered_df["y"] - orig_data["y"]
# py_diff  = filtered_df["py"] - orig_data["py"]

# print("x_diff mean", x_diff_n.abs().mean(), "±", x_diff_n.std())
# print("x calc std", np.sqrt(filtered_df["var_x"]).mean())

# print("px_diff mean", px_diff.abs().mean(), "±", px_diff.std())
# print("px calc std", np.sqrt(filtered_df["var_px"]).mean())

# print("y_diff mean", y_diff.abs().mean(), "±", y_diff.std())
# print("y calc std", np.sqrt(filtered_df["var_y"]).mean())

# print("py_diff mean", py_diff.abs().mean(), "±", py_diff.std())
# print("py calc std", np.sqrt(filtered_df["var_py"]).mean())

# # 2. Write out as Feather with LZ4 compression
# filtered_df.to_feather(FILTERED_FILE, compression="lz4")
print("→ Saved:", FILTERED_FILE)
