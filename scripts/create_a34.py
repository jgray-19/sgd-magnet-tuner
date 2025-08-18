import concurrent.futures  # for parallel processing of track results

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BEAM_ENERGY,
    # FILTERED_FILE,
    FLATTOP_TURNS,
    # KALMAN_FILE,
    MAGNET_RANGE,
    NOISE_FILE,
    NUM_TRACKS,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
    TRUE_STRENGTHS,
    TUNE_KNOBS_FILE,
)

# from aba_optimiser.ellipse_filtering import filter_noisy_data
# from aba_optimiser.kalman_filtering import BPMKalmanFilter
from aba_optimiser.make_noisy_track_data import make_noisy_track_data


# Define a helper to process a single track result in parallel
def process_track(ntrk, mad_trk):
    """
    Process a single MAD tracking result: extract DataFrame, apply noise, run Kalman filter, apply ellipse filtering.
    Returns tuple of (df_track, noisy_df, kalman_df, filtered_df).
    """
    # Retrieve tracked data and convert to DataFrame
    trk = mad_trk["trk"]
    df = trk.to_df(columns=["name", "turn", "x", "px", "y", "py"])
    # Adjust turn count
    df["turn"] += ntrk * FLATTOP_TURNS
    # Downcast for memory
    df["name"] = df["name"].astype("category")
    df["turn"] = df["turn"].astype(np.int32)
    # Add noise and run Kalman filter
    noisy_df, _ = make_noisy_track_data(df)
    # kalman_df = BPMKalmanFilter().run(noisy_df)
    # Apply ellipse-based filtering
    # filtered_df = filter_noisy_data(noisy_df)
    # filtered_df["name"] = filtered_df["name"].astype("category")
    # filtered_df["turn"] = filtered_df["turn"].astype(np.int32)
    return df.copy(), noisy_df  # , kalman_df, filtered_df


# Initialize MAD-NG interface
mad = MAD(debug=False)

# Load the sequence
mad.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
seq = mad.MADX[SEQ_NAME]
mad["SEQ_NAME"] = SEQ_NAME

# Set beam
seq.beam = mad.beam(particle='"proton"', energy=BEAM_ENERGY)

# Run an initial twiss
ini_tws, _ = mad.twiss(sequence=seq)
ini_tws: tfs.TfsDataFrame = ini_tws.to_df()

rng = np.random.default_rng(42)  # reproducibility

# Define parameters
tunes = [0.28, 0.31]
knob_start, end_pos = MAGNET_RANGE.split("/")
mad["knob_range"] = MAGNET_RANGE

# Collect quadrupole element names
main_quad_names = []
start, end, _ = mad.MADX[SEQ_NAME].range_of("knob_range")

for elm in seq:
    if elm.kind == "quadrupole" and elm.k1 != 0 and "MQ." in elm.name:
        main_quad_names.append(elm.name)
        noise = REL_K1_STD_DEV * rng.normal(0, 1)
        elm.k1 = elm.k1 + noise  # Add noise

print(f"Found {len(main_quad_names)} quadrupoles in the sequence.")

# Run a twiss after changing the quad strengths.
changed_tws, _ = mad.twiss(sequence=seq)
changed_tws: tfs.TfsDataFrame = changed_tws.to_df()

beta11_beating = (changed_tws["beta11"] - ini_tws["beta11"]) / ini_tws["beta11"]
beta22_beating = (changed_tws["beta22"] - ini_tws["beta22"]) / ini_tws["beta22"]
print(
    f"Beta11 beating: {beta11_beating.mean() * 100:.2f}% ± {beta11_beating.std() * 100:.2f}%"
)
print(
    f"Beta22 beating: {beta22_beating.mean() * 100:.2f}% ± {beta22_beating.std() * 100:.2f}%"
)
print(
    f"Old tunes: {ini_tws.q1}, {ini_tws.q2}. New tunes: {changed_tws.q1}, {changed_tws.q2}"
)

# Match tunes
mad["result"] = mad.match(
    command=r"\ -> twiss{sequence=MADX[SEQ_NAME]}",
    variables=[
        {"var": "'MADX.dqx_b1_op'", "name": "'dQx.b1_op'"},
        {"var": "'MADX.dqy_b1_op'", "name": "'dQy.b1_op'"},
    ],
    equalities=[
        {"expr": f"\\t -> math.abs(t.q1)-(62+{tunes[0]})", "name": "'q1'"},
        {"expr": f"\\t -> math.abs(t.q2)-(60+{tunes[1]})", "name": "'q2'"},
    ],
    objective={"fmin": 1e-18},
    info=2,
)

# Store matched tunes in Python variables
matched_tunes = {key: mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")}

# Save matched tunes to file using a loop
with TUNE_KNOBS_FILE.open("w") as f:
    for key, val in matched_tunes.items():
        f.write(f"{key}\t{val: .15e}\n")

# Save final strengths to Python
true_strengths = {qname: mad[f"MADX['{qname}'].k1"] for qname in main_quad_names}
with TRUE_STRENGTHS.open("w") as f:
    for name, val in true_strengths.items():
        f.write(f"{name}_k1\t{val: .15e}\n")

# Twiss before tracking
tw = mad.twiss(sequence=seq)[0]
df_twiss = tw.to_df()
print(df_twiss.columns)
del tw

# Tracking using action-angle
action_list = [8e-9]  # np.linspace(6e-9, 1.5e-8, num=5)
angle_list = [0]  # np.linspace(0, 2 * np.pi, endpoint=False, num=2)
mad_processes = []
num_tracks = len(action_list) * len(angle_list)
assert num_tracks == NUM_TRACKS, f"Expected {NUM_TRACKS} tracks, got {num_tracks}."

for ntrk in range(NUM_TRACKS):
    # 1. New MAD process
    mad_trk = MAD(debug=False, redirect_stderr=True)
    mad_trk.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
    this_seq = mad_trk.MADX[SEQ_NAME]
    this_seq.beam = mad_trk.beam(particle='"proton"', energy=BEAM_ENERGY)
    this_seq.deselect(mad_trk.element.flags.observed)
    this_seq.select(mad_trk.element.flags.observed, {"pattern": "'BPM'"})
    del this_seq

    # 2. Set tune knobs
    for key, val in matched_tunes.items():
        mad_trk[f"MADX['{key}']"] = val

    # 3. Set ALL quadrupole strengths (true_strengths: dict base -> k1, quad_all: all quad names)
    for name in main_quad_names:
        mad_trk[f"MADX['{name}'].k1"] = mad[f"MADX['{name}'].k1"]

    mad_processes.append(mad_trk)

# 4. Start tracking in each process (no data retrieved yet)
for ntrk, mad_trk in enumerate(mad_processes):
    # Determine action and angle indices
    idx_action = ntrk // len(angle_list)
    idx_angle = ntrk % len(angle_list)
    action = action_list[idx_action]
    angle = angle_list[idx_angle]
    # You may need twiss results for beta11, beta22 for each process
    beta11 = df_twiss.loc[0, "beta11"]
    beta22 = df_twiss.loc[0, "beta22"]
    alfa11 = df_twiss.loc[0, "alfa11"]
    alfa22 = df_twiss.loc[0, "alfa22"]

    # Compute normalized coordinates from action and angle
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    # Convert to real space coordinates
    x = np.sqrt(action * beta11) * cos_ang
    px = -np.sqrt(action / beta11) * (sin_ang + alfa11 * cos_ang)
    y = np.sqrt(action * beta22) * cos_ang
    py = -np.sqrt(action / beta22) * (sin_ang + alfa22 * cos_ang)
    X0 = {
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "t": 0,
        "pt": 0,
    }
    # Send tracking command
    mad_trk.send(
        f"""trk, mflw = track{{sequence=MADX['{SEQ_NAME}'], X0={{x={X0["x"]}, px={X0["px"]}, y={X0["y"]}, py={X0["py"]}, t=0, pt=0}}, nturn={FLATTOP_TURNS}, cmap=true, info=1}}"""
    )

# 5. After all tracks are launched, retrieve and process each result in parallel
track_dfs = []
noisy_dfs = []
kalman_dfs = []
filtered_dfs = []

# Use a thread pool to leverage independent MAD-NG results
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Map each (index, process) pair to the helper
    results = executor.map(lambda args: process_track(*args), enumerate(mad_processes))
    # for df_t, n_df, k_df, f_df in results:
    for df_t, n_df in results:
        track_dfs.append(df_t)
        noisy_dfs.append(n_df)
        # kalman_dfs.append(k_df)
        # filtered_dfs.append(f_df)

# Concatenate all results once
combined_df = pd.concat(track_dfs, ignore_index=True)
combined_noisy = pd.concat(noisy_dfs, ignore_index=True)
# combined_kalman = pd.concat(kalman_dfs, ignore_index=True)
# combined_filtered = pd.concat(filtered_dfs, ignore_index=True)
del track_dfs, noisy_dfs, kalman_dfs, filtered_dfs

# Save combined tracking data
# Ensure TFS writer compatibility: convert any categorical 'name' to string
combined_df["name"] = combined_df["name"].astype(str)
tfs.write(TRACK_DATA_FILE, combined_df)
print(f"→ Saved combined tracking data: {TRACK_DATA_FILE}")
del combined_df

combined_noisy.to_feather(NOISE_FILE, compression="lz4")
print(f"→ Saved combined noisy data: {NOISE_FILE}")
del combined_noisy

# combined_kalman.to_feather(KALMAN_FILE, compression="lz4")
# print(f"→ Saved combined Kalman-filtered data: {KALMAN_FILE}")
# del combined_kalman

# combined_filtered.to_feather(FILTERED_FILE, compression="lz4")
# print(f"→ Saved combined filtered data: {FILTERED_FILE}")
# del combined_filtered

# final status
print(
    f"All calculations and filtering completed successfully with {NUM_TRACKS} parallel MAD-NG processes."
)
