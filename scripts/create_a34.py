import concurrent.futures
import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pymadng import MAD

from aba_optimiser.config import (
    BEAM_ENERGY,
    DELTAP,
    EMINUS_NOISY_FILE,
    EMINUS_NONOISE_FILE,
    EPLUS_NOISY_FILE,
    EPLUS_NONOISE_FILE,
    # BPM_START_POINTS,
    # FILTERED_FILE,
    FLATTOP_TURNS,
    # KALMAN_FILE,
    MAGNET_RANGE,
    NO_NOISE_FILE,
    NOISY_FILE,
    NUM_TRACKS,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRUE_STRENGTHS,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.physics.transverse_momentum import calculate_pz

if TYPE_CHECKING:
    import pandas as pd


def single_writer_loop(queue: "mp.Queue", out_path: str) -> None:
    """Dedicated writer: consumes Arrow Tables and writes row groups to one Parquet file."""
    writer = None
    try:
        while True:
            table = queue.get()
            if table is None:  # STOP sentinel
                break
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def process_track_with_queue(ntrk, mad_trk, track_q: "mp.Queue", noise_q: "mp.Queue"):
    """
    Process a single MAD tracking result and enqueue Arrow tables for a
    dedicated writer process to persist. No file reads, no locks.
    """
    # Retrieve tracked data and convert to DataFrame
    trk = mad_trk["trk"]
    df: pd.DataFrame = trk.to_df(
        columns=["name", "turn", "x", "px", "y", "py"], force_pandas=True
    )
    # Adjust turn count
    df["turn"] += ntrk * FLATTOP_TURNS
    # Downcast for memory
    df["turn"] = df["turn"].astype(np.int32)

    # Enqueue tracking data as Arrow table
    track_q.put(pa.Table.from_pandas(df, preserve_index=False))

    # Add noise and enqueue noisy data
    df["name"] = df["name"].astype("category")
    tws, _ = mad_trk.twiss(
        sequence=mad_trk.MADX[SEQ_NAME], deltap=mad_trk.deltap, observe=1
    )
    _, _, noisy_df = calculate_pz(df, tws=tws.to_df().set_index("name"))
    del tws
    # noisy_df, _, _ = calculate_pz(df, low_noise_bpms=BPM_START_POINTS)
    noisy_df["name"] = noisy_df["name"].astype(str)
    noise_q.put(pa.Table.from_pandas(noisy_df, preserve_index=False))

    # Free memory immediately
    del df, noisy_df


def process_track(ntrk, mad_trk, track_q: "mp.Queue", noise_q: "mp.Queue"):
    """
    Process a single MAD tracking result.
    Routes to the appropriate implementation based on configuration.
    """
    # Use the single-writer queue approach for better I/O efficiency
    return process_track_with_queue(ntrk, mad_trk, track_q, noise_q)


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
ini_tws = ini_tws.to_df()

rng = np.random.default_rng(42)  # reproducibility

# Define parameters
tunes = [0.28, 0.31]
knob_start, end_pos = MAGNET_RANGE.split("/")
mad["knob_range"] = MAGNET_RANGE

# Collect quadrupole element names
main_quad_names = []
main_sext_names = []
start, end, _ = mad.MADX[SEQ_NAME].range_of("knob_range")

true_strengths = {}
for elm in seq:
    if elm.kind == "quadrupole" and elm.k1 != 0 and elm.name[:3] == "MQ.":
        # before = elm.k1
        elm.k1 = elm.k1 + rng.normal(0, abs(elm.k1 * REL_K1_STD_DEV))  # Add noise
        # after = elm.k1
        # print(f"Adjusted {elm.name}: {before:.6e} -> {after:.6e}")
        main_quad_names.append(elm.name)
        true_strengths[elm.name] = elm.k1
    elif elm.kind == "sextupole" and elm.k2 != 0 and elm.name[:3] == "MS.":
        # before = elm.k2
        elm.k2 = elm.k2 + rng.normal(0, abs(elm.k2 * REL_K1_STD_DEV))  # Add noise
        # after = elm.k2
        # print(f"Adjusted {elm.name}: {before:.6e} -> {after:.6e}")
        main_sext_names.append(elm.name)
        true_strengths[elm.name] = elm.k2

print(f"Found {len(main_quad_names)} quadrupoles in the sequence.")

# Run a twiss after changing the quad strengths.
changed_tws, _ = mad.twiss(sequence=seq)
changed_tws = changed_tws.to_df()

# Select the dataframe that matches name = r"^BPM\.\d\d.*" and then calculate beta beating
changed_tws = changed_tws[changed_tws["name"].str.match(r"^BPM\.\d\d.*")]

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
        {"expr": f"\\t -> t.q1%1-{tunes[0]}", "name": "'q1'"},
        {"expr": f"\\t -> t.q2%1-{tunes[1]}", "name": "'q2'"},
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
with TRUE_STRENGTHS.open("w") as f:
    for name, val in true_strengths.items():
        f.write(f"{name}_k1\t{val: .15e}\n")

# Tracking using action-angle
# action_list = [10e-9, 12.5e-9, 15e-9]  # np.linspace(6e-9, 1.5e-8, num=5)
action_list = np.linspace(8e-9, 12e-9, num=NUM_TRACKS)
angle_list = [1.4]
num_tracks = len(action_list) * len(angle_list)
assert num_tracks == NUM_TRACKS, f"Expected {NUM_TRACKS} tracks, got {num_tracks}."

no_noise_files = {
    -DELTAP: EMINUS_NONOISE_FILE,
    0: NO_NOISE_FILE,
    DELTAP: EPLUS_NONOISE_FILE,
}
noise_files = {-DELTAP: EMINUS_NOISY_FILE, 0: NOISY_FILE, DELTAP: EPLUS_NOISY_FILE}

for deltap in [-DELTAP, 0, DELTAP]:
    mad_processes = []
    # Twiss before tracking
    tw, _ = mad.twiss(sequence=seq, deltap=deltap)
    df_twiss = tw.to_df()
    del tw
    for ntrk in range(NUM_TRACKS):
        # 1. New MAD process
        mad_trk = MAD(debug=False, redirect_stderr=True)
        mad_trk.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
        mad_trk["deltap"] = deltap
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

        for name in main_sext_names:
            mad_trk[f"MADX['{name}'].k2"] = mad[f"MADX['{name}'].k2"]

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
            f"""
trk, mflw = track{{
    sequence=MADX['{SEQ_NAME}'],
    X0={{x={X0["x"]}, px={X0["px"]}, y={X0["y"]}, py={X0["py"]}, t=0, pt=0}},
    deltap=deltap,
    nturn={FLATTOP_TURNS},
    cmap=true,
    info=1
}}
"""
        )

    # 5. After all tracks are launched, retrieve and process each result in parallel
    # Use lock-based approach for efficient I/O

    # Start dedicated writer processes and bounded queues for backpressure
    track_queue: "mp.Queue" = mp.Queue(maxsize=32)
    noise_queue: "mp.Queue" = mp.Queue(maxsize=32)

    track_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(track_queue, str(no_noise_files[deltap])),
        daemon=True,
    )
    noise_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(noise_queue, str(noise_files[deltap])),
        daemon=True,
    )
    track_writer_proc.start()
    noise_writer_proc.start()

    # Use a thread pool to leverage independent MAD-NG results; enqueue to writers
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each (index, process, queue) to the helper and ensure all tasks complete
        list(
            executor.map(
                lambda args: process_track(*args),
                ((i, p, track_queue, noise_queue) for i, p in enumerate(mad_processes)),
            )
        )

    # Signal writers to stop and wait for them to finish
    track_queue.put(None)
    noise_queue.put(None)
    track_writer_proc.join()
    noise_writer_proc.join()
    track_queue.close()
    noise_queue.close()
    track_queue.join_thread()
    noise_queue.join_thread()

    # Report final status
    print(f"→ Saved tracking data: {no_noise_files[deltap]}")
    print(f"→ Saved noisy data: {noise_files[deltap]}")

    # Clean up
    del mad_processes

# Final status
print("→ All tracking and noisy data written to Parquet files.")

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
