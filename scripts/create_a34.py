import concurrent.futures
import logging
import multiprocessing as mp
import time
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BEAM_ENERGY,
    # BPM_START_POINTS,
    CLEANED_FILE,
    CORRECTOR_STRENGTHS,
    DELTAP,
    EMINUS_NOISY_FILE,
    EMINUS_NONOISE_FILE,
    EPLUS_NOISY_FILE,
    EPLUS_NONOISE_FILE,
    FLATTOP_TURNS,
    KICK_BOTH_PLANES,
    MACHINE_DELTAP,
    # KALMAN_FILE,
    NO_NOISE_FILE,
    NOISY_FILE,
    NUM_TRACKS,
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRUE_STRENGTHS,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.filtering.svd import svd_clean_measurements

# from aba_optimiser.physics.transverse_momentum import calculate_pz
from aba_optimiser.physics.dispersive_momentum_reconstruction import calculate_pz

if TYPE_CHECKING:
    import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("create_a34.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Log script start
logger.info("=== Starting create_a34.py script execution ===")
script_start_time = time.time()


def single_writer_loop(queue: "mp.Queue", out_path: str) -> None:
    """Dedicated writer: consumes Arrow Tables and writes row groups to one Parquet file."""
    logger.info(f"Starting dedicated writer process for {out_path}")
    writer = None
    try:
        while True:
            table = queue.get()
            if table is None:  # STOP sentinel
                logger.info(f"Writer process received stop signal for {out_path}")
                break
            if writer is None:
                logger.debug(f"Initializing ParquetWriter for {out_path}")
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            logger.debug(f"Written table with {table.num_rows} rows to {out_path}")
    except Exception as e:
        logger.error(f"Error in writer process for {out_path}: {e}")
        raise
    finally:
        if writer is not None:
            logger.info(f"Closing ParquetWriter for {out_path}")
            writer.close()


def process_track_with_queue(
    ntrk, mad_trk, track_q: "mp.Queue", noise_q: "mp.Queue", cleaned_q: "mp.Queue"
):
    """
    Process a single MAD tracking result and enqueue Arrow tables for a
    dedicated writer process to persist. No file reads, no locks.
    """
    logger.debug(f"Processing track {ntrk}")
    start_time = time.time()

    try:
        # Retrieve tracked data and convert to DataFrame
        trk = mad_trk["trk"]
        true_df: pd.DataFrame = trk.to_df(
            columns=["name", "turn", "x", "px", "y", "py"], force_pandas=True
        )
        logger.debug(f"Track {ntrk}: Retrieved {len(true_df)} rows of tracking data")

        # Adjust turn count
        true_df["turn"] += ntrk * FLATTOP_TURNS
        # Downcast for memory
        true_df["turn"] = true_df["turn"].astype(np.int32)

        # Add a new category column that indicates if it is a x or y kick
        if KICK_BOTH_PLANES:
            true_df["kick_plane"] = "xy"
        else:
            true_df["kick_plane"] = "x" if ntrk % 2 == 0 else "y"
        true_df["x_weight"] = 1.0
        true_df["y_weight"] = 1.0

        # Enqueue tracking data as Arrow table
        track_q.put(pa.Table.from_pandas(true_df, preserve_index=False))
        logger.debug(f"Track {ntrk}: Enqueued tracking data table")

        # Add noise and enqueue noisy data
        true_df["name"] = true_df["name"].astype("category")

        true_df["kick_plane"] = true_df["kick_plane"].astype("category")

        tws, _ = mad_trk.twiss(
            sequence=mad_trk.MADX[SEQ_NAME], deltap=mad_trk.input_deltap, observe=1
        )
        tws = tws.to_df().set_index("name")
        noisy_df = calculate_pz(true_df, tws=tws, info=False)
        noisy_df["name"] = noisy_df["name"].astype(str)
        noisy_df["kick_plane"] = noisy_df["kick_plane"].astype(str)
        noise_q.put(pa.Table.from_pandas(noisy_df, preserve_index=False))

        # Filter the noisy data and enqueue cleaned data
        cleaned_df = svd_clean_measurements(noisy_df)
        cleaned_df = calculate_pz(cleaned_df, inject_noise=False, tws=tws, info=True)
        del tws

        cleaned_q.put(pa.Table.from_pandas(cleaned_df, preserve_index=False))

        # Print the differences in x, y, px, py between noisy and cleaned for debugging (note they may have different indices)
        diff_x = true_df["x"] - cleaned_df["x"]
        diff_y = true_df["y"] - cleaned_df["y"]
        diff_px = true_df["px"] - cleaned_df["px"]
        diff_py = true_df["py"] - cleaned_df["py"]
        logger.info(
            f"Track {ntrk}: Noisy vs cleaned x diff: mean={diff_x.mean():.3e}, std={diff_x.std():.3e}; "
            f"y diff: mean={diff_y.mean():.3e}, std={diff_y.std():.3e}; "
            f"px diff: mean={diff_px.mean():.3e}, std={diff_px.std():.3e}; "
            f"py diff: mean={diff_py.mean():.3e}, std={diff_py.std():.3e}"
        )
        del diff_x, diff_y, diff_px, diff_py

        logger.debug(f"Track {ntrk}: Enqueued noisy data table")
        processing_time = time.time() - start_time
        logger.info(f"Track {ntrk}: Processing completed in {processing_time:.2f}s")

        # Free memory immediately
        del true_df, noisy_df, cleaned_df

    except Exception as e:
        logger.error(f"Error processing track {ntrk}: {e}")
        raise


def process_track(
    ntrk, mad_trk, track_q: "mp.Queue", noise_q: "mp.Queue", cleaned_q: "mp.Queue"
):
    """
    Process a single MAD tracking result.
    Routes to the appropriate implementation based on configuration.
    """
    # Use the single-writer queue approach for better I/O efficiency
    return process_track_with_queue(ntrk, mad_trk, track_q, noise_q, cleaned_q)


def select_bpms(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """Select only BPM rows from a TFS DataFrame."""
    return df[df["name"].str.match(r"^BPM\.\d\d.*")].reset_index(drop=True)


# Initialize MAD-NG interface
logger.info("Initializing MAD-NG interface")
mad = MAD(debug=False)

# Load the sequence
logger.info(f"Loading sequence from {SEQUENCE_FILE.absolute()}")
mad.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
mad["seq"] = mad.MADX[SEQ_NAME]
mad["SEQ_NAME"] = SEQ_NAME

# Set beam
logger.info(f"Setting beam parameters: particle=proton, energy={BEAM_ENERGY}")
mad.seq.beam = mad.beam(particle='"proton"', energy=BEAM_ENERGY)

# Run an initial twiss
logger.info("Running initial twiss calculation")
ini_tws, _ = mad.twiss(sequence=mad.seq)
ini_tws = ini_tws.to_df()
ini_tws = select_bpms(ini_tws)
logger.info(f"Initial twiss completed. Found {len(ini_tws)} BPMs")

rng = np.random.default_rng(42)  # reproducibility

# Define parameters
tunes = [0.28, 0.31]
logger.info(f"Target tunes: Qx={62 + tunes[0]}, Qy={60 + tunes[1]}")

# Collect quadrupole element names
logger.info("Scanning sequence for quadrupoles and sextupoles")
main_bend_names = []
main_quad_names = []
main_sext_names = []
true_strengths = {}
for elm in mad.seq:
    # if elm.kind == "sbend" and elm.k0 != 0 and elm.name[:3] == "MB.":
    # before = elm.k0
    # elm.k0 = elm.k0 + rng.normal(0, abs(elm.k0 * 1e-4))  # Add noise
    # after = elm.k0
    # print(f"Adjusted {elm.name}: {before:.6e} -> {after:.6e}")
    # main_bend_names.append(elm.name)
    # true_strengths[elm.name] = elm.k0
    if elm.kind == "quadrupole" and elm.k1 != 0 and elm.name[:3] == "MQ.":
        # before = elm.k1
        elm.k1 = elm.k1 + rng.normal(0, abs(elm.k1 * REL_K1_STD_DEV))  # Add noise
        # after = elm.k1
        # print(f"Adjusted {elm.name}: {before:.6e} -> {after:.6e}")
        main_quad_names.append(elm.name)
        true_strengths[elm.name] = elm.k1
    elif elm.kind == "sextupole" and elm.k2 != 0 and elm.name[:3] == "MS.":
        # before = elm.k2
        elm.k2 = elm.k2 + rng.normal(0, abs(elm.k2 * 1e-4))  # Add noise
        # after = elm.k2
        # print(f"Adjusted {elm.name}: {before:.6e} -> {after:.6e}")
        main_sext_names.append(elm.name)
        true_strengths[elm.name] = elm.k2

logger.info(
    f"Found {len(main_quad_names)} quadrupoles and {len(main_sext_names)} sextupoles"
)

if main_quad_names:
    logger.info(
        f"Applied relative K1 noise with std dev: {REL_K1_STD_DEV} to {len(main_quad_names)} quadrupoles"
    )

if main_bend_names:
    logger.info(
        f"Applied relative K0 noise with std dev: 1e-4 to {len(main_bend_names)} dipoles"
    )
if main_sext_names:
    logger.info(
        f"Applied absolute K2 noise with std dev: 1e-4 to {len(main_sext_names)} sextupoles"
    )

# Run a twiss after changing the quad strengths.
logger.info("Running twiss calculation after quadrupole strength adjustments")
changed_tws, _ = mad.twiss(sequence=mad.seq)
changed_tws = changed_tws.to_df()

# Select the dataframe that matches name = r"^BPM\.\d\d.*" and then calculate beta beating
changed_tws = select_bpms(changed_tws)

beta11_beating = (changed_tws["beta11"] - ini_tws["beta11"]) / ini_tws["beta11"]
beta22_beating = (changed_tws["beta22"] - ini_tws["beta22"]) / ini_tws["beta22"]
logger.info(
    f"Beta11 beating: {beta11_beating.mean() * 100:.2f}% ± {beta11_beating.std() * 100:.2f}%"
)
logger.info(
    f"Beta22 beating: {beta22_beating.mean() * 100:.2f}% ± {beta22_beating.std() * 100:.2f}%"
)
logger.info(
    f"Old tunes: {ini_tws.q1:.4f}, {ini_tws.q2:.4f}. New tunes: {changed_tws.q1:.4f}, {changed_tws.q2:.4f}"
)

# Match tunes before adding deltap
logger.info("Starting tune matching for initial conditions")
mad["result"] = mad.match(
    command=r"\ -> twiss{sequence=MADX[SEQ_NAME]}",
    variables=[
        {"var": "'MADX.dqx_b1_op'", "name": "'dQx.b1_op'"},
        {"var": "'MADX.dqy_b1_op'", "name": "'dQy.b1_op'"},
    ],
    equalities=[
        {"expr": f"\\t -> t.q1-(62+{tunes[0]})", "name": "'q1'"},
        {"expr": f"\\t -> t.q2-(60+{tunes[1]})", "name": "'q2'"},
    ],
    objective={"fmin": 1e-18},
    info=2,
)
logger.info("Initial tune matching completed")

# Now add a deltap to correct and match
logger.info(f"Setting machine deltap: {MACHINE_DELTAP}")
mad["machine_deltap"] = MACHINE_DELTAP
mad["qx"] = tunes[0]
mad["qy"] = tunes[1]
mad["correct_file"] = str(CORRECTOR_STRENGTHS.absolute())
logger.info(f"Starting orbit correction with corrector file: {CORRECTOR_STRENGTHS}")
mad.send(r"""
local correct, option in MAD

io.write("*** orbit correction using off momentum twiss\n")
local tbl = twiss { sequence=seq, deltap=machine_deltap }

! Increase file numerical formatting
local fmt = option.numfmt ; option.numfmt = "% -.16e"
correct { sequence=seq, model=tbl, method="micado", info=1 } :write(correct_file)
option.numfmt = fmt ! restore formatting

io.write("*** rematching tunes for off-momentum twiss\n")
match {
  command := twiss {sequence=seq, observe=1, deltap=machine_deltap},
  variables = { rtol=1e-6, -- 1 ppm
    { var = 'MADX.dqx_b1_op', name='dQx.b1_op' },
    { var = 'MADX.dqy_b1_op', name='dQy.b1_op' },
  },
  equalities = { tol = 1e-6,
    { expr = \t -> t.q1-62-qx, name='q1' },
    { expr = \t -> t.q2-60-qy, name='q2' },
  },
  info=2
}
""")
logger.info("Orbit correction and tune rematching completed")

# Run a twiss after adding deltap and correction
logger.info("Running final twiss calculation after orbit correction")
changed_tws, _ = mad.twiss(sequence=mad.seq)
changed_tws = changed_tws.to_df()

# Select the dataframe that matches name = r"^BPM\.\d\d.*" and then calculate beta beating
changed_tws = select_bpms(changed_tws)

beta11_beating = (changed_tws["beta11"] - ini_tws["beta11"]) / ini_tws["beta11"]
beta22_beating = (changed_tws["beta22"] - ini_tws["beta22"]) / ini_tws["beta22"]
logger.info(
    f"Beta11 beating: {beta11_beating.mean() * 100:.2f}% ± {beta11_beating.std() * 100:.2f}%"
)
logger.info(
    f"Beta22 beating: {beta22_beating.mean() * 100:.2f}% ± {beta22_beating.std() * 100:.2f}%"
)

# Plot the beta11 and beta22 beating in the magnet range
# import matplotlib.pyplot as plt

# from aba_optimiser.config import MAGNET_RANGE

# plt.figure(figsize=(10, 6))
# start_bpm, end_bpm = MAGNET_RANGE.split("/")
# start_idx = changed_tws[changed_tws["name"] == start_bpm].index[0]
# end_idx = changed_tws[changed_tws["name"] == end_bpm].index[0]
# print(start_idx, end_idx, len(changed_tws))
# plt.plot(
#     changed_tws["s"][start_idx:end_idx],
#     beta11_beating[start_idx:end_idx] * 100,
#     label="Beta11 Beating (%)",
# )
# plt.plot(
#     changed_tws["s"][start_idx:end_idx],
#     beta22_beating[start_idx:end_idx] * 100,
#     label="Beta22 Beating (%)",
# )
# plt.show()

# Store matched tunes in Python variables
matched_tunes = {key: mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")}
logger.info(f"Matched tune knobs: {matched_tunes}")

# Save matched tunes to file using a loop
logger.info(f"Saving matched tunes to {TUNE_KNOBS_FILE}")
with TUNE_KNOBS_FILE.open("w") as f:
    for key, val in matched_tunes.items():
        f.write(f"{key}\t{val: .15e}\n")

# Save final strengths to Python
logger.info(f"Saving true magnet strengths to {TRUE_STRENGTHS}")
with TRUE_STRENGTHS.open("w") as f:
    for name, val in true_strengths.items():
        f.write(f"{name}_k\t{val: .15e}\n")

# Tracking using action-angle
logger.info("Setting up action-angle tracking parameters")
# action_list = [10e-9, 12.5e-9, 15e-9]  # np.linspace(6e-9, 1.5e-8, num=5)
num_actions = max(1, NUM_TRACKS // 5)
num_angles = max(1, NUM_TRACKS // num_actions)
action_list = np.linspace(6e-9, 8e-9, num=num_actions)
# angle_list = [1.4]
angle_list = np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False)
num_tracks = len(action_list) * len(angle_list)
assert num_tracks == NUM_TRACKS, f"Expected {NUM_TRACKS} tracks, got {num_tracks}."
logger.info(
    f"Generated {len(action_list)} action values and {len(angle_list)} angle values"
)
logger.info(f"Total tracks to process: {num_tracks}")

no_noise_files = {
    -DELTAP: EMINUS_NONOISE_FILE,
    0: NO_NOISE_FILE,
    DELTAP: EPLUS_NONOISE_FILE,
}
noise_files = {
    -DELTAP: EMINUS_NOISY_FILE,
    0: NOISY_FILE,
    DELTAP: EPLUS_NOISY_FILE,
}
logger.info("Loading corrector strengths table")
corrector_table = tfs.read(CORRECTOR_STRENGTHS)

# Remove all rows that have monitor in the column kind
corrector_table = corrector_table[corrector_table["kind"] != "monitor"]
logger.info(f"Loaded {len(corrector_table)} corrector elements")

# for input_deltap in [-DELTAP, 0, DELTAP]:
for input_deltap in [0]:
    logger.info(f"Starting tracking for input_deltap = {input_deltap}")
    start_time = time.time()

    mad_processes = []
    # Twiss before tracking
    true_deltap = input_deltap + MACHINE_DELTAP
    logger.info(f"True deltap for this iteration: {true_deltap}")
    tw, _ = mad.twiss(sequence=mad.seq, deltap=true_deltap)
    df_twiss = tw.to_df()
    del tw
    logger.info(f"Pre-tracking twiss completed with {len(df_twiss)} elements")

    logger.info(f"Initializing {NUM_TRACKS} MAD processes for parallel tracking")
    for ntrk in range(NUM_TRACKS):
        # 1. New MAD process
        mad_trk = MAD(debug=False, stdout="/dev/null", redirect_stderr=True)
        mad_trk.MADX.load(f"'{SEQUENCE_FILE.absolute()}'")
        mad_trk["true_deltap"] = true_deltap
        mad_trk["input_deltap"] = input_deltap
        this_seq = mad_trk.MADX[SEQ_NAME]
        this_seq.beam = mad_trk.beam(particle='"proton"', energy=BEAM_ENERGY)
        this_seq.deselect(mad_trk.element.flags.observed)
        this_seq.select(mad_trk.element.flags.observed, {"pattern": "'BPM'"})
        del this_seq

        # 2. Set tune knobs
        for key, val in matched_tunes.items():
            mad_trk[f"MADX['{key}']"] = val

        # 3. Set ALL quadrupole strengths (true_strengths: dict base -> k1, quad_all: all quad names)
        for name in main_bend_names:
            mad_trk[f"MADX['{name}'].k0"] = mad[f"MADX['{name}'].k0"]

        for name in main_quad_names:
            mad_trk[f"MADX['{name}'].k1"] = mad[f"MADX['{name}'].k1"]

        for name in main_sext_names:
            mad_trk[f"MADX['{name}'].k2"] = mad[f"MADX['{name}'].k2"]

        # Loop through every corrector in the table and set the h and v kicks
        for _, row in corrector_table.iterrows():
            mad_trk.send(f"MADX.{SEQ_NAME}['{row['ename']}'].hkick = py:recv()")
            mad_trk.send(row["hkick"])
            mad_trk.send(f"MADX.{SEQ_NAME}['{row['ename']}'].vkick = py:recv()")
            mad_trk.send(row["vkick"])

        mad_processes.append(mad_trk)

    logger.info(f"All {NUM_TRACKS} MAD processes initialized")

    # Progress tracking
    progress_interval = max(1, NUM_TRACKS // 10)  # Log progress every 10%

    # 4. Start tracking in each process (no data retrieved yet)
    logger.info("Starting tracking commands in all MAD processes")
    for ntrk, mad_trk in enumerate(mad_processes):
        if ntrk % progress_interval == 0:
            logger.info(
                f"Starting tracking command for process {ntrk}/{NUM_TRACKS - 1} ({ntrk / NUM_TRACKS * 100:.1f}%)"
            )

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

        # Compute normalised coordinates from action and angle
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        # Convert to real space coordinates
        x = np.sqrt(action * beta11) * cos_ang
        px = -np.sqrt(action / beta11) * (sin_ang + alfa11 * cos_ang)
        y = np.sqrt(action * beta22) * cos_ang
        py = -np.sqrt(action / beta22) * (sin_ang + alfa22 * cos_ang)
        # Set x, px and y, py to zero depending on whether ntrk is even or odd
        if not KICK_BOTH_PLANES:
            if ntrk % 2 == 0:
                y = 0.0
                py = 0.0
            else:
                x = 0.0
                px = 0.0

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
    deltap=true_deltap,
    nturn={FLATTOP_TURNS},
    cmap=true,
    info=1
}}
"""
        )
        logger.debug(
            f"Track {ntrk}: Started with action={action:.2e}, angle={angle:.3f}"
        )

    logger.info("All tracking commands sent, starting data processing")

    # 5. After all tracks are launched, retrieve and process each result in parallel
    # Use lock-based approach for efficient I/O

    # Start dedicated writer processes and bounded queues for backpressure
    logger.info("Starting dedicated writer processes")
    track_queue: "mp.Queue" = mp.Queue(maxsize=32)
    noise_queue: "mp.Queue" = mp.Queue(maxsize=32)
    cleaned_queue: "mp.Queue" = mp.Queue(maxsize=32)

    track_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(track_queue, str(no_noise_files[input_deltap])),
        daemon=True,
    )
    noise_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(noise_queue, str(noise_files[input_deltap])),
        daemon=True,
    )
    cleaned_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(cleaned_queue, str(CLEANED_FILE)),
        daemon=True,
    )
    track_writer_proc.start()
    noise_writer_proc.start()
    cleaned_writer_proc.start()

    # Use a thread pool to leverage independent MAD-NG results; enqueue to writers
    logger.info("Starting parallel data processing with thread pool")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map each (index, process, queue) to the helper and ensure all tasks complete
        list(
            executor.map(
                lambda args: process_track(*args),
                (
                    (i, p, track_queue, noise_queue, cleaned_queue)
                    for i, p in enumerate(mad_processes)
                ),
            )
        )

    # Signal writers to stop and wait for them to finish
    logger.info("Signaling writer processes to stop")
    track_queue.put(None)
    noise_queue.put(None)
    cleaned_queue.put(None)
    track_writer_proc.join()
    noise_writer_proc.join()
    cleaned_writer_proc.join()
    track_queue.close()
    noise_queue.close()
    cleaned_queue.close()
    track_queue.join_thread()
    noise_queue.join_thread()
    cleaned_queue.join_thread()

    iteration_time = time.time() - start_time
    logger.info(
        f"Tracking and processing for input_deltap = {input_deltap} completed in {iteration_time:.2f}s"
    )
    # Report final status
    logger.info(f"→ Saved tracking data: {no_noise_files[input_deltap]}")
    logger.info(f"→ Saved noisy data: {noise_files[input_deltap]}")
    logger.info(f"→ Saved cleaned data: {CLEANED_FILE}")

    # Clean up
    del mad_processes

# Final status
logger.info("All tracking and noisy data written to Parquet files.")

# combined_kalman.to_feather(KALMAN_FILE, compression="lz4")
# print(f"→ Saved combined Kalman-cleaned data: {KALMAN_FILE}")
# del combined_kalman

# combined_cleaned.to_feather(cleaned_FILE, compression="lz4")
# print(f"→ Saved combined cleaned data: {cleaned_FILE}")
# del combined_cleaned

# final status
logger.info(
    f"All calculations and filtering completed successfully with {NUM_TRACKS} parallel MAD-NG processes."
)

# Log total execution time
total_execution_time = time.time() - script_start_time
logger.info(f"Total script execution time: {total_execution_time:.2f}s")
logger.info("=== Script execution completed successfully ===")
