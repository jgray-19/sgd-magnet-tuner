"""
Data processing utilities for accelerator simulation data.

This module provides functions for processing tracking data, adding noise,
applying SVD cleaning, and writing data to files.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aba_optimiser.filtering.svd import svd_clean_measurements

# from aba_optimiser.physics.dispersive_momentum_reconstruction import calculate_pz
from aba_optimiser.momentum_recon.transverse import calculate_pz, inject_noise_xy

from .coordinates import get_kick_plane_category

if TYPE_CHECKING:
    import multiprocessing as mp

    import pandas as pd
    import tfs

logger = logging.getLogger(__name__)


def single_writer_loop(queue: mp.Queue, out_path: str) -> None:
    """
    Dedicated writer: consumes Arrow Tables and writes row groups to one Parquet file.

    Args:
        queue: Multiprocessing queue with Arrow tables
        out_path: Output file path
    """
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
            # Explicitly clear the table reference to help with memory cleanup
            del table
    except Exception as e:
        logger.error(f"Error in writer process for {out_path}: {e}")
        raise
    finally:
        if writer is not None:
            logger.info(f"Closing ParquetWriter for {out_path}")
            writer.close()


def process_track_with_queue(
    ntrk: int,
    true_df: pd.DataFrame,
    # tws: tfs.TfsDataFrame,
    track_q: mp.Queue,
    noise_q: mp.Queue,
    cleaned_q: mp.Queue,
    flattop_turns: int,
    kick_both_planes: bool,
) -> None:
    """
    Process a single MAD tracking result and enqueue Arrow tables for a
    dedicated writer process to persist. Receives true_df and tws directly.

    Args:
        ntrk: Track number
        true_df: DataFrame with tracking data
        tws: Twiss parameters indexed by element name
        track_q: Queue for clean tracking data
        noise_q: Queue for noisy data
        cleaned_q: Queue for cleaned data
        flattop_turns: Number of turns per track
        kick_both_planes: Whether to kick both planes
    """
    logger.debug(f"Processing track {ntrk}")
    start_time = time.time()

    try:
        # Adjust turn count
        true_df["turn"] += ntrk * flattop_turns
        # Downcast for memory
        true_df["turn"] = true_df["turn"].astype(np.int32)

        # Add a new category column that indicates if it is a x or y kick
        true_df["kick_plane"] = get_kick_plane_category(ntrk, kick_both_planes)
        true_df["x_weight"] = 1.0
        true_df["y_weight"] = 1.0

        # Enqueue tracking data as Arrow table
        track_table = pa.Table.from_pandas(true_df, preserve_index=False)
        track_q.put(track_table)
        logger.debug(f"Track {ntrk}: Enqueued tracking data table")
        del track_table

        # Add noise and enqueue noisy data
        true_df["name"] = true_df["name"].astype("category")
        true_df["kick_plane"] = true_df["kick_plane"].astype("category")

        # noisy_df = calculate_pz(true_df, tws=tws, info=True)
        # noisy_df["name"] = noisy_df["name"].astype(str)
        # noisy_df["kick_plane"] = noisy_df["kick_plane"].astype(str)
        # noise_table = pa.Table.from_pandas(noisy_df, preserve_index=False)
        # noise_q.put(noise_table)
        # del noise_table

        noisy_df = true_df.copy()
        inject_noise_xy(noisy_df, true_df, np.random.default_rng(), [])

        # Filter the noisy data and enqueue cleaned data
        cleaned_df = svd_clean_measurements(noisy_df)
        cleaned_df = calculate_pz(
            cleaned_df, inject_noise=False, info=False, subtract_mean=True
        )
        del noisy_df  # Clean up noisy_df after its last use

        cleaned_df["name"] = cleaned_df["name"].astype(str)
        cleaned_df["kick_plane"] = cleaned_df["kick_plane"].astype(str)

        cleaned_table = pa.Table.from_pandas(cleaned_df, preserve_index=False)
        cleaned_q.put(cleaned_table)
        del cleaned_table

        # Print the differences in x, y, px, py between noisy and cleaned for debugging
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
        del (
            diff_x,
            diff_y,
            diff_px,
            diff_py,
            true_df,
            cleaned_df,
        )  # Clean up difference arrays

        logger.debug(f"Track {ntrk}: Enqueued noisy data table")
        processing_time = time.time() - start_time
        logger.info(f"Track {ntrk}: Processing completed in {processing_time:.2f}s")

        gc.collect()

    except Exception as e:
        logger.error(f"Error processing track {ntrk}: {e}")
        raise


def process_track(
    ntrk: int,
    true_df: pd.DataFrame,
    # tws: tfs.TfsDataFrame,
    track_q: mp.Queue,
    noise_q: mp.Queue,
    cleaned_q: mp.Queue,
) -> None:
    """
    Process a single MAD tracking result.
    Routes to the appropriate implementation based on configuration.

    Args:
        ntrk: Track number
        true_df: DataFrame with tracking data
        tws: Twiss parameters indexed by element name
        track_q: Queue for clean tracking data
        noise_q: Queue for noisy data
        cleaned_q: Queue for cleaned data
    """
    # Import configuration at runtime to avoid circular imports
    from aba_optimiser.config import FLATTOP_TURNS, KICK_BOTH_PLANES

    # Use the single-writer queue approach for better I/O efficiency
    return process_track_with_queue(
        ntrk, true_df, track_q, noise_q, cleaned_q, FLATTOP_TURNS, KICK_BOTH_PLANES
    )


def prepare_track_dataframe(
    true_df: pd.DataFrame, ntrk: int, flattop_turns: int, kick_both_planes: bool
) -> pd.DataFrame:
    """
    Prepare tracking dataframe with metadata.

    Args:
        true_df: Raw tracking dataframe
        ntrk: Track number
        flattop_turns: Number of turns per track
        kick_both_planes: Whether to kick both planes

    Returns:
        Prepared dataframe with metadata
    """
    # Adjust turn count
    true_df["turn"] += ntrk * flattop_turns
    true_df["turn"] = true_df["turn"].astype(np.int32)

    # Add kick plane category
    true_df["kick_plane"] = get_kick_plane_category(ntrk, kick_both_planes)
    true_df["x_weight"] = 1.0
    true_df["y_weight"] = 1.0

    return true_df


def add_noise_and_clean(
    true_df: pd.DataFrame, tws: tfs.TfsDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add noise to tracking data and apply SVD cleaning.

    Args:
        true_df: Clean tracking dataframe
        tws: Twiss parameters indexed by element name

    Returns:
        Tuple of (noisy_df, cleaned_df)
    """
    # Prepare data types for noise injection
    true_df["name"] = true_df["name"].astype("category")
    true_df["kick_plane"] = true_df["kick_plane"].astype("category")

    # Add noise
    noisy_df = calculate_pz(true_df, tws=tws, info=False)
    noisy_df["name"] = noisy_df["name"].astype(str)
    noisy_df["kick_plane"] = noisy_df["kick_plane"].astype(str)

    # Apply SVD cleaning
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_df = calculate_pz(cleaned_df, inject_noise=False, tws=tws, info=True)

    return noisy_df, cleaned_df


def calculate_differences(
    true_df: pd.DataFrame, cleaned_df: pd.DataFrame, ntrk: int
) -> None:
    """
    Calculate and log differences between true and cleaned data.

    Args:
        true_df: True tracking data
        cleaned_df: Cleaned tracking data
        ntrk: Track number
    """
    diff_x = true_df["x"] - cleaned_df["x"]
    diff_y = true_df["y"] - cleaned_df["y"]
    diff_px = true_df["px"] - cleaned_df["px"]
    diff_py = true_df["py"] - cleaned_df["py"]

    logger.info(
        f"Track {ntrk}: True vs cleaned x diff: mean={diff_x.mean():.3e}, std={diff_x.std():.3e}; "
        f"y diff: mean={diff_y.mean():.3e}, std={diff_y.std():.3e}; "
        f"px diff: mean={diff_px.mean():.3e}, std={diff_px.std():.3e}; "
        f"py diff: mean={diff_py.mean():.3e}, std={diff_py.std():.3e}"
    )
