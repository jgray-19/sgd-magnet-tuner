"""
Data processing utilities for accelerator simulation data.

This module provides functions for processing tracking data, adding noise,
applying weighted SVD cleaning, and writing data to files.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aba_optimiser.config import MOMENTUM_STD_DEV, POSITION_STD_DEV

if TYPE_CHECKING:
    import multiprocessing as mp

    import pandas as pd

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


def prepare_track_dataframe(
    true_df: pd.DataFrame, ntrk: int, flattop_turns: int
) -> pd.DataFrame:
    """
    Prepare tracking dataframe with metadata.

    Args:
        true_df: Raw tracking dataframe
        ntrk: Track number
        flattop_turns: Number of turns per track

    Returns:
        Prepared dataframe with metadata
    """
    # Adjust turn count
    true_df["turn"] += ntrk * flattop_turns
    true_df["turn"] = true_df["turn"].astype(np.int32)
    true_df["bunch_number"] = np.int32(ntrk)

    # Add kick plane category
    true_df["var_x"] = POSITION_STD_DEV**2
    true_df["var_y"] = POSITION_STD_DEV**2
    true_df["var_px"] = MOMENTUM_STD_DEV**2
    true_df["var_py"] = MOMENTUM_STD_DEV**2

    return true_df
