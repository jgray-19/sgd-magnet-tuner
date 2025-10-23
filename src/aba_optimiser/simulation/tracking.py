"""
Parallel tracking utilities for accelerator simulations.

This module provides functions for setting up multiprocessing tracking,
managing queues, and handling batch processing of MAD tracking jobs.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import tfs

from aba_optimiser.mad.tracking_interface import TrackingMadInterface
from aba_optimiser.simulation.coordinates import create_initial_conditions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    import tfs


def setup_writer_processes(
    no_noise_file: Path, noise_file: Path, cleaned_file: Path
) -> tuple[mp.Process, mp.Process, mp.Process, mp.Queue, mp.Queue, mp.Queue]:
    """
    Set up dedicated writer processes and bounded queues for data output.

    Args:
        no_noise_file: Path for no-noise data output
        noise_file: Path for noisy data output
        cleaned_file: Path for cleaned data output

    Returns:
        Tuple of (track_writer, noise_writer, cleaned_writer, track_queue, noise_queue, cleaned_queue)
    """
    from .data_processing import single_writer_loop

    logger.info("Starting dedicated writer processes")

    # Create bounded queues for backpressure
    track_queue = mp.Queue(maxsize=32)
    noise_queue = mp.Queue(maxsize=32)
    cleaned_queue = mp.Queue(maxsize=32)

    # Create writer processes
    track_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(track_queue, str(no_noise_file)),
        daemon=True,
    )
    noise_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(noise_queue, str(noise_file)),
        daemon=True,
    )
    cleaned_writer_proc = mp.Process(
        target=single_writer_loop,
        args=(cleaned_queue, str(cleaned_file)),
        daemon=True,
    )

    # Start processes
    track_writer_proc.start()
    noise_writer_proc.start()
    cleaned_writer_proc.start()

    return (
        track_writer_proc,
        noise_writer_proc,
        cleaned_writer_proc,
        track_queue,
        noise_queue,
        cleaned_queue,
    )


def cleanup_writer_processes(writers_and_queues: tuple) -> None:
    """
    Signal writers to stop and clean up processes and queues.

    Args:
        writers_and_queues: Tuple returned from setup_writer_processes
    """
    (
        track_writer_proc,
        noise_writer_proc,
        cleaned_writer_proc,
        track_queue,
        noise_queue,
        cleaned_queue,
    ) = writers_and_queues

    logger.info("Signaling writer processes to stop")

    # Send stop signals
    track_queue.put(None)
    noise_queue.put(None)
    cleaned_queue.put(None)

    # Wait for processes to finish
    track_writer_proc.join()
    noise_writer_proc.join()
    cleaned_writer_proc.join()

    # Close and cleanup queues
    track_queue.close()
    noise_queue.close()
    cleaned_queue.close()
    track_queue.join_thread()
    noise_queue.join_thread()
    cleaned_queue.join_thread()


def initialise_mad_batch(
    batch_start: int,
    batch_end: int,
    sequence_file: Path,
    seq_name: str,
    beam_energy: float,
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
) -> list[TrackingMadInterface]:
    """
    Initialise a batch of MAD processes for parallel tracking.

    Args:
        batch_start: Starting track index
        batch_end: Ending track index
        sequence_file: Path to sequence file
        seq_name: Sequence name
        beam_energy: Beam energy in GeV
        matched_tunes: Dictionary of matched tune knobs
        magnet_strengths: Dictionary of magnet strengths
        corrector_table: DataFrame with corrector strengths

    Returns:
        List of initialised TrackingMadInterface instances
    """
    batch_size = batch_end - batch_start
    logger.info(f"Initializing {batch_size} MAD interfaces for batch")

    interfaces = []

    for _ in range(batch_start, batch_end):
        # Initialise new tracking interface
        interface = TrackingMadInterface(
            debug=False, stdout="/dev/null", redirect_stderr=True
        )

        # Load sequence and setup beam
        interface.load_sequence(sequence_file, seq_name)
        interface.setup_beam(beam_energy)

        # Set up observation for BPMs
        interface.observe_elements("BPM")

        # Set tune knobs
        interface.set_madx_variables(**matched_tunes)

        # Set magnet strengths
        for str_name, strength in magnet_strengths.items():
            magnet_name, var = str_name.rsplit(".", 1)
            logger.debug(f"Setting {magnet_name.lower()} {var} to {strength}")
            interface.set_variables(**{f"MADX['{magnet_name}'].{var}": strength})

        # Apply corrector strengths
        interface.apply_corrector_strengths(corrector_table)

        interfaces.append(interface)

    return interfaces


def start_tracking_batch(
    interfaces: list[TrackingMadInterface],
    batch_start: int,
    action_list: list[float],
    angle_list: list[float],
    twiss_data: tfs.TfsDataFrame,
    kick_both_planes: bool,
    flattop_turns: int,
    progress_interval: int,
    num_tracks: int,
    true_deltap: float,
) -> None:
    """
    Start tracking commands for a batch of MAD interfaces.

    Args:
        interfaces: List of TrackingMadInterface instances
        batch_start: Starting track index
        action_list: List of action values
        angle_list: List of angle values
        twiss_data: Twiss parameters
        kick_both_planes: Whether to kick both planes
        flattop_turns: Number of turns to track
        progress_interval: Interval for progress logging
        num_tracks: Total number of tracks
    """
    logger.info("Starting tracking commands for batch")

    for batch_idx, interface in enumerate(interfaces):
        ntrk = batch_start + batch_idx

        if ntrk % progress_interval == 0:
            logger.info(
                f"Starting tracking command for process {ntrk}/{num_tracks - 1} "
                f"({ntrk / num_tracks * 100:.1f}%)"
            )

        # Create initial conditions
        x0_data = create_initial_conditions(
            ntrk, action_list, angle_list, twiss_data, kick_both_planes
        )

        # Run tracking using interface
        interface.run_tracking(
            x0=x0_data["x"],
            px0=x0_data["px"],
            y0=x0_data["y"],
            py0=x0_data["py"],
            t0=0,
            pt0=interface.dp2pt(true_deltap),
            nturns=flattop_turns,
        )

        action = action_list[ntrk]
        angle = angle_list[ntrk]
        logger.debug(
            f"Track {ntrk}: Started with action={action:.2e}, angle={angle:.3f}"
        )


def retrieve_tracking_data(interfaces: list[TrackingMadInterface]) -> list:
    """
    Retrieve tracking data from completed MAD tracking interfaces.

    Args:
        interfaces: List of TrackingMadInterface instances

    Returns:
        List of tracking data DataFrames
    """
    logger.info("Retrieving tracking data from all interfaces")

    data_list = []
    for interface in interfaces:
        # Get tracking data using interface method
        tracking_data = interface.get_tracking_data()
        data_list.append(tracking_data)

    return data_list


def run_parallel_tracking(
    batch_start: int,
    batch_end: int,
    sequence_file: Path,
    seq_name: str,
    beam_energy: float,
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
    true_deltap: float,
    action_list: list[float],
    angle_list: list[float],
    twiss_data: tfs.TfsDataFrame,
    kick_both_planes: bool,
    flattop_turns: int,
    track_queue: mp.Queue,
    noise_queue: mp.Queue,
    cleaned_queue: mp.Queue,
    progress_interval: int,
    num_tracks: int,
    use_xsuite: bool = False,
) -> None:
    """
    Run parallel tracking for a batch of tracks.

    Args:
        batch_start: Starting track index
        batch_end: Ending track index
        sequence_file: Path to sequence file
        seq_name: Sequence name
        beam_energy: Beam energy in GeV
        matched_tunes: Dictionary of matched tune knobs
        magnet_strengths: Dictionary of magnet strengths
        corrector_table: DataFrame with corrector strengths
        true_deltap: True momentum deviation
        action_list: List of action values
        angle_list: List of angle values
        twiss_data: Twiss parameters
        kick_both_planes: Whether to kick both planes
        flattop_turns: Number of turns to track
        track_queue: Queue for tracking data
        noise_queue: Queue for noisy data
        cleaned_queue: Queue for cleaned data
        progress_interval: Interval for progress logging
        num_tracks: Total number of tracks
    """
    from .data_processing import process_track

    if not use_xsuite:
        # Initialise MAD interfaces
        interfaces = initialise_mad_batch(
            batch_start,
            batch_end,
            sequence_file,
            seq_name,
            beam_energy,
            matched_tunes,
            magnet_strengths,
            corrector_table,
        )

        # Start tracking
        start_tracking_batch(
            interfaces,
            batch_start,
            action_list,
            angle_list,
            twiss_data,
            kick_both_planes,
            flattop_turns,
            progress_interval,
            num_tracks,
            true_deltap,
        )

        # Retrieve data
        true_dfs = retrieve_tracking_data(interfaces)
        interfaces.clear()
    else:
        from aba_optimiser.xsuite.xsuite_tools import (
            initialise_env,
            line_to_dataframes,
            start_tracking_xsuite_batch,
        )

        # Create xsuite environment
        logger.info("Creating xsuite environment")
        env = initialise_env(matched_tunes, magnet_strengths, corrector_table)

        # Set up the beam line with corrections
        logger.info("Setting up beam line with orbit correction")
        tracked_line = start_tracking_xsuite_batch(
            env=env,
            batch_start=batch_start,
            batch_end=batch_end,
            action_list=action_list,
            angle_list=angle_list,
            twiss_data=twiss_data,
            kick_both_planes=kick_both_planes,
            flattop_turns=flattop_turns,
            progress_interval=progress_interval,
            num_tracks=num_tracks,
            true_deltap=true_deltap,
        )

        true_dfs = line_to_dataframes(tracked_line)

    # Process data in parallel
    logger.info("Starting parallel data processing with thread pool for batch")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            executor.map(
                lambda args: process_track(*args),
                (
                    (
                        batch_start + i,
                        true_dfs[i],
                        # twiss_data,
                        track_queue,
                        noise_queue,
                        cleaned_queue,
                    )
                    for i in range(len(true_dfs))
                ),
            )
        )

    # Clean up
    del true_dfs
    del twiss_data
    gc.collect()

    # Log memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug(
        f"Memory usage after batch: RSS={mem_info.rss / 1024 / 1024:.1f}MB, "
        f"VMS={mem_info.vms / 1024 / 1024:.1f}MB"
    )
