import argparse
import contextlib
import gc
import logging
import time

import tfs

from aba_optimiser.config import (
    BEAM_ENERGY,
    CLEANED_FILE,
    CORRECTOR_STRENGTHS,
    LHCB1_SEQ_NAME,
    NO_NOISE_FILE,
    NOISY_FILE,
    TRUE_STRENGTHS_FILE,
    TUNE_KNOBS_FILE,
)
from aba_optimiser.io.utils import get_lhc_file_path, save_knobs
from aba_optimiser.simulation import (
    apply_magnet_perturbations,
    cleanup_writer_processes,
    create_mad_interface,
    generate_action_angle_coordinates,
    match_tunes,
    perform_orbit_correction,
    run_initial_twiss_analysis,
    run_parallel_tracking,
    select_bpms,
    setup_writer_processes,
    validate_coordinate_generation,
)

# Note: the config constants are intentionally not imported here so that the
# main `create_a34` function can accept them as parameters. The legacy
# behaviour is preserved by importing the config values in the __main__ guard
# and passing them through.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("create_a34.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_a34(
    flattop_turns,
    kick_both_planes,
    machine_deltap,
    num_tracks,
    rel_k1_std_dev,
    track_batch_size,
    use_xsuite,
):
    """Run the create_a34 workflow.

    Parameters correspond to the former imports from aba_optimiser.config. All
    file path arguments (sequence file and SEQ_NAME) are still supplied so the
    function can be called programmatically.
    """
    # Log script start
    logger.info("=== Starting create_a34.py script execution ===")
    logger.info(f"Using tracking backend: {'xsuite' if use_xsuite else 'MAD-NG'}")
    script_start_time = time.time()

    # MAD-NG backend initialization
    logger.info("Initializing MAD-NG interface")
    mad_interface = create_mad_interface(
        debug=False, stdout="/dev/null", redirect_stderr=True
    )
    mad = mad_interface.mad  # For backward compatibility with existing code

    # Load the sequence
    mad_interface.load_sequence(get_lhc_file_path(1), LHCB1_SEQ_NAME)

    # Set beam
    mad_interface.setup_beam(BEAM_ENERGY)
    mad_interface.observe_elements()

    # Run an initial twiss
    logger.info("Running initial twiss calculation")
    ini_tws = mad_interface.run_twiss()
    ini_tws = select_bpms(ini_tws)
    logger.info(f"Initial twiss completed. Found {len(ini_tws.index)} BPMs")
    logger.info(f"Initial tunes: Qx={ini_tws.q1}, Qy={ini_tws.q2}")

    # Apply magnet perturbations
    tunes = [0.28, 0.31]
    magnet_strengths, true_strengths = apply_magnet_perturbations(
        mad,
        rel_k1_std_dev,
        seed=43,
        # seed=time.time_ns(),
    )

    # Perform orbit correction and tune rematching
    matched_tunes = perform_orbit_correction(
        mad, 0, tunes[0], tunes[1], CORRECTOR_STRENGTHS
    )

    # Run twiss after magnet perturbations and analyze
    changed_tws = run_initial_twiss_analysis(mad_interface.run_twiss(), ini_tws)

    # Define parameters
    logger.info(f"Target tunes: Qx={62 + tunes[0]}, Qy={60 + tunes[1]}")

    # Match tunes before adding deltap
    match_tunes(mad, tunes[0], tunes[1], deltap=0)

    # Perform orbit correction and tune rematching
    matched_tunes = perform_orbit_correction(
        mad, machine_deltap, tunes[0], tunes[1], CORRECTOR_STRENGTHS
    )

    # Save matched tunes to file
    save_knobs(matched_tunes, TUNE_KNOBS_FILE)

    # Run final twiss calculation after orbit correction
    logger.info("Running final twiss calculation after orbit correction")
    changed_tws = mad_interface.run_twiss(deltap=machine_deltap)
    changed_tws = select_bpms(changed_tws)
    logger.info(f"Final twiss completed. Qx={changed_tws.q1}, Qy={changed_tws.q2}")

    # Save true magnet strengths
    true_strengths = {k + "_k": v for k, v in true_strengths.items()}
    save_knobs(true_strengths, TRUE_STRENGTHS_FILE)

    # Generate action-angle coordinates for tracking
    logger.info("Setting up action-angle tracking parameters")
    action_range = (1e-8, 1e-8)
    action_list, angle_list = generate_action_angle_coordinates(
        num_tracks, action_range
    )
    logging.info(
        f"Generated {action_list[0]:.2e} to {action_list[-1]:.2e} m action values"
    )
    logging.info(
        f"Generated {angle_list[0]:.2f} to {angle_list[-1]:.2f} rad angle values"
    )
    validate_coordinate_generation(num_tracks, action_list, angle_list)

    no_noise_files = {
        0: NO_NOISE_FILE,
    }
    noise_files = {
        0: NOISY_FILE,
    }

    cleaned_files = {
        0: CLEANED_FILE,
    }

    logger.info("Loading corrector strengths table")
    corrector_table = tfs.read(CORRECTOR_STRENGTHS)
    # Remove all rows that have monitor in the column kind
    corrector_table = corrector_table[corrector_table["kind"] != "monitor"]
    logger.info(f"Loaded {len(corrector_table)} corrector elements")

    # Tracking loop for different deltap values
    # for input_deltap in [-DELTAP, 0, DELTAP]:
    for input_deltap in [0.0]:
        logger.info(f"Starting tracking for input_deltap = {input_deltap}")

        # Calculate twiss before tracking
        true_deltap = input_deltap + machine_deltap
        logger.info(f"True deltap for this iteration: {true_deltap}")
        df_twiss = mad_interface.run_twiss(deltap=true_deltap, observe=0)
        logger.info(df_twiss.headers)
        logger.info(f"Pre-tracking twiss completed with {len(df_twiss)} elements")

        # Set up writer processes and queues
        writers_and_queues = setup_writer_processes(
            no_noise_files[input_deltap],
            noise_files[input_deltap],
            cleaned_files[input_deltap],
        )

        # Progress tracking
        progress_interval = max(1, num_tracks // 10)  # Log progress every 10%

        # Process tracks in batches to save RAM
        for batch_start in range(0, num_tracks, track_batch_size):
            batch_end = min(batch_start + track_batch_size, num_tracks)
            logger.info(
                f"Processing batch {batch_start // track_batch_size + 1}: tracks {batch_start} to {batch_end - 1}"
            )

            # Extract queues from the tuple
            _, _, _, track_queue, noise_queue, cleaned_queue = writers_and_queues

            # Run parallel tracking for this batch
            run_parallel_tracking(
                batch_start,
                batch_end,
                get_lhc_file_path(beam=1),
                LHCB1_SEQ_NAME,
                BEAM_ENERGY,
                matched_tunes,
                magnet_strengths,
                corrector_table,
                true_deltap,
                action_list,
                angle_list,
                df_twiss,
                kick_both_planes=kick_both_planes,
                flattop_turns=flattop_turns,
                track_queue=track_queue,
                noise_queue=noise_queue,
                cleaned_queue=cleaned_queue,
                progress_interval=progress_interval,
                num_tracks=num_tracks,
                use_xsuite=use_xsuite,
            )

            logger.info(
                f"Completed processing batch {batch_start // track_batch_size + 1}"
            )
            gc.collect()  # Clean up memory after each batch

        # Clean up writer processes
        cleanup_writer_processes(writers_and_queues)

        # Clean up per-input_deltap data
        del df_twiss

    # Final status
    logger.info("All tracking and noisy data written to Parquet files.")

    # Final status
    logger.info(
        f"All calculations and filtering completed successfully with {num_tracks} parallel MAD-NG processes."
    )

    # Log total execution time
    total_execution_time = time.time() - script_start_time
    logger.info(f"Total script execution time: {total_execution_time:.2f}s")
    logger.info("=== Script execution completed successfully ===")

    # Clean up main MAD instance
    logger.info("Cleaning up main MAD instance")
    del mad

    # Clean up local data structures
    del corrector_table
    del action_list
    del angle_list
    del true_strengths
    del matched_tunes
    del magnet_strengths
    del ini_tws
    del tunes

    # Clean up any remaining twiss data
    with contextlib.suppress(NameError):
        del changed_tws

    # Final memory cleanup
    gc.collect()


if __name__ == "__main__":
    # Parse command line arguments (only for xsuite flag default)
    parser = argparse.ArgumentParser(
        description="Create A34 tracking data using MAD-NG or xsuite"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=6600,
        help="Number of flattop turns for tracking (default: 6600)",
    )
    parser.add_argument(
        "--kick-both-planes",
        action="store_true",
        default=True,
        help="Kick in both planes during tracking (default: True)",
    )
    parser.add_argument(
        "--machine-deltap",
        type=float,
        default=0.0,
        help="Machine deltap value (default: 0.0)",
    )
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=3,
        help="Number of parallel tracks (could be considered bunches) to simulate (default: 3)",
    )
    parser.add_argument(
        "--rel-k1-std-dev",
        type=float,
        default=1e-4,
        help="Relative K1 standard deviation for magnet perturbations (default: 1e-4)",
    )
    parser.add_argument(
        "--track-batch-size",
        type=int,
        default=3,
        help="Number of tracks to process in each batch to save RAM (default: 3)",
    )
    parser.add_argument(
        "--xsuite",
        action="store_true",
        default=True,
        help="Use xsuite for tracking instead of MAD-NG",
    )
    args = parser.parse_args()

    create_a34(
        args.turns,
        args.kick_both_planes,
        args.machine_deltap,
        args.num_tracks,
        args.rel_k1_std_dev,
        args.track_batch_size,
        args.xsuite,
    )
