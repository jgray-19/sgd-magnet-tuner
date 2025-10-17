"""
Simulation module for creating simulated accelerator data.

This module contains components for setting up MAD-NG simulations,
applying magnet perturbations, calculating beam optics, generating
tracking coordinates, and processing simulation data.

Recommended approach: Use the interface classes directly from aba_optimiser.mad
for MAD operations, which provide better encapsulation and object-oriented design.
"""

from .coordinates import (
    create_initial_conditions,
    generate_action_angle_coordinates,
    validate_coordinate_generation,
)
from .data_processing import process_track, single_writer_loop
from .mad_setup import (
    create_mad_interface,
    select_bpms,
    setup_tracking_interface,
)
from .magnet_perturbations import apply_magnet_perturbations
from .optics import (
    calculate_beta_beating,
    match_tunes,
    perform_orbit_correction,
    run_initial_twiss_analysis,
    save_knobs,
)
from .tracking import (
    cleanup_writer_processes,
    run_parallel_tracking,
    setup_writer_processes,
)

__all__ = [
    # Modern interface-based functions
    "create_mad_interface",
    "setup_tracking_interface",
    "select_bpms",
    # Simulation functions
    "apply_magnet_perturbations",
    "calculate_beta_beating",
    "calculate_twiss",
    "match_tunes",
    "perform_orbit_correction",
    "run_initial_twiss_analysis",
    "save_knobs",
    "generate_action_angle_coordinates",
    "create_initial_conditions",
    "validate_coordinate_generation",
    "run_parallel_tracking",
    "setup_writer_processes",
    "cleanup_writer_processes",
    "process_track",
    "single_writer_loop",
]
