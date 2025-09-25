"""
Coordinate generation utilities for accelerator tracking simulations.

This module provides functions for generating action-angle coordinates
and creating initial conditions for particle tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tfs

logger = logging.getLogger(__name__)


def generate_action_angle_coordinates(
    num_tracks: int,
    action_range: tuple[float, float],
) -> tuple[list[float], list[float]]:
    """
    Generate action-angle coordinate pairs for tracking.

    Args:
        num_tracks: Total number of tracks needed
        action_range: Tuple of (min_action, max_action) in meters
        num_actions: Number of action values (optional, calculated if not provided)

    Returns:
        Tuple of (action_list, angle_list) where both lists have length num_tracks
    """
    # Ensure we get exactly the requested number of tracks
    action_values = np.linspace(action_range[0], action_range[1], num=num_tracks)
    angle_values = np.linspace(0.1, 2 * np.pi, num=num_tracks, endpoint=False)

    logger.info(
        f"Generated {len(action_values)} action-angle pairs for {num_tracks} tracks"
    )

    return action_values.tolist(), angle_values.tolist()


def create_initial_conditions(
    ntrk: int,
    action_list: list[float],
    angle_list: list[float],
    twiss_data: tfs.TfsDataFrame,
    kick_both_planes: bool = True,
    starting_bpm: str | int = 0,
) -> dict[str, float]:
    """
    Create initial conditions for a specific track from action-angle coordinates.

    Args:
        ntrk: Track number
        action_list: List of action values (same length as angle_list)
        angle_list: List of angle values (same length as action_list)
        twiss_data: Twiss parameters at starting point
        kick_both_planes: Whether to kick both x and y planes

    Returns:
        Dictionary with initial coordinates (x, px, y, py, t, pt)
    """
    # Direct indexing since action_list and angle_list have the same length
    action = action_list[ntrk]
    angle = angle_list[ntrk]

    # Get beta and alpha functions at starting point (first BPM)
    first_bpm = starting_bpm
    if isinstance(starting_bpm, int):
        first_bpm = twiss_data.index[starting_bpm]

    beta11 = twiss_data.loc[first_bpm, "beta11"]
    beta22 = twiss_data.loc[first_bpm, "beta22"]
    alfa11 = twiss_data.loc[first_bpm, "alfa11"]
    alfa22 = twiss_data.loc[first_bpm, "alfa22"]

    # Compute normalized coordinates from action and angle
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)

    # Convert to real space coordinates
    x = np.sqrt(action * beta11) * cos_ang
    px = -np.sqrt(action / beta11) * (sin_ang + alfa11 * cos_ang)
    y = np.sqrt(action * beta22) * cos_ang
    py = -np.sqrt(action / beta22) * (sin_ang + alfa22 * cos_ang)

    # Set coordinates to zero depending on kick plane strategy
    if not kick_both_planes:
        if ntrk % 2 == 0:
            y = 0.0
            py = 0.0
        else:
            x = 0.0
            px = 0.0

    logger.debug(
        f"Track {ntrk}: Created initial conditions with action={action:.2e}, angle={angle:.3f}"
    )

    return {
        "x": x,
        "px": px,
        "y": y,
        "py": py,
        "t": 0.0,
        "pt": 0.0,
    }


def get_kick_plane_category(ntrk: int, kick_both_planes: bool) -> str:
    """
    Determine kick plane category for a track.

    Args:
        ntrk: Track number
        kick_both_planes: Whether to kick both planes

    Returns:
        Kick plane category string ("xy", "x", or "y")
    """
    if kick_both_planes:
        return "xy"
    return "x" if ntrk % 2 == 0 else "y"


def validate_coordinate_generation(
    num_tracks: int, action_list: list[float], angle_list: list[float]
) -> bool:
    """
    Validate that action-angle coordinate generation produces the expected number of tracks.

    Args:
        num_tracks: Expected number of tracks
        action_list: Generated action values
        angle_list: Generated angle values

    Returns:
        True if validation passes

    Raises:
        AssertionError: If validation fails
    """
    assert len(action_list) == num_tracks, (
        f"Expected {num_tracks} action values, got {len(action_list)}."
    )
    assert len(angle_list) == num_tracks, (
        f"Expected {num_tracks} angle values, got {len(angle_list)}."
    )
    assert len(action_list) == len(angle_list), (
        f"Action and angle lists must have the same length, got {len(action_list)} and {len(angle_list)}."
    )
    return True
