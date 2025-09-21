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
    num_tracks: int, action_range: tuple[float, float], num_actions: int = None
) -> tuple[list[float], list[float]]:
    """
    Generate action-angle coordinate pairs for tracking.

    Args:
        num_tracks: Total number of tracks needed
        action_range: Tuple of (min_action, max_action) in meters
        num_actions: Number of action values (optional, calculated if not provided)

    Returns:
        Tuple of (action_list, angle_list)
    """
    if num_actions is None:
        num_actions = max(1, num_tracks // 5)

    num_angles = max(1, num_tracks // num_actions)

    # Ensure we get exactly the requested number of tracks
    actual_tracks = num_actions * num_angles
    if actual_tracks != num_tracks:
        logger.warning(
            f"Requested {num_tracks} tracks, but action-angle grid gives {actual_tracks} tracks"
        )

    action_list = np.linspace(action_range[0], action_range[1], num=num_actions)
    angle_list = np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False)

    logger.info(
        f"Generated {len(action_list)} action values and {len(angle_list)} angle values"
    )
    logger.info(f"Total tracks to process: {len(action_list) * len(angle_list)}")

    return action_list.tolist(), angle_list.tolist()


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
        action_list: List of action values
        angle_list: List of angle values
        twiss_data: Twiss parameters at starting point
        kick_both_planes: Whether to kick both x and y planes

    Returns:
        Dictionary with initial coordinates (x, px, y, py, t, pt)
    """
    # Determine action and angle indices
    idx_action = ntrk // len(angle_list)
    idx_angle = ntrk % len(angle_list)
    action = action_list[idx_action]
    angle = angle_list[idx_angle]

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
    actual_tracks = len(action_list) * len(angle_list)
    assert actual_tracks == num_tracks, (
        f"Expected {num_tracks} tracks, got {actual_tracks}."
    )
    return True
