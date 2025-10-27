"""
Tests for aba_optimiser.simulation.coordinates module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tfs

from aba_optimiser.simulation.coordinates import (
    create_initial_conditions,
    generate_action_angle_coordinates,
    get_kick_plane_category,
    validate_coordinate_generation,
)


# Tests for generate_action_angle_coordinates function
@pytest.mark.parametrize("num_tracks", [1, 5, 10])
def test_generate_action_angle_coordinates_correct_number(num_tracks: int) -> None:
    """Test that function generates the requested number of coordinate pairs."""
    action_range = (1e-6, 1e-4)
    actions, angles = generate_action_angle_coordinates(num_tracks, action_range)

    assert len(actions) == num_tracks
    assert len(angles) == num_tracks


def test_generate_action_angle_coordinates_actions_span_range() -> None:
    """Test that generated actions cover the specified range."""
    action_range = (1e-6, 1e-4)
    actions, angles = generate_action_angle_coordinates(5, action_range)

    assert min(actions) == action_range[0]
    assert max(actions) == action_range[1]


def test_generate_action_angle_coordinates_angles_valid_range() -> None:
    """Test that angles are in valid range (0, 2π)."""
    actions, angles = generate_action_angle_coordinates(10, (1e-6, 1e-4))

    assert all(0 < angle < 2 * np.pi for angle in angles)


# Tests for create_initial_conditions function
@pytest.fixture
def twiss_data_multiple_bpms() -> tfs.TfsDataFrame:
    """Create Twiss DataFrame with multiple BPMs for testing starting_bpm parameter."""
    data = {
        "beta11": [10.0, 12.0],
        "beta22": [8.0, 9.0],
        "alfa11": [0.1, 0.2],
        "alfa22": [-0.05, 0.1],
        "x": [1e-6, 2e-6],
        "px": [0.0, 1e-6],
        "y": [0.5e-6, 1e-6],
        "py": [-0.2e-6, 0.1e-6],
    }
    df = pd.DataFrame(data, index=["BPM1", "BPM2"])
    return tfs.TfsDataFrame(df)


@pytest.mark.parametrize(
    "starting_bpm,kick_both_planes",
    [
        ("BPM1", True),
        ("BPM1", False),
        ("BPM2", True),
        ("BPM2", False),
        (0, True),  # Test integer indexing
        (1, False),
    ],
)
def test_create_initial_conditions_closed_orbit_and_starting_bpm(
    twiss_data_multiple_bpms: tfs.TfsDataFrame, starting_bpm, kick_both_planes: bool
) -> None:
    """Test that coordinates properly include closed orbit and starting_bpm works."""
    action_list = [1e-6]
    angle_list = [0.5]

    # Determine expected BPM name
    if isinstance(starting_bpm, str):
        expected_bpm = starting_bpm
    else:
        expected_bpm = twiss_data_multiple_bpms.index[starting_bpm]

    # Create Twiss data without closed orbit (all zeros)
    twiss_data_no_co = twiss_data_multiple_bpms.copy()
    twiss_data_no_co.loc[:, ["x", "px", "y", "py"]] = 0.0

    # Get results with and without closed orbit
    result_with_co = create_initial_conditions(
        0,
        action_list,
        angle_list,
        twiss_data_multiple_bpms,
        kick_both_planes=kick_both_planes,
        starting_bpm=starting_bpm,
    )
    result_without_co = create_initial_conditions(
        0,
        action_list,
        angle_list,
        twiss_data_no_co,
        kick_both_planes=kick_both_planes,
        starting_bpm=starting_bpm,
    )

    # Check that function returns all required coordinates
    required_keys = {"x", "px", "y", "py", "t", "pt"}
    assert set(result_with_co.keys()) == required_keys
    assert all(isinstance(v, float | np.floating) for v in result_with_co.values())

    # Check that t and pt are zero as per implementation
    assert result_with_co["t"] == 0.0
    assert result_with_co["pt"] == 0.0

    # Extract closed orbit values for the expected BPM
    cox = twiss_data_multiple_bpms.loc[expected_bpm, "x"]
    copx = twiss_data_multiple_bpms.loc[expected_bpm, "px"]
    coy = twiss_data_multiple_bpms.loc[expected_bpm, "y"]
    copy = twiss_data_multiple_bpms.loc[expected_bpm, "py"]

    # Verify: result_with_co = result_without_co + closed_orbit
    assert np.isclose(result_with_co["x"], result_without_co["x"] + cox)
    assert np.isclose(result_with_co["px"], result_without_co["px"] + copx)
    assert np.isclose(result_with_co["y"], result_without_co["y"] + coy)
    assert np.isclose(result_with_co["py"], result_without_co["py"] + copy)


# Tests for get_kick_plane_category function
@pytest.mark.parametrize(
    "ntrk,kick_both_planes,expected",
    [
        (0, True, "xy"),
        (1, True, "xy"),
        (0, False, "x"),
        (1, False, "y"),
        (2, False, "x"),
        (3, False, "y"),
    ],
)
def test_get_kick_plane_category(
    ntrk: int, kick_both_planes: bool, expected: str
) -> None:
    """Test kick plane category determination."""
    assert get_kick_plane_category(ntrk, kick_both_planes) == expected


# Tests for validate_coordinate_generation function
def test_validate_coordinate_generation_valid_input() -> None:
    """Test that valid inputs pass validation."""
    assert validate_coordinate_generation(3, [1, 2, 3], [0.1, 0.2, 0.3]) is True


@pytest.mark.parametrize(
    "invalid_input",
    [
        (3, [1, 2], [0.1, 0.2, 0.3]),  # Wrong action length
        (3, [1, 2, 3], [0.1, 0.2]),  # Wrong angle length
        (3, [1, 2, 3], [0.1, 0.2, 0.3, 0.4]),  # Mismatched lengths
    ],
)
def test_validate_coordinate_generation_invalid_input(invalid_input) -> None:
    """Test that invalid inputs raise AssertionError."""
    with pytest.raises(AssertionError):
        validate_coordinate_generation(*invalid_input)
