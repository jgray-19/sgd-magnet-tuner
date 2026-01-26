import numpy as np
import pandas as pd
import pytest

from src.aba_optimiser.physics.bpm_phases import (
    next_bpm_to_pi,
    next_bpm_to_pi_2,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)


@pytest.fixture
def next_mu():
    return pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.75],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8"],
    )


@pytest.fixture
def prev_mu():
    return pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8"],
    )


expected_results = {
    "next_bpm_to_pi_2": {
        "names": ["BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8", "BPM8", "BPM1"],
        "deltas": [-0.05, -0.05, -0.05, -0.05, -0.05, 0.00, -0.1, 0.0],
    },
    "prev_bpm_to_pi_2": {
        "names": ["BPM7", "BPM8", "BPM1", "BPM2", "BPM3", "BPM5", "BPM6", "BPM6"],
        "deltas": [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, 0.05],
    },
    "next_bpm_to_pi": {
        # . FROM   BPM1.   BPM2.   BPM3.   BPM4.   BPM5.   BPM6.   BPM7.   BPM8
        "names": ["BPM6", "BPM7", "BPM7", "BPM8", "BPM1", "BPM1", "BPM2", "BPM3"],
        "deltas": [0, 0, -0.1, -0.05, 0.1, 0, 0, -0.05],
    },
    "prev_bpm_to_pi": {
        "names": ["BPM4", "BPM5", "BPM6", "BPM6", "BPM1", "BPM2", "BPM4", "BPM5"],
        "deltas": [0, 0, -0.1, 0, -0.1, 0, 0, 0],
    },
}


@pytest.mark.parametrize(
    "func, mu_fixture",
    [
        (next_bpm_to_pi_2, "next_mu"),
        (prev_bpm_to_pi_2, "prev_mu"),
    ],
)
def test_bpm_to_pi_2(func, mu_fixture, request):
    mu = request.getfixturevalue(mu_fixture)
    tune = 1.0

    result = func(mu, tune)

    key = "next_bpm" if "next" in func.__name__ else "prev_bpm"
    expected = expected_results[func.__name__]
    assert all(result[key] == expected["names"])
    assert np.allclose(result["delta"], expected["deltas"])


@pytest.mark.parametrize(
    "func, mu_fixture, tune",
    [
        (next_bpm_to_pi, "next_mu", 1.0),
        (prev_bpm_to_pi, "prev_mu", 0.8),
    ],
)
def test_bpm_to_pi(func, mu_fixture, tune, request):
    mu = request.getfixturevalue(mu_fixture)

    result = func(mu, tune)

    key = "next_bpm" if "next" in func.__name__ else "prev_bpm"
    expected = expected_results[func.__name__]
    assert all(result[key] == expected["names"])
    assert np.allclose(result["delta"], expected["deltas"])


def test_bpm_distance_limit_pi_2():
    """
    Test that BPM distance limits are respected for π/2 phase advance (max 11 BPMs).

    Create a scenario with many BPMs where a perfect match exists beyond the 11 BPM limit,
    and verify it is not selected.
    """
    # Create BPMs at regular intervals, the ideal next BPM at π/2 is every 13/14 bpms, it should select within 11
    mu = pd.Series(
        np.linspace(0, 0.9, 50),  # 50 BPMs over nearly 1 full turn.
        index=[f"BPM{i}" for i in range(50)],
    )
    tune = 1.0
    result = next_bpm_to_pi_2(mu, tune)

    # For each BPM, verify that the selected next BPM is within 11 steps
    for i, bpm in enumerate(mu.index):
        matched_bpm = result.loc[bpm, "next_bpm"]
        if pd.notna(matched_bpm):
            matched_idx = list(mu.index).index(matched_bpm)
            distance = (matched_idx - i) % len(mu)
            assert distance <= 11, f"BPM{i} matched with BPM at distance {distance} > 11"


def test_bpm_distance_limit_pi():
    """
    Test that BPM distance limits are respected for π phase advance (max 20 BPMs).

    Create a scenario with many BPMs where a perfect match exists beyond the 20 BPM limit,
    and verify it is not selected.
    """
    # Create BPMs at regular intervals, the ideal next BPM at π is every 26 bpms, it should select within 20
    mu = pd.Series(
        np.linspace(0, 0.95, 50),  # 50 BPMs over nearly 1 full turn
        index=[f"BPM{i}" for i in range(50)],
    )
    tune = 1.0
    result = next_bpm_to_pi(mu, tune)

    # For each BPM, verify that the selected next BPM is within 20 steps
    for i, bpm in enumerate(mu.index):
        matched_bpm = result.loc[bpm, "next_bpm"]
        if pd.notna(matched_bpm):
            matched_idx = list(mu.index).index(matched_bpm)
            distance = (matched_idx - i) % len(mu)
            assert distance <= 20, f"BPM{i} matched with BPM at distance {distance} > 20"
