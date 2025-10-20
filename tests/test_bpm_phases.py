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
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7"],
    )


@pytest.fixture
def prev_mu():
    return pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7"],
    )


expected_results = {
    "next_bpm_to_pi_2": {
        "names": ["BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM7", "BPM1"],
        "deltas": [-0.05, -0.05, -0.05, -0.05, -0.05, -0.15, 0.15],
    },
    "prev_bpm_to_pi_2": {
        "names": ["BPM7", "BPM7", "BPM1", "BPM2", "BPM3", "BPM5", "BPM6"],
        "deltas": [-0.15, -0.05, -0.05, -0.05, -0.05, -0.05, 0.05],
    },
    "next_bpm_to_pi": {
        "names": ["BPM6", "BPM7", "BPM7", "BPM1", "BPM1", "BPM1", "BPM2"],
        "deltas": [0, 0, -0.1, 0.2, 0.1, 0, 0],
    },
    "prev_bpm_to_pi": {
        "names": ["BPM4", "BPM5", "BPM6", "BPM6", "BPM1", "BPM7", "BPM5"],
        "deltas": [0, 0, -0.1, 0, -0.1, 0, 0],
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
