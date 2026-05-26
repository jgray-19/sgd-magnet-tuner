"""Shared fixtures for controller integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture(params=["loss_regression", "run_converges"])
def controller_test_mode(request: pytest.FixtureRequest) -> str:
    """Run controller integration tests in fast loss and full optimisation modes."""
    return str(request.param)
