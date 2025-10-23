"""
Tests for TrackingMadInterface.

This module contains pytest tests for the TrackingMadInterface class.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.mad.tracking_interface import (
    TrackingMadInterface,
)
from tests.mad.helpers import check_element_observations

if TYPE_CHECKING:
    from pathlib import Path


def test_init_default() -> None:
    """Test initialization of TrackingMadInterface with default parameters."""
    interface = TrackingMadInterface()
    assert interface.py_name == "py"
    assert hasattr(interface, "mad")
    print(dir(interface.mad))
    assert interface.mad._MAD__process.stdout_file.writable(), (
        "STDOUT file is not writable"
    )
    assert str(interface.mad._MAD__process.stdout_file_path) == "mad_stdout.log", (
        "STDOUT file path is incorrect"
    )
    assert interface.mad._MAD__process.debug is True, "Debug mode should be enabled"
    with contextlib.suppress(Exception):
        del interface


def test_init_without_logging() -> None:
    """Test initialization with logging enabled."""
    interface = TrackingMadInterface(enable_logging=False)
    # Check that logging files are set up
    assert str(interface.mad._MAD__process.stdout_file_path) == "/dev/null"
    assert interface.mad._MAD__process.debug is False, "Debug mode should be disabled"

    with contextlib.suppress(Exception):
        del interface


@pytest.mark.parametrize(
    "beam_energy, observe_pattern",
    [
        (450.0, "BPM"),
        (6500.0, "MB.A33R2.B1"),
    ],
)
def test_setup_for_tracking(
    loaded_tracking_interface: TrackingMadInterface,
    sequence_file: Path,
    beam_energy: float,
    observe_pattern: str,
) -> None:
    """Test setup_for_tracking method."""
    interface = loaded_tracking_interface
    interface.setup_for_tracking(
        sequence_file=sequence_file,
        seq_name="lhcb1",
        beam_energy=beam_energy,
        observe_pattern=observe_pattern,
    )

    # Check that sequence is loaded
    assert interface.mad.SEQ_NAME == "lhcb1"
    assert interface.mad.loaded_sequence is not None

    # Check beam setup
    assert interface.mad.loaded_sequence.beam.particle == "proton"
    assert interface.mad.loaded_sequence.beam.energy == beam_energy

    # Check BPMs are observed
    # This would require checking the observation status, similar to base tests
    check_element_observations(interface, f"elm.name:match('{observe_pattern}')")
