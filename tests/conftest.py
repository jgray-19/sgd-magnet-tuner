"""
Common pytest fixtures for MAD interface tests.

This module contains shared fixtures used across MAD interface test modules.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.mad.base_mad_interface import BaseMadInterface
from aba_optimiser.mad.tracking_interface import TrackingMadInterface

if TYPE_CHECKING:
    from collections.abc import Generator

# Configure logging for tests
logging.getLogger("xdeps").setLevel(logging.WARNING)


@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Path to the example corrector file used by several tests."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def corrector_file(data_dir: Path) -> Path:
    """Path to the example corrector strengths file used by several tests."""
    return data_dir / "corrector_strengths.tfs"


@pytest.fixture(scope="module")
def corrector_table(corrector_file: Path) -> tfs.TfsDataFrame:
    """Load and filter corrector table, removing monitor elements."""
    corrector_table = tfs.read(corrector_file)
    # Filter out monitor elements from the corrector table
    return corrector_table[corrector_table["kind"] != "monitor"]


@pytest.fixture(scope="module")
def sequence_file(data_dir: Path) -> Path:
    """Path to the example sequence file used by several tests."""
    return data_dir / "lhcb1.seq"


@pytest.fixture(scope="function")
def interface() -> Generator[BaseMadInterface, None, None]:
    """Create a fresh BaseMadInterface for each test."""
    iface = BaseMadInterface()
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def loaded_interface(
    interface: BaseMadInterface, sequence_file: Path
) -> BaseMadInterface:
    """Fixture that returns an interface with the example sequence loaded."""
    interface.load_sequence(sequence_file, "lhcb1")
    return interface


@pytest.fixture(scope="function")
def loaded_interface_with_beam(loaded_interface: BaseMadInterface) -> BaseMadInterface:
    """Fixture that returns an interface with the example sequence loaded and beam set up."""
    loaded_interface.setup_beam(particle="proton", beam_energy=6800.0)
    return loaded_interface


@pytest.fixture(scope="function")
def tracking_interface() -> Generator[TrackingMadInterface, None, None]:
    """Create a fresh TrackingMadInterface for each test."""
    iface = TrackingMadInterface()
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def loaded_tracking_interface(
    tracking_interface: TrackingMadInterface, sequence_file: Path
) -> TrackingMadInterface:
    """Fixture that returns a tracking interface with the example sequence loaded."""
    tracking_interface.load_sequence(sequence_file, "lhcb1")
    return tracking_interface


@pytest.fixture(scope="function")
def loaded_tracking_interface_with_beam(
    loaded_tracking_interface: TrackingMadInterface,
) -> TrackingMadInterface:
    """Fixture that returns a tracking interface with the example sequence loaded and beam set up."""
    loaded_tracking_interface.setup_beam(particle="proton", beam_energy=6800.0)
    return loaded_tracking_interface
