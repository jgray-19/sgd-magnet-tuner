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
def seq_b1(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 used by several tests."""
    return data_dir / "sequences" / "lhcb1.seq"


@pytest.fixture(scope="module")
def json_b1(data_dir: Path) -> Path:
    """Path to the pre-generated xsuite JSON for beam 1."""
    return data_dir / "sequences" / "lhcb1.json"


@pytest.fixture(scope="module")
def json_b1_corrected(data_dir: Path) -> Path:
    """Path to the corrected xsuite JSON for beam 1."""
    return data_dir / "sequences" / "lhcb1_corrected.json"


@pytest.fixture(scope="module")
def tune_knobs_file(data_dir: Path) -> Path:
    """Path to the tune knobs file."""
    return data_dir / "strengths" / "tune_knobs.txt"


@pytest.fixture(scope="module")
def corrector_knobs_file(data_dir: Path) -> Path:
    """Path to the corrector knobs file."""
    return data_dir / "correctors" / "corrector_knobs.txt"


@pytest.fixture(scope="module")
def corrector_file(data_dir: Path) -> Path:
    """Path to the corrector table file."""
    return data_dir / "correctors" / "corrector_table.tfs"


@pytest.fixture(scope="module")
def tracking_path(data_dir: Path) -> Path:
    """Path to the tracking data directory."""
    return data_dir / "analysis" / "tracking"


@pytest.fixture(scope="module")
def estimated_strengths_file(data_dir: Path) -> Path:
    """Path to the estimated quadrupole strengths file."""
    return data_dir / "strengths" / "estimated_quad_strengths.json"


@pytest.fixture(scope="module")
def model_dir_b1() -> Path:
    """Path to the beam 1 model directory."""
    return Path(__file__).parent.parent / "models" / "lhcb1_12cm"


@pytest.fixture(scope="module")
def corrector_table(corrector_file: Path) -> tfs.TfsDataFrame:
    """Load and filter corrector table, removing monitor elements."""
    corrector_table = tfs.read(corrector_file)
    # Filter out monitor elements from the corrector table
    return corrector_table[corrector_table["kind"] != "monitor"]


@pytest.fixture(scope="module")
def sequence_file(seq_b1: Path) -> Path:
    """Path to the example sequence file used by several tests."""
    return seq_b1


@pytest.fixture(scope="function")
def interface() -> Generator[BaseMadInterface, None, None]:
    """Create a fresh BaseMadInterface for each test."""
    iface = BaseMadInterface()
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def loaded_interface(interface: BaseMadInterface, sequence_file: Path) -> BaseMadInterface:
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
