"""
Tests for aba_optimiser.io.utils module.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from aba_optimiser.io.utils import (
    read_results,
    save_results,
    scientific_notation,
)


@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    """Fixture to create and clean up temporary files."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


class TestSaveResults:
    """Tests for save_results function."""

    @pytest.mark.parametrize(
        "knob_names,knob_strengths,uncertainties,expected_content",
        [
            (
                ["knob1", "knob2"],
                {"knob1": 1.23e-05, "knob2": -2.35e-04},
                [0.001, 0.002],
                "Knob Name\tStrength\tUncertainty\nknob1\t1.230000000000000e-05\t1.000000000000000e-03\nknob2\t-2.350000000000000e-04\t2.000000000000000e-03\n",
            ),
            (
                [],
                {},
                [],
                "Knob Name\tStrength\tUncertainty\n",
            ),
        ],
    )
    def test_save_results(
        self,
        temp_file: Path,
        knob_names: list[str],
        knob_strengths: dict[str, float],
        uncertainties: list[float],
        expected_content: str,
    ) -> None:
        """Test saving results to file."""
        save_results(knob_names, knob_strengths, uncertainties, str(temp_file))
        content = temp_file.read_text()
        assert content == expected_content


class TestReadResults:
    """Tests for read_results function."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            (
                "Knob Name\tStrength\tUncertainty\nknob1\t1.230000e-05\t1.000000e-03\nknob2\t-2.350000e-04\t2.000000e-03\n",
                (["knob1", "knob2"], [1.23e-05, -2.35e-04], [0.001, 0.002]),
            ),
            ("", ([], [], [])),
            (
                "Knob Name\tStrength\tUncertainty\nknob1\t1.0\t0.1\ninvalid_line\nknob2\t2.0\t0.2\n",
                (["knob1", "knob2"], [1.0, 2.0], [0.1, 0.2]),
            ),
        ],
    )
    def test_read_results(
        self,
        temp_file: Path,
        content: str,
        expected: tuple[list[str], list[float], list[float]],
    ) -> None:
        """Test reading results from files."""
        temp_file.write_text(content)
        result = read_results(str(temp_file))
        assert result == expected


class TestScientificNotation:
    """Tests for scientific_notation function."""

    @pytest.mark.parametrize(
        "num,precision,expected",
        [
            (123.456, 2, "$1.23\\times10^{2}$"),
            (0.00123456, 3, "$1.235\\times10^{-3}$"),
            (-123.456, 2, "$-1.23\\times10^{2}$"),
            (-0.00123456, 3, "$-1.235\\times10^{-3}$"),
            (0.0, 2, "0"),
            (1.0, 2, "1.00"),
            (123.456, 0, "$1\\times10^{2}$"),
            (1.23456789, 6, "1.234568"),
            (float("nan"), 2, "nan"),
            (float("inf"), 2, "inf"),
            (float("-inf"), 2, "-inf"),
            ("not_a_number", 2, "not_a_number"),
        ],
    )
    def test_scientific_notation(self, num: Any, precision: int, expected: str) -> None:
        """Test scientific notation formatting."""
        assert scientific_notation(num, precision) == expected
