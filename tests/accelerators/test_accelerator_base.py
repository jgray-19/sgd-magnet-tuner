"""Tests for base Accelerator class and its interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from aba_optimiser.accelerators.base import Accelerator


class ConcreteAccelerator(Accelerator):
    """Concrete implementation of Accelerator for testing."""

    def get_seq_name(self) -> str:
        """Return a test sequence name."""
        return "test_seq"

    def get_supported_knob_specs(self) -> list[tuple[str, str, str, bool, bool]]:
        """Return a list of supported knob specifications."""
        return [("quadrupole", "k1", "MQ", True, True)]

class TestAcceleratorBase:
    """Tests for base Accelerator functionality."""

    @pytest.fixture
    def test_sequence_file(self, tmp_path: Path) -> Path:
        """Create a dummy sequence file for testing."""
        seq_file = tmp_path / "test.seq"
        seq_file.write_text("! Dummy sequence file\n")
        return seq_file

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic initialization with minimal parameters."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        assert acc.sequence_file == test_sequence_file
        assert acc.beam_energy == 6800.0
        assert acc.seq_name is None
        assert acc.optimise_energy is False
        assert acc.optimise_quadrupoles is False
        assert acc.optimise_sextupoles is False
        assert acc.custom_knobs_to_optimise is None

    def test_init_with_seq_name(self, test_sequence_file: Path) -> None:
        """Test initialization with sequence name."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            seq_name="lhcb1",
        )
        assert acc.seq_name == "lhcb1"

    def test_init_with_optimise_energy(self, test_sequence_file: Path) -> None:
        """Test initialization with energy optimization enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
        )
        assert acc.optimise_energy is True

    def test_init_with_all_optimisation_flags(self, test_sequence_file: Path) -> None:
        """Test initialization with all optimisation flags enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
            optimise_quadrupoles=True,
            optimise_sextupoles=True,
            custom_knobs_to_optimise=["K1", "K2"],
        )
        assert acc.optimise_energy is True
        assert acc.optimise_quadrupoles is True
        assert acc.optimise_sextupoles is True
        assert acc.custom_knobs_to_optimise == ["K1", "K2"]

    def test_sequence_file_as_string(self, test_sequence_file: Path) -> None:
        """Test that sequence file can be provided as string."""
        acc = ConcreteAccelerator(
            sequence_file=str(test_sequence_file),
            beam_energy=6800.0,
        )
        assert isinstance(acc.sequence_file, Path)
        assert acc.sequence_file == test_sequence_file

    def test_has_any_optimisation_false(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns False when no optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        assert acc.has_any_optimisation() is False

    def test_has_any_optimisation_energy(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when energy optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_quadrupoles(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when quad optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_quadrupoles=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_sextupoles(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when sextupole optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_sextupoles=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_custom_knobs(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when custom knobs provided."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            custom_knobs_to_optimise=["K1"],
        )
        assert acc.has_any_optimisation() is True

    def test_get_bend_lengths_returns_none(self, test_sequence_file: Path) -> None:
        """Test that base get_bend_lengths returns None."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        result = acc.get_bend_lengths(None)
        assert result is None

    def test_normalise_true_strengths_returns_unchanged(
        self, test_sequence_file: Path
    ) -> None:
        """Test that base normalise_true_strengths returns unchanged dictionary."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        test_strengths = {"K1": 0.5, "K2": -0.3}
        result = acc.normalise_true_strengths(test_strengths, None)
        assert result == test_strengths
        assert result is test_strengths  # Should return same object

    def test_normalise_true_strengths_with_bend_lengths(
        self, test_sequence_file: Path
    ) -> None:
        """Test normalise_true_strengths with bend lengths provided."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        test_strengths = {"K0": 1.0}
        bend_lengths = {"K0": 2.0}
        result = acc.normalise_true_strengths(test_strengths, bend_lengths)
        assert result == test_strengths

    def test_format_result_knob_names_without_energy(
        self, test_sequence_file: Path
    ) -> None:
        """Test format_result_knob_names without energy optimisation."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=False,
        )
        knob_names = ["K1.b1", "K2.b1", "pt"]
        result = acc.format_result_knob_names(knob_names)
        assert result == ["K1.b1", "K2.b1", "pt"]

    def test_format_result_knob_names_with_energy_no_pt(
        self, test_sequence_file: Path
    ) -> None:
        """Test format_result_knob_names with energy but no pt knob."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
        )
        knob_names = ["K1.b1", "K2.b1"]
        result = acc.format_result_knob_names(knob_names)
        assert result == ["K1.b1", "K2.b1"]

    def test_format_result_knob_names_with_energy_and_pt(
        self, test_sequence_file: Path
    ) -> None:
        """Test format_result_knob_names converts pt to deltap when energy optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
        )
        knob_names = ["K1.b1", "K2.b1", "pt"]
        result = acc.format_result_knob_names(knob_names)
        assert "pt" not in result
        assert "deltap" in result
        assert "K1.b1" in result
        assert "K2.b1" in result

    def test_format_result_knob_names_returns_copy(
        self, test_sequence_file: Path
    ) -> None:
        """Test that format_result_knob_names returns a new list."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        original = ["K1.b1", "K2.b1"]
        result = acc.format_result_knob_names(original)
        assert result is not original
        assert result == original

    def test_log_optimisation_targets_none(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets with no optimisations."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
        )
        acc.log_optimisation_targets()
        assert "No optimisation targets set" in caplog.text

    def test_log_optimisation_targets_energy(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets logs energy."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
        )
        acc.log_optimisation_targets()
        assert "beam energy" in caplog.text

    def test_log_optimisation_targets_multiple(
        self, test_sequence_file: Path, caplog
    ) -> None:
        """Test log_optimisation_targets logs multiple targets."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            beam_energy=6800.0,
            optimise_energy=True,
            optimise_quadrupoles=True,
            custom_knobs_to_optimise=["K1"],
        )
        acc.log_optimisation_targets()
        assert "beam energy" in caplog.text
        assert "quadrupoles" in caplog.text
        assert "custom knobs" in caplog.text
