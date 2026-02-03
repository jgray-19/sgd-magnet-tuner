"""Tests for LHC accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from aba_optimiser.accelerators import LHC

if TYPE_CHECKING:
    from pathlib import Path


class TestLHCBPMPlaneMasks:
    """Tests for LHC BPM plane mask functionality."""

    @pytest.fixture
    def lhc(self, tmp_path: Path) -> LHC:
        """Create an LHC accelerator instance for testing."""
        seq_file = tmp_path / "test.seq"
        seq_file.write_text("! Dummy sequence file\n")
        return LHC(beam=1, beam_energy=6800.0, sequence_file=str(seq_file))

    def test_parse_bad_bpm_dual_plane(self, lhc: LHC) -> None:
        """Test parsing dual-plane BPM specification."""
        base_name, plane = lhc.parse_bad_bpm_specification("BPM.14L1.B1")
        assert base_name == "BPM.14L1.B1"
        assert plane is None

    def test_parse_bad_bpm_horizontal_only(self, lhc: LHC) -> None:
        """Test parsing horizontal-only BPM specification."""
        base_name, plane = lhc.parse_bad_bpm_specification("BPM.14L1.B1.H")
        assert base_name == "BPM.14L1.B1"
        assert plane == "H"

    def test_parse_bad_bpm_vertical_only(self, lhc: LHC) -> None:
        """Test parsing vertical-only BPM specification."""
        base_name, plane = lhc.parse_bad_bpm_specification("BPM.14L1.B1.V")
        assert base_name == "BPM.14L1.B1"
        assert plane == "V"

    def test_plane_mask_all_dual_plane_no_bad(self, lhc: LHC) -> None:
        """Test plane mask with all dual-plane BPMs and no bad BPMs."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1", "BPM.12L1.B1"]
        bad_bpms: list[str] = []
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        assert h_mask == [True, True, True]
        assert v_mask == [True, True, True]

    def test_plane_mask_single_plane_horizontal(self, lhc: LHC) -> None:
        """Test plane mask with horizontal-only BPM."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1.H", "BPM.12L1.B1"]
        bad_bpms: list[str] = []
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1.H only measures horizontal
        assert h_mask == [True, True, True]
        assert v_mask == [True, False, True]

    def test_plane_mask_single_plane_vertical(self, lhc: LHC) -> None:
        """Test plane mask with vertical-only BPM."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1.V", "BPM.12L1.B1"]
        bad_bpms: list[str] = []
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1.V only measures vertical
        assert h_mask == [True, False, True]
        assert v_mask == [True, True, True]

    def test_plane_mask_bad_both_planes(self, lhc: LHC) -> None:
        """Test plane mask with dual-plane BPM marked as bad."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1", "BPM.12L1.B1"]
        bad_bpms = ["BPM.11L1.B1"]  # Both planes bad
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1 is completely bad
        assert h_mask == [True, False, True]
        assert v_mask == [True, False, True]

    def test_plane_mask_bad_horizontal_only(self, lhc: LHC) -> None:
        """Test plane mask with only horizontal plane bad."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1", "BPM.12L1.B1"]
        bad_bpms = ["BPM.11L1.B1.H"]  # Only horizontal plane bad
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1 horizontal is bad, vertical is good
        assert h_mask == [True, False, True]
        assert v_mask == [True, True, True]

    def test_plane_mask_bad_vertical_only(self, lhc: LHC) -> None:
        """Test plane mask with only vertical plane bad."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1", "BPM.12L1.B1"]
        bad_bpms = ["BPM.11L1.B1.V"]  # Only vertical plane bad
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1 vertical is bad, horizontal is good
        assert h_mask == [True, True, True]
        assert v_mask == [True, False, True]

    def test_plane_mask_mixed_scenario(self, lhc: LHC) -> None:
        """Test complex scenario with single-plane BPMs and bad BPMs."""
        bpm_list = [
            "BPM.10L1.B1",  # Dual-plane, good
            "BPM.11L1.B1.H",  # Horizontal-only
            "BPM.12L1.B1",  # Dual-plane, will be marked bad on H
            "BPM.13L1.B1.V",  # Vertical-only
            "BPM.14L1.B1",  # Dual-plane, completely bad
        ]
        bad_bpms = [
            "BPM.12L1.B1.H",  # Only horizontal of 12L1 is bad
            "BPM.14L1.B1",  # 14L1 is completely bad
        ]
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        assert h_mask == [True, True, False, False, False]
        assert v_mask == [True, False, True, True, False]

    def test_plane_mask_single_plane_bpm_marked_bad(self, lhc: LHC) -> None:
        """Test single-plane BPM that is also marked as bad."""
        bpm_list = ["BPM.10L1.B1", "BPM.11L1.B1.H"]
        bad_bpms = ["BPM.11L1.B1.H"]  # Single-plane BPM is bad
        h_mask, v_mask = lhc.get_bpm_plane_mask(bpm_list, bad_bpms)

        # BPM.11L1.B1.H is horizontal-only and bad
        assert h_mask == [True, False]
        assert v_mask == [True, False]  # Already False because it's H-only


class TestLHCAccelerator:
    """Tests for LHC accelerator class."""

    @pytest.fixture
    def test_sequence_file(self, tmp_path: Path) -> Path:
        """Create a dummy sequence file for testing."""
        seq_file = tmp_path / "test.seq"
        seq_file.write_text("! Dummy sequence file\n")
        return seq_file

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic LHC initialization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.beam == 1
        assert lhc.beam_energy == 6800.0
        assert lhc.optimise_bends is False
        assert lhc.optimise_correctors is False
        assert lhc.normalise_bends is False
        assert lhc.optimise_energy is False

    def test_init_beam_2(self, test_sequence_file: Path) -> None:
        """Test LHC initialization with beam 2."""
        lhc = LHC(
            beam=2,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.beam == 2
        assert lhc.get_seq_name() == "lhcb2"

    def test_init_invalid_beam(self, test_sequence_file: Path) -> None:
        """Test that invalid beam numbers raise ValueError."""
        with pytest.raises(ValueError, match="LHC beam must be 1 or 2"):
            LHC(
                beam=3,
                beam_energy=6800.0,
                sequence_file=str(test_sequence_file),
            )

    def test_init_beam_zero(self, test_sequence_file: Path) -> None:
        """Test that beam 0 raises ValueError."""
        with pytest.raises(ValueError, match="LHC beam must be 1 or 2"):
            LHC(
                beam=0,
                beam_energy=6800.0,
                sequence_file=str(test_sequence_file),
            )

    def test_init_with_optimise_energy(self, test_sequence_file: Path) -> None:
        """Test initialization with energy optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        assert lhc.optimise_energy is True

    def test_init_with_optimise_bends(self, test_sequence_file: Path) -> None:
        """Test initialization with bend optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
        )
        assert lhc.optimise_bends is True
        assert lhc.normalise_bends is True  # Should default to True when optimise_bends is True

    def test_init_with_optimise_bends_no_normalise(self, test_sequence_file: Path) -> None:
        """Test initialization with bend optimization but no normalization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
            normalise_bends=False,
        )
        assert lhc.optimise_bends is True
        assert lhc.normalise_bends is False

    def test_init_normalise_bends_none_without_bends(self, test_sequence_file: Path) -> None:
        """Test normalise_bends defaults to False when optimise_bends is False."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=False,
            normalise_bends=None,
        )
        assert lhc.normalise_bends is False

    def test_init_with_optimise_correctors(self, test_sequence_file: Path) -> None:
        """Test initialization with corrector optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_correctors=True,
        )
        assert lhc.optimise_correctors is True

    def test_init_with_all_optimisations(self, test_sequence_file: Path) -> None:
        """Test initialization with all optimizations enabled."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
            optimise_bends=True,
            optimise_quadrupoles=True,
            optimise_sextupoles=True,
            optimise_correctors=True,
        )
        assert lhc.optimise_energy is True
        assert lhc.optimise_bends is True
        assert lhc.optimise_quadrupoles is True
        assert lhc.optimise_sextupoles is True
        assert lhc.optimise_correctors is True

    def test_get_seq_name_beam_1(self, test_sequence_file: Path) -> None:
        """Test get_seq_name returns correct sequence name for beam 1."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.get_seq_name() == "lhcb1"

    def test_get_seq_name_beam_2(self, test_sequence_file: Path) -> None:
        """Test get_seq_name returns correct sequence name for beam 2."""
        lhc = LHC(
            beam=2,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.get_seq_name() == "lhcb2"

    def test_has_any_optimisation_false(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns False with no optimizations."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        assert lhc.has_any_optimisation() is False

    def test_has_any_optimisation_energy(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with energy optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        assert lhc.has_any_optimisation() is True

    def test_has_any_optimisation_bends(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with bend optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
        )
        assert lhc.has_any_optimisation() is True

    def test_has_any_optimisation_correctors(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True with corrector optimization."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_correctors=True,
        )
        assert lhc.has_any_optimisation() is True

    def test_log_optimisation_targets_none(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets with no optimizations."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
        )
        lhc.log_optimisation_targets()
        assert "No optimisation targets set" in caplog.text

    def test_log_optimisation_targets_all(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets logs all optimizations."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
            optimise_bends=True,
            optimise_quadrupoles=True,
            optimise_sextupoles=True,
            optimise_correctors=True,
        )
        lhc.log_optimisation_targets()
        assert "bends" in caplog.text
        assert "quadrupoles" in caplog.text
        assert "sextupoles" in caplog.text
        assert "correctors" in caplog.text
        assert "beam energy" in caplog.text

    def test_get_bend_lengths_returns_none_no_bends(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns None when bends not optimized."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=False,
        )
        result = lhc.get_bend_lengths(MagicMock())
        assert result is None

    def test_get_bend_lengths_returns_none_no_normalise(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns None when normalisation disabled."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
            normalise_bends=False,
        )
        result = lhc.get_bend_lengths(MagicMock())
        assert result is None

    def test_get_bend_lengths_returns_from_interface(self, test_sequence_file: Path) -> None:
        """Test get_bend_lengths returns bend_lengths from interface."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
            normalise_bends=True,
        )
        mock_interface = MagicMock()
        mock_interface.bend_lengths = {"MB.A1.k0": 2.0, "MB.A2.k0": 2.0}
        result = lhc.get_bend_lengths(mock_interface)
        assert result == {"MB.A1.k0": 2.0, "MB.A2.k0": 2.0}

    def test_get_bend_lengths_handles_missing_attribute(
        self, test_sequence_file: Path
    ) -> None:
        """Test get_bend_lengths handles interface without bend_lengths attribute."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
            normalise_bends=True,
        )
        mock_interface = MagicMock(spec=[])  # Empty spec means no attributes
        result = lhc.get_bend_lengths(mock_interface)
        assert result is None

    def test_normalise_true_strengths_no_bends(self, test_sequence_file: Path) -> None:
        """Test normalise_true_strengths returns unchanged when bends not optimized."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=False,
        )
        test_strengths = {"K1": 0.5}
        result = lhc.normalise_true_strengths(test_strengths, None)
        assert result == test_strengths

    def test_normalise_true_strengths_no_bend_lengths(self, test_sequence_file: Path) -> None:
        """Test normalise_true_strengths returns unchanged when no bend_lengths provided."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
        )
        test_strengths = {"MB.k0": 1.0}
        result = lhc.normalise_true_strengths(test_strengths, None)
        assert result == test_strengths

    def test_normalise_true_strengths_with_bend_lengths(
        self, test_sequence_file: Path
    ) -> None:
        """Test normalise_true_strengths calls normalise_lhcbend_magnets."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_bends=True,
            normalise_bends=True,
        )
        # These would be the actual bend strengths from the sequence
        test_strengths = {
            "MB.A1[ABCD]1L1.B1.k0": 1.0,
            "MB.A1[ABCD]1L2.B1.k0": 1.0,
        }
        bend_lengths = {
            "MB.A1[ABCD]1L1.B1.k0": 14.2,
            "MB.A1[ABCD]1L2.B1.k0": 14.2,
        }
        result = lhc.normalise_true_strengths(test_strengths, bend_lengths)
        # The result should be a dict (we're not testing the actual normalisation logic here)
        assert isinstance(result, dict)

    def test_format_result_knob_names_with_energy(self, test_sequence_file: Path) -> None:
        """Test format_result_knob_names converts pt to deltap."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=True,
        )
        knob_names = ["K1.b1", "pt"]
        result = lhc.format_result_knob_names(knob_names)
        assert "deltap" in result
        assert "pt" not in result

    def test_format_result_knob_names_without_energy(self, test_sequence_file: Path) -> None:
        """Test format_result_knob_names leaves knobs unchanged without energy."""
        lhc = LHC(
            beam=1,
            beam_energy=6800.0,
            sequence_file=str(test_sequence_file),
            optimise_energy=False,
        )
        knob_names = ["K1.b1", "pt"]
        result = lhc.format_result_knob_names(knob_names)
        assert result == knob_names
