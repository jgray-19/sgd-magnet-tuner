"""Tests for PSB accelerator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from aba_optimiser.accelerators import PSB

if TYPE_CHECKING:
    from pathlib import Path


class TestPSBAccelerator:
    """Tests for the PSB accelerator class."""

    @pytest.fixture
    def test_sequence_file(self, data_dir: Path) -> Path:
        """Use the psb sequence in the data directory for testing."""
        return data_dir / "sequences" / "psb1.seq"

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic PSB initialisation."""
        psb = PSB(ring=1, sequence_file=test_sequence_file)

        expected_energy = 0.160 + 0.9382720813  # kinetic + proton mass = total energy
        assert psb.ring == 1
        assert psb.sequence_file == test_sequence_file
        assert psb.pc == pytest.approx(expected_energy)
        assert psb.bpm_pattern == "^BR1%.BPM"
        assert psb.optimise_quadrupoles is False
        assert psb.optimise_correctors is False
        assert psb.optimise_energy is False

    @pytest.mark.parametrize("ring", [1, 2, 3, 4])
    def test_seq_name_uses_ring_number(self, test_sequence_file: Path, ring: int) -> None:
        """Test sequence name follows the PSB ring convention."""
        psb = PSB(ring=ring, sequence_file=test_sequence_file)
        assert psb.seq_name == f"psb{ring}"

    @pytest.mark.parametrize("ring", [0, 5])
    def test_init_invalid_ring(self, test_sequence_file: Path, ring: int) -> None:
        """Test invalid ring numbers raise ValueError."""
        with pytest.raises(ValueError, match="PSB ring must be 1, 2, 3, or 4"):
            PSB(ring=ring, sequence_file=test_sequence_file)

    def test_init_custom_bpm_pattern(self, test_sequence_file: Path) -> None:
        """Test a custom BPM pattern overrides the ring default."""
        psb = PSB(
            ring=2,
            sequence_file=test_sequence_file,
            bpm_pattern="^CUSTOM%.BPM",
        )
        assert psb.bpm_pattern == "^CUSTOM%.BPM"

    def test_get_supported_knob_specs(self, test_sequence_file: Path) -> None:
        """Test PSB exposes quadrupole knob specs."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            optimise_quadrupoles=True,
        )

        assert ("quadrupole", "k1", "^BR%.Q[FD][OE]%d+$", "k1", True) in psb.get_supported_knob_specs()

    def test_init_with_optimise_correctors(self, test_sequence_file: Path) -> None:
        """Test initialization with corrector optimization."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            optimise_correctors=True,
        )
        assert psb.optimise_correctors is True

    def test_get_supported_knob_specs_with_correctors(self, test_sequence_file: Path) -> None:
        """Test PSB exposes sequence-name patterns for horizontal and vertical correctors."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            optimise_correctors=True,
        )

        assert ("hkicker", "kick", "^B[RE]%d+%.DHZ%d+L%d+$", None, True) in psb.get_supported_knob_specs()
        assert ("vkicker", "kick", "^B[RE]%d+%.DVT%d+L%d+$", None, True) in psb.get_supported_knob_specs()

    def test_get_perturbation_families(self, test_sequence_file: Path) -> None:
        """Test PSB perturbation metadata is available for quadrupoles."""
        psb = PSB(ring=1, sequence_file=test_sequence_file)
        assert psb.get_perturbation_families() == {
            "q": {
                "default_rel_std": 2e-4,
                "pattern": r"^BR\.Q(?:FO\d+|DE\d+)$",
            },
        }

    @pytest.mark.parametrize(
        "monitor_name",
        [
            "BR1.BPM1L3",
            "BR1.BPMT3L1",
            "BR1.BWSH4L1",
            "BR1.BPP1L5",
        ],
    )
    def test_infer_monitor_plane(self, monitor_name: str) -> None:
        """Test PSB monitors are treated as dual-plane."""
        assert PSB.infer_monitor_plane(monitor_name) == "HV"

    def test_infer_monitor_plane_invalid(self) -> None:
        """Test unsupported PSB monitor names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported PSB monitor name"):
            PSB.infer_monitor_plane("BR1.QFO11")

    def test_tune_configuration(self, test_sequence_file: Path) -> None:
        """Test PSB tune variable names and integer tunes."""
        psb = PSB(ring=1, sequence_file=test_sequence_file)
        assert psb.tune_variables == ("kbrqf", "kbrqd")
        assert psb.tune_integers == (4, 4)

    def test_has_any_optimisation(self, test_sequence_file: Path) -> None:
        """Test generic optimisation flags work for PSB."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            optimise_quadrupoles=True,
            optimise_energy=True,
            custom_knobs_to_optimise=["BR.QFO11.dk1l"],
        )
        assert psb.has_any_optimisation() is True

    def test_has_any_optimisation_correctors(self, test_sequence_file: Path) -> None:
        """Test corrector optimisation contributes to PSB optimisation state."""
        psb = PSB(
            ring=1,
            sequence_file=test_sequence_file,
            optimise_correctors=True,
        )
        assert psb.has_any_optimisation() is True

    def test_format_result_knob_names_maps_indexed_sextupoles(self, test_sequence_file: Path) -> None:
        """Test PSB rewrites indexed sextupole knob names to public dk forms."""
        psb = PSB(ring=1, sequence_file=test_sequence_file)
        assert psb.format_result_knob_names(["br3.xnoh0.4l1.knl[3]"]) == ["br3.xnoh0.4l1.dk2l"]
        assert psb.format_result_knob_names(["br3.osk4l1.ksl[3]"]) == ["br3.osk4l1.dk2sl"]
