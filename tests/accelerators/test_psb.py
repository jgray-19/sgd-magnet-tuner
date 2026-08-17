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
        return data_dir / "sequences" / "psb3_saved.seq"

    def test_init_basic(self, test_sequence_file: Path) -> None:
        """Test basic PSB initialisation."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)

        assert psb.ring == 3
        assert psb.sequence_file == test_sequence_file
        assert psb.kinetic_energy == pytest.approx(0.160)
        assert psb.energy == pytest.approx(0.160 + 0.9382720813)
        assert psb.bpm_pattern == "^BR3%.BPM%d+L3$"
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

        assert ("quadrupole", "k1", "^BR%.Q[FD][OE]%d+$", "k1", True, "quadrupoles") in psb.get_supported_knob_specs()

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

        assert ("hkicker", "kick", "^B[RE]%d+%.DHZ%d+L%d+$", None, True, "correctors") in psb.get_supported_knob_specs()
        assert ("vkicker", "kick", "^B[RE]%d+%.DVT%d+L%d+$", None, True, "correctors") in psb.get_supported_knob_specs()

    def test_get_perturbation_families(self, test_sequence_file: Path) -> None:
        """Test PSB perturbation metadata is available for bends and quadrupoles."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.get_perturbation_families() == {
            "d": {
                "default_rel_std": 8e-4,
                "pattern": r"(?i)^BR\.(?:BHZ\d+|BSW\d+L\d+\.\d+)$",
            },
            "q": {
                "default_rel_std": 2e-3,
                "pattern": r"(?i)^BR\.Q(?:FO\d+|DE\d+)$",
            },
        }

    @pytest.mark.parametrize(
        "monitor_name",
        [
            "BR3.BPM1L3",
            "BR3.BPMT3L1",
            "BR3.BWSH4L1",
            "BR3.BPP1L5",
        ],
    )
    def test_infer_monitor_plane(self, monitor_name: str) -> None:
        """Test PSB monitors are treated as dual-plane."""
        assert PSB.infer_monitor_plane(monitor_name) == "HV"

    def test_infer_monitor_plane_invalid(self) -> None:
        """Test unsupported PSB monitor names raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported PSB monitor name"):
            PSB.infer_monitor_plane("BR3.QFO11")

    def test_tune_configuration(self, test_sequence_file: Path) -> None:
        """Test PSB tune variable names and integer tunes."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.tune_variables == ("kBRQF", "kBRQD")
        assert psb.tune_integers == (4, 4)

    def test_has_any_optimisation(self, test_sequence_file: Path) -> None:
        """Test generic optimisation flags work for PSB."""
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            optimise_quadrupoles=True,
            optimise_energy=True,
            custom_knobs_to_optimise=["BR.QFO11.dk1l"],
        )
        assert psb.has_any_optimisation() is True

    def test_has_any_optimisation_correctors(self, test_sequence_file: Path) -> None:
        """Test corrector optimisation contributes to PSB optimisation state."""
        psb = PSB(
            ring=3,
            sequence_file=test_sequence_file,
            optimise_correctors=True,
        )
        assert psb.has_any_optimisation() is True

    def test_format_result_knob_names_maps_indexed_sextupoles(self, test_sequence_file: Path) -> None:
        """Test PSB rewrites indexed sextupole knob names to public dk forms."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.format_result_knob_names(["br3.xnoh0.4l1.knl[3]"]) == ["br3.xnoh0.4l1.dk2l"]
        assert psb.format_result_knob_names(["br3.osk4l1.ksl[3]"]) == ["br3.osk4l1.dk2sl"]

    def test_init_with_optimise_bpm_dx(self, test_sequence_file: Path) -> None:
        """Test initialization with BPM horizontal displacement optimization."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dx=True)
        assert psb.optimise_bpm_dx is True
        assert psb.optimise_bpm_dy is False

    def test_init_with_optimise_bpm_dy(self, test_sequence_file: Path) -> None:
        """Test initialization with BPM vertical displacement optimization."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dy=True)
        assert psb.optimise_bpm_dy is True
        assert psb.optimise_bpm_dx is False

    def test_init_bpm_flags_default_false(self, test_sequence_file: Path) -> None:
        """Test BPM displacement flags default to False."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        assert psb.optimise_bpm_dx is False
        assert psb.optimise_bpm_dy is False

    def test_bpm_misalignment_patterns_ring3(self, test_sequence_file: Path) -> None:
        """Test PSB returns the correct BPM pattern for ring 3."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        patterns = psb.bpm_misalignment_patterns
        assert patterns["dx"] == ("^BR3%.BPM%d+L3$",)
        assert patterns["dy"] == ("^BR3%.BPM%d+L3$",)

    @pytest.mark.parametrize("ring", [1, 2, 4])
    def test_bpm_misalignment_patterns_other_rings(self, test_sequence_file: Path, ring: int) -> None:
        """Test PSB returns ring-specific BPM patterns."""
        psb = PSB(ring=ring, sequence_file=test_sequence_file)
        patterns = psb.bpm_misalignment_patterns
        assert patterns["dx"] == (f"^BR{ring}%.BPM%d+L{ring}$",)
        assert patterns["dy"] == (f"^BR{ring}%.BPM%d+L{ring}$",)

    def test_get_supported_knob_specs_bpm_dx(self, test_sequence_file: Path) -> None:
        """Test BPM horizontal displacement spec is included when enabled."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dx=True)
        specs = psb.get_supported_knob_specs()
        assert ("monitor", "dx", "^BR3%.BPM%d+L3$", None, True, "BPM horizontal offsets") in specs

    def test_get_supported_knob_specs_bpm_dy(self, test_sequence_file: Path) -> None:
        """Test BPM vertical displacement spec is included when enabled."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dy=True)
        specs = psb.get_supported_knob_specs()
        assert ("monitor", "dy", "^BR3%.BPM%d+L3$", None, True, "BPM vertical offsets") in specs

    def test_get_supported_knob_specs_bpm_disabled(self, test_sequence_file: Path) -> None:
        """Test BPM displacement specs are present but disabled when flags are off."""
        psb = PSB(ring=3, sequence_file=test_sequence_file)
        specs = psb.get_supported_knob_specs()
        assert ("monitor", "dx", "^BR3%.BPM%d+L3$", None, False, "BPM horizontal offsets") in specs
        assert ("monitor", "dy", "^BR3%.BPM%d+L3$", None, False, "BPM vertical offsets") in specs

    def test_has_any_optimisation_bpm_dx(self, test_sequence_file: Path) -> None:
        """Test BPM horizontal displacement contributes to has_any_optimisation."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dx=True)
        assert psb.has_any_optimisation() is True

    def test_has_any_optimisation_bpm_dy(self, test_sequence_file: Path) -> None:
        """Test BPM vertical displacement contributes to has_any_optimisation."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dy=True)
        assert psb.has_any_optimisation() is True

    def test_copy_with_bpm_flags(self, test_sequence_file: Path) -> None:
        """Test copy_with correctly copies and overrides BPM displacement flags."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dx=True, optimise_bpm_dy=False)
        copy = psb.copy_with(optimise_bpm_dy=True)
        assert copy.optimise_bpm_dx is True
        assert copy.optimise_bpm_dy is True

    def test_copy_with_preserves_bpm_flags(self, test_sequence_file: Path) -> None:
        """Test copy_with preserves BPM displacement flags when not overridden."""
        psb = PSB(ring=3, sequence_file=test_sequence_file, optimise_bpm_dx=True, optimise_bpm_dy=True)
        copy = psb.copy_with(optimise_quadrupoles=True)
        assert copy.optimise_bpm_dx is True
        assert copy.optimise_bpm_dy is True
        assert copy.optimise_quadrupoles is True
