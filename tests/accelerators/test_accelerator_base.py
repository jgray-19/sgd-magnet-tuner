"""Tests for base Accelerator class and its interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pymadng_utils.accelerators.base import PROTON_MASS_GEV

from aba_optimiser.accelerators.base import Accelerator, KnobSpec

if TYPE_CHECKING:
    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


class ConcreteAccelerator(Accelerator):
    """Concrete implementation of Accelerator for testing."""

    def __init__(self, *args, bpm_pattern: str = "^BPM", **kwargs):
        super().__init__(*args, bpm_pattern=bpm_pattern, **kwargs)  # ty:ignore[parameter-already-assigned]

    @property
    def seq_name(self) -> str:
        """Return a test sequence name."""
        return "test_seq"

    def get_supported_knob_specs(self) -> list[KnobSpec]:
        """Return a simple knob specification for testing."""
        """Return a list of supported knob specifications."""
        return [
            KnobSpec(
                "quadrupole",
                "k1",
                "MQ",
                "k1",
                enabled=self.optimise_quadrupoles,
                label="quadrupoles",
            ),
            KnobSpec(
                "sextupole",
                "k2",
                "MS",
                "k2",
                enabled=self.optimise_sextupoles,
                label="sextupoles",
            ),
        ]

    def ac_dipole_location(self) -> str | None:
        """Return None for ac dipole location in base class."""

    def copy_with(self, **overrides) -> ConcreteAccelerator:
        """Return a copy with selected constructor parameters overridden."""
        params = {
            "sequence_file": self.sequence_file,
            "kinetic_energy": self.kinetic_energy,
            "optimise_energy": self.optimise_energy,
            "optimise_quadrupoles": self.optimise_quadrupoles,
            "optimise_sextupoles": self.optimise_sextupoles,
            "optimise_quad_dx": self.optimise_quad_dx,
            "optimise_quad_dy": self.optimise_quad_dy,
            "custom_knobs_to_optimise": self.custom_knobs_to_optimise,
        }
        params.update(overrides)
        return type(self)(**params)

    # def get_exciter_bpm(self) -> dict[str, float] | None:
    #     """Return None for exciter BPM in base class."""

    def apply_accelerator_specific_errors(self, mad_iface: AbaMadInterface) -> None:
        """No accelerator-specific errors to apply in base class."""

    @staticmethod
    def infer_monitor_plane(bpm_name: str) -> str:
        del bpm_name
        return "HV"

    @property
    def tune_variables(self) -> tuple[str, str]:
        """Return dummy tune variable names for abstract interface compliance."""
        return "qx_test", "qy_test"

    @property
    def tune_integers(self) -> tuple[int, int]:
        """Return dummy integer tune parts for abstract interface compliance."""
        return 62, 60


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
            kinetic_energy=6800.0,
        )
        assert acc.sequence_file == test_sequence_file
        assert acc.kinetic_energy == pytest.approx(6800.0)
        assert acc.energy == pytest.approx(6800.0 + PROTON_MASS_GEV)
        assert acc.seq_name == "test_seq"
        assert acc.optimise_energy is False
        assert acc.optimise_quadrupoles is False
        assert acc.optimise_sextupoles is False
        assert acc.custom_knobs_to_optimise is None

    def test_init_with_seq_name(self, test_sequence_file: Path) -> None:
        """Test initialization with sequence name."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
        )
        assert acc.seq_name == "test_seq"

    def test_init_with_optimise_energy(self, test_sequence_file: Path) -> None:
        """Test initialization with energy optimization enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_energy=True,
        )
        assert acc.optimise_energy is True

    def test_init_with_all_optimisation_flags(self, test_sequence_file: Path) -> None:
        """Test initialization with all optimisation flags enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
        )
        assert isinstance(acc.sequence_file, Path)
        assert acc.sequence_file == test_sequence_file

    def test_has_any_optimisation_false(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns False when no optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
        )
        assert acc.has_any_optimisation() is False

    def test_has_any_optimisation_energy(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when energy optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_energy=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_quadrupoles(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when quad optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_quadrupoles=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_sextupoles(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when sextupole optimisation enabled."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_sextupoles=True,
        )
        assert acc.has_any_optimisation() is True

    def test_has_any_optimisation_custom_knobs(self, test_sequence_file: Path) -> None:
        """Test has_any_optimisation returns True when custom knobs provided."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            custom_knobs_to_optimise=["K1"],
        )
        assert acc.has_any_optimisation() is True

    def test_rejects_legacy_dknl_custom_knob_names(self, test_sequence_file: Path) -> None:
        """Legacy dknl knob names without the trailing length suffix are rejected."""
        with pytest.raises(ValueError, match=r"\.dk1l"):
            ConcreteAccelerator(
                sequence_file=test_sequence_file,
                kinetic_energy=6800.0,
                custom_knobs_to_optimise=["MQ.1L1.B1.dk1"],
            )

    def test_get_bend_lengths_returns_none(self, test_sequence_file: Path) -> None:
        """Test that base get_bend_lengths returns None."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
        )
        result = acc.get_bend_lengths()
        assert result is None

    def test_normalise_true_strengths_returns_unchanged(
        self, test_sequence_file: Path
    ) -> None:
        """Test that base normalise_true_strengths returns unchanged dictionary."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
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
            kinetic_energy=6800.0,
        )
        original = ["K1.b1", "K2.b1"]
        result = acc.format_result_knob_names(original)
        assert result is not original
        assert result == original

    def test_format_result_knob_names_rewrites_indexed_multipoles(
        self, test_sequence_file: Path
    ) -> None:
        """Test indexed knl/ksl knob names are rewritten to public dk forms."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
        )
        knob_names = ["mq.knl[3]", "ms.ksl[3]"]
        result = acc.format_result_knob_names(knob_names)
        assert result == ["mq.dk2l", "ms.dk2sl"]

    def test_log_optimisation_targets_none(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets with no optimisations."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
        )
        with caplog.at_level(logging.INFO):
            acc.log_optimisation_targets()
        assert "No optimisation targets set" in caplog.text

    def test_log_optimisation_targets_energy(self, test_sequence_file: Path, caplog) -> None:
        """Test log_optimisation_targets logs energy."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_energy=True,
        )
        with caplog.at_level(logging.INFO):
            acc.log_optimisation_targets()
        assert "beam energy" in caplog.text

    def test_log_optimisation_targets_multiple(
        self, test_sequence_file: Path, caplog
    ) -> None:
        """Test log_optimisation_targets logs multiple targets."""
        acc = ConcreteAccelerator(
            sequence_file=test_sequence_file,
            kinetic_energy=6800.0,
            optimise_energy=True,
            optimise_quadrupoles=True,
            custom_knobs_to_optimise=["K1"],
        )
        with caplog.at_level(logging.INFO):
            acc.log_optimisation_targets()
        assert "beam energy" in caplog.text
        assert "quadrupoles" in caplog.text
        assert "custom knobs" in caplog.text
