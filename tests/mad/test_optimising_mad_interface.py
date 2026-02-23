"""Tests for MAD interfaces.

This module contains pytest tests for GenericMadInterface and
LHCOptimisationMadInterface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymadng_utils.io.utils import read_knobs

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.lhc_optimising_interface import LHCOptimisationMadInterface
from aba_optimiser.mad.optimising_mad_interface import GenericMadInterface
from tests.mad.helpers import (
    check_beam_setup,
    check_corrector_strengths,
    check_corrector_strengths_zero,
    check_element_observations,
    check_interface_basic_init,
    check_sequence_loaded,
    cleanup_interface,
)

BEAM_ENERGY = 6800  # Beam energy in GeV

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    import pandas as pd


def setup_and_check_interface(
    accelerator: LHC,
    magnet_range: str = "$start/$end",
    bpm_range: str = "$start/$end",
    bpm_pattern: str = r"^BPM",
) -> tuple[GenericMadInterface, pd.DataFrame]:
    """Set up interface with given parameters and perform common validation checks.

    Args:
        accelerator: LHC accelerator instance
        magnet_range: Range specification for magnets
        bpm_range: Range specification for BPMs
        bpm_pattern: Regex pattern for BPM matching

    Returns:
        Tuple of (interface, twiss_dataframe) after setup and validation
    """
    interface = GenericMadInterface(
        accelerator=accelerator,
        magnet_range=magnet_range,
        bpm_range=bpm_range,
        bpm_pattern=bpm_pattern,
    )

    # Verify that MAD variables are set correctly
    assert interface.mad["magnet_range"] == magnet_range
    assert interface.mad["bpm_range"] == bpm_range
    assert interface.mad["bpm_pattern"] == bpm_pattern
    assert len(interface.bpms_in_range) == interface.nbpms

    # Check that BPMs matching the pattern are observed
    check_element_observations(interface, condition=f"elm.name:match('{bpm_pattern}')")

    # Run twiss calculation to get BPM data
    twiss_df = interface.run_twiss()

    return interface, twiss_df


@pytest.fixture(scope="function")
def optimising_interface(seq_b1: Path) -> Generator[LHCOptimisationMadInterface, None, None]:
    """Create a fresh LHCOptimisationMadInterface for each test."""
    accelerator = LHC(
        beam=1,
        beam_energy=BEAM_ENERGY,
        sequence_file=str(seq_b1),
    )
    iface = LHCOptimisationMadInterface(
        accelerator=accelerator,
    )
    yield iface
    cleanup_interface(iface)


class TestOptimisationMadInterfaceInit:
    @pytest.mark.parametrize(
        "beam_energy, seq_name",
        [
            (None, None),
            (6500, "lhcb1"),
        ],
    )
    def test_default(self, seq_b1: Path, beam_energy: float | None, seq_name: str | None) -> None:
        """Test initialisation of GenericMadInterface with default parameters."""
        accelerator = LHC(
            beam=1,
            beam_energy=beam_energy or BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface = GenericMadInterface(
            accelerator=accelerator,
        )
        bpm_pattern = r"^BPM"
        check_interface_basic_init(interface, "py")

        # Default discard_mad_output=False, so no stdout_file attributes
        assert not hasattr(interface.mad._MAD__process, 'stdout_file')
        assert not hasattr(interface.mad._MAD__process, 'stdout_file_path')
        assert interface.mad._MAD__process.debug is False  # default debug=False

        assert interface.magnet_range == "$start/$end"
        assert interface.bpm_range == "$start/$end"
        assert interface.bpm_pattern == bpm_pattern
        assert interface.nbpms == 563, f"Expected 563 BPMs, got {interface.nbpms}"

        assert isinstance(interface.bpms_in_range, list)
        assert len(interface.bpms_in_range) == interface.nbpms
        assert isinstance(interface.all_bpms, list)
        assert len(interface.all_bpms) == 563  # Total BPMs in sequence

        # Check sequence loading
        check_sequence_loaded(interface, "lhcb1")

        # Check beam setup
        check_beam_setup(interface, particle="proton", energy=beam_energy or BEAM_ENERGY)

        assert interface.knob_names == []
        assert interface.elem_spos == []

        cleanup_interface(interface)

    def test_bpm_magnet_pattern(self, seq_b1: Path) -> None:
        """Test that MAD variables are set correctly for default ranges and patterns."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface = GenericMadInterface(
            accelerator=accelerator,
            discard_mad_output=True,
        )

        # Verify that MAD variables are set correctly
        assert interface.mad["magnet_range"] == "$start/$end"
        assert interface.mad["bpm_range"] == "$start/$end"
        assert interface.mad["bpm_pattern"] == "^BPM"
        assert len(interface.bpms_in_range) == interface.nbpms

        # Check that BPMs matching the pattern are observed
        check_element_observations(interface, condition=f"elm.name:match('{interface.bpm_pattern}')")

        # Run twiss calculation to get BPM data
        twiss_df = interface.run_twiss()

        # Verify that the twiss dataframe includes all BPMs in the range
        assert list(twiss_df.index) == interface.bpms_in_range

        cleanup_interface(interface)

    def test_custom_bpm_pattern(self, seq_b1: Path) -> None:
        """Test that MAD variables are set correctly for custom patterns."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface, twiss_df = setup_and_check_interface(
            accelerator, "BPM.10L1.B1/BPM.20L1.B1", "$start/$end", r"^BPM%.10.*"
        )

        # Filter BPMs to match the custom pattern (BPMs starting with "BPM.10")
        expected_bpms = [name for name in interface.bpms_in_range if name.startswith("BPM.10")]

        # Verify that twiss dataframe contains exactly the expected filtered BPMs
        assert len(twiss_df.index) == len(expected_bpms)
        assert list(twiss_df.index) == expected_bpms
        assert len(twiss_df.index) == interface.nbpms

        cleanup_interface(interface)

    def test_custom_bpm_range(self, seq_b1: Path) -> None:
        """Test that MAD variables are set correctly for custom BPM ranges."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface, twiss_df = setup_and_check_interface(
            accelerator, "$start/$end", "BPM.10L1.B1/BPM.10R1.B1", r"^BPM"
        )

        # Extract the BPM range boundaries
        first_bpm, second_bpm = "BPM.10L1.B1", "BPM.10R1.B1"

        # Verify that bpms_in_range is correctly sliced to the specified range
        assert interface.bpms_in_range[0] == first_bpm
        assert interface.bpms_in_range[-1] == second_bpm

        start_idx = interface.all_bpms.index(first_bpm)
        end_idx = interface.all_bpms.index(second_bpm) + 1
        assert len(interface.bpms_in_range) == end_idx - start_idx

        # Verify that all_bpms contains all BPMs in the sequence
        assert len(interface.all_bpms) == 563
        # Verify that twiss dataframe contains all observed BPMs (not just those in range)
        assert len(twiss_df.index) == len(interface.all_bpms)

        cleanup_interface(interface)

    @pytest.mark.parametrize(
        "optimise_energy, optimise_quadrupoles, optimise_bends",
        [(True, False, False), (False, True, False), (False, False, True), (True, True, False)],
        ids=["opt-energy_only", "opt-quad_only", "opt-bend_only", "opt-energy_quad"],
    )
    def test_with_knob_config(
        self,
        seq_b1: Path,
        optimise_energy: bool,
        optimise_quadrupoles: bool,
        optimise_bends: bool,
    ) -> None:
        """Test initialisation with knob configuration."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
            optimise_energy=optimise_energy,
            optimise_quadrupoles=optimise_quadrupoles,
            optimise_bends=optimise_bends,
        )
        interface = LHCOptimisationMadInterface(
            accelerator=accelerator,
            discard_mad_output=True,
        )
        check_interface_basic_init(interface, "py")
        if optimise_energy:
            assert "pt" in interface.knob_names

        allowed_substrings = []
        if optimise_energy:
            allowed_substrings.append("pt")
        if optimise_quadrupoles:
            allowed_substrings.append("MQ")
        if optimise_bends:
            allowed_substrings.append("MB")

        assert all(any(sub in name for sub in allowed_substrings) for name in interface.knob_names)

        if optimise_energy and not (optimise_quadrupoles or optimise_bends):
            assert len(interface.knob_names) == 1
            assert len(interface.elem_spos) == 0
        else:
            if optimise_energy:
                assert len(interface.elem_spos) == len(interface.knob_names) - 1
            else:
                assert len(interface.elem_spos) == len(interface.knob_names)
        cleanup_interface(interface)

    @pytest.mark.parametrize("apply_correctors", [True, False])
    def test_with_corrector_settings(
        self,
        seq_b1: Path,
        corrector_file: Path,
        corrector_table,
        apply_correctors: bool,
    ) -> None:
        """Test initialization with different corrector strength settings."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface = GenericMadInterface(
            accelerator=accelerator,
            corrector_strengths=corrector_file if apply_correctors else None,
            tune_knobs_file=None,
        )
        # Check that strengths match expectations
        if apply_correctors:
            check_corrector_strengths(interface, corrector_table)
        else:
            check_corrector_strengths_zero(interface, corrector_table)

        cleanup_interface(interface)

    def test_knob_files(
        self, seq_b1: Path, corrector_knobs_file: Path, tune_knobs_file: Path
    ) -> None:
        """Test initialization with knob for tunes and corrector files."""
        corrector_knob_file = corrector_knobs_file
        tune_knob_file = tune_knobs_file

        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        no_knob_interface = GenericMadInterface(
            accelerator=accelerator,
            corrector_strengths=None,
            tune_knobs_file=None,
        )
        original_mqt_strength = no_knob_interface.mad["MADX['MQT.14R3.B1'].k1"]

        knob_interface = GenericMadInterface(
            accelerator=accelerator,
            corrector_strengths=corrector_knob_file,
            tune_knobs_file=tune_knob_file,
        )
        corrector_knobs = read_knobs(corrector_knob_file)
        tune_knobs = read_knobs(tune_knob_file)
        all_knobs = {**corrector_knobs, **tune_knobs}
        for name in all_knobs:
            assert knob_interface.mad[f"MADX['{name}']"] == all_knobs[name]
            assert knob_interface.mad[f"MADX['{name}']"] != no_knob_interface.mad[f"MADX['{name}']"]
        # Check that the mqt strength has changed, not just that the knobs exists in the MAD interface
        new_mqt_strength = knob_interface.mad["MADX['MQT.14R3.B1'].k1"]
        assert new_mqt_strength != original_mqt_strength
        print(f"Original MQT.14R3.B1 strength: {original_mqt_strength}, New strength: {new_mqt_strength}")

        cleanup_interface(knob_interface)
        cleanup_interface(no_knob_interface)

    @pytest.mark.parametrize(
        "bad_bpms",
        [
            None,
            [],
            ["BPM.10L1.B1"],
            ["BPM.10L1.B1", "BPM.10R1.B1"],
            ["BPM.10L1.B1", "BPM.10R1.B1", "BPM.11L1.B1"],
        ],
        ids=["none", "empty_list", "single_bpm", "two_bpms", "three_bpms"],
    )
    def test_bad_bpms(self, seq_b1: Path, bad_bpms: list[str] | None) -> None:
        """Test that bad_bpms are properly unobserved."""
        accelerator = LHC(
            beam=1,
            beam_energy=BEAM_ENERGY,
            sequence_file=str(seq_b1),
        )
        interface = GenericMadInterface(
            accelerator=accelerator,
            corrector_strengths=None,
            tune_knobs_file=None,
            bad_bpms=bad_bpms,
        )

        # Determine expected number of bad BPMs
        num_bad_bpms = len(bad_bpms) if bad_bpms else 0
        expected_nbpms = 563 - num_bad_bpms

        # Check that nbpms is reduced by the number of bad BPMs
        assert interface.nbpms == expected_nbpms, f"Expected {expected_nbpms} BPMs, got {interface.nbpms}"

        # Check that bad_bpms are not in bpms_in_range
        if bad_bpms:
            for bpm in bad_bpms:
                assert bpm not in interface.bpms_in_range, f"Bad BPM {bpm} should not be in bpms_in_range"

        # Run twiss and check that bad_bpms are not in the dataframe
        twiss_df = interface.run_twiss()
        if bad_bpms:
            for bpm in bad_bpms:
                assert bpm not in twiss_df.index, f"Bad BPM {bpm} should not be in twiss dataframe"

        # Check that the length of twiss_df matches nbpms
        assert len(twiss_df.index) == interface.nbpms

        cleanup_interface(interface)


@pytest.mark.parametrize(
    "bpm_range",
    [
        "$start/$end",
        "BPM.10L1.B1/BPM.10R1.B1",
        "BPM.7L4.B1/BPM.20R7.B1",
    ],
    ids=["full_range", "custom_range", "wider_custom_range"],
)
def test_count_bpms(optimising_interface: LHCOptimisationMadInterface, bpm_range: str) -> None:
    """Test counting BPMs in the sequence with different ranges."""
    full_bpms = optimising_interface.all_bpms

    if bpm_range == "$start/$end":
        expected_bpms_in_range = optimising_interface.bpms_in_range
    else:
        start, end = bpm_range.split("/")
        expected_bpms_in_range = full_bpms[full_bpms.index(start) : full_bpms.index(end) + 1]

    bpms_in_range, nbpms, all_bpms = optimising_interface.count_bpms(bpm_range)
    assert nbpms == len(expected_bpms_in_range)
    assert bpms_in_range == expected_bpms_in_range
    assert all_bpms == full_bpms  # Should always return full BPM list


def test_recv_update_knob_values(
    optimising_interface: LHCOptimisationMadInterface,
) -> None:
    """Test receiving current knob values."""
    values = optimising_interface.receive_knob_values()
    # No knobs are created, so we expect empty array
    assert len(values) == 0

    # Now add some knobs and verify they can be received
    optimising_interface.knob_names.extend(["a", "b", "c"])
    optimising_interface.mad.send("MADX.a = 1.0; MADX.b = 2.1; MADX.c = -3.2")
    values = optimising_interface.receive_knob_values()
    assert all(values == [1.0, 2.1, -3.2]), f"Unexpected knob values: {values}"

    # Update knobs (including a non-existent one "pt" that should be ignored)
    update_table = {"a": 4.5, "b": -6.7, "c": 8.9, "pt": 1.0}
    optimising_interface.update_knob_values(update_table)
    values = optimising_interface.receive_knob_values()
    # Only the 3 existing knobs should be updated, "pt" is ignored with a warning
    assert all(values == [4.5, -6.7, 8.9]), f"Unexpected knob values after update: {values}"
