"""Tests for magnet perturbations using real loaded MAD sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface


def _get_element_attr(interface: AbaMadInterface, element_name: str, attr: str) -> float:
    """Read one element attribute from the currently loaded MAD sequence."""
    return float(interface.mad.loaded_sequence[element_name][attr])


def _get_effective_strength(interface: AbaMadInterface, element_name: str, attr: str) -> float:
    """Read the effective strength reported by the interface."""
    return interface.get_magnet_strengths([f"{element_name}.{attr}"])[f"{element_name}.{attr}"]


def _get_element_dknl(interface: AbaMadInterface, element_name: str, index: int) -> float:
    """Read one dknl component (0-based, Python indexing)."""
    try:
        return float(interface.mad.loaded_sequence[element_name].dknl[index])
    except IndexError:
        # If the dknl array is empty or shorter than expected, treat missing components as zero
        return 0


def test_effective_strength_matches_base_when_dknl_not_created(
    loaded_interface: AbaMadInterface,
    seq_b1: Path,
) -> None:
    """Effective strength lookup should fall back to the base strength before any perturbation."""
    del seq_b1
    quad_name = "MQY.B5L2.B1"

    assert len(loaded_interface.mad.loaded_sequence[quad_name].dknl) == 0
    assert np.isclose(
        _get_effective_strength(loaded_interface, quad_name, "k1"),
        _get_element_attr(loaded_interface, quad_name, "k1"),
    )


@pytest.mark.parametrize(
    ("rel_error", "expect_non_table_changed"),
    [(None, False), (1e-2, True)],
    ids=["table_relative_errors", "global_relative_error"],
)
def test_lhc_quadrupole_perturbation_modes(
    loaded_interface: AbaMadInterface,
    seq_b1: Path,
    rel_error: float | None,
    expect_non_table_changed: bool,
) -> None:
    """LHC quadrupole perturbation should support table mode and global relative-error mode."""
    table_family_quad = "MQY.B5L2.B1"  # Covered by QUAD_ERROR_TABLE
    non_table_quad = "MQT.12R2.B1"  # Not covered by QUAD_ERROR_TABLE

    k1_table_before = _get_effective_strength(loaded_interface, table_family_quad, "k1")
    k1_non_table_before = _get_effective_strength(loaded_interface, non_table_quad, "k1")

    magnet_strengths, _ = loaded_interface.apply_magnet_perturbations(
        rel_error=rel_error,
        seed=42,
        magnet_type="q",
    )

    k1_table_after = _get_effective_strength(loaded_interface, table_family_quad, "k1")
    k1_non_table_after = _get_effective_strength(loaded_interface, non_table_quad, "k1")

    assert not np.isclose(k1_table_after, k1_table_before)
    non_table_rel_change = abs(k1_non_table_after - k1_non_table_before) / max(
        abs(k1_non_table_before), 1e-12
    )
    non_table_changed = non_table_rel_change > 1e-3
    assert non_table_changed == expect_non_table_changed
    assert f"{table_family_quad}.k1" in magnet_strengths
    if expect_non_table_changed:
        assert f"{non_table_quad}.k1" in magnet_strengths


def test_sps_quadrupole_only_perturbation(
    loaded_sps_interface: AbaMadInterface,
    seq_sps: Path,
) -> None:
    """SPS magnet_type='q' should perturb quadrupoles but leave sextupoles/dipoles unchanged."""
    quad_name = "QF.13010"
    sext_name = "LSF.13205"
    dip_name = "MBA.13030"

    quad_before = _get_effective_strength(loaded_sps_interface, quad_name, "k1")
    sext_before = _get_element_attr(loaded_sps_interface, sext_name, "k2")
    dip_before = _get_element_attr(loaded_sps_interface, dip_name, "k0")

    magnet_strengths, _ = loaded_sps_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="q",
    )

    quad_after = _get_effective_strength(loaded_sps_interface, quad_name, "k1")
    sext_after = _get_element_attr(loaded_sps_interface, sext_name, "k2")
    dip_after = _get_element_attr(loaded_sps_interface, dip_name, "k0")

    assert not np.isclose(quad_after, quad_before)
    assert np.isclose(sext_after, sext_before)
    assert np.isclose(dip_after, dip_before)
    assert f"{quad_name}.k1" in magnet_strengths
    assert f"{sext_name}.k2" not in magnet_strengths
    assert f"{dip_name}.k0" not in magnet_strengths


def test_sps_perturbation_sets_dknl(
    loaded_sps_interface: AbaMadInterface,
    seq_sps: Path,
) -> None:
    """SPS perturbation should set dknl and keep base sextupole strength."""
    sext_name = "LSF.13205"

    k2_before = _get_element_attr(loaded_sps_interface, sext_name, "k2")
    dknl_before = _get_element_dknl(loaded_sps_interface, sext_name, 2)

    magnet_strengths, true_strengths = loaded_sps_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=77,
        magnet_type="s",
    )

    k2_after = _get_element_attr(loaded_sps_interface, sext_name, "k2")
    dknl_after = _get_element_dknl(loaded_sps_interface, sext_name, 2)

    assert np.isclose(k2_after, k2_before)
    assert dknl_before is None or np.isclose(dknl_before, 0.0)
    assert dknl_after is not None and not np.isclose(dknl_after, 0.0)
    assert f"{sext_name}.k2" in magnet_strengths
    assert sext_name in true_strengths


def test_sps_perturbations_preserve_previous_dknl_components(
    loaded_sps_interface: AbaMadInterface,
    seq_sps: Path,
) -> None:
    """SPS perturbations should preserve previously written dknl components."""
    quad_name = "QF.13010"
    sext_name = "LSF.13205"

    quad_dknl1_before = _get_element_dknl(loaded_sps_interface, quad_name, 1)

    loaded_sps_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=42,
        magnet_type="q",
    )
    quad_dknl1_after_q = _get_element_dknl(loaded_sps_interface, quad_name, 1)
    assert not np.isclose(quad_dknl1_after_q, quad_dknl1_before)

    loaded_sps_interface.apply_magnet_perturbations(
        rel_error=None,
        seed=77,
        magnet_type="s",
    )

    quad_dknl1_after_q_then_s = _get_element_dknl(loaded_sps_interface, quad_name, 1)
    sext_dknl2_after_q_then_s = _get_element_dknl(loaded_sps_interface, sext_name, 2)

    assert np.isclose(quad_dknl1_after_q_then_s, quad_dknl1_after_q)
    assert not np.isclose(sext_dknl2_after_q_then_s, 0.0)
