from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.optimising_mad_interface import GenericMadInterface
from aba_optimiser.measurements.b2_errors import (
    b2_errors_to_magnet_strengths,
    resolve_b2_error_table,
)
from tests.mad.helpers import cleanup_interface

if TYPE_CHECKING:
    from pathlib import Path


def _get_bend_sample(
    interface: GenericMadInterface, *, beam: int = 2, limit: int = 20
) -> tuple[list[str], dict[str, float]]:
    """Return the names and lengths of the first ``limit`` arc MB bends."""
    interface.mad.send(
        f"""
local names = {{}}
local lengths = {{}}

for _, e in loaded_sequence:iter() do
    if e.kind == "sbend"
        and e.l ~= nil
        and e.l > 0
        and string.match(e.name, "^MB%.[ABC]?%d+[LR][1-8]%.B{beam}$") then
        table.insert(names, e.name)
        lengths[e.name] = e.l
        if #names >= {limit} then
            break
        end
    end
end

{interface.py_name}:send(names, true)
{interface.py_name}:send(lengths, true)
"""
    )
    names = interface.mad.recv()
    lengths = interface.mad.recv()
    return names, lengths


def _write_b2_error_table(path: Path, names: list[str], lengths: dict[str, float]) -> dict[str, float]:
    k1l_values = {name: 2e-4 * float(lengths[name]) for name in names}
    df = tfs.TfsDataFrame({"K1L": [k1l_values[name] for name in names]}, index=names)
    df.index.name = "NAME"
    tfs.write(path, df, save_index="NAME")
    return k1l_values


def _get_bend_dknl2(interface: GenericMadInterface, name: str) -> float:
    """Return the quadrupole perturbation dknl[2] applied to a bend (0 if none)."""
    interface.mad.send(
        f"""
local element = loaded_sequence['{name}']
local value = 0
if MAD.typeid.is_deferred(element.dknl) then
    value = element.dknl[2] or 0
end
{interface.py_name}:send(value)
"""
    )
    return float(interface.mad.recv())


def _find_untouched_bend(interface: GenericMadInterface, excluded_names: list[str]) -> str:
    interface.mad.send(
        f"""
local excluded = {interface.py_name}:recv()

for _, e in loaded_sequence:iter() do
    if e.kind == "sbend"
        and e.l ~= nil
        and e.l > 0
        and string.match(e.name, "^MB%.[ABC]?%d+[LR][1-8]%.B2$")
        and not excluded[e.name] then
        {interface.py_name}:send(e.name)
        return
    end
end

error("Could not find untouched LHC bend")
"""
    )
    interface.mad.send(dict.fromkeys(excluded_names, True))
    return str(interface.mad.recv())


def test_resolve_b2_error_table_picks_closest_energy(tmp_path: Path) -> None:
    beam_root = tmp_path / "Beam2"
    beam_root.mkdir(parents=True)
    for name in (
        "MB2022_0450.0GeV_1100cm.errors",
        "MB2022_6500.0GeV_0133cm.errors",
        "MB2022_6800.0GeV_0133cm.errors",
    ):
        (beam_root / name).write_text("")

    assert resolve_b2_error_table(2, 6799.0, errors_root=tmp_path) == (
        beam_root / "MB2022_6800.0GeV_0133cm.errors"
    )
    assert resolve_b2_error_table(2, 6510.0, errors_root=tmp_path) == (
        beam_root / "MB2022_6500.0GeV_0133cm.errors"
    )


def test_b2_errors_to_magnet_strengths_routes_k1l_to_dk1l_suffix() -> None:
    strengths = b2_errors_to_magnet_strengths({"MB.A12L1.B2": 14.3, "MB.B7R4.B2": -2.1})

    assert strengths == {"MB.A12L1.B2.dk1l": 14.3, "MB.B7R4.B2.dk1l": -2.1}


def test_b2_errors_to_magnet_strengths_drops_zero_entries() -> None:
    strengths = b2_errors_to_magnet_strengths({"MB.A12L1.B2": 0.0, "MB.B7R4.B2": 3.5})

    assert strengths == {"MB.B7R4.B2.dk1l": 3.5}


@pytest.mark.slow
def test_lhc_b2_errors_require_tune_knobs_file(seq_b2: Path, tmp_path: Path) -> None:
    """b2 errors shift the tunes, so the MAD interface must receive a tune knobs file."""
    error_file = tmp_path / "b2.errors"
    _write_b2_error_table(error_file, ["MB.A12L1.B2"], {"MB.A12L1.B2": 14.3})

    with pytest.raises(ValueError, match="tune knobs are designed to compensate"):
        GenericMadInterface(
            accelerator=LHC(
                beam=2,
                kinetic_energy=6800.0,
                sequence_file=seq_b2,
            ),
            b2_errors=error_file,
        )


@pytest.mark.slow
def test_lhc_b2_errors_route_to_dknl_and_keep_twiss_stable(
    seq_b2: Path, tune_knobs: Path, tmp_path: Path
) -> None:
    clean = GenericMadInterface(
        accelerator=LHC(beam=2, kinetic_energy=6800.0, sequence_file=seq_b2)
    )
    try:
        names, lengths = _get_bend_sample(clean, beam=2, limit=20)
        assert names, "Expected to find at least one LHC bend"
        untouched_name = _find_untouched_bend(clean, names)
        clean_twiss = clean.run_twiss(observe=0)
        assert clean_twiss is not None
    finally:
        cleanup_interface(clean)

    error_file = tmp_path / "b2.errors"
    k1l_values = _write_b2_error_table(error_file, names, lengths)

    with_errors = GenericMadInterface(
        accelerator=LHC(
            beam=2,
            kinetic_energy=6800.0,
            sequence_file=seq_b2,
        ),
        b2_errors=error_file,
        tune_knobs=tune_knobs,
    )
    try:
        # Each errored bend carries its K1L in the dknl[2] (quadrupole) slot.
        for name in names:
            assert _get_bend_dknl2(with_errors, name) == pytest.approx(
                k1l_values[name], rel=0.0, abs=1e-15
            ), name
        # Bends absent from the table keep an empty perturbation table.
        assert _get_bend_dknl2(with_errors, untouched_name) == 0.0
        # Small b2 errors shift the tunes but must NOT destabilise twiss.
        errored_twiss = with_errors.run_twiss(observe=0)
        assert errored_twiss is not None
    finally:
        cleanup_interface(with_errors)
