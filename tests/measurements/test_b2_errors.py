from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tfs

from aba_optimiser.accelerators import LHC
from aba_optimiser.mad.optimising_mad_interface import GenericMadInterface
from aba_optimiser.measurements.b2_errors import resolve_b2_error_table
from tests.mad.helpers import cleanup_interface

if TYPE_CHECKING:
    from pathlib import Path


def _get_bend_sample(
    interface: GenericMadInterface, *, limit: int = 20
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    interface.mad.send(
        f"""
local limit = {limit}
local names = {{}}
local lengths = {{}}
local values = {{}}

for _, e in loaded_sequence:iter() do
    if e.kind == "sbend"
        and e.l ~= nil
        and e.l > 0
        and string.match(e.name, "^MB%.[ABC]?%d+[LR][1-8]%.B2$") then
        if e.knl == nil then
            error("Expected bend to expose knl for K1L application: " .. e.name)
        end
        table.insert(names, e.name)
        lengths[e.name] = e.l
        values[e.name] = e.knl[2] or 0

        if #names >= limit then
            break
        end
    end
end

{interface.py_name}:send(names, true)
{interface.py_name}:send(lengths, true)
{interface.py_name}:send(values, true)
"""
    )
    names = interface.mad.recv()
    lengths = interface.mad.recv()
    values = interface.mad.recv()
    return names, lengths, values


def _write_b2_error_table(path: Path, names: list[str], lengths: dict[str, float]) -> dict[str, float]:
    k1l_values = {name: 2e-8 * float(lengths[name]) for name in names}
    df = tfs.TfsDataFrame({"K1L": [k1l_values[name] for name in names]}, index=names)
    df.index.name = "NAME"
    tfs.write(path, df, save_index="NAME")
    return k1l_values


def _get_bend_component(interface: GenericMadInterface, name: str) -> float:
    interface.mad.send(
        f"""
local element = loaded_sequence['{name}']
if element.knl == nil then
    error("Expected bend to expose knl for K1L application: " .. '{name}')
end
{interface.py_name}:send(element.knl[2] or 0)
"""
    )
    return float(interface.mad.recv())


def _find_untouched_bend(interface: GenericMadInterface, excluded_names: list[str]) -> str:
    interface.mad.send(
        f"""
local excluded = {interface.py_name}:recv()

for _, e in loaded_sequence:iter() do
    if e.kind == "sbend"
        and e.knl ~= nil
        and e.l ~= nil
        and e.l > 0
        and string.match(e.name, "^MB%.[ABC]?%d+[LR][1-8]%.B2$")
        and not excluded[e.name] then
        {interface.py_name}:send(e.name)
        return
    end
end

error("Could not find untouched LHC bend with knl")
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


@pytest.mark.slow
def test_lhc_b2_errors_modify_bend_strengths_and_destabilise_twiss(
    seq_b2: Path, tmp_path: Path
) -> None:
    clean = GenericMadInterface(
        accelerator=LHC(beam=2, kinetic_energy=6800.0, sequence_file=seq_b2)
    )
    try:
        names, lengths, base_values = _get_bend_sample(clean, limit=20)
        assert names, "Expected to find at least one LHC bend"
        untouched_name = _find_untouched_bend(clean, names)
        untouched_before = _get_bend_component(clean, untouched_name)
        assert clean.run_twiss(observe=0) is not None
    finally:
        cleanup_interface(clean)

    error_file = tmp_path / "b2_errors.tfs"
    k1l_values = _write_b2_error_table(error_file, names, lengths)

    with_errors = GenericMadInterface(
        accelerator=LHC(
            beam=2,
            kinetic_energy=6800.0,
            sequence_file=seq_b2,
            b2_errors=error_file,
        ),
    )
    try:
        _, _, applied_values = _get_bend_sample(with_errors, limit=20)
        untouched_after = _get_bend_component(with_errors, untouched_name)
        with pytest.raises(RuntimeError, match="Twiss failed"):
            with_errors.run_twiss(observe=0)
    finally:
        cleanup_interface(with_errors)

    for name in names:
        delta = float(applied_values[name]) - float(base_values[name])
        assert delta == pytest.approx(k1l_values[name], rel=0.0, abs=1e-15), name

    assert untouched_after == pytest.approx(untouched_before, rel=0.0, abs=1e-15)
