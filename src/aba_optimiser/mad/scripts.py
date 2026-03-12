"""Helpers for loading or generating MAD-NG tracking scripts."""

from __future__ import annotations

import logging
from pathlib import Path

mad_dir = Path(__file__).absolute().parent
MAD_SCRIPTS_DIR = mad_dir / "mad_scripts"

TRACK_OPTICS_INIT = MAD_SCRIPTS_DIR / "run_optics_track_init.mad"
TRACK_OPTICS_SCRIPT = MAD_SCRIPTS_DIR / "run_optics_track.mad"

TRACKING_OBSERVABLES = ("x", "y", "px", "py")
LOGGER = logging.getLogger(__name__)
TAB = "\t"


def dump_debug_script(
    script_name: str,
    text: str,
    *,
    debug: bool,
    mad_logfile: Path | None = None,
    worker_id: int | None = None,
) -> Path | None:
    """Write a generated MAD script to disk when debug mode is enabled."""
    if not debug:
        return None

    base_dir = (
        mad_logfile.parent / "generated_mad_scripts"
        if mad_logfile is not None
        else Path.cwd() / "generated_mad_scripts"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_worker_{worker_id}" if worker_id is not None else ""
    output_path = base_dir / f"{script_name}{suffix}.mad"
    output_path.write_text(text)
    LOGGER.debug("Wrote generated MAD script to %s", output_path)
    return output_path


def _validate_observables(observables: tuple[str, ...]) -> tuple[str, ...]:
    invalid = set(observables) - set(TRACKING_OBSERVABLES)
    if invalid:
        raise ValueError(f"Unsupported tracking observables: {sorted(invalid)}")
    if not observables:
        raise ValueError("At least one observable is required")
    return observables


def _join_lines(lines: list[str]) -> str:
    """Join script lines while keeping the assembly logic easy to read."""
    return "\n".join(lines)


def _indent(level: int, text: str) -> str:
    """Indent one generated Lua line with tabs."""
    return f"{TAB * level}{text}"


def _table_definitions(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(f"{observable} = table.new(batch_size, 0)")
    for observable in observables:
        lines.append(f"d{observable}_dk = table.new(batch_size, 0)")
    return _join_lines(lines)


def _allocation_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(_indent(1, f"{observable}[i] = vector(nbpms * n_run_turns)"))
    for observable in observables:
        lines.append(_indent(1, f"d{observable}_dk[i] = matrix(matrix_size, nbpms * n_run_turns)"))
    return _join_lines(lines)


def _save_scalar_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(
            _indent(3, f"{observable}[i]:seti(observe_count, mflw[i].{observable}:get0())")
        )
    return _join_lines(lines)


def _save_knob_derivative_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(
            _indent(
                4,
                f"d{observable}_dk[i]:setsub(set_range, observe_count, get_knob_vec(mflw[i].{observable}))",
            )
        )
    return _join_lines(lines)


def _save_energy_derivative_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(
            _indent(
                4,
                f"d{observable}_dk[i]:setsub(dpt_idx, observe_count, mflw[i].{observable}:get(7))",
            )
        )
    return _join_lines(lines)


def _reset_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(_indent(2, f"{observable}[i]:zeros()"))
    for observable in observables:
        lines.append(_indent(2, f"d{observable}_dk[i]:zeros()"))
    return _join_lines(lines)


def _send_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(f"python:send({observable}, true)")
    for observable in observables:
        lines.append(f"python:send(d{observable}_dk, true)")
    return _join_lines(lines)


def _hessian_asserts(observables: tuple[str, ...]) -> str:
    return " and ".join(f"weights_{observable}" for observable in observables)


def _hessian_weight_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(f"local W_{observable} = vector(weights_{observable}):diag()")
    return _join_lines(lines)


def _hessian_accumulation_block(observables: tuple[str, ...]) -> str:
    lines: list[str] = []
    for observable in observables:
        lines.append(_indent(2, f"local j_{observable} = d{observable}_dk[part]"))
    for observable in observables:
        lines.append(
            _indent(2, f"Htot = Htot + j_{observable} * (W_{observable} * j_{observable}:t())")
        )
    return _join_lines(lines)


def build_tracking_init_script(observables: tuple[str, ...]) -> str:
    """Build the tracking initialisation script for the requested observables."""
    observables = _validate_observables(observables)

    return f"""! Generated tracking init script
assert(
    batch_size and nbpms and knob_monomials and n_run_turns and sdir and
    knob_names and vector and matrix and python,
    "Missing required variables for initialising"
)

num_knobs = #knob_names
local matrix_size = optimise_energy and (num_knobs + 1) or num_knobs
local set_range, get_range, get_knob_vec

if num_knobs > 0 then
    set_range = vector(num_knobs):seq()
    get_range = vector(num_knobs):seq(7)

    function get_knob_vec(coord_tpsa)
        local all_first_deriv = coord_tpsa:getvec(1, num_knobs + 7)
        return all_first_deriv:getvec(get_range)
    end
end

{_table_definitions(observables)}

for i=1,batch_size do
{_allocation_block(observables)}
end

observe_count = nbpms + 1
function save_data(elm, mflw, _, slc)
    if slc == -2 and elm:is_observed() then
        for i=1,batch_size do
{_save_scalar_block(observables)}

            if set_range then
{_save_knob_derivative_block(observables)}
            end

            if optimise_energy then
                local dpt_idx = num_knobs + 1
{_save_energy_derivative_block(observables)}
            end
        end
        observe_count = observe_count + 1
    end
end

function reset_before_tracking()
    observe_count = 1
    for i=1,batch_size do
{_reset_block(observables)}
    end
end
"""


def build_tracking_script(observables: tuple[str, ...]) -> str:
    """Build the tracking script for the requested observables."""
    observables = _validate_observables(observables)
    return f"""! Generated tracking script
reset_before_tracking()
local _, mflw= track{{
    sequence=loaded_sequence,
    X0=da_x0_c[batch],
    nturn=n_run_turns,
    save=false,
    atexit=save_data,
    range=tracking_range,
    dir=sdir,
}}

{_send_block(observables)}
"""


def build_tracking_hessian_script(observables: tuple[str, ...]) -> str:
    """Build the Hessian script for the requested observables."""
    observables = _validate_observables(observables)
    return f"""! Generated tracking Hessian script
assert(
    loaded_sequence and batch_size and nbpms and reset_before_tracking and
    knob_names and coord_names and da_x0_c and knob_monomials and
    {_hessian_asserts(observables)} and vector and matrix and python and n_run_turns,
    "Missing required variables for tracking"
)

local matrix, vector in MAD
local matrix_size = optimise_energy and (num_knobs + 1) or num_knobs
local Htot = matrix(matrix_size, matrix_size):zeros()
{_hessian_weight_block(observables)}
collectgarbage("collect")

for batch=1,num_batches do
    reset_before_tracking()
    local _, mflw= track{{
        sequence = loaded_sequence,
        X0 = da_x0_c[batch],
        nturn = n_run_turns,
        save=false,
        atexit=save_data,
        range=tracking_range,
        dir=sdir,
    }}

    for part = 1, batch_size do
{_hessian_accumulation_block(observables)}
    end
end
python:send(Htot, true)
"""
