from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xtrack as xt
from xobjects import ContextCpu as Context

from .action_angle import _build_coords_from_action_angle
from .env import create_xsuite_environment
from .monitors import insert_particle_monitors_at_pattern, process_tracking_data
from .tracking import run_tracking

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


def insert_ac_dipole(
    line: xt.Line,
    tws: xt.TwissTable,
    beam: int,
    acd_ramp: int,
    total_turns: int,
    driven_tunes: list[float],
    lag: float = 0.0,
) -> xt.Line:
    """Insert an AC dipole at the marked location in the line."""
    line = line.copy()
    acd_marker = f"mkqa.6l4.b{beam}"
    betxac = tws.rows[acd_marker]["betx"]
    betyac = tws.rows[acd_marker]["bety"]
    logger.info(f"Inserting AC dipole at {acd_marker} with betx={betxac}, bety={betyac}")

    driven_tunes = [q % 1 for q in driven_tunes]
    qxd_qx = driven_tunes[0] - tws["qx"] % 1
    qyd_qx = driven_tunes[1] - tws["qy"] % 1
    logger.info(f"Qxd/Qx: {qxd_qx}, Qyd/Qx: {qyd_qx}")
    pbeam = line.particle_ref.p0c / 1e9

    line.env.elements[f"mkach.6l4.b{beam}"] = xt.ACDipole(  # type: ignore[attr-defined]
        plane="x",
        volt=2 * 0.042 * pbeam * abs(qxd_qx) / np.sqrt(180.0 * betxac),
        freq=driven_tunes[0],
        lag=lag,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    line.env.elements[f"mkacv.6l4.b{beam}"] = xt.ACDipole(  # type: ignore[attr-defined]
        plane="y",
        volt=2 * 0.042 * pbeam * abs(qyd_qx) / np.sqrt(177.0 * betyac),
        freq=driven_tunes[1],
        lag=lag,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    placement = line.get_s_position(acd_marker)
    line.insert(f"mkacv.6l4.b{beam}", at=placement)
    line.insert(f"mkach.6l4.b{beam}", at=placement)
    return line


def _prepare_acd_line_with_monitors(
    line: xt.Line,
    tws: xt.TwissTable | None,
    beam: int,
    ramp_turns: int,
    flattop_turns: int,
    driven_tunes: list[float],
    lag: float,
    bpm_pattern: str,
    num_particles: int,
    inplace: bool = False,
) -> tuple[xt.Line, int]:
    """Insert AC dipole and monitors; return monitored line and total turns."""
    total_turns = ramp_turns + flattop_turns
    if tws is None:
        tws = line.twiss(method="4d")
    ac_line = insert_ac_dipole(
        line=line,
        tws=tws,
        beam=beam,
        acd_ramp=ramp_turns,
        total_turns=total_turns,
        driven_tunes=driven_tunes,
        lag=lag,
    )
    monitored_line = insert_particle_monitors_at_pattern(
        line=ac_line,
        pattern=bpm_pattern,
        num_turns=total_turns,
        num_particles=num_particles,
        inplace=inplace,
    )
    return monitored_line, total_turns


def run_acd_twiss(line: xt.Line, beam: int, dpp: float, driven_tunes: list[float]) -> xt.TwissTable:
    """Run twiss calculation with AC dipole elements."""
    line_acd = line.copy()
    before_acd_tws = line_acd.twiss(method="4d", delta0=dpp)
    acd_marker = f"mkqa.6l4.b{beam}"
    if acd_marker not in line.element_names:
        raise ValueError(f"AC dipole marker '{acd_marker}' not found in the line.")

    bet_at_acdipole = before_acd_tws.rows[acd_marker]
    logger.info(
        f"Running twiss with AC dipole at {acd_marker} with betx={bet_at_acdipole['betx']}, bety={bet_at_acdipole['bety']}"
    )

    line_acd.env.elements[f"mkach.6l4.b{beam}"] = xt.ACDipole(  # type: ignore[attr-defined]
        plane="x",
        natural_q=before_acd_tws["qx"] % 1,
        freq=driven_tunes[0],
        beta_at_acdipole=bet_at_acdipole["betx"],
        twiss_mode=True,
    )
    line_acd.env.elements[f"mkacv.6l4.b{beam}"] = xt.ACDipole(  # type: ignore[attr-defined]
        plane="y",
        natural_q=before_acd_tws["qy"] % 1,
        freq=driven_tunes[1],
        beta_at_acdipole=bet_at_acdipole["bety"],
        twiss_mode=True,
    )

    placement = line_acd.get_s_position(acd_marker)
    line_acd.insert(f"mkach.6l4.b{beam}", at=placement)
    line_acd.insert(f"mkacv.6l4.b{beam}", at=placement)
    return line_acd.twiss(method="4d", delta0=dpp)


def run_acd_track(
    json_path: Path,
    sequence_file: Path,
    delta_p: float = 0.0,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    driven_tunes: list[float] | None = None,
    beam: int = 1,
    bpm_pattern: str = r"(?i)bpm.*",
    add_variance_columns: bool = True,
) -> tuple[pd.DataFrame, xt.TwissTable, xt.Line]:
    """Set up xsuite tracking with AC dipole and return tracking data + twiss."""
    if driven_tunes is None:
        driven_tunes = [0.27, 0.322]

    env = create_xsuite_environment(
        json_file=json_path,
        sequence_file=sequence_file,
        seq_name=f"lhcb{beam}",
    )
    baseline_line: xt.Line = env[f"lhcb{beam}"].copy()  # ty:ignore[not-subscriptable]
    tws_input: xt.TwissTable = baseline_line.twiss4d(delta0=delta_p)

    qx = float(tws_input.qx % 1)
    qy = float(tws_input.qy % 1)
    logger.info(f"Natural tunes: Qx = {qx:.6f}, Qy = {qy:.6f}")
    if not (np.isclose(qx, 0.28, atol=1e-3) and np.isclose(qy, 0.31, atol=1e-3)):
        logger.warning(f"Tunes (Qx={qx:.6f}, Qy={qy:.6f}) differ from expected (0.28, 0.31)")

    monitored_line, total_turns = _prepare_acd_line_with_monitors(
        line=baseline_line,
        tws=tws_input,
        beam=beam,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=driven_tunes,
        lag=0.0,
        bpm_pattern=bpm_pattern,
        num_particles=1,
        inplace=False,
    )

    ctx = Context()
    particles: xt.Particles = monitored_line.build_particles(
        _context=ctx,
        x=0,
        y=0,
        px=0,
        py=0,
        delta=delta_p,
    )

    logger.info(f"Tracking {total_turns} turns with AC dipole")
    monitored_line.track(particles, num_turns=total_turns, with_progress=False)

    tracking_df = process_tracking_data(
        monitored_line, ramp_turns, flattop_turns, delta_p, add_variance_columns
    )

    return tracking_df, tws_input, baseline_line


def run_ac_dipole_tracking_with_particles(
    line: xt.Line,
    tws: xt.TwissTable | None = None,
    beam: int = 1,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    driven_tunes: list[float] | None = None,
    lag: float = 0.0,
    bpm_pattern: str = r"bpm.*[^k]",
    particle_coords: dict[str, list[float]] | None = None,
    action_list: list[float] | None = None,
    angle_list: list[float] | None = None,
    kick_both_planes: bool = True,
    start_marker: str | None = None,
    delta_values: list[float] | None = None,
) -> xt.Line:
    """Run AC dipole tracking with multiple particles using various initial conditions."""
    if driven_tunes is None:
        driven_tunes = [0.27, 0.322]

    if particle_coords is not None:
        num_particles = len(particle_coords["x"])
        xs = particle_coords["x"]
        pxs = particle_coords["px"]
        ys = particle_coords["y"]
        pys = particle_coords["py"]
        deltas = particle_coords.get("delta", [0.0] * num_particles)
    elif action_list is not None and angle_list is not None:
        working_line = line.copy()
        if start_marker is not None:
            working_line.cycle(name_first_element=start_marker.lower(), inplace=True)
            start_elem = working_line.element_names[0].upper()
        else:
            bpm_names = [name for name in working_line.element_names if "bpm" in name.lower()]
            start_elem = bpm_names[0].upper() if bpm_names else None

        if tws is None:
            tws = working_line.twiss4d()
        xs, pxs, ys, pys = _build_coords_from_action_angle(
            action_list,
            angle_list,
            tws,
            kick_both_planes,
            start_elem,
        )
        num_particles = len(xs)
        deltas = delta_values if delta_values is not None else [0.0] * num_particles
    else:
        raise ValueError("Provide particle_coords or both action_list and angle_list")

    monitored_line, total_turns = _prepare_acd_line_with_monitors(
        line=line,
        tws=tws,
        beam=beam,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=driven_tunes,
        lag=lag,
        bpm_pattern=bpm_pattern,
        num_particles=num_particles,
        inplace=False,
    )

    ctx = Context()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=xs,
        px=pxs,
        y=ys,
        py=pys,
        delta=deltas,
    )

    logger.info(f"Tracking {num_particles} particles for {total_turns} turns")
    run_tracking(line=monitored_line, particles=particles, nturns=total_turns)

    return monitored_line
