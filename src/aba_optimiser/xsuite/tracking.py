from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xobjects import ContextCpu as Context

from aba_optimiser.simulation.coordinates import create_initial_conditions

from .action_angle import _build_coords_from_action_angle
from .monitors import insert_particle_monitors_at_pattern

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tfs
    import xtrack as xt


def run_tracking(line: xt.Line, particles: xt.Particles, nturns: int) -> None:
    """Run tracking simulation for given particles."""
    logger.debug(f"Starting tracking for {nturns} turns")
    line.track(particles, num_turns=nturns, with_progress=True)
    if particles.state[0] == 1:
        logger.debug("Tracking completed successfully!")
        return
    raise RuntimeError("Tracking failed. Please check the input parameters.")


def run_tracking_without_ac_dipole(
    line: xt.Line,
    tws: xt.TwissTable,
    flattop_turns: int,
    bpm_pattern: str = r"bpm.*[^k]",
    particle_coords: dict[str, list[float]] | None = None,
    action_list: list[float] | None = None,
    angle_list: list[float] | None = None,
    kick_both_planes: bool = True,
    start_marker: str | None = None,
    delta_values: list[float] | None = None,
    context: Context | None = None,
) -> xt.Line:
    """Track particles without inserting an AC dipole, using monitors at BPMs."""

    working_line = line.copy()
    if start_marker is not None:
        working_line.cycle(name_first_element=start_marker.lower(), inplace=True)
        start_elem = working_line.element_names[0].upper()
    else:
        bpm_names = [name for name in working_line.element_names if "bpm" in name.lower()]
        start_elem = bpm_names[0].upper() if bpm_names else None

    if particle_coords is not None:
        num_particles = len(particle_coords["x"])
        xs = particle_coords["x"]
        pxs = particle_coords["px"]
        ys = particle_coords["y"]
        pys = particle_coords["py"]
        deltas = particle_coords.get("delta", [0.0] * num_particles)
    elif action_list is not None and angle_list is not None:
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

    monitored_line = insert_particle_monitors_at_pattern(
        line=working_line,
        pattern=bpm_pattern,
        num_turns=flattop_turns,
        num_particles=num_particles,
        inplace=True,
    )

    ctx = context or Context()
    particles = monitored_line.build_particles(
        _context=ctx,
        x=xs,
        px=pxs,
        y=ys,
        py=pys,
        delta=deltas,
    )

    run_tracking(line=monitored_line, particles=particles, nturns=flattop_turns)
    return monitored_line


def start_tracking_xsuite_batch(
    env: xt.Environment,
    batch_start: int,
    batch_end: int,
    action_list: list[float],
    angle_list: list[float],
    twiss_data: tfs.TfsDataFrame,
    kick_both_planes: bool,
    flattop_turns: int,
    progress_interval: int,
    num_tracks: int,
    true_deltap: float,
    seq_name: str,
) -> xt.Line:
    """Start tracking for a batch of particles using action-angle coordinates."""

    line: xt.Line = env[seq_name]  # type: ignore[index]

    x_list: list[float] = []
    px_list: list[float] = []
    y_list: list[float] = []
    py_list: list[float] = []

    for batch_idx in range(batch_end - batch_start):
        ntrk = batch_start + batch_idx

        if ntrk % progress_interval == 0:
            logger.info(
                f"Starting tracking command for process {ntrk}/{num_tracks - 1} "
                f"({ntrk / num_tracks * 100:.1f}%)"
            )

        x0_data: dict[str, int | float] = create_initial_conditions(
            ntrk,
            action_list,
            angle_list,
            twiss_data,
            kick_both_planes,
            starting_bpm=line.element_names[0].upper(),
        )
        x_list.append(x0_data["x"])
        px_list.append(x0_data["px"])
        y_list.append(x0_data["y"])
        py_list.append(x0_data["py"])

    num_particles = len(x_list)
    deltas = [true_deltap] * num_particles

    insert_particle_monitors_at_pattern(
        line=line,
        pattern="bpm.*[^k]",
        num_turns=flattop_turns,
        num_particles=num_particles,
        inplace=True,
    )

    ctx = Context(32)
    particles = line.build_particles(
        _context=ctx,
        x=x_list,
        px=px_list,
        y=y_list,
        py=py_list,
        delta=deltas,
    )

    run_tracking(
        line=line,
        particles=particles,
        nturns=flattop_turns,
    )
    return line
