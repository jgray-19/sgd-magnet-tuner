"""
Xsuite tools for LHC beam simulation and tracking.

This module contains utility functions for:
- Creating and managing xsuite environments
- Inserting particle monitors and AC dipoles
- Running tracking simulations
- Analyzing tracking data
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xpart as xp
import xtrack as xt
from xobjects import ContextCpu as Context
from xtrack.mad_parser.loader import load_madx_lattice

from aba_optimiser.config import (
    BEAM_ENERGY,
    BEAM_NUMBER,
    SEQ_NAME,
    SEQUENCE_FILE,
    XSUITE_JSON,
)
from aba_optimiser.simulation.tracking import create_initial_conditions

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)


def create_xsuite_environment(rerun_madx: bool = False) -> xt.Environment:
    """
    Run MADX to create a saved sequence then load this into xsuite.

    Args:
        rerun_madx: Whether to force re-running MADX

    Returns:
        xsuite Environment object
    """
    if not XSUITE_JSON.exists() or rerun_madx:
        if not SEQUENCE_FILE.exists():
            raise FileNotFoundError(f"Sequence file not found: {SEQUENCE_FILE}")

        env: xt.Environment = load_madx_lattice(file=SEQUENCE_FILE)
        env.to_json(XSUITE_JSON)
        logging.info(f"xsuite environment saved to {XSUITE_JSON}")
    else:
        logging.info(f"Loading existing xsuite environment from {XSUITE_JSON}")
        env = xt.Environment.from_json(XSUITE_JSON)

    env[SEQ_NAME].particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=BEAM_ENERGY * 1e9,
    )
    return env


def insert_particle_monitors_at_pattern(
    line: xt.Line,
    pattern: str,
    num_turns: int = 10_000,
    num_particles: int = 1,
    inplace: bool = False,
) -> xt.Line:
    """
    Returns a copy of the given line with particle monitors inserted at all elements
    whose names match the given regex pattern.

    Args:
        line: xsuite Line object
        pattern: Regex pattern to match element names
        num_turns: Number of turns to record
        num_particles: Number of particles to monitor

    Returns:
        Modified line with particle monitors
    """
    monitored_line = line
    if not inplace:
        monitored_line = line.copy()

    # Find all element names matching the pattern (e.g., all BPMs)
    selected_list = [
        name for name in monitored_line.element_names if re.match(pattern, name)
    ]
    if not selected_list:
        logger.warning(f"No elements found matching pattern '{pattern}'.")
        return monitored_line

    # Get the s-positions of these elements
    s_positions = monitored_line.get_s_position(selected_list)

    # Register a monitor element for each BPM, and collect placement instructions
    inserts = []
    for name, s in zip(selected_list, s_positions):
        name_upper = name.upper()
        monitored_line.env._element_dict[name_upper] = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=num_turns, num_particles=num_particles
        )

        # Only insert if the name was not already uppercase
        if name != name_upper:
            inserts.append(monitored_line.env.place(name_upper, at=s))

    # Insert all monitors at once for efficiency
    monitored_line.insert(inserts)
    return monitored_line


def insert_ac_dipole(
    line: xt.Line,
    tws: xt.TwissTable,
    beam: int,
    acd_ramp: int,
    total_turns: int,
    driven_tunes: list[float],
) -> xt.Line:
    """
    Inserts an AC dipole at the marked location in the line.

    Args:
        line: xsuite Line object
        tws: Twiss table
        beam: Beam number
        acd_ramp: AC dipole ramp time
        total_turns: Total number of turns
        driven_tunes: List of driven tunes [qx, qy]

    Returns:
        Modified line with AC dipole
    """
    line = line.copy()
    acd_marker = f"mkqa.6l4.b{beam}"
    betxac = tws.rows[acd_marker]["betx"]
    betyac = tws.rows[acd_marker]["bety"]
    logger.info(
        f"Inserting AC dipole at {acd_marker} with betx={betxac}, bety={betyac}"
    )

    driven_tunes = [q % 1 for q in driven_tunes]  # Ensure tunes are in [0, 1)
    logger.info(f"Driven tunes: {driven_tunes}")
    qxd_qx = driven_tunes[0] - tws["qx"] % 1
    qyd_qx = driven_tunes[1] - tws["qy"] % 1
    logger.info(f"Qxd/Qx: {qxd_qx}, Qyd/Qx: {qyd_qx}")
    logger.info(f"tws['qx']={tws['qx']}, tws['qy']={tws['qy']}")
    pbeam = line.particle_ref.p0c / 1e9  # Convert to GeV

    line.env.elements[f"mkach.6l4.b{beam}"] = xt.ACDipoleThickHorizontal(
        volt=0.042 * pbeam * abs(qxd_qx) / np.sqrt(180.0 * betxac),
        freq=driven_tunes[0],
        lag=0,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    line.env.elements[f"mkacv.6l4.b{beam}"] = xt.ACDipoleThickVertical(
        volt=0.042 * pbeam * abs(qyd_qx) / np.sqrt(177.0 * betyac),
        freq=driven_tunes[1],
        lag=0,
        ramp=[0, acd_ramp, total_turns, total_turns + acd_ramp],
    )
    placement = line.get_s_position(acd_marker)
    line.insert(f"mkacv.6l4.b{beam}", at=placement)
    line.insert(f"mkach.6l4.b{beam}", at=placement)
    return line


def run_acd_twiss(
    line: xt.Line, beam: int, dpp: float, driven_tunes: list[float]
) -> xt.TwissTable:
    """
    Run twiss calculation with AC dipole elements.

    Args:
        line: xsuite Line object
        beam: Beam number
        dpp: Delta p/p value
        driven_tunes: List of driven tunes [qx, qy, qz]

    Returns:
        Twiss table with AC dipole
    """
    line_acd = line.copy()
    before_acd_tws = line_acd.twiss(method="4d", delta0=dpp)
    acd_marker = f"mkqa.6l4.b{beam}"
    # if acd_marker not in before_acd_tws.rows:
    #     raise ValueError(f"AC dipole marker '{acd_marker}' not found in Twiss table.")

    bet_at_acdipole = before_acd_tws.rows[acd_marker]
    logger.info(
        f"Running twiss with AC dipole at {acd_marker} with betx={bet_at_acdipole['betx']}, bety={bet_at_acdipole['bety']}"
    )

    line_acd.env.elements[f"mkach.6l4.b{beam}"] = xt.ACDipoleThinHorizontal(
        natural_qx=before_acd_tws["qx"] % 1,
        driven_qx=driven_tunes[0],
        betx_at_acdipole=bet_at_acdipole["betx"],
    )
    line_acd.env.elements[f"mkacv.6l4.b{beam}"] = xt.ACDipoleThinVertical(
        natural_qy=before_acd_tws["qy"] % 1,
        driven_qy=driven_tunes[1],
        bety_at_acdipole=bet_at_acdipole["bety"],
    )

    # Insert the ACDipole elements at the correct position
    placement = line_acd.get_s_position(acd_marker)
    line_acd.insert(f"mkach.6l4.b{beam}", at=placement)
    line_acd.insert(f"mkacv.6l4.b{beam}", at=placement)
    return line_acd.twiss(method="4d", delta0=dpp)


def run_xsuite_tracking(
    line: xt.Line,
    beam: int,
    dpp: float,
    total_turns: int,
    acd_ramp: int,
    driven_tunes: list[float],
    output_dir: Path,
    model_dir: Path,
) -> None:
    """
    Run complete tracking simulation with AC dipole.

    Args:
        line: xsuite Line object
        beam: Beam number
        dpp: Delta p/p value
        total_turns: Total number of turns
        acd_ramp: AC dipole ramp time
        driven_tunes: List of driven tunes [qx, qy, qz]
        noise_level: Noise level to apply.
        n_noise_seeds: Number of noise seeds to generate.
        output_dir: Directory for output files
        model_dir: Model directory for saving twiss data
    """
    logger.info(f"Starting tracking simulation for dpp={dpp:.1e}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of turns: {total_turns}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tws = line.twiss(method="4d", delta0=dpp)
    twiss_file = model_dir / f"twiss_dpp_{dpp:.1e}_xsuite.dat"
    tws.rows["bpm.*[^k]"].to_tfs(
        filename=twiss_file,
        columns=["name", "s", "x", "y", "betx", "bety", "mux", "muy"],
    )
    logger.info(f"Twiss table saved to {twiss_file}")

    # Track with ACD
    logger.info("=== Tracking WITH AC dipole ===")
    acd_twiss = run_acd_twiss(line, beam, dpp, driven_tunes)
    acd_twiss_file = model_dir / f"twiss_acd_dpp_{dpp:.1e}_xsuite.dat"
    acd_twiss.rows["bpm.*[^k]"].to_tfs(
        filename=acd_twiss_file,
        columns=["name", "s", "x", "y", "betx", "bety", "mux", "muy"],
    )
    logger.info(
        f"AC dipole twiss calculated (Qx={acd_twiss['qx']:.5f}, Qy={acd_twiss['qy']:.5f}), saved to {acd_twiss_file}"
    )
    line_acd = insert_ac_dipole(line, tws, beam, acd_ramp, total_turns, driven_tunes)
    logger.info("AC dipole inserted into the line for tracking.")

    line_acd = insert_particle_monitors_at_pattern(
        line_acd, "bpm.*[^k]", num_turns=total_turns
    )
    x, px, y, py = 0, 0, 0, 0
    particles_acd: xt.Particles = line_acd.build_particles(
        x=x, px=px, y=y, py=py, delta=dpp
    )
    line_acd.build_tracker()

    line_acd.track(particles_acd, num_turns=total_turns, with_progress=False)
    if particles_acd.state[0] == 1:
        logger.info("Tracking simulation WITH ACD completed successfully!")
        return line_acd
    raise RuntimeError(
        f"Tracking failed for DPP={dpp:.1e} in beam {beam} WITH ACD. "
        "Please check the input parameters and model configuration."
    )


def run_tracking(
    line: xt.Line,
    particles: xt.Particles,
    nturns: int,
) -> None:
    """
    Run tracking simulation for given particles.

    Args:
        particle: xsuite Particles object
        nturns: Number of turns to track
    """
    logger.debug(f"Starting tracking for {nturns} turns")
    line.track(particles, num_turns=nturns, with_progress=True)
    if particles.state[0] == 1:
        logger.debug("Tracking completed successfully!")
        return
    raise RuntimeError("Tracking failed. Please check the input parameters.")


def _set_corrector_strengths(
    env: xt.Environment, corrector_table: tfs.TfsDataFrame
) -> None:
    logger.debug(f"Applying corrector strengths to {len(corrector_table)} elements")
    for _, row in corrector_table.iterrows():
        env.set(row["ename"].lower(), knl=[-row["hkick"]], ksl=[row["vkick"]])


def initialise_env(
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
) -> xt.Environment:
    """
    Initialize a batch of MAD processes for parallel tracking.

    Args:
        matched_tunes: Dictionary of matched tune knobs
        magnet_strengths: Dictionary of magnet strengths
        corrector_table: DataFrame with corrector strengths

    Returns:
        Configured xsuite Environment object
    """
    # logger.info(f"Initializing {batch_size} MAD interfaces for batch")
    base_env = create_xsuite_environment(rerun_madx=False)

    for k, v in matched_tunes.items():
        base_env.set(k, v)

    for str_name, strength in magnet_strengths.items():
        magnet_name, var = str_name.rsplit(".", 1)
        logger.debug(f"Setting {magnet_name.lower()} {var} to {strength}")
        base_env.set(magnet_name.lower(), **{var: strength})

    _set_corrector_strengths(base_env, corrector_table)
    return base_env


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
) -> None:
    """
    Start tracking commands for a batch of MAD interfaces.

    Args:
        interfaces: List of TrackingMadInterface instances
        batch_start: Starting track index
        action_list: List of action values
        angle_list: List of angle values
        twiss_data: Twiss parameters
        kick_both_planes: Whether to kick both planes
        flattop_turns: Number of turns to track
        progress_interval: Interval for progress logging
        num_tracks: Total number of tracks
    """
    # logger.info("Starting tracking commands for batch")
    x_list = []
    px_list = []
    y_list = []
    py_list = []
    deltas = []

    for batch_idx in range(batch_end - batch_start):
        ntrk = batch_start + batch_idx

        if ntrk % progress_interval == 0:
            logger.info(
                f"Starting tracking command for process {ntrk}/{num_tracks - 1} "
                f"({ntrk / num_tracks * 100:.1f}%)"
            )

        # breakpoint()
        # Create initial conditions
        x0_data = create_initial_conditions(
            ntrk,
            action_list,
            angle_list,
            twiss_data,
            kick_both_planes,
            starting_bpm=env[SEQ_NAME].element_names[0].upper(),
        )
        x_list.append(x0_data["x"])
        px_list.append(x0_data["px"])
        y_list.append(x0_data["y"])
        py_list.append(x0_data["py"])
        deltas.append(true_deltap)

    ctx = Context(32)

    particles = env[SEQ_NAME].build_particles(
        _context=ctx,
        x=x_list,
        px=px_list,
        y=y_list,
        py=py_list,
        delta=deltas,
    )

    insert_particle_monitors_at_pattern(
        env[SEQ_NAME],
        pattern="bpm.*[^k]",
        num_turns=flattop_turns,
        num_particles=len(deltas),
        inplace=True,
    )

    # Run tracking using interface
    run_tracking(
        line=env[SEQ_NAME],
        particles=particles,
        nturns=flattop_turns,
    )
    return env[SEQ_NAME]


def line_to_dataframes(tracked_line: xt.Line) -> list[pd.DataFrame]:
    """
    Convert xsuite tracked line to list of DataFrames.

    Args:
        tracked_line: xsuite Line object after tracking

    Returns:
        List of DataFrames, one per particle
    """
    # Collect monitor names and monitor objects in order from the line
    monitor_pairs: list[tuple[str, xt.ParticlesMonitor]] = [
        (name, elem)
        for name, elem in zip(tracked_line.element_names, tracked_line.elements)
        if isinstance(elem, xt.ParticlesMonitor)
    ]
    # Check that we have at least one monitor
    if not monitor_pairs:
        raise ValueError(
            "No ParticlesMonitor found in the Line. Please add a ParticlesMonitor to the Line."
        )
    monitor_names, monitors = zip(*monitor_pairs)

    # First check that no particles were lost during tracking. There will be trailing
    # zeros in the data if particles were lost. This might be difficult to detect.
    assert all(
        mon.data.particle_id[-1] == mon.data.particle_id.max() for mon in monitors
    ), (
        "Some particles were lost during tracking, which is not supported by this function. "
        "Ensure that all particles are tracked through the entire line without loss."
    )

    # Check that all monitors have the same number of particles
    npart_set = {len(set(mon.data.particle_id)) for mon in monitors}
    if len(npart_set) != 1:
        raise ValueError(
            "Monitors have different number of particles, maybe some lost particles?"
        )
    npart = npart_set.pop()

    num_turns = len(monitors[0].data.x) // npart
    particle_masks = [
        mon.data.particle_id[:, None] == np.arange(npart)[None, :] for mon in monitors
    ]

    tracking_dataframes = []
    # Loop over each particle ID (pid)
    for pid in range(npart):
        # Build a long-format DataFrame with monitor, turn, x, y, px, py
        monitor_names_rep = np.tile(monitor_names, num_turns)
        turns_rep = np.repeat(np.arange(num_turns), len(monitor_names))

        # This will produce arrays of shape (num_monitors, num_turns)
        x_data = np.vstack(
            [mon.data.x[particle_masks[i][:, pid]] for i, mon in enumerate(monitors)]
        )
        y_data = np.vstack(
            [mon.data.y[particle_masks[i][:, pid]] for i, mon in enumerate(monitors)]
        )
        px_data = np.vstack(
            [mon.data.px[particle_masks[i][:, pid]] for i, mon in enumerate(monitors)]
        )
        py_data = np.vstack(
            [mon.data.py[particle_masks[i][:, pid]] for i, mon in enumerate(monitors)]
        )
        tracking_data = pd.DataFrame(
            {
                "name": monitor_names_rep,
                "turn": turns_rep + 1,
                "x": x_data.T.flatten(),
                "px": px_data.T.flatten(),
                "y": y_data.T.flatten(),
                "py": py_data.T.flatten(),
            }
        )
        # breakpoint()
        tracking_dataframes.append(tracking_data)
    return tracking_dataframes


def random_zero_mean_unit_std_np(n, rng, desired_std=1e-3, max_abs=None):
    """
    Generate n random numbers with zero mean and desired standard deviation.
    Optionally clip the values to [-max_abs, max_abs] if max_abs is not None.
    """
    if max_abs is None:
        x = rng.standard_normal(n)  # noqa: NPY002
        x -= x.mean()  # mean → 0
        x *= desired_std / x.std()  # std → desired_std
        return x.tolist()

    # Ensure all returned values are within [-max_abs, max_abs] after mean subtraction and scaling
    max_attempts = 100
    for _ in range(max_attempts):
        x = rng.standard_normal(n)
        x -= x.mean()
        x *= desired_std / x.std()
        if np.all(np.abs(x) <= max_abs):
            return x.tolist()
    raise RuntimeError(
        f"Could not generate {n} values with zero mean, std={desired_std}, and all |x| <= {max_abs} after {max_attempts} attempts."
    )


def add_phase_errors(line: xt.Line, dpp: float, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    dmux = random_zero_mean_unit_std_np(8, rng, desired_std=3e-2, max_abs=2e-1)
    dmuy = random_zero_mean_unit_std_np(8, rng, desired_std=3e-2, max_abs=2e-1)

    assert sum(dmux) < 1e-16, "dmux should sum to zero"
    assert sum(dmuy) < 1e-16, "dmuy should sum to zero"

    pre_shift_tws = line.twiss(method="4d", delta0=dpp)
    muxs = pre_shift_tws.rows["ip[0-9]*"]["mux"]
    muys = pre_shift_tws.rows["ip[0-9]*"]["muy"]
    muxs = np.roll(muxs, -1)
    muys = np.roll(muys, -1)
    muxs[-1] += pre_shift_tws.qx
    muys[-1] += pre_shift_tws.qy
    starts = [2, 3, 4, 5, 6, 7, 8, 1]
    ends = [3, 4, 5, 6, 7, 8, 1, 2]
    for i, arc_start, arc_end in zip(range(8), starts, ends):
        start_elm = f"ip{arc_start}"
        end_elm = f"ip{arc_end}"

        a_str = f"a{arc_start}{(arc_end)}"
        defoc = f"kqd.{a_str}"
        foc = f"kqf.{a_str}"

        start_defoc = line.env[defoc]
        start_foc = line.env[foc]

        logging.warning(
            f"Matching {defoc} and {foc} for beam {BEAM_NUMBER} at {start_elm} to {end_elm}"
        )
        logging.warning(f"Starting values, {defoc}: {start_defoc}, {foc}: {start_foc}")

        line.match(
            start=start_elm,
            end=end_elm,
            vary=[
                xt.Vary(defoc),
                xt.Vary(foc),
            ],
            targets=[
                xt.TargetSet(mux=muxs[i] + dmux[i], at=end_elm, tol=1e-8),
                xt.TargetSet(muy=muys[i] + dmuy[i], at=end_elm, tol=1e-8),
            ],
            init_at=start_elm,
            init=pre_shift_tws,
            method="4d",
            solve=True,
            n_steps_max=200,
        )
        logging.warning(
            f"After matching, {defoc}: {line.env[defoc]}, {foc}: {line.env[foc]}"
        )
        logging.warning(
            f"The difference for {defoc}: {line.env[defoc] - start_defoc}, {foc}: {line.env[foc] - start_foc}"
        )
        logging.warning(
            f"The relative difference for {defoc}: {(line.env[defoc] - start_defoc) / start_defoc}, {foc}: {(line.env[foc] - start_foc) / start_foc}"
        )
    line.match(
        vary=[
            xt.Vary(f"dqx.b{BEAM_NUMBER}_op"),
            xt.Vary(f"dqy.b{BEAM_NUMBER}_op"),
        ],
        targets=[
            xt.Target("qx", 62.28, tol=1e-8),
            xt.Target("qy", 60.31, tol=1e-8),
        ],
        method="4d",
        delta0=dpp,
        n_steps_max=300,
    )
