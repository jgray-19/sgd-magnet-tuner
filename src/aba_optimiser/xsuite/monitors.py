from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import tfs
import xtrack as xt

logger = logging.getLogger(__name__)


def insert_particle_monitors_at_pattern(
    line: xt.Line,
    pattern: str,
    num_turns: int = 10_000,
    num_particles: int = 1,
    inplace: bool = False,
) -> xt.Line:
    """Insert particle monitors at elements whose names match a regex pattern."""
    monitored_line = line if inplace else line.copy()

    selected_list = [name for name in monitored_line.element_names if re.match(pattern, name)]
    if not selected_list:
        logger.warning(f"No elements found matching pattern '{pattern}'.")
        return monitored_line

    s_positions = monitored_line.get_s_position(selected_list)

    inserts = []
    for name, s in zip(selected_list, s_positions):
        name_upper = name.upper()
        if name_upper in monitored_line.element_names:
            logger.warning(
                f"Element '{name_upper}' already exists and is being replaced with a monitor."
            )
        monitored_line.env._element_dict[name_upper] = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=num_turns, num_particles=num_particles
        )

        if name_upper not in monitored_line.element_names:
            inserts.append(monitored_line.env.place(name_upper, at=s))

    monitored_line.insert(inserts)  # ty:ignore[possibly-missing-attribute]
    return monitored_line


def line_to_dataframes(tracked_line: xt.Line) -> list[pd.DataFrame]:
    """Convert tracked Line monitor data to one DataFrame per particle."""
    monitor_pairs: list[tuple[str, xt.ParticlesMonitor]] = [
        (name, elem)
        for name, elem in zip(tracked_line.element_names, tracked_line.elements)
        if isinstance(elem, xt.ParticlesMonitor)
    ]
    if not monitor_pairs:
        raise ValueError(
            "No ParticlesMonitor found in the Line. Please add a ParticlesMonitor to the Line."
        )
    monitor_names, monitors = zip(*monitor_pairs)

    assert all(mon.data.particle_id[-1] == mon.data.particle_id.max() for mon in monitors), (
        "Some particles were lost during tracking, which is not supported by this function. "
        "Ensure that all particles are tracked through the entire line without loss."
    )

    npart_set = {len(set(mon.data.particle_id)) for mon in monitors}
    if len(npart_set) != 1:
        raise ValueError("Monitors have different number of particles, maybe some lost particles?")
    npart = npart_set.pop()

    num_turns = len(monitors[0].data.x) // npart
    particle_masks = [
        mon.data.particle_id[:, None] == np.arange(npart)[None, :] for mon in monitors
    ]

    tracking_dataframes = []
    for pid in range(npart):
        monitor_names_rep = np.tile(monitor_names, num_turns)
        turns_rep = np.repeat(np.arange(num_turns), len(monitor_names))

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
        tracking_dataframes.append(tracking_data)
    return tracking_dataframes


def process_tracking_data(
    monitored_line: xt.Line,
    ramp_turns: int = 1000,
    flattop_turns: int = 100,
    delta_p: float = 0.0,
    add_variance_columns: bool = True,
) -> pd.DataFrame:
    """Process raw tracking data from monitors, trimming ramp and adding variance cols."""
    tracking_df = line_to_dataframes(monitored_line)[0]
    tracking_df = tracking_df[tracking_df["turn"] >= ramp_turns].copy()
    tracking_df["turn"] = tracking_df["turn"] - ramp_turns
    tracking_df = tracking_df.reset_index(drop=True)

    if add_variance_columns:
        tracking_df["var_x"] = 1.0
        tracking_df["var_y"] = 1.0
        tracking_df["var_px"] = 1.0
        tracking_df["var_py"] = 1.0
        tracking_df["kick_plane"] = "both"

    return tracking_df


def xsuite_tws_to_ng(tws) -> pd.DataFrame:
    """Convert xsuite twiss object to ng-compatible format."""
    tws_df = tws.to_pandas()
    tws_df = tfs.TfsDataFrame(tws_df)
    tws_df.columns = [c.lower() for c in tws_df.columns]
    tws_df["name"] = tws_df["name"].str.upper()  # ty:ignore[unresolved-attribute]
    tws_df = tws_df.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws_df.headers = {"q1": float(tws.qx % 1), "q2": float(tws.qy % 1)}
    return tws_df.set_index("name")
