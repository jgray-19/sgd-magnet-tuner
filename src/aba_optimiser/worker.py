from multiprocessing import Process
from multiprocessing.connection import Connection

import numpy as np
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BPM_RANGE,
    SEQ_NAME,
    SEQUENCE_FILE,
    TRACK_DATA_FILE,
)
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.utils import filter_out_marker, select_marker


class Worker(Process):
    """
    A worker process that runs a batch of tracking simulations in parallel and
    communicates gradients and loss back to the master process.
    """

    def __init__(
        self,
        worker_id: int,
        indices: list[int],
        conn: Connection,
        elem_names: list[str],
    ):
        super().__init__()
        self.worker_id = worker_id
        self.indices = indices
        self.conn = conn
        self.elem_names = elem_names

    @staticmethod
    def build_batch_tracking_code(num_tracks: int) -> str:
        """
        Build Lua code for full-arc tracking on the given initial-condition indices.
        Instead of aggregating gradients and loss inside MAD-NG, we send full derivative
        and difference matrices.
        """
        lines = []
        for i in range(1, num_tracks + 1):
            lines.append(f"""-- Track for initial condition {i}
sim_trk[{i}] = track{{sequence = MADX.{SEQ_NAME}, X0 = da_x0_c[{i}], nturn = 1, savemap=true, range=range}}
assert(sim_trk[{i}].lost == 0, "Tracking failed for initial condition {i}")
for j = 1, nbpms do
    local map = sim_trk[{i}].__map[j]
    for k, param in ipairs(knob_names) do
        dx_dk[{i}]:set(k, j, map.x:get(knob_monomials[param]))
        dy_dk[{i}]:set(k, j, map.y:get(knob_monomials[param]))
    end
end
x_diffs[{i}] = sim_trk[{i}].x - meas_trk[{i}].x
y_diffs[{i}] = sim_trk[{i}].y - meas_trk[{i}].y
""")
        # After processing all tracks, send full arrays
        lines.append("py:send(dx_dk)")  # send full (n_quads, n_bpms) array
        lines.append("py:send(dy_dk)")  # send full (n_quads, n_bpms) array
        lines.append("py:send(x_diffs)")  # send full (n_bpms,) array
        lines.append("py:send(y_diffs)")  # send full (n_bpms,) array
        return "\n".join(lines)

    @staticmethod
    def uncertainty_code() -> str:
        """
        Returns the Lua code for calculating aggregated uncertainty per knob
        across all tracks. For each knob, uncertainty (sigma) is computed from
        the Hessian matrix for each track and then aggregated by taking the square root
        of the sum of the squares.
        """
        code = f"""
local monomial, vector in MAD

local num_knobs = #knob_names
local ncoord = #coord_names
local Htot = matrix(num_knobs, num_knobs):zeros()

local sigma = 1e-4
local W_vec = vector(nbpms)
for b = 1, nbpms do W_vec[b] = 1/(sigma^2) end
local W = W_vec:diag()

local Htot = matrix(num_knobs, num_knobs):zeros()
for idx = 1, #da_x0_c do
    local trk = track{{sequence=MADX.{SEQ_NAME}, X0=da_x0_c[idx],
                       nturn=1, savemap=true, range=range}}

    -- build first-derivative matrices dx, dy -----------------------------
    local dx = matrix(num_knobs, nbpms):zeros()
    local dy = matrix(num_knobs, nbpms):zeros()

    for i, param in ipairs(knob_names) do
        local mono = knob_monomials[param]
        for b = 1, nbpms do
            local map = trk.__map[b]
            dx:set(i,b, map.x:get(mono))
            dy:set(i,b, map.y:get(mono))
        end
    end

    local Gx = dx * (W * dx:t())
    local Gy = dy * (W * dy:t())
    Htot = Htot + Gx + Gy
end

py:send(Htot)
"""
        return code

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo=2, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def send_initial_conditions(self, mad: MAD, selected_init: list[dict]) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send(f"da_x0_c = table.new({len(selected_init)}, 0)")
        for idx, x0 in enumerate(selected_init):
            mad.send(f"da_x0_c[{idx + 1}] = da_x0_base:copy()")
            mad.send(
                f"da_x0_c[{idx + 1}]:set0({{{x0['x']}, {x0['px']}, {x0['y']}, {x0['py']}, {x0['t']}, {x0['pt']}}})"
            )

    def run(self) -> None:
        # Load initial conditions
        init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
        # Remove all rows that are not the BPM s.ds.r3.b1
        start_bpm, end_bpm = BPM_RANGE.split("/")
        init_coords = select_marker(init_coords, start_bpm)

        selected_init = []
        for turn_idx in self.indices:
            row = init_coords.iloc[turn_idx]
            selected_init.append(
                {
                    "x": row["x"],
                    "px": row["px"],
                    "y": row["y"],
                    "py": row["py"],
                    # t and pt unavailable; default to zero
                    "t": 0,
                    "pt": 0,
                }
            )
        num_tracks = len(selected_init)

        # Load measurement tracks
        all_data = tfs.read(TRACK_DATA_FILE)
        if "BPM" not in start_bpm:
            all_data = filter_out_marker(all_data, start_bpm)
        if "BPM" not in end_bpm:
            all_data = filter_out_marker(all_data, end_bpm)

        meas_trk: list[dict[str, np.ndarray]] = []
        for i, turn_idx in enumerate(self.indices):
            filtered = all_data[all_data["turn"] == turn_idx + 1]
            meas_trk.append(
                {
                    "x": filtered["x"].to_numpy().reshape(-1, 1),
                    "y": filtered["y"].to_numpy().reshape(-1, 1),
                }
            )

        # Initialise MAD interface and load sequence
        mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
        mad = mad_iface.mad
        mad_iface.make_knobs(self.elem_names)

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix")
        mad.load("MAD.utility", "tblcat")
        mad.send('coord_names = { "x", "px", "y", "py", "t", "pt" }')

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad)
        mad.send(f"""
knob_monomials = {{}}
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

        mad.send(f"""
dx_dk    = table.new({num_tracks}, 0)
dy_dk    = table.new({num_tracks}, 0)
meas_trk = table.new({num_tracks}, 0)
sim_trk  = table.new({num_tracks}, 0)
x_diffs  = table.new({num_tracks}, 0)
y_diffs  = table.new({num_tracks}, 0)
""")

        self.send_initial_conditions(mad, selected_init)

        # Allocate derivative matrices
        for idx in range(1, num_tracks + 1):
            mad.send(f"dx_dk[{idx}] = matrix(#knob_names, nbpms)")
            mad.send(f"dy_dk[{idx}] = matrix(#knob_names, nbpms)")

        # Send measurement data into MAD-NG
        for idx in range(num_tracks):
            mad.send(f"meas_trk[{idx + 1}] = {{x = py:recv(), y = py:recv()}}")
            mad.send(meas_trk[idx]["x"])
            mad.send(meas_trk[idx]["y"])

        # Clean up python objects that are no longer needed in the worker loop
        del init_coords
        del all_data
        del meas_trk

        # Main loop: receive messages, run tracking, and send full per-BPM data
        track_code = self.build_batch_tracking_code(num_tracks)
        while True:
            knob_updates, slc = self.conn.recv()  # assume dict[name->value]
            if knob_updates is None:
                break
            for name, val in knob_updates.items():
                mad.send(f"MADX.{SEQ_NAME}['{name}']:set0({val})")
            # Run full-arc tracking code and get all per-BPM data.
            mad.send(track_code)
            dx_dk = np.array(mad.recv())
            dy_dk = np.array(mad.recv())
            x_diffs = np.array(mad.recv())
            y_diffs = np.array(mad.recv())

            # 2) Slice out only the BPMs in current window
            sub_dx = dx_dk[:, :, slc]
            sub_dy = dy_dk[:, :, slc]
            x_col = x_diffs[:, slc, :]
            y_col = y_diffs[:, slc, :]

            # 4) Batched mat-vec:
            grad = (sub_dx @ x_col + sub_dy @ y_col)[..., 0]
            grad = np.sum(grad, axis=0)
            loss = np.sum(x_col**2 + y_col**2)

            self.conn.send((self.worker_id, grad, loss))

        # Update the knobs to order 2.
        self.create_base_damap(mad, knob_order=2)
        self.send_initial_conditions(mad, selected_init)
        mad.send(f"""
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param]:get0() + da_x0_base[param]
end
""")

        mad_iface.mad.send(self.uncertainty_code())
        H_part = mad_iface.mad.recv()
        self.conn.send(H_part)  # shape (n_knobs, n_knobs)
