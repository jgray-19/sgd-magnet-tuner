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
    def build_batch_tracking_code(num_particles: int) -> str:
        """
        Build Lua code for full-arc tracking on the given initial-condition indices.
        Instead of aggregating gradients and loss inside MAD-NG, we send full derivative
        and difference matrices.
        """
        return f"""
num_particles = {num_particles:d}
sim_trk = track{{sequence = MADX.{SEQ_NAME}, X0 = da_x0_c, nturn = 1, savemap=true, range=range}}
assert(sim_trk.lost == 0, "Tracking failed for initial conditions")
for i = 1, num_particles do
    for j = 1, nbpms do
        local map = sim_trk.__map[i + (j - 1) * num_particles]
!        assert(sim_trk.id[i + (j - 1) * num_particles] == i, "Mismatch between track ID and index")
        for k, param in ipairs(knob_names) do
            dx_dk[i]:set(k, j, map.x:get(knob_monomials[param]))
            dy_dk[i]:set(k, j, map.y:get(knob_monomials[param]))
        end
    end
end
py:send(dx_dk)
py:send(dy_dk)
py:send(sim_trk.x - meas_trk.x)
py:send(sim_trk.y - meas_trk.y)
"""

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
        track_data = tfs.read(TRACK_DATA_FILE, index="turn")
        # Remove all rows that are not the BPM s.ds.r3.b1
        start_bpm, end_bpm = BPM_RANGE.split("/")
        track_data = select_marker(track_data, start_bpm)

        init_coords = []
        for turn_idx in self.indices:
            row = track_data.iloc[turn_idx]
            init_coords.append(
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
        num_particles = len(init_coords)

        # Load measurement tracks
        all_data = tfs.read(TRACK_DATA_FILE)
        if "BPM" not in start_bpm:
            all_data = filter_out_marker(all_data, start_bpm)
        if "BPM" not in end_bpm:
            all_data = filter_out_marker(all_data, end_bpm)

        # Initialise MAD interface and load sequence
        mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE)
        mad = mad_iface.mad
        

        # Run a twiss and to get the beta functions
        mad.send(f"""
tws = twiss{{sequence=MADX.{SEQ_NAME}, observe=1}}
        """)
        tws: tfs.TfsDataFrame = mad.tws.to_df().set_index("name")
        
        # Remove all rows that are before the start BPM and after the end BPM
        start_bpm_idx = tws.index.get_loc(start_bpm)
        end_bpm_idx = tws.index.get_loc(end_bpm)
        tws = tws.iloc[start_bpm_idx:end_bpm_idx + 1]

        beta_x = np.sqrt(tws["beta11"].to_numpy())
        beta_y = np.sqrt(tws["beta22"].to_numpy())

        assert len(beta_x) == len(beta_y), "Beta functions are not the same length"
        assert len(beta_x) == mad_iface.nbpms, "Beta functions do not match the number of BPMs"
        
        mad_iface.make_knobs(self.elem_names)

        x_array = np.empty((num_particles, mad_iface.nbpms))
        y_array = np.empty((num_particles, mad_iface.nbpms))
        for i, turn_idx in enumerate(self.indices):
            filtered = all_data[all_data["turn"] == turn_idx + 1] # A pandas dataframe with nbpms rows
            x_array[i, :] = filtered["x"].to_numpy()
            y_array[i, :] = filtered["y"].to_numpy()
        
        meas_trk = { # Reshape to have a 2D array of shape (num_particles*nbpms, 1)
            "x": x_array.flatten(order="F").reshape(-1, 1), 
            "y": y_array.flatten(order="F").reshape(-1, 1),
        }

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
dx_dk    = table.new({num_particles}, 0)
dy_dk    = table.new({num_particles}, 0)
for i=1,{num_particles} do
    dx_dk[i] = matrix(#knob_names, nbpms)
    dy_dk[i] = matrix(#knob_names, nbpms)
end
""")

        self.send_initial_conditions(mad, init_coords)

        mad.send("meas_trk = {x = py:recv(), y = py:recv()}")
        mad.send(meas_trk['x'])
        mad.send(meas_trk['y'])

        # Clean up python objects that are no longer needed in the worker loop
        del track_data
        del all_data
        del meas_trk

        # Main loop: receive messages, run tracking, and send full per-BPM data
        track_code = self.build_batch_tracking_code(num_particles)
        knob_updates, slc = self.conn.recv()  # assume dict[name->value]
        while knob_updates is not None:
            for name, val in knob_updates.items():
                mad.send(f"MADX.{SEQ_NAME}['{name}']:set0({val})")
            
            # 1) Run full-arc tracking code and get all per-BPM data.
            mad.send(track_code)

            # This is a list of matrices, one for each particle, each with shape (n_knobs, nbpms)
            # Stack per-particle matrices along a new axis so that dx_stack has shape (num_particles, n_knobs, nbpms)
            dx_stack = np.stack(mad.recv(), axis=0)
            dy_stack = np.stack(mad.recv(), axis=0)

            # Reshape differences into (num_particles, nbpms)
            x_diffs = mad.recv().reshape(num_particles, -1, order="F") / beta_x
            y_diffs = mad.recv().reshape(num_particles, -1, order="F") / beta_y
            
            # Select the slice of interest
            dx_slc = dx_stack[:, :, slc]      # shape: (num_particles, n_knobs, len(slc))
            dy_slc = dy_stack[:, :, slc]

            x_slc = x_diffs[:, slc]           # shape: (num_particles, len(slc))
            y_slc = y_diffs[:, slc]

            # Compute the dot product for each particle and sum over particles.
            # This avoids using an explicit transpose by aligning the axes in einsum.
            grad = np.einsum('ink,ik->n', dx_slc, x_slc) + np.einsum('ink,ik->n', dy_slc, y_slc)
            loss = np.sum(x_slc ** 2) + np.sum(y_slc ** 2)
                
            # 5) Send back the gradient and loss
            self.conn.send((self.worker_id, grad, loss))

            # 6) Receive the knob updates and new slice
            knob_updates, slc = self.conn.recv()

        # Update the knobs to order 2.
        self.create_base_damap(mad, knob_order=2)
        self.send_initial_conditions(mad, init_coords)
        mad.send(f"""
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param]:get0() + da_x0_base[param]
end
""")

        mad_iface.mad.send(self.uncertainty_code())
        H_part = mad_iface.mad.recv()
        self.conn.send(H_part)  # shape (n_knobs, n_knobs)
