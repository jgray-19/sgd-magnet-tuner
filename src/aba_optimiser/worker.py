from multiprocessing import Process
from multiprocessing.connection import Connection

import numpy as np
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BPM_RANGE,
    SEQ_NAME,
    SEQUENCE_FILE,
    WINDOWS,
    X_BPM_START,
    # Y_BPM_START,
)
from aba_optimiser.mad_interface import MadInterface

class Worker(Process):
    """
    A worker process that runs a batch of tracking simulations in parallel and
    communicates gradients and loss back to the master process.
    """

    def __init__(
        self,
        worker_id: int,
        x_indices: list[int],
        y_indices: list[int],
        conn: Connection,
        comparison_data: tfs.TfsDataFrame,
        start_data_dict: dict[str, tfs.TfsDataFrame],
        window_slices: list[slice],  
        beta_x: np.ndarray,
        beta_y: np.ndarray,
    ) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.x_indices = x_indices
        self.y_indices = y_indices
        self.num_particles = len(x_indices)  # Assuming x_indices and y_indices are of the same length
        assert len(y_indices) == self.num_particles, "x_indices and y_indices must have the same length."
        self.conn = conn
        comparison_data = comparison_data.reset_index().set_index('turn')
        self.comparison_data = comparison_data
        self.start_data_dict = start_data_dict
        assert len(self.start_data_dict.keys()) == 2, "Expected two sets of initial conditions for x and y tracking."

        self.window_slices = window_slices
        self.beta_x = beta_x
        self.beta_y = beta_y

    @staticmethod
    def build_batch_tracking_code() -> str:
        """
        Build Lua code for full-arc tracking on the given initial-condition indices.
        Instead of aggregating gradients and loss inside MAD-NG, we send full derivative
        and difference matrices.
        """
        return f"""
dx_dk    = table.new(num_particles, 0)
dy_dk    = table.new(num_particles, 0)
for i=1,num_particles do
    dx_dk[i] = matrix(#knob_names, win_lengths[win_idx])
    dy_dk[i] = matrix(#knob_names, win_lengths[win_idx])
end
sim_trk = track{{sequence = MADX.{SEQ_NAME}, X0 = da_x0_c[starting_bpm], nturn = 1, savemap=true, range=range}}
assert(sim_trk.lost == 0, "Tracking failed for initial conditions")
for i = 1, num_particles do
    for j = 1, win_lengths[win_idx] do
        local map = sim_trk.__map[i + (j - 1) * num_particles]
        for k, param in ipairs(knob_names) do
            dx_dk[i]:set(k, j, map.x:get(knob_monomials[param]))
            dy_dk[i]:set(k, j, map.y:get(knob_monomials[param]))
        end
    end
end
py:send(dx_dk)
py:send(dy_dk)

py:send(sim_trk.x  - meas_trk.x [win_idx])
py:send(sim_trk.y  - meas_trk.y [win_idx])
collectgarbage()
"""

    @staticmethod
    def estimate_hessian(diagonal_only: bool = False) -> str:
        """
        Returns the Lua code for calculating aggregated uncertainty per knob
        across all tracks. For each knob, uncertainty (sigma) is computed from
        the Hessian matrix for each track and then aggregated by taking the square root
        of the sum of the squares.
        """
        code = f"""
local monomial, matrix in MAD

local num_knobs = #knob_names
local ncoord = #coord_names
local Htot = matrix(num_knobs, num_knobs):zeros()

local trk = track{{sequence=MADX.{SEQ_NAME}, X0=da_x0_c[starting_bpm], nturn=1, savemap=true, range=range}}
for idx = 1, num_particles do
    -- build jacobian matrices jx, jy -----------------------------
    local jx = matrix(num_knobs, nbpms):zeros()
    local jy = matrix(num_knobs, nbpms):zeros()
    for b = 1, nbpms do
        local map = trk.__map[idx + (b - 1) * num_particles]
        assert(trk.id[idx + (b - 1) * num_particles] == idx, "Mismatch between track ID and index")
        for k, param in ipairs(knob_names) do
            local mono = knob_monomials[param]
            jx:set(k,b, map.x:get(mono))
            jy:set(k,b, map.y:get(mono))
        end
    end

    local Gx = jx * (W_x * jx:t())
    local Gy = jy * (W_y * jy:t())
    Htot = Htot + (Gx + Gy)
end
if {str(diagonal_only).lower()} then
    py:send(Htot:diag() / num_particles)
else
    py:send(Htot)
end
"""
        return code

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo={knob_order}, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def send_initial_conditions(self, mad: MAD, selected_init: dict[str, list[list[float]]]) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send("da_x0_c = {}")
        for key, value in selected_init.items():
            mad.send(f"""
    da_x0_c['{key}'] = table.new(num_particles, 0)
    init_coords = py:recv()
    for i=1,num_particles do 
        da_x0_c['{key}'][i] = da_x0_base:copy() 
        da_x0_c['{key}'][i]:set0(init_coords[i])
    end
            """)
            mad.send(value)

    def run(self) -> None:
        # Load initial conditions
        init_coords: dict[str, list[float]] = {}
        init_vars: dict[str, list[float]] = {}
        for start_bpm, start_df in self.start_data_dict.items():
            init_coords[start_bpm] = []
            init_vars[start_bpm] = []
            indices = self.x_indices if start_bpm == X_BPM_START else self.y_indices

            turn_start_df = start_df.reset_index().set_index('turn')
            for turn in indices:
                row = turn_start_df.loc[turn]
                init_coords[start_bpm].append(
                    [row["x"], row["px"], row["y"], row["py"], 0, 0]
                )
                init_vars[start_bpm].append(
                    [row["var_x"], row["var_y"]]
                )
        del self.start_data_dict

        # Initialise MAD interface and load sequence
        mad_iface = MadInterface(
            SEQUENCE_FILE, BPM_RANGE
        )
        mad = mad_iface.mad
        BPM_VERY_START = BPM_RANGE.split("/")[0]
        mad.send(f"starting_bpm = '{BPM_VERY_START}'")
            
        win_lengths = []

        for i, win_slice in enumerate(self.window_slices):
            win_length = win_slice.stop - win_slice.start
            win_lengths.append(win_length)

        mad_iface.mad["win_lengths"] = win_lengths
        mad_iface.mad["num_particles"] = self.num_particles

        x_comparisons  = []
        y_comparisons  = []
        x_masks = []
        y_masks = []

        for i, win_slice in enumerate(self.window_slices):
            x_comparisons.append(np.empty((self.num_particles, win_lengths[i])))
            y_comparisons.append(np.empty((self.num_particles, win_lengths[i])))
            x_masks.append(np.ones((self.num_particles, win_lengths[i]), dtype=bool))
            y_masks.append(np.ones((self.num_particles, win_lengths[i]), dtype=bool))

            start = WINDOWS[i][0]
            if start == X_BPM_START:
                x_masks[i][:, 1::2] = False  # x BPMs are even indexed
                y_masks[i][:, ::2] = False  # y BPMs are odd indexed
                indices = self.x_indices
            else:
                x_masks[i][:, ::2] = False  # x BPMs are odd indexed
                y_masks[i][:, 1::2] = False  # y BPMs are even indexed
                indices = self.y_indices

            for j, turn in enumerate(indices):
                filtered = self.comparison_data.loc[turn].iloc[win_slice]

                # Skip if filtered is empty
                if filtered.shape[0] == 0:
                    raise ValueError(f"No data available for turn {turn} in window {i + 1}")

                x_comparisons[i][j, :] = filtered["x"].to_numpy()
                y_comparisons[i][j, :] = filtered["y"].to_numpy()

        del self.comparison_data

        meas_trk = {  # Reshape to have a 2D array of shape (num_particles*nbpms, 1)
            "x":  [x.flatten(order="F").reshape(-1, 1) for x in x_comparisons],
            "y":  [y.flatten(order="F").reshape(-1, 1) for y in y_comparisons],
        }
        del x_comparisons
        del y_comparisons

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix")
        mad.load("MAD.utility", "tblcat")

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad)
        mad.send(f"""
knob_monomials = {{}}
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")
        
        self.send_initial_conditions(mad, init_coords)

        mad.send("meas_trk = {}")
        for key, value in meas_trk.items():
            mad.send(f"meas_trk['{key}'] = py:recv()").send(value)

        # Clean up python objects that are no longer needed in the worker loop
        del meas_trk

        # Main loop: receive messages, run tracking, and send full per-BPM data
        track_code = self.build_batch_tracking_code()
        hessian_var_x, hessian_var_y = self.conn.recv()
        mad_iface.mad.send("""
local vector in MAD
W_x = vector(py:recv()):diag()
W_y = vector(py:recv()):diag()
""")
        mad_iface.mad.send((1/hessian_var_x).tolist())
        mad_iface.mad.send((1/hessian_var_y).tolist())
        
        # mad_iface.mad.send(self.estimate_hessian(diagonal_only=True))
        # H_part_diag = mad_iface.mad.recv()
        # self.conn.send(H_part_diag)  # shape (n_knobs,)

        knob_updates = self.conn.recv()  # assume dict[name->value]
        while knob_updates is not None:
            grad = np.zeros(len(knob_updates))
            loss = 0.0
            for win_idx, (start_bpm, end_bpm) in enumerate(WINDOWS):

                update_string = [f"MADX.{SEQ_NAME}['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]
                update_string.extend([
                    f'range = "{start_bpm}/{end_bpm}"',
                    f"win_idx = {win_idx+1:d}",
                    f"starting_bpm = '{start_bpm}'"
                ])

                # Run full-arc tracking code and get all per-BPM data.
                mad.send("\n".join(update_string))
                mad.send(track_code)

                # This is a list of matrices, one for each particle, each with shape (n_knobs, nbpms)
                # Stack per-particle matrices along a new axis so that dx_stack has shape (num_particles, n_knobs, nbpms)
                dx_stack = np.stack(mad.recv(), axis=0)
                dy_stack = np.stack(mad.recv(), axis=0)

                # Replace the recv/reshape for differences:
                x_diffs  = mad.recv().reshape(self.num_particles, -1, order="F") * x_masks[win_idx]
                y_diffs  = mad.recv().reshape(self.num_particles, -1, order="F") * y_masks[win_idx]

                # Build gradient and loss
                grad += (
                    np.einsum('ijk,ik->j', dx_stack, x_diffs) +
                    np.einsum('ijk,ik->j', dy_stack, y_diffs) #+
                )
                loss += (
                    np.sum(abs(x_diffs)) + 
                    np.sum(abs(y_diffs))# + 
                )
                
            # 5) Send back the gradient and loss
            self.conn.send((self.worker_id, grad, loss))

            # 6) Receive the knob updates and new slice
            knob_updates = self.conn.recv()

        # Update the knobs to order 2.
        self.create_base_damap(mad, knob_order=2)
        self.send_initial_conditions(mad, init_coords)
        mad.send(f"""
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param]:get0() + da_x0_base[param]
end
""")
        mad.send(f'range = "{BPM_RANGE}"') # Reset the range to the full BPM range
        mad.send(f'starting_bpm = "{BPM_VERY_START}"')
        mad_iface.mad.send(self.estimate_hessian())
        H_part = mad_iface.mad.recv()
        self.conn.send(H_part)  # shape (n_knobs, n_knobs)
