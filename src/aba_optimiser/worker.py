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
    Y_BPM_START,
)
from aba_optimiser.mad_interface import MadInterface

class Worker(Process):
    """
    A worker process that runs a batch of tracking simulations in parallel and
    communicates gradients and loss back to the master process.
    """

    def __init__(
        self,
        conn: Connection,
        worker_id: int,
        x_indices: list[int],
        y_indices: list[int],
        comparison_data: tfs.TfsDataFrame,
        start_data_dict: dict[str, tfs.TfsDataFrame],
        window_slices: dict[str, slice],
    ) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn

        self.x_indices = x_indices
        self.y_indices = y_indices
        self.num_particles = len(x_indices)  # Assuming x_indices and y_indices are of the same length
        assert len(y_indices) == self.num_particles, "x_indices and y_indices must have the same length."
        
        comparison_data = comparison_data.reset_index().set_index('turn')
        self.comparison_data = comparison_data
        self.start_data_dict = start_data_dict
        assert len(self.start_data_dict.keys()) == 2, "Expected two sets of initial conditions for x and y tracking."

        self.window_slices = window_slices

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
    dx_dk[i] = matrix(#knob_names, win_lengths[starting_bpm])
    dy_dk[i] = matrix(#knob_names, win_lengths[starting_bpm])
end
local sim_trk = track{{sequence = MADX.{SEQ_NAME}, X0 = da_x0_c[starting_bpm], nturn = 1, savemap=true, range=range}}
assert(sim_trk.lost == 0, "Tracking failed for initial conditions")
for i = 1, num_particles do
    for j = 1, win_lengths[starting_bpm] do
        local map = sim_trk.__map[i + (j - 1) * num_particles]
        for k, param in ipairs(knob_names) do
            dx_dk[i]:set(k, j, map.x:get(knob_monomials[param]))
            dy_dk[i]:set(k, j, map.y:get(knob_monomials[param]))
        end
    end
end
py:send(dx_dk)
py:send(dy_dk)

py:send(sim_trk.x)
py:send(sim_trk.y)
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
        assert WINDOWS[0][0] == X_BPM_START, "First window must start with X BPMs"
        assert WINDOWS[1][0] == Y_BPM_START, "Second window must start with Y BPMs"

        init_coords: dict[str, list[float]] = {}
        win_lengths: dict[str, int] = {}
        x_comparisons: dict[str, np.ndarray] = {}
        y_comparisons: dict[str, np.ndarray] = {}
        x_masks: dict[str, np.ndarray] = {}
        y_masks: dict[str, np.ndarray] = {}
        for start_bpm, start_df in self.start_data_dict.items():
            init_coords[start_bpm] = []
            if start_bpm == X_BPM_START:
                # x_masks[i][:, 1::2] = False  # x BPMs are even indexed
                # y_masks[i][:, ::2] = False  # y BPMs are odd indexed
                indices = self.x_indices
            else:
                # x_masks[i][:, ::2] = False  # x BPMs are odd indexed
                # y_masks[i][:, 1::2] = False  # y BPMs are even indexed
                indices = self.y_indices
                   
            win_slice = self.window_slices[start_bpm]
            win_length = win_slice.stop - win_slice.start
            x_comparisons[start_bpm] = np.empty((self.num_particles, win_length))
            y_comparisons[start_bpm] = np.empty((self.num_particles, win_length))

            x_masks[start_bpm] = np.ones((self.num_particles, win_length), dtype=bool)
            y_masks[start_bpm] = np.ones((self.num_particles, win_length), dtype=bool)

            win_lengths[start_bpm] = win_length

            turn_start_df = start_df.reset_index().set_index('turn')
            for j, turn in enumerate(indices):
                row = turn_start_df.loc[turn]
                init_coords[start_bpm].append(
                    [row["x"], row["px"], row["y"], row["py"], 0, 0]
                )
                filtered = self.comparison_data.loc[turn].iloc[win_slice]
                if filtered.shape[0] == 0:
                    raise ValueError(f"No data available for turn {turn} in window {start_bpm}")
                x_comparisons[start_bpm][j, :] = filtered["x"].to_numpy()
                y_comparisons[start_bpm][j, :] = filtered["y"].to_numpy()

        del self.start_data_dict
        del self.comparison_data

        # Initialise MAD interface and load sequence
        mad_iface = MadInterface(
            SEQUENCE_FILE, BPM_RANGE, #discard_mad_output=False
        )
        mad = mad_iface.mad
        mad_iface.mad["num_particles"] = self.num_particles

        mad.send("win_lengths = {}")
        for start_bpm, win_length in win_lengths.items():
            mad.send(f"win_lengths['{start_bpm}'] = py:recv()")
            mad.send(win_length)

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
        
        knob_updates = self.conn.recv()  # shape (n_knobs,)
        while knob_updates is not None:
            grad = np.zeros(len(knob_updates))
            loss = 0.0
            for start_bpm, end_bpm in WINDOWS:

                update_string = [f"MADX.{SEQ_NAME}['{name}']:set0({val:.15e})" for name, val in knob_updates.items()]
                update_string.extend([
                    f'range = "{start_bpm}/{end_bpm}"',
                    f"starting_bpm = '{start_bpm}'"
                ])

                # Run full-arc tracking code and get all per-BPM data.
                mad.send("\n".join(update_string))
                mad.send(track_code)

                # This is a list of matrices, one for each particle, each with shape (n_knobs, nbpms)
                # Stack per-particle matrices along a new axis so that dx/y_stack has shape (num_particles, n_knobs, nbpms)
                dx_stack = np.stack(mad.recv(), axis=0)
                dy_stack = np.stack(mad.recv(), axis=0)
                
                x_res = mad.recv().reshape(self.num_particles, win_lengths[start_bpm], order='F')
                y_res = mad.recv().reshape(self.num_particles, win_lengths[start_bpm], order='F')
                x_diffs = x_res - x_comparisons[start_bpm]
                y_diffs = y_res - y_comparisons[start_bpm]

                # Build gradient and loss
                grad += (
                    np.einsum('ijk,ik->j', dx_stack, x_diffs) +
                    np.einsum('ijk,ik->j', dy_stack, y_diffs)
                )
                loss += (
                    np.sum(abs(x_diffs)) + 
                    np.sum(abs(y_diffs))
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
        mad.send(f'starting_bpm = "{BPM_RANGE.split("/")[0]}"')
        mad_iface.mad.send(self.estimate_hessian())
        H_part = mad_iface.mad.recv()
        self.conn.send(H_part)  # shape (n_knobs, n_knobs)
