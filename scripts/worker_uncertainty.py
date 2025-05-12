from multiprocessing import Process
from multiprocessing.connection import Connection

import numpy as np
import tfs
from pymadng import MAD

from aba_optimiser.config import (
    BPM_RANGE,
    SEQ_NAME,
    SEQUENCE_FILE,
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
        indices: list[int],
        conn: Connection,
        comparison_data: tfs.TfsDataFrame,
        start_data: tfs.TfsDataFrame,
        beta_x: np.ndarray,
        beta_y: np.ndarray,
    ) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.indices = indices
        self.conn = conn
        self.comparison_data = comparison_data
        self.start_data = start_data
        self.beta_x = beta_x
        self.beta_y = beta_y

    @staticmethod
    def uncertainty_code() -> str:
        """
        Returns the Lua code for calculating aggregated uncertainty per knob
        across all tracks. For each knob, uncertainty (sigma) is computed from
        the Hessian matrix for each track and then aggregated by taking the square root
        of the sum of the squares.
        """
        code = f"""
local monomial, vector, matrix in MAD

local num_knobs = #knob_names
local ncoord = #coord_names
local Htot = matrix(num_knobs, num_knobs):zeros()
local num_particles = #da_x0_c

local sigma = 1e-4
local W_vec = vector(nbpms)
for b = 1, nbpms do W_vec[b] = 1/(sigma^2) end
local W = W_vec:diag()

local Htot = matrix(num_knobs, num_knobs):zeros()
local trk = track{{sequence=MADX.{SEQ_NAME}, X0=da_x0_c,
                    nturn=1, savemap=true, range=range}}
for idx = 1, num_particles do
    -- build jacobian matrices jx, jy -----------------------------
    local jx = matrix(num_knobs, nbpms):zeros()
    local jy = matrix(num_knobs, nbpms):zeros()
    for b = 1, nbpms do
        local map = trk.__map[idx + (b - 1) * num_particles]
        for k, param in ipairs(knob_names) do
            local mono = knob_monomials[param]
            jx:set(k,b, map.x:get(mono))
            jy:set(k,b, map.y:get(mono))
        end
    end

    local Gx = jx * (W * jx:t())
    local Gy = jy * (W * jy:t())
    Htot = Htot + Gx + Gy
end

py:send(Htot)
"""
        return code

    def create_base_damap(self, mad: MAD, knob_order: int = 1) -> None:
        """
        Create a base damap object in MAD-NG with the given knob order.
        """
        mad.load("MAD.utility", "tblcat")
        mad.send(
            f"da_x0_base = damap{{nv=#coord_names, np=#knob_names, mo=2, po={knob_order}, vn=tblcat(coord_names, knob_names)}}"
        )

    def send_initial_conditions(self, mad: MAD, selected_init: list[list[float]]) -> None:
        """
        Sets the initial conditions for each track in MAD-NG.
        """
        mad.send("da_x0_c = table.new(num_particles, 0)")
        mad.send("""
init_coords = py:recv()
for i=1,num_particles do 
    da_x0_c[i] = da_x0_base:copy() 
    da_x0_c[i]:set0(init_coords[i])
end
        """)
        mad.send(selected_init)

    def run(self) -> None:
        # Load initial conditions
        init_coords = []
        for turn_idx in self.indices:
            row = self.start_data.iloc[turn_idx]
            init_coords.append(
                [row["x"], row["px"], row["y"], row["py"], 0, 0]
            )
        del self.start_data
        num_particles = len(init_coords)

        # Initialise MAD interface and load sequence
        mad_iface = MadInterface(SEQUENCE_FILE, BPM_RANGE, stdout="/dev/null", redirect_sterr=True)
        mad = mad_iface.mad

        # Import required MAD-NG modules
        mad.load("MAD", "damap", "matrix")

        # Pre-allocate TPSA and derivative matrices
        self.create_base_damap(mad)
        mad.send(f"""
knob_monomials = {{}}
for i,param in ipairs(knob_names) do
    MADX.{SEQ_NAME}[param] = MADX.{SEQ_NAME}[param] + da_x0_base[param]
    knob_monomials[param] = string.rep("0", 6 + i - 1) .. "1"
end
""")

        mad.send(f"num_particles = {num_particles}")
        self.send_initial_conditions(mad, init_coords)

        # Update the knobs to order 2.
        mad_iface.mad.send(self.uncertainty_code())
        H_part = mad_iface.mad.recv()
        self.conn.send(H_part)  # shape (n_knobs, n_knobs)

        del mad_iface
