from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.config import (
    N_COMPARE_TURNS,
    N_RUN_TURNS,
    OBSERVE_TURNS_FROM,
)
from aba_optimiser.workers.base import BaseWorker

if TYPE_CHECKING:
    from pymadng import MAD


class RingWorker(BaseWorker):
    """
    Worker that runs full ring tracking simulations.
    """

    def get_n_data_points(self, nbpms: int) -> int:
        """Get the number of data points for ring mode."""
        return nbpms * N_COMPARE_TURNS

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Setup MAD sequence for ring mode."""
        mad["n_run_turns"] = N_RUN_TURNS
        mad["observe_from_turn"] = OBSERVE_TURNS_FROM
        mad["n_compare_turns"] = N_COMPARE_TURNS
        # No need to explicitly set None/nil -> would just clear the variable.
        # mad["tracking_range"] = None
        self.cycle_to_bpm(mad, self.start_bpm)

    def cycle_to_bpm(self, mad: MAD, bpm_name: str) -> None:
        """
        Cycles the MAD-NG sequence to the specified BPM.
        """
        if mad.loaded_sequence[bpm_name].kind == "monitor":
            # MAD-NG must cycle to a marker not a monitor
            marker_name = bpm_name.replace("BPM", "MARKER")
            mad.send(f"""
loaded_sequence:install{{
MAD.element.marker '{marker_name}' {{ at=-1e-10, from="{bpm_name}" }} ! 1e-12 is too small for a drift but ensures we cycle to before the BPM
}}
                     """)
            mad.loaded_sequence.cycle(mad.quote_strings(marker_name))
        else:
            mad.loaded_sequence.cycle(mad.quote_strings(bpm_name))

    @staticmethod
    def get_observation_turns(turn: int) -> list[int]:
        """Get the list of observation turns for ring mode."""
        return [
            t
            for t in range(turn, turn + N_RUN_TURNS)
            if t >= turn + OBSERVE_TURNS_FROM - 1
        ]

    @staticmethod
    def get_bpm_range(start_bpm) -> str:
        """Get the magnet range for ring mode."""
        return "$start/$end"
