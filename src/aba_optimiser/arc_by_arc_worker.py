from __future__ import annotations

from typing import TYPE_CHECKING

from aba_optimiser.base_worker import BaseWorker
from aba_optimiser.config import (
    MAGNET_RANGE,
)

if TYPE_CHECKING:
    from pymadng import MAD


class ArcByArcWorker(BaseWorker):
    """
    Worker that runs arc-by-arc tracking simulations.
    """

    def get_bpm_range(self) -> str:
        """Get the magnet range for arc-by-arc mode."""
        return self.start_bpm + "/" + MAGNET_RANGE.split("/")[1]

    def get_n_data_points(self, nbpms: int) -> int:
        """Get the number of data points for arc-by-arc mode."""
        return nbpms

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Setup MAD sequence for arc-by-arc mode."""
        mad["n_run_turns"] = 1
        mad["n_compare_turns"] = 1
        # Set to 0, as with a range on the tracking, we start at turn 0
        mad["observe_from_turn"] = 0
        mad["tracking_range"] = self.get_bpm_range()

    def get_observation_turns(self, turn: int) -> list[int]:
        """Get the list of observation turns for arc-by-arc mode."""
        return [turn]
