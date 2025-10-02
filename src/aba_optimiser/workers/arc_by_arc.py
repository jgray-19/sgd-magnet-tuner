from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aba_optimiser.workers.base_worker import BaseWorker

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)


class ArcByArcWorker(BaseWorker):
    """
    Worker that runs arc-by-arc tracking simulations.
    """

    def setup_mad_sequence(self, mad: MAD) -> None:
        """Setup MAD sequence for arc-by-arc mode."""
        mad["n_run_turns"] = 1
        mad["n_compare_turns"] = 1
        # Set to 0, as with a range on the tracking, we start at turn 0
        mad["observe_from_turn"] = 0
        mad["tracking_range"] = self.get_bpm_range(self.sdir)

    @staticmethod
    def get_observation_turns(turn: int) -> list[int]:
        """Get the list of observation turns for arc-by-arc mode."""
        return [turn]

    def get_bpm_range(self, sdir: int) -> str:
        """Get the magnet range for arc-by-arc mode."""
        if sdir == -1:
            return self.end_bpm + "/" + self.start_bpm
        return self.start_bpm + "/" + self.end_bpm

    @staticmethod
    def get_n_data_points(nbpms: int) -> int:
        """Get the number of data points for arc-by-arc mode."""
        return nbpms
