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
        # Set to 0, as with a range on the tracking, we start at turn 0
        mad["tracking_range"] = self.get_bpm_range(self.config.sdir)

    def get_bpm_range(self, sdir: int) -> str:
        """Get the magnet range for arc-by-arc mode."""
        if sdir == -1:
            return self.config.end_bpm + "/" + self.config.start_bpm
        return self.config.start_bpm + "/" + self.config.end_bpm

    @staticmethod
    def get_n_data_points(nbpms: int) -> int:
        """Get the number of data points for arc-by-arc mode."""
        return nbpms
