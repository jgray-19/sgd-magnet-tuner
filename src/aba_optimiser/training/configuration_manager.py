"""Configuration and setup management for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.config import BEAM_ENERGY
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.workers import TrackingWorker

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


def circular_far_samples(
    arr: np.ndarray, k: int, start_offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick k indices as far apart as possible on a circular array.
    k must be <= len(arr). start_offset rotates the selection.
    """
    n = len(arr)
    if k > n:
        raise ValueError("k must be <= len(arr)")
    step = n / k
    # bin centers → unique indices even when step is non-integer
    idx = (np.floor((np.arange(k) + 0.5) * step + start_offset) % n).astype(int)
    return arr[idx], idx


class ConfigurationManager:
    """Manages configuration and setup for the optimisation process."""

    def __init__(
        self,
        simulation_config: SimulationConfig,
        magnet_range: str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        bpm_range: str | None = None,
        optimise_knobs: list[str] | None = None,
    ):
        self.mad_iface: OptimisationMadInterface = None
        self.knob_names: list[str] = []
        self.elem_spos: list[float] = []
        self.all_bpms: list[str] = []
        self.initial_strengths: np.ndarray = np.array([])
        self.bpm_ranges: list[str] = []

        self.start_bpms = bpm_start_points
        self.end_bpms = bpm_end_points
        self.magnet_range = magnet_range
        self.simulation_config = simulation_config
        self.bpm_range = bpm_range
        self.optimise_knobs = optimise_knobs

    def setup_mad_interface(
        self,
        sequence_file_path: str,
        first_bpm: str | None,
        bad_bpms: list[str] | None,
        seq_name: str | None = None,
        beam_energy: float = BEAM_ENERGY,
        debug: bool = False,
        mad_logfile: Path | None = None,
    ) -> None:
        """Initialise the MAD-NG interface and get basic model parameters."""
        self.mad_iface = OptimisationMadInterface(
            sequence_file_path,
            seq_name=seq_name,
            start_bpm=first_bpm,
            magnet_range=self.magnet_range,
            bpm_range=self.bpm_range,
            simulation_config=self.simulation_config,
            corrector_strengths=None,
            tune_knobs_file=None,
            bad_bpms=bad_bpms,
            beam_energy=beam_energy,
            debug=debug,
            mad_logfile=mad_logfile,
            optimise_knobs=self.optimise_knobs,
        )
        self.knob_names = self.mad_iface.knob_names

        self.elem_spos = self.mad_iface.elem_spos
        self.all_bpms = self.mad_iface.all_bpms
        self.start_bpms = [bpm for bpm in self.start_bpms if bpm in self.all_bpms]
        self.end_bpms = [bpm for bpm in self.end_bpms if bpm in self.all_bpms]
        if self.simulation_config.optimise_bends:
            self.bend_lengths = self.mad_iface.bend_lengths

        # Use bpm_range to determine fixed start and end points, defaulting to magnet_range
        effective_bpm_range = self.bpm_range or self.magnet_range
        self.fixed_start, self.fixed_end = effective_bpm_range.split("/", 1)

        # Validate fixed points are in the model
        if self.fixed_start not in self.all_bpms or self.fixed_end not in self.all_bpms:
            LOGGER.warning(
                f"Fixed BPMs from range {effective_bpm_range} not found in model, using first available"
            )
            self.fixed_start = self.start_bpms[0] if self.start_bpms else self.fixed_start
            self.fixed_end = self.end_bpms[0] if self.end_bpms else self.fixed_end

    @property
    def bpm_pairs(self) -> list[tuple[str, str]]:
        """Return BPM ranges as explicit (start, end) tuples.

        When use_fixed_bpm is True (default), creates pairs by varying starts
        with fixed end and varying ends with fixed start.

        When use_fixed_bpm is False, creates all combinations (Cartesian product)
        of start_bpms with end_bpms (every start with every end).
        """
        if self.simulation_config.use_fixed_bpm:
            return [(s, self.fixed_end) for s in self.start_bpms] + [
                (self.fixed_start, e) for e in self.end_bpms
            ]
        # Cartesian product: every start with every end
        return [(s, e) for s in self.start_bpms for e in self.end_bpms]

    def initialise_knob_strengths(
        self,
        true_strengths: dict[str, float],
        provided_initial_knobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Initialise knob strengths from MAD and filter true strengths."""
        if self.mad_iface is None:
            raise ValueError("MAD interface must be setup first")

        if provided_initial_knobs is not None:
            # Use provided initial knobs where available, fill missing ones from MAD
            LOGGER.info("Using provided initial knob strengths from previous optimisation")
            # First get the default MAD values for all knobs
            self.mad_iface.update_knob_values(provided_initial_knobs)
        initial_strengths = self.mad_iface.receive_knob_values()

        self.initial_strengths = initial_strengths
        current_knobs = dict(zip(self.knob_names, initial_strengths))

        # Restrict true strengths to knobs we actually have in model
        if len(true_strengths) == 0:
            LOGGER.warning("No true strengths provided, skipping filtering")
            filtered_true_strengths = {}
        else:
            filtered_true_strengths = {knob: true_strengths[knob] for knob in self.knob_names}
        return current_knobs, filtered_true_strengths

    def calculate_n_data_points(self) -> dict[tuple[str, str], int]:
        """Calculate number of data points for each BPM pair."""
        n_data_points = {}
        for start, end in self.bpm_pairs:
            n_bpms, _ = self.mad_iface.count_bpms(f"{start}/{end}")
            n_data_points[(start, end)] = TrackingWorker.get_n_data_points(n_bpms)
            logging.info(f"{start}/{end}: {n_data_points[(start, end)]} data points")
        return n_data_points
