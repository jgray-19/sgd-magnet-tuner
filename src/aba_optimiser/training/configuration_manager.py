"""Configuration and setup management for the optimisation controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.mad import GradientDescentMadInterface, get_mad_interface
from aba_optimiser.workers import TrackingWorker

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages configuration and setup for the optimisation process."""

    def __init__(
        self,
        accelerator: Accelerator,
        simulation_config: SimulationConfig,
        magnet_range: str,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        optimise_knobs: list[str] | None = None,
    ):
        self.mad_iface: GradientDescentMadInterface = None
        self.knob_names: list[str] = []
        self.elem_spos: list[float] = []
        self.all_bpms: list[str] = []
        self.initial_strengths: np.ndarray = np.array([])
        self.fixed_start: str = ""
        self.fixed_end: str = ""
        self.bend_lengths: dict | None = None

        self.accelerator = accelerator
        self.start_bpms = bpm_start_points
        self.end_bpms: list[str] = bpm_end_points
        self.magnet_range = magnet_range
        self.simulation_config = simulation_config

    def setup_mad_interface(
        self,
        first_bpm: str | None,
        bad_bpms: list[str] | None,
        debug: bool = False,
        mad_logfile: Path | None = None,
    ) -> None:
        """Initialise the MAD-NG interface and get basic model parameters."""

        self.mad_iface = get_mad_interface(self.accelerator)(
            accelerator=self.accelerator,
            start_bpm=first_bpm,
            magnet_range=self.magnet_range,
            corrector_strengths=None,
            tune_knobs_file=None,
            bad_bpms=bad_bpms,
            debug=debug,
            mad_logfile=mad_logfile,
        )
        self.knob_names = self.mad_iface.knob_names

        self.elem_spos: list[int | float] = self.mad_iface.elem_spos

        self.all_bpms = self.mad_iface.all_bpms
        self.bpms_in_range = self.mad_iface.bpms_in_range
        LOGGER.warning(f"Total BPMs in model: {len(self.all_bpms)}, BPMs in specified range {self.magnet_range}: {len(self.bpms_in_range)}")

        self.start_bpms = [bpm for bpm in self.start_bpms if bpm in self.bpms_in_range]
        self.end_bpms = [bpm for bpm in self.end_bpms if bpm in self.bpms_in_range]

        # Accelerator-specific bend normalisation (no-op by default)
        self.bend_lengths = self.accelerator.get_bend_lengths(self.mad_iface)

        # When use_fixed_bpm is True we derive a fixed BPM window from magnet_range and
        # store its start/end in fixed_start/fixed_end. When it is False we intentionally
        # leave fixed_start/fixed_end at their default values (empty strings), which
        # indicates to downstream code that no fixed BPM window should be enforced and
        # that the active BPM range should instead be taken from start_bpms/end_bpms or
        # other model-derived information.
        if self.simulation_config.use_fixed_bpm:
            # Use magnet_range to determine fixed start and end points
            self.fixed_start, self.fixed_end = self.magnet_range.split("/", 1)

            # Validate fixed points are in the model
            if (
                self.fixed_start not in self.bpms_in_range
                or self.fixed_end not in self.bpms_in_range
            ):
                LOGGER.warning(
                    f"Fixed BPMs from range {self.magnet_range} not found in model, using first available"
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
            _, n_bpms, _ = self.mad_iface.count_bpms(f"{start}/{end}")
            n_data_points[(start, end)] = TrackingWorker.get_n_data_points(n_bpms)
            logging.info(f"{start}/{end}: {n_data_points[(start, end)]} data points")
        return n_data_points
