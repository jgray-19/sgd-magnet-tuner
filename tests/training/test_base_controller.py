from __future__ import annotations

from types import SimpleNamespace

import pytest

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.configuration_manager import ConfigurationManager


class DummyController(BaseController):
    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        return {}, {}


def test_base_controller_raises_when_no_knobs_created(
    monkeypatch: pytest.MonkeyPatch,
    seq_b1,
) -> None:
    def fake_setup_mad_interface(
        self,
        first_bpm,
        bad_bpms,
        debug=False,
        mad_logfile=None,
    ) -> None:
        del first_bpm, bad_bpms, debug, mad_logfile
        self.mad_iface = SimpleNamespace()
        self.knob_names = []
        self.elem_spos = []
        self.all_bpms = ["BPM.9R1.B1", "BPM.9L2.B1"]
        self.bpms_in_range = ["BPM.9R1.B1", "BPM.9L2.B1"]
        self.bend_lengths = None

    def fake_initialise_knob_strengths(
        self,
        true_strengths,
        provided_initial_knobs=None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        del self, true_strengths, provided_initial_knobs
        return {}, {}

    monkeypatch.setattr(
        ConfigurationManager,
        "setup_mad_interface",
        fake_setup_mad_interface,
    )
    monkeypatch.setattr(
        ConfigurationManager,
        "initialise_knob_strengths",
        fake_initialise_knob_strengths,
    )

    accelerator = LHC(
        beam=1,
        beam_energy=6800,
        sequence_file=seq_b1,
        optimise_quadrupoles=True,
    )
    optimiser_config = OptimiserConfig(
        max_epochs=1,
        warmup_epochs=0,
        warmup_lr_start=1e-6,
        max_lr=1e-6,
        min_lr=1e-6,
        gradient_converged_value=1e-12,
    )
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=1,
        num_batches=1,
    )

    with pytest.raises(ValueError, match="No optimisation knobs were created for this controller configuration"):
        DummyController(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            magnet_range="BPM.9R1.B1/BPM.9L2.B1",
            bpm_start_points=["BPM.9R1.B1"],
            bpm_end_points=["BPM.9L2.B1"],
            show_plots=False,
            write_tensorboard_logs=False,
        )
