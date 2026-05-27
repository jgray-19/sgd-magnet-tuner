from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.controller_config import MeasurementConfig, OutputConfig, SequenceConfig


class DummyController(BaseController):
    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        return {}, {}


class EmptyKnobConfigurationManager(ConfigurationManager):
    def setup_mad_interface(
        self,
        debug: bool = False,
        mad_logfile=None,
        corrector_strengths=None,
        tune_knobs_file=None,
    ) -> None:
        del debug, mad_logfile, corrector_strengths, tune_knobs_file
        self.mad_iface = SimpleNamespace(  # ty:ignore[invalid-assignment]
            dp2pt=lambda value: value,
            pt2dp=lambda value: value,
        )
        self.knob_names = []
        self.elem_spos = []
        self.all_bpms = ["BPM.9R1.B1", "BPM.9L2.B1"]
        self.bpms_in_range = ["BPM.9R1.B1", "BPM.9L2.B1"]

    def initialise_knob_strengths(
        self,
        true_strengths: dict[str, float] | None = None,
        provided_initial_knobs: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        del true_strengths, provided_initial_knobs
        self.initial_strengths = np.array([])
        return {}, {}


class EmptyKnobController(DummyController):
    _configuration_manager_cls = EmptyKnobConfigurationManager


def test_measurement_config_expands_single_file_scoped_values(tmp_path) -> None:
    config = MeasurementConfig(
        measurement_files=[tmp_path / "m0.parquet", tmp_path / "m1.parquet"],
        corrector_files=tmp_path / "correctors.tfs",
        tune_knobs_files=None,
        machine_deltaps=1e-4,
    ).expanded_for_measurements()

    assert config.corrector_files == [tmp_path / "correctors.tfs", tmp_path / "correctors.tfs"]
    assert config.tune_knobs_files == [None, None]
    assert config.machine_deltaps == [1e-4, 1e-4]


def test_base_controller_raises_when_no_knobs_created(seq_b1) -> None:
    accelerator = LHC(
        beam=1,
        kinetic_energy=6800,
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
        EmptyKnobController(
            accelerator=accelerator,
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_config=SequenceConfig("BPM.9R1.B1/BPM.9L2.B1"),
            bpm_start_points=["BPM.9R1.B1"],
            bpm_end_points=["BPM.9L2.B1"],
            output_config=OutputConfig(write_tensorboard_logs=False),
        )
