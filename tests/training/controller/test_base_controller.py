from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_controller import BaseController
from aba_optimiser.training.config.manager import ConfigurationManager
from aba_optimiser.training.config.models import MeasurementConfig, OutputConfig, SequenceConfig


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


def test_configuration_manager_preserves_model_defaults_for_missing_initial_knobs(seq_b1) -> None:
    accelerator = LHC(
        beam=1,
        kinetic_energy=6800,
        sequence_file=seq_b1,
        optimise_quadrupoles=True,
    )
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=1,
        num_batches=1,
    )
    manager = ConfigurationManager(
        accelerator=accelerator,
        simulation_config=simulation_config,
        sequence_config=SequenceConfig("BPM.9R1.B1/BPM.9L2.B1"),
        bpm_start_points=["BPM.9R1.B1"],
        bpm_end_points=["BPM.9L2.B1"],
    )
    initial_state = {"k1": 1.25, "k2": -0.75, "k3": 0.5}

    class FakeMadInterface:
        def __init__(self, initial_values: dict[str, float]) -> None:
            self.values = initial_values.copy()

        def receive_knob_values(self) -> np.ndarray:
            return np.asarray([self.values[name] for name in manager.knob_names], dtype=float)

        def update_knob_values(self, new_values: dict[str, float]) -> None:
            self.values.update(new_values)

    manager.knob_names = list(initial_state)
    manager.mad_iface = FakeMadInterface(initial_state)  # ty:ignore[invalid-assignment]

    initial_knobs, filtered_true = manager.initialise_knob_strengths(
        true_strengths={"k1": 9.0, "outside": 2.0},
        provided_initial_knobs={"k2": 4.0},
    )

    assert initial_knobs == {"k1": 1.25, "k2": 4.0, "k3": 0.5}
    assert filtered_true == {"k1": 9.0}
    np.testing.assert_allclose(manager.initial_strengths, np.array([1.25, 4.0, 0.5]))


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
