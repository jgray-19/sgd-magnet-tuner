from __future__ import annotations

import json

import numpy as np
import pytest

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.optimisation_loop import OptimisationLoop


def _make_loop(knob_names: list[str]) -> OptimisationLoop:
    optimiser_config = OptimiserConfig(
        max_epochs=2,
        warmup_epochs=1,
        warmup_lr_start=1e-3,
        max_lr=1e-3,
        min_lr=1e-3,
        gradient_converged_value=1e-6,
        optimiser_type="adam",
    )
    simulation_config = SimulationConfig(
        tracks_per_worker=1,
        num_workers=1,
        num_batches=1,
    )
    initial_strengths = np.zeros(len(knob_names), dtype=float)
    return OptimisationLoop(
        initial_strengths=initial_strengths,
        knob_names=knob_names,
        true_strengths={},
        optimiser_config=optimiser_config,
        simulation_config=simulation_config,
    )


def test_load_checkpoint_allows_current_knob_superset(tmp_path) -> None:
    loop = _make_loop(["k1", "k2", "k3"])

    checkpoint_payload = {
        "saved_epoch": 3,
        "next_epoch": 4,
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": 1.5, "k2": -2.0},
        "best_knobs": {"k1": 1.0, "k2": -1.0},
        "best_loss": 0.25,
        "prev_loss": 0.3,
        "smoothed_grad_norm": 1e-3,
        "smoothed_loss_change": 2e-3,
        "max_clipping_ratio": 1.2,
        "optimiser_state": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "m": [1.0, 2.0],
            "v": [3.0, 4.0],
            "t": 7,
        },
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    base_current = {"k1": 10.0, "k2": 20.0, "k3": 30.0}
    checkpoint_state = loop._load_checkpoint(checkpoint_path, base_current_knobs=base_current)

    assert checkpoint_state["saved_epoch"] == 3
    assert checkpoint_state["next_epoch"] == 4
    assert checkpoint_state["current_knobs"] == {"k1": 1.5, "k2": -2.0, "k3": 30.0}
    assert checkpoint_state["prev_loss"] == 0.3

    assert loop.best_knobs == {"k1": 1.0, "k2": -1.0, "k3": 30.0}
    assert loop.best_loss == 0.25

    # Optimiser state should be remapped and padded for the extra knob.
    assert loop.optimiser.t == 7
    assert np.allclose(loop.optimiser.m, [1.0, 2.0, 0.0])
    assert np.allclose(loop.optimiser.v, [3.0, 4.0, 0.0])


def test_load_checkpoint_rejects_missing_current_checkpoint_knobs(tmp_path) -> None:
    loop = _make_loop(["k1"])  # current setup is missing k2 from checkpoint

    checkpoint_payload = {
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": 1.0, "k2": 2.0},
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    with pytest.raises(ValueError, match="missing checkpoint knobs"):
        loop._load_checkpoint(checkpoint_path)


def test_load_checkpoint_rejects_non_finite_knob_values(tmp_path) -> None:
    loop = _make_loop(["k1", "k2"])

    checkpoint_payload = {
        "knob_names": ["k1", "k2"],
        "current_knobs": {"k1": float("nan"), "k2": 2.0},
        "best_knobs": {"k1": 1.0, "k2": 2.0},
    }
    checkpoint_path = tmp_path / "checkpoint_nan.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    with pytest.raises(ValueError, match="non-finite knob values"):
        loop._load_checkpoint(checkpoint_path)


def test_load_checkpoint_remaps_and_pads_in_current_knob_order(tmp_path) -> None:
    # Current optimisation order differs from checkpoint order and adds one extra knob.
    loop = _make_loop(["k3", "k1", "k2", "k4"])

    checkpoint_payload = {
        "saved_epoch": 5,
        "next_epoch": 6,
        "knob_names": ["k1", "k2", "k3"],
        "current_knobs": {"k1": 10.0, "k2": 20.0, "k3": 30.0},
        "best_knobs": {"k1": 1.0, "k2": 2.0, "k3": 3.0},
        "optimiser_state": {
            "type": "adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "m": [100.0, 200.0, 300.0],
            "v": [1.0, 2.0, 3.0],
            "t": 11,
        },
    }
    checkpoint_path = tmp_path / "checkpoint_order.json"
    checkpoint_path.write_text(json.dumps(checkpoint_payload))

    base_current = {"k3": -3.0, "k1": -1.0, "k2": -2.0, "k4": 99.0}
    checkpoint_state = loop._load_checkpoint(checkpoint_path, base_current_knobs=base_current)

    # Values should be in current order: [k3, k1, k2, k4].
    assert checkpoint_state["current_knobs"] == {
        "k3": 30.0,
        "k1": 10.0,
        "k2": 20.0,
        "k4": 99.0,
    }
    assert loop.best_knobs == {
        "k3": 3.0,
        "k1": 1.0,
        "k2": 2.0,
        "k4": 99.0,
    }

    # Optimiser vectors are remapped by knob name then padded for k4.
    # checkpoint m/v order was [k1, k2, k3] = [100, 200, 300] / [1, 2, 3]
    # current order is [k3, k1, k2, k4] -> [300, 100, 200, 0] / [3, 1, 2, 0]
    assert np.allclose(loop.optimiser.m, [300.0, 100.0, 200.0, 0.0])
    assert np.allclose(loop.optimiser.v, [3.0, 1.0, 2.0, 0.0])
    assert loop.optimiser.t == 11
