"""Tests for the initial-condition update path.

Covers:
- TrackingWorker._prepare_batches stores flat numpy arrays
- TrackingWorker._send_init_condition_update updates _init_coords_np in Python
- TrackingWorker._handle_control_command dispatches 'update_init_coords'
- WorkerManager.send_init_condition_updates validates shape and sends per-worker slices
- Controller._make_epoch_end_hook returns None when no callback is given
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from aba_optimiser.workers.tracking import TrackingWorker

# ---------------------------------------------------------------------------
# Stubs (no mocking library)
# ---------------------------------------------------------------------------


class FakeMAD:
    """Records every object passed to .send() so tests can inspect sent values."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, obj: object) -> FakeMAD:
        self.sent.append(obj)
        return self  # support chaining


class FakeConn:
    """Minimal pipe-like object that stores the last sent message."""

    def __init__(self) -> None:
        self.last_sent = None

    def send(self, obj) -> None:
        self.last_sent = obj


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_worker_with_init_coords(n_particles: int = 6, num_batches: int = 2) -> TrackingWorker:
    """Return a TrackingWorker with _prepare_batches already called (no subprocess)."""
    worker = object.__new__(TrackingWorker)
    worker.worker_id = 0
    worker.observables = ("x", "px")
    worker.hessian_weight_order = ("x", "px")
    worker.simulation_config = SimpleNamespace(num_batches=num_batches)
    worker.config = SimpleNamespace(kick_plane="x")

    init_coords = np.zeros((n_particles, 6), dtype=np.float64)
    init_coords[:, 0] = np.arange(n_particles, dtype=float)        # x
    init_coords[:, 1] = np.arange(n_particles, dtype=float) * 0.1  # px
    init_coords[:, 3] = np.arange(n_particles, dtype=float) * 0.01 # py
    init_pts = np.ones(n_particles, dtype=np.float64) * 1e-3

    worker.comparison_arrays = {
        "x": np.zeros((n_particles, 2)),
        "px": np.zeros((n_particles, 2)),
    }
    worker.weight_arrays = {
        "x": np.ones((n_particles, 2)),
        "px": np.ones((n_particles, 2)),
    }

    worker._prepare_batches(init_coords, init_pts, num_batches)
    return worker


# ---------------------------------------------------------------------------
# _prepare_batches stores flat numpy arrays
# ---------------------------------------------------------------------------

def test_prepare_batches_stores_flat_numpy_init_coords() -> None:
    worker = _make_worker_with_init_coords(n_particles=4, num_batches=2)

    assert hasattr(worker, "_init_coords_np")
    assert hasattr(worker, "_init_pts_np")
    assert worker._init_coords_np.shape == (4, 6)
    assert worker._init_pts_np.shape == (4,)
    assert worker._init_coords_np.dtype == np.float64
    assert worker._init_coords_np.flags["C_CONTIGUOUS"]


def test_prepare_batches_flat_arrays_match_batched_lists() -> None:
    n = 6
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=3)

    # Reconstruct the flat array from the nested lists.
    reconstructed = np.array(
        [coord for batch in worker.init_coords for coord in batch], dtype=np.float64
    )
    assert np.allclose(worker._init_coords_np, reconstructed)


# ---------------------------------------------------------------------------
# _send_init_condition_update updates _init_coords_np in Python
# ---------------------------------------------------------------------------

def test_send_init_condition_update_patches_px_py_in_python() -> None:
    n = 6
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    expected_x = worker._init_coords_np[:, 0].copy()

    new_px = np.linspace(1.0, 2.0, n)
    new_py = np.linspace(-1.0, -2.0, n)

    worker._send_init_condition_update(FakeMAD(), new_px, new_py)  # type: ignore[arg-type]

    assert np.allclose(worker._init_coords_np[:, 1], new_px)
    assert np.allclose(worker._init_coords_np[:, 3], new_py)
    # x column (index 0) must be untouched.
    assert np.allclose(worker._init_coords_np[:, 0], expected_x)


def test_send_init_condition_update_sends_column_matrices_to_mad() -> None:
    n = 4
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    new_px = np.ones(n)
    new_py = np.ones(n) * 2

    mad = FakeMAD()
    worker._send_init_condition_update(mad, new_px, new_py)  # type: ignore[arg-type]

    # The Lua script string is sent first, then px, then py.
    assert len(mad.sent) >= 3
    px_sent = mad.sent[-2]
    py_sent = mad.sent[-1]
    assert isinstance(px_sent, np.ndarray) and px_sent.shape == (n, 1)
    assert isinstance(py_sent, np.ndarray) and py_sent.shape == (n, 1)
    assert np.allclose(px_sent[:, 0], new_px)
    assert np.allclose(py_sent[:, 0], new_py)


# ---------------------------------------------------------------------------
# _handle_control_command dispatches update_init_coords
# ---------------------------------------------------------------------------

def test_handle_control_command_update_init_coords_updates_arrays_and_acks() -> None:
    n = 4
    worker = _make_worker_with_init_coords(n_particles=n, num_batches=2)
    conn = FakeConn()
    worker.conn = conn

    new_px = np.linspace(0.1, 0.4, n)
    new_py = np.linspace(-0.1, -0.4, n)
    command = {"cmd": "update_init_coords", "px": new_px, "py": new_py}

    worker._handle_control_command(FakeMAD(), command)  # type: ignore[arg-type]

    assert np.allclose(worker._init_coords_np[:, 1], new_px)
    assert np.allclose(worker._init_coords_np[:, 3], new_py)
    assert conn.last_sent == {"worker_id": 0, "status": "ok"}


# ---------------------------------------------------------------------------
# WorkerManager.send_init_condition_updates — real pipes + threads
# ---------------------------------------------------------------------------

def _recv_and_ack(child_conn, received_store: list, idx: int) -> None:
    """Thread target: receive one message from the child end and send ack back."""
    msg = child_conn.recv()
    received_store[idx] = msg
    child_conn.send({"worker_id": idx, "status": "ok"})


def _make_real_channels(counts: list[int]):
    """Build a WorkerManager stub backed by real mp.Pipe() connections."""
    from aba_optimiser.training.worker_manager import WorkerManager
    from aba_optimiser.workers.protocol import WorkerChannels

    parent_conns, child_conns = zip(*[mp.Pipe() for _ in counts])

    channels = object.__new__(WorkerChannels)
    channels.parent_conns = tuple(parent_conns)
    # workers just need a .exitcode attribute for error-handling
    channels.workers = tuple(SimpleNamespace(pid=i, exitcode=None) for i in range(len(counts)))
    channels._count = len(counts)
    channels._conn_index = {conn: i for i, conn in enumerate(parent_conns)}

    wm = object.__new__(WorkerManager)
    wm._worker_particle_counts = list(counts)
    wm.channels = channels
    wm._channels = lambda: channels

    return wm, list(child_conns)


def test_send_init_condition_updates_slices_correctly() -> None:
    counts = [3, 2, 4]
    wm, child_conns = _make_real_channels(counts)

    total = sum(counts)
    new_px_py = np.column_stack([
        np.arange(total, dtype=float),
        -np.arange(total, dtype=float),
    ])

    received: list = [None] * len(counts)
    threads = [
        threading.Thread(target=_recv_and_ack, args=(child_conns[i], received, i))
        for i in range(len(counts))
    ]
    for t in threads:
        t.start()

    wm.send_init_condition_updates(new_px_py)

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "Worker thread did not finish in time"

    offset = 0
    for i, n in enumerate(counts):
        msg = received[i]
        assert isinstance(msg, dict), f"Worker {i} received unexpected value: {msg!r}"
        assert msg["cmd"] == "update_init_coords"
        assert np.allclose(msg["px"], new_px_py[offset : offset + n, 0])
        assert np.allclose(msg["py"], new_px_py[offset : offset + n, 1])
        offset += n


def test_send_init_condition_updates_rejects_wrong_shape() -> None:
    counts = [3, 2]
    wm, _ = _make_real_channels(counts)

    with pytest.raises(ValueError, match="shape"):
        wm.send_init_condition_updates(np.zeros((4, 2)))  # wrong total (should be 5)


# ---------------------------------------------------------------------------
# Controller._make_epoch_end_hook — no subprocess needed
# ---------------------------------------------------------------------------

def test_make_epoch_end_hook_returns_none_when_no_callback() -> None:
    from aba_optimiser.training.controller import Controller

    ctrl = object.__new__(Controller)
    ctrl.initial_conditions_callback = None
    assert ctrl._make_epoch_end_hook() is None


def test_make_epoch_end_hook_calls_callback_and_dispatches() -> None:
    from aba_optimiser.training.controller import Controller

    dispatched: list[np.ndarray] = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            dispatched.append(arr)

    ctrl = object.__new__(Controller)
    ctrl.worker_manager = FakeWorkerManager()

    new_px_py = np.zeros((5, 2))
    ctrl.initial_conditions_callback = lambda knobs: new_px_py

    hook = ctrl._make_epoch_end_hook()
    assert hook is not None

    hook({"k1": 1.0})
    assert len(dispatched) == 1
    assert dispatched[0] is new_px_py


def test_make_epoch_end_hook_skips_dispatch_when_callback_returns_none() -> None:
    from aba_optimiser.training.controller import Controller

    dispatched: list = []

    class FakeWorkerManager:
        def send_init_condition_updates(self, arr: np.ndarray) -> None:
            dispatched.append(arr)

    ctrl = object.__new__(Controller)
    ctrl.worker_manager = FakeWorkerManager()
    ctrl.initial_conditions_callback = lambda knobs: None

    hook = ctrl._make_epoch_end_hook()
    assert hook is not None

    hook({"k1": 1.0})
    assert len(dispatched) == 0
