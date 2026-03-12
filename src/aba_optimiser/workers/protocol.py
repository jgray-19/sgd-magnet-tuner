"""Shared worker communication helpers for parent-side IPC."""

from __future__ import annotations

from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING, NoReturn, TypedDict

if TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection


class WorkerErrorPayload(TypedDict):
    """Structured worker failure payload sent across the pipe."""

    worker_id: int
    status: str
    phase: str
    error_type: str
    error: str
    traceback: str

def raise_for_worker_error_payload(
    payload: object,
    worker: Process | None = None,
) -> NoReturn:
    """Raise a RuntimeError from a worker payload."""
    if isinstance(payload, dict) and payload.get("status") == "error":
        worker_id = payload.get("worker_id", "?")
        phase = payload.get("phase", "unknown")
        error_type = payload.get("error_type", "Exception")
        error = payload.get("error", "unknown worker error")
        raise RuntimeError(f"Worker {worker_id} failed during {phase}: {error_type}: {error}")

    exitcode = None if worker is None else worker.exitcode
    raise RuntimeError(f"Unexpected worker payload: {payload!r} (worker exitcode={exitcode})")


class WorkerChannels:
    """Reusable worker communication state for fast send/receive rounds."""

    __slots__ = ("parent_conns", "workers", "_count", "_conn_index")

    def __init__(self, parent_conns: list[Connection], workers: list[Process]) -> None:
        if len(parent_conns) != len(workers):
            raise ValueError(
                f"Connection/worker count mismatch: {len(parent_conns)} != {len(workers)}"
            )
        self.parent_conns = tuple(parent_conns)
        self.workers = tuple(workers)
        self._count = len(self.parent_conns)
        self._conn_index = {conn: idx for idx, conn in enumerate(self.parent_conns)}

    def send_all(self, message: object) -> None:
        """Send one message to every worker."""
        payload = ForkingPickler.dumps(message)
        for conn in self.parent_conns:
            conn.send_bytes(payload)

    @staticmethod
    def _recv(conn: Connection, worker: Process) -> object:
        try:
            payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"Worker process {worker.pid} closed its pipe before sending a response "
                f"(exitcode={worker.exitcode})"
            ) from exc
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise_for_worker_error_payload(payload, worker)
        return payload

    def recv_all(self) -> list[object]:
        """Receive one message from each worker, preserving connection order."""
        if self._count == 0:
            return []
        if self._count == 1:
            wait(self.parent_conns)
            return [self._recv(self.parent_conns[0], self.workers[0])]

        results: list[object] = [None] * self._count
        seen = bytearray(self._count)
        remaining = self._count

        while remaining:
            for conn in wait(self.parent_conns):
                idx = self._conn_index[conn]
                if seen[idx]:
                    continue
                results[idx] = self._recv(conn, self.workers[idx])
                seen[idx] = 1
                remaining -= 1

        return results
