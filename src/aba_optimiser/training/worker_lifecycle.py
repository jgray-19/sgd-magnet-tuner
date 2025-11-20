"""Worker lifecycle management for multiprocessing workers."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

LOGGER = logging.getLogger(__name__)

WorkerType = TypeVar("WorkerType", bound=mp.Process)
PayloadType = TypeVar("PayloadType")


class WorkerLifecycleManager(Generic[WorkerType, PayloadType]):
    """Manages the lifecycle of multiprocessing workers.

    Handles:
    - Worker creation with Pipe communication
    - Worker startup and handshake
    - Worker termination and cleanup
    """

    def __init__(self, worker_class: type[WorkerType]):
        """Initialize the worker lifecycle manager.

        Args:
            worker_class: The worker Process class to instantiate
        """
        self.worker_class = worker_class
        self.workers: list[WorkerType] = []
        self.parent_conns: list[Connection] = []

    def create_and_start_workers(
        self,
        payloads: list[tuple[Any, ...]],
        send_handshake: bool = True,
    ) -> None:
        """Create workers with payloads and start them.

        Args:
            payloads: List of tuples containing arguments for each worker
            send_handshake: Whether to send initial handshake (None) to each worker
        """
        LOGGER.info(f"Starting {len(payloads)} workers...")

        for worker_id, payload_args in enumerate(payloads):
            parent, child = mp.Pipe()

            # Unpack payload and create worker
            # Assumes worker constructor is: (conn, worker_id, *payload_args)
            worker = self.worker_class(child, worker_id, *payload_args)
            worker.start()

            self.parent_conns.append(parent)
            self.workers.append(worker)

            # Send initial handshake if requested
            if send_handshake:
                parent.send(None)

        LOGGER.info(f"Successfully started {len(self.workers)} workers")

    def terminate_workers(self, termination_signal: Any = (None, None)) -> None:
        """Terminate all workers gracefully.

        Args:
            termination_signal: Signal to send to workers to indicate termination
        """
        LOGGER.info("Terminating workers...")

        # Send termination signal to all workers
        for conn in self.parent_conns:
            try:
                conn.send(termination_signal)
            except Exception as e:
                LOGGER.warning(f"Error sending termination signal: {e}")

        # Wait for all workers to finish
        for worker in self.workers:
            try:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    LOGGER.warning(f"Worker {worker.name} did not terminate, forcing...")
                    worker.terminate()
            except Exception as e:
                LOGGER.warning(f"Error terminating worker {worker.name}: {e}")

        # Close connections
        for conn in self.parent_conns:
            with contextlib.suppress(Exception):
                conn.close()

        LOGGER.info("All workers terminated")

    def send_to_all(self, message: Any) -> None:
        """Send a message to all workers.

        Args:
            message: Message to send to all workers
        """
        for conn in self.parent_conns:
            conn.send(message)

    def collect_results(self) -> list[Any]:
        """Collect results from all workers.

        Returns:
            List of results from each worker
        """
        results = []
        for conn in self.parent_conns:
            results.append(conn.recv())
        return results

    def cleanup(self) -> None:
        """Clean up resources."""
        self.workers.clear()
        self.parent_conns.clear()
