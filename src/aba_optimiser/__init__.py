"""Public interface for the ABA optimiser package.

The package bundles configuration helpers, optimiser implementations, and
simulation orchestrators for magnet knob tuning studies.
"""

from __future__ import annotations

import logging
import os

# The top-level package intentionally does not re-export a large symbol surface.
__all__ = []


def _configure_repo_logging() -> None:
    """Enable package-local INFO logging outside GitHub environments."""
    if os.getenv("GITHUB_ACTIONS"):
        return

    package_logger = logging.getLogger("aba_optimiser")
    package_logger.setLevel(logging.INFO)
    # Keep propagation enabled so pytest caplog and user/root logging config
    # can observe aba_optimiser logs without installing a package-specific
    # handler that duplicates output.
    package_logger.propagate = True


_configure_repo_logging()
