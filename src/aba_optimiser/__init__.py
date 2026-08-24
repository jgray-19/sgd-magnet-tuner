"""Public interface for the ABA optimiser package.

The package bundles configuration helpers, optimiser implementations, and
simulation orchestrators for magnet knob tuning studies.
"""

# from __future__ import annotations

# import logging
# import os
# import warnings

# The top-level package intentionally does not re-export a large symbol surface.
__all__ = []


# def _suppress_third_party_warning_noise() -> None:
#     """Silence known dependency deprecation spam during matplotlib imports.

#     The current matplotlib/pyparsing combination emits a large volume of
#     PyparsingDeprecationWarning messages at import time. These warnings come
#     from third-party code rather than this package and can drown out real test
#     failures or CLI output.
#     """
#     try:
#         from pyparsing import PyparsingDeprecationWarning
#     except ImportError:
#         return

#     warnings.filterwarnings(
#         "ignore",
#         category=PyparsingDeprecationWarning,
#         message=r".*deprecated - use .*",
#     )


# def _configure_repo_logging() -> None:
#     """Enable package-local INFO logging outside GitHub environments."""
#     if os.getenv("GITHUB_ACTIONS"):
#         return

#     package_logger = logging.getLogger("aba_optimiser")
#     package_logger.setLevel(logging.DEBUG)
#     # Keep propagation enabled so pytest caplog and user/root logging config
#     # can observe aba_optimiser logs without installing a package-specific
#     # handler that duplicates output.
#     package_logger.propagate = True


# _suppress_third_party_warning_noise()
# _configure_repo_logging()
