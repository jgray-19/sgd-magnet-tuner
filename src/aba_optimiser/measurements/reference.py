"""Construction helpers for tmom-recon coordinate frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tmom_recon import ReconstructionFrame

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


def reconstruction_frame(
    closed_orbit: pd.DataFrame,
    *,
    dynamic_planes: Iterable[str] = (),
) -> ReconstructionFrame:
    """Build a frame from measured positions and, for retained planes, fitted angles."""
    orbit = closed_orbit.copy()
    if "name" in orbit.columns:
        orbit = orbit.set_index("name")
    orbit.columns = [str(column).lower() for column in orbit.columns]
    dynamic = tuple(str(plane).lower() for plane in dynamic_planes)
    retained = tuple(plane for plane in ("x", "y") if plane not in dynamic)
    fitted_momenta = orbit[[f"p{plane}" for plane in retained]] if retained else None
    return ReconstructionFrame(
        orbit_zero=orbit[["x", "y"]],
        dynamic_planes=dynamic,
        fitted_momenta=fitted_momenta,
    )


__all__ = ["reconstruction_frame"]
