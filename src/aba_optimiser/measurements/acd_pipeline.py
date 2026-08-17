"""Reusable orchestration helpers for end-to-end AC-dipole pipelines.

The accelerator-specific parts of an integration pipeline are model creation,
the accelerator options passed to omc3, and the fitted accelerator class.  The
data plumbing is identical for PSB and LHC: convert long-form turn-by-turn data
to Harpy input, run driven and compensated optics, combine measured positions
with fitted model angles, and merge reconstructed momenta. This module keeps
that common path out of accelerator tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


@dataclass(frozen=True)
class ACDOpticsAnalysisConfig:
    """Configuration shared by the Harpy and two optics-analysis stages."""

    model_dir: Path
    harpy_options: Mapping[str, Any]
    optics_options: Mapping[str, Any]


def long_frame_to_tbt_data(frame: pd.DataFrame, *, source_file: Path) -> TbtData:
    """Convert long-form ``name/turn/x/y`` data into Harpy's in-memory type."""
    names = list(dict.fromkeys(frame["name"].astype(str)))
    turns = sorted(int(turn) for turn in frame["turn"].unique())

    def matrix(plane: str) -> pd.DataFrame:
        result = (
            frame.pivot(index="name", columns="turn", values=plane)
            .reindex(index=names, columns=turns)
            .astype(float)
        )
        if result.isna().any().any():
            raise ValueError(f"Incomplete {plane}-plane turn-by-turn matrix")
        return result

    return TbtData(
        matrices=[TransverseData(X=matrix("x"), Y=matrix("y"))],
        nturns=len(turns),
        bunch_ids=[0],
        meta={"file": str(source_file)},
    )


def run_driven_and_compensated_optics(
    frame: pd.DataFrame,
    *,
    source_file: Path,
    output_dir: Path,
    config: ACDOpticsAnalysisConfig,
) -> tuple[Path, Path]:
    """Run Harpy once, then driven and equation-compensated optics analyses."""
    harpy_options = dict(config.harpy_options)
    if harpy_options.get("clean", False):
        raise ValueError(
            "Harpy/OMC3 cleaning is disabled for ACD pipelines; clean the "
            "turn-by-turn data before passing it to Harpy instead."
        )
    harpy_options["clean"] = False
    lin_dir = output_dir / "lin_files"
    driven_dir = output_dir / "driven"
    compensated_dir = output_dir / "compensated"
    lin_dir.mkdir(parents=True, exist_ok=True)
    input_data = long_frame_to_tbt_data(frame, source_file=source_file)

    hole_in_one_entrypoint(
        harpy=True,
        optics=False,
        files=[input_data],
        outputdir=lin_dir,
        tbt_datatype="tbt_data",
        model_dir=config.model_dir,
        **harpy_options,
    )
    lin_base = lin_dir / source_file.name
    common = dict(config.optics_options)
    for destination, compensation in (
        (driven_dir, "none"),
        (compensated_dir, "equation"),
    ):
        hole_in_one_entrypoint(
            harpy=False,
            optics=True,
            files=[lin_base],
            outputdir=destination,
            model_dir=config.model_dir,
            compensation=compensation,
            **common,
        )
    return driven_dir, compensated_dir


def build_mixed_closed_orbit_reference(
    measured_orbit: pd.DataFrame,
    fitted_orbit: pd.DataFrame,
) -> pd.DataFrame:
    """Combine measured ``x/y`` with fitted-model ``px/py`` at common BPMs."""
    measured = measured_orbit.set_index("name") if "name" in measured_orbit else measured_orbit
    fitted = fitted_orbit.set_index("name") if "name" in fitted_orbit else fitted_orbit
    common = measured.index.intersection(fitted.index)
    if common.empty:
        raise ValueError("Measured and fitted closed orbits have no common BPMs")
    result = pd.DataFrame(
        {
            "x": measured.loc[common, "x"].astype(float),
            "y": measured.loc[common, "y"].astype(float),
            "px": fitted.loc[common, "px"].astype(float),
            "py": fitted.loc[common, "py"].astype(float),
        },
        index=common,
    )
    if result.isna().any().any():
        raise ValueError("Mixed closed-orbit reference contains missing values")
    return result


def merge_reconstructed_momenta(current: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    """Patch px/py by case-insensitive ``(turn, name)`` while preserving names."""
    base = current.reset_index().copy()
    refreshed = pd.DataFrame(reconstructed).copy()
    offset = int(base["turn"].min())
    refreshed["turn"] = refreshed["turn"].astype(int) + offset
    base["_match_name"] = base["name"].astype(str).str.upper()
    refreshed["_match_name"] = refreshed["name"].astype(str).str.upper()
    columns = ["turn", "_match_name", "px", "py", "var_px", "var_py"]
    merged = base.merge(
        refreshed[columns], on=["turn", "_match_name"], how="left", suffixes=("", "_new")
    )
    for column in ("px", "py", "var_px", "var_py"):
        merged[column] = merged[f"{column}_new"].fillna(merged[column])
        merged.drop(columns=f"{column}_new", inplace=True)
    merged.drop(columns="_match_name", inplace=True)
    return merged.set_index(["turn", "name"])


def subtract_closed_orbit(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Subtract a named closed-orbit state from matching tracking rows."""
    result = frame.copy()
    closed_orbit = reference.copy()
    if "name" in closed_orbit:
        closed_orbit = closed_orbit.set_index("name")
    closed_orbit.index = closed_orbit.index.astype(str).str.upper()
    names = result["name"].astype(str).str.upper()
    for coordinate in ("x", "px", "y", "py"):
        if coordinate not in result or coordinate not in closed_orbit:
            continue
        offset = names.map(closed_orbit[coordinate])
        matched = offset.notna()
        result.loc[matched, coordinate] -= offset.loc[matched].to_numpy(dtype=float)
    return result


def make_live_marker_momentum_callback(
    *,
    controller: Any,
    generators: Mapping[int, Any],
    pts: Mapping[int, float],
    refresh_every: int = 1,
    recoverable_exceptions: tuple[type[Exception], ...] = (),
) -> Callable[[dict[str, float], dict[str, float]], Any]:
    """Refresh marker momenta periodically as the fitted lattice changes."""
    if refresh_every < 1:
        raise ValueError("refresh_every must be at least one")
    track_data = controller.data_manager.track_data
    calls = 0

    def refresh(current: dict[str, float], _best: dict[str, float]):
        nonlocal calls
        calls += 1
        optimisation_loop = controller.optimisation_loop
        if not current or calls % refresh_every or calls >= optimisation_loop.max_epochs:
            return None
        updated = {}
        try:
            for file_index, existing_data in track_data.items():
                reconstructed = generators[file_index].update(
                    magnet_strengths=current,
                    pt=float(pts[file_index]),
                )
                updated[file_index] = merge_reconstructed_momenta(existing_data, reconstructed)
        except recoverable_exceptions:
            return None
        # A marker refresh changes the objective. A loss recorded against the
        # previous marker coordinates is not comparable with subsequent losses.
        optimisation_loop.best_loss = float("inf")
        optimisation_loop.best_knobs = current.copy()
        return controller.worker_manager.build_update_coords(updated)

    return refresh
