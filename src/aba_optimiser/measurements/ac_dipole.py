"""Shared AC-dipole optimisation window definition.

Single source of truth for the AC-dipole window used by both the closed-orbit
optimisation workflow and the squeeze quadrupole tuning pipeline.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tfs


@dataclass(frozen=True)
class ACDipoleOptimisationWindow:
    """Window definition for full-turn arc-by-arc tracking around the AC dipole."""

    bpm_upstream: str
    bpm_downstream: str


def window_from_attrs(attrs: dict) -> ACDipoleOptimisationWindow | None:
    """Build an :class:`ACDipoleOptimisationWindow` from a DataFrame's attrs.

    Returns ``None`` when the upstream/downstream BPM metadata is absent.
    """
    upstream = attrs.get("ac_dipole_bpm_upstream")
    downstream = attrs.get("ac_dipole_bpm_downstream")
    if not upstream or not downstream:
        return None
    return ACDipoleOptimisationWindow(
        bpm_upstream=str(upstream),
        bpm_downstream=str(downstream),
    )


def normalise_model_name(name: object) -> str:
    return str(name).strip().strip('"').upper()


def find_name_column(df: pd.DataFrame) -> str:
    for candidate in ("name", "NAME"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not find a NAME/name column")


def find_s_column(df: pd.DataFrame) -> str:
    for candidate in ("s", "S"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not find an S/s column")


def model_search_roots(model_dir: Path) -> list[Path]:
    roots: list[Path] = [model_dir]
    linked_models = model_dir / "acc-models-psb"
    if linked_models.exists():
        roots.append(linked_models)
        with contextlib.suppress(OSError):
            roots.append(linked_models.resolve())
    return roots


def find_model_file(model_dir: Path, filename: str) -> Path:
    for root in model_search_roots(model_dir):
        candidate = root / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {filename} under {model_dir}")


def load_twiss_table(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(tfs.read(path))
    name_col = find_name_column(df)
    s_col = find_s_column(df)
    df = df.copy()
    df["name"] = df[name_col].map(normalise_model_name)
    df["s"] = pd.to_numeric(df[s_col], errors="coerce")
    return df.set_index("name", drop=False)


def load_model_twiss(model_dir: Path, filename: str) -> pd.DataFrame:
    return load_twiss_table(find_model_file(model_dir, filename))


def infer_ac_dipole_s(model_dir: Path, ac_dipole_name: str) -> float:
    """Return the longitudinal position [m] of the AC dipole.

    ``ac_dipole_name`` is the exciter element name, taken from the accelerator
    class (e.g. ``PSB.ac_dipole_name``).
    """
    ac_dipole_name = normalise_model_name(ac_dipole_name)
    twiss_elements = load_model_twiss(model_dir, "twiss_elements.dat")
    ac_mask = (twiss_elements["name"] == ac_dipole_name) & twiss_elements["s"].notna()
    if not ac_mask.any():
        raise ValueError(
            f"No {ac_dipole_name} element found in "
            f"{find_model_file(model_dir, 'twiss_elements.dat')}"
        )
    return float(twiss_elements.loc[ac_mask, "s"].iloc[0])
