"""Helpers for reading LHC dipole b2 error tables."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import tfs
from omc3.optics_measurements.constants import NAME

_B2_ERRORS_ROOT = Path("/afs/cern.ch/eng/sl/lintrack/error_tables")
_ENERGY_FILE_PATTERN = re.compile(r"_(?P<energy>\d+(?:\.\d+)?)GeV_.*\.errors$")
LOGGER = logging.getLogger(__name__)


def resolve_b2_error_table(
    beam: int,
    kinetic_energy: float,
    *,
    errors_root: Path = _B2_ERRORS_ROOT,
) -> Path:
    """Resolve the closest b2 error table for the given beam and energy."""
    beam_root = errors_root / f"Beam{beam}"
    if not beam_root.is_dir():
        raise FileNotFoundError(f"LHC b2 error table directory not found: {beam_root}")

    candidates: list[tuple[float, Path]] = []
    for f in beam_root.iterdir():
        match = _ENERGY_FILE_PATTERN.search(f.name)
        if match is None:
            continue
        candidates.append((float(match.group("energy")), f))

    if not candidates:
        raise FileNotFoundError(f"No *GeV_*.errors files found in {beam_root}")

    _, best_file = min(candidates, key=lambda item: (abs(item[0] - kinetic_energy), item[0]))
    return best_file


def read_b2_error_table(path: Path | str) -> dict[str, float]:
    """Read an OMC3-style b2 error table as element-name to K1L mapping."""
    assert Path(path).suffix == ".errors", f"Expected a .errors file, got {path}"
    table = tfs.read(path, index=NAME)
    LOGGER.info(f"Read b2 error table from {path} with {len(table)} entries")
    if "K1L" not in table.columns:
        raise KeyError(f"Column 'K1L' not found in b2 error table: {path}")
        print(f"Available columns in b2 error table: {table.columns}")
    return {str(name): float(value) for name, value in table["K1L"].items()}


def b2_errors_to_magnet_strengths(b2_errors: dict[str, float]) -> dict[str, float]:
    """Convert a b2 K1L error mapping to ``AcceleratorMadInterface.set_magnet_strengths`` keys.

    Routes each K1L value into the quadrupole perturbation slot (``dknl[2]``),
    leaving the dipole slot (``dknl[1]``) untouched, matching the effect of the
    previous raw ``mad.send`` implementation.
    """
    return {f"{name}.dk1l": k1l for name, k1l in b2_errors.items() if k1l != 0}
