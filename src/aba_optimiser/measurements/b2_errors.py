"""Helpers for reading LHC dipole b2 error tables."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import tfs
from omc3.model.constants import AFS_B2_ERRORS_ROOT
from omc3.optics_measurements.constants import NAME

if TYPE_CHECKING:
    from os import PathLike

_B2_ERRORS_PATTERN = re.compile(r"^MB2022_(?P<energy>\d+(?:\.\d+)?)GeV_.+\.errors$")
LOGGER = logging.getLogger(__name__)


def resolve_b2_error_table(
    beam: int,
    kinetic_energy: float,
    *,
    errors_root: str | PathLike[str] | Path = AFS_B2_ERRORS_ROOT,
) -> Path:
    """Resolve the closest OMC3/WISE b2 error table for the given beam energy."""
    beam_root = Path(errors_root) / f"Beam{beam}"
    if not beam_root.is_dir():
        raise FileNotFoundError(f"LHC b2 error table directory not found: {beam_root}")

    candidates: list[tuple[float, Path]] = []
    for path in beam_root.glob("MB2022_*.errors"):
        match = _B2_ERRORS_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append((float(match.group("energy")), path))

    if not candidates:
        raise FileNotFoundError(f"No MB2022 b2 error tables found in {beam_root}")

    _, best_path = min(candidates, key=lambda item: (abs(item[0] - kinetic_energy), item[0]))
    return best_path


def read_b2_error_table(path: Path | str) -> dict[str, float]:
    """Read an OMC3-style b2 error table as element-name to K1L mapping."""
    table = tfs.read(path, index=NAME)
    LOGGER.info(f"Read b2 error table from {path} with {len(table)} entries")
    if "K1L" not in table.columns:
        raise KeyError(f"Column 'K1L' not found in b2 error table: {path}")
        print(f"Available columns in b2 error table: {table.columns}")
    return {str(name): float(value) for name, value in table["K1L"].items()}
