from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xpart as xp
import xtrack as xt
from xtrack import load_madx_lattice  # ty:ignore[unresolved-import]

from aba_optimiser.config import BEAM_ENERGY, PROJECT_ROOT
from aba_optimiser.io.utils import get_lhc_file_path

if TYPE_CHECKING:
    from pathlib import Path

    import tfs

logger = logging.getLogger(__name__)


def get_default_json_path(sequence_file: Path, model_dir: Path | None = None) -> Path:
    """Compute the default xsuite JSON path for a MAD-X sequence file."""
    if model_dir is None:
        model_dir = PROJECT_ROOT / "models"
    xsuite_dir = model_dir / "xsuite"
    xsuite_dir.mkdir(exist_ok=True)
    return xsuite_dir / f"{sequence_file.stem}.json"


def create_xsuite_environment(
    beam: int | None = None,
    sequence_file: Path | None = None,
    beam_energy: float = BEAM_ENERGY,
    seq_name: str | None = None,
    rerun_madx: bool = False,
    json_file: Path | None = None,
) -> xt.Environment:
    """Run MAD-X (if needed) and load an xsuite environment."""
    if json_file is None:
        raise ValueError("json_file parameter is required and cannot be None")

    if sequence_file is None:
        if beam is None:
            raise ValueError("Either beam or sequence_file must be provided.")
        sequence_file = get_lhc_file_path(beam)

    if seq_name is None:
        seq_name = sequence_file.stem

    needs_regen = rerun_madx or not json_file.exists()
    if not needs_regen:
        needs_regen = sequence_file.stat().st_mtime > json_file.stat().st_mtime

    if needs_regen:
        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")
        env: xt.Environment = load_madx_lattice(file=sequence_file)
        env.to_json(json_file)
        logger.info(f"xsuite environment saved to {json_file}")
    else:
        logger.info(f"Loading existing xsuite environment from {json_file}")
        env = xt.Environment.from_json(json_file)  # type: ignore[attr-defined]

    env[seq_name].particle_ref = xt.Particles(
        mass=xp.PROTON_MASS_EV,
        energy0=beam_energy * 1e9,
    )
    return env


def _set_corrector_strengths(env: xt.Environment, corrector_table: tfs.TfsDataFrame) -> None:
    logger.debug(f"Applying corrector strengths to {len(corrector_table)} elements")
    for _, row in corrector_table.iterrows():
        mag_name = row["ename"].lower()
        assert (
            np.isclose(env[mag_name].knl[0], -row["hkick_old"], atol=1e-10)  # ty:ignore[not-subscriptable]
            and np.isclose(env[mag_name].ksl[0], row["vkick_old"], atol=1e-10)  # ty:ignore[not-subscriptable]
        ), (
            f"Corrector {row['ename']} has different initial strengths in environment: "
            f"knl_env={env[mag_name].knl[0]}, expected={-row['hkick_old']}, "  # ty:ignore[not-subscriptable]
            f"ksl_env={env[mag_name].ksl[0]}, expected={row['vkick_old']}"  # ty:ignore[not-subscriptable]
        )
        knl_str: float = -row["hkick"] if abs(row["hkick"]) > 1e-10 else 0.0
        ksl_str: float = row["vkick"] if abs(row["vkick"]) > 1e-10 else 0.0
        env.set(mag_name, knl=[knl_str], ksl=[ksl_str])  # type: ignore[attr-defined]


def initialise_env(
    matched_tunes: dict[str, float],
    magnet_strengths: dict[str, float],
    corrector_table: tfs.TfsDataFrame,
    beam: int | None = None,
    sequence_file: Path | None = None,
    beam_energy: float = BEAM_ENERGY,
    seq_name: str | None = None,
    json_file: Path | None = None,
) -> xt.Environment:
    """Initialise xsuite environment with tune knobs, magnets, and correctors."""
    if json_file is None:
        raise ValueError("json_file parameter is required and cannot be None")

    base_env = create_xsuite_environment(
        beam=beam,
        sequence_file=sequence_file,
        beam_energy=beam_energy,
        seq_name=seq_name,
        rerun_madx=False,
        json_file=json_file,
    )

    for k, v in matched_tunes.items():
        knob = k[:3] + "." + k[4:]
        base_env.set(knob, v)  # type: ignore[attr-defined]
        import re

        if not re.match(r"dq[xy]\.b[12]_op", knob):
            raise ValueError(f"Unexpected tune knob name format: {knob}")

    for str_name, strength in magnet_strengths.items():
        magnet_name, var = str_name.rsplit(".", 1)
        logger.debug(f"Setting {magnet_name.lower()} {var} to {strength}")
        base_env.set(magnet_name.lower(), **{var: strength})  # type: ignore[attr-defined]

    _set_corrector_strengths(base_env, corrector_table)
    return base_env
