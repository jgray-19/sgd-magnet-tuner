"""Helpers for rebuilding optics analysis without AC-dipole compensation."""

from __future__ import annotations

import ast
import configparser
import logging
import shutil
from pathlib import Path

from omc3.hole_in_one import hole_in_one_entrypoint

LOGGER = logging.getLogger(__name__)

NO_COMPENSATION_SUFFIX = "_no_compensation"

def get_uncompensated_analysis_dir(analysis_dir: str | Path) -> Path:
    """Return the sibling directory used for optics analysis without compensation."""
    analysis_dir = Path(analysis_dir)
    return analysis_dir.with_name(f"{analysis_dir.name}{NO_COMPENSATION_SUFFIX}")


def rerun_optics_analysis_without_compensation(
    source_analysis_dir: str | Path,
    *,
    model_dir: str | Path | None = None,
    beam: int | None = None,
    target_analysis_dir: str | Path | None = None,
    force: bool = False,
) -> tuple[Path, list[str]]:
    """Rebuild optics outputs from the latest saved analysis ini, changing only compensation."""
    source_analysis_dir = Path(source_analysis_dir)
    if target_analysis_dir is None:
        target_analysis_dir = get_uncompensated_analysis_dir(source_analysis_dir)
    target_analysis_dir = Path(target_analysis_dir)

    if force and target_analysis_dir.exists():
        shutil.rmtree(target_analysis_dir)
    target_analysis_dir.mkdir(parents=True, exist_ok=True)

    ini_file = _find_latest_analysis_ini(source_analysis_dir)
    run_config = _load_analysis_ini(ini_file)
    run_config["outputdir"] = target_analysis_dir
    run_config["compensation"] = "none"
    run_config = _remove_empty_defaulted_kwargs(run_config)

    LOGGER.info(
        "Running optics analysis without compensation from %s into %s",
        ini_file,
        target_analysis_dir,
    )
    hole_in_one_entrypoint(**run_config)

    analysed_files = _resolve_analysed_files(
        target_analysis_dir,
        configured_files=run_config.get("files"),
        source_analysis_dir=source_analysis_dir,
    )
    bad_bpms = _collect_bad_bpms(analysed_files)
    LOGGER.info(
        "Completed uncompensated optics analysis with %d bad BPMs in %s",
        len(bad_bpms),
        target_analysis_dir,
    )
    return target_analysis_dir, bad_bpms


def _find_latest_analysis_ini(source_analysis_dir: Path) -> Path:
    ini_files = sorted(
        source_analysis_dir.glob("analysis*.ini"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not ini_files:
        raise FileNotFoundError(f"No analysis*.ini file found in {source_analysis_dir}")
    return ini_files[-1]


def _load_analysis_ini(ini_file: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(ini_file)
    defaults = dict(config.defaults())
    if not defaults:
        raise ValueError(f"No DEFAULT section found in {ini_file}")
    return {key: _parse_ini_value(value) for key, value in defaults.items()}


def _remove_empty_defaulted_kwargs(run_config: dict) -> dict:
    """Drop empty tune placeholders so omc3 falls back to its own defaults."""
    new_config = {}
    for key, value in run_config.items():
        if value != "":
            new_config[key] = value
    return new_config


def _parse_ini_value(value: str):
    value = value.strip()
    if value == "":
        return ""
    try:
        node = ast.parse(value, mode="eval").body
    except SyntaxError:
        return value
    return _eval_ini_ast(node)


def _eval_ini_ast(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_ini_ast(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ini_ast(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _eval_ini_ast(key): _eval_ini_ast(value) for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ini_ast(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
        node.func.id in {"Path", "PosixPath", "WindowsPath"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return Path(_eval_ini_ast(node.args[0]))
    raise ValueError(f"Unsupported ini expression: {ast.unparse(node)}")


def _resolve_analysed_files(
    analysis_dir: Path,
    *,
    configured_files: list[Path | str] | None = None,
    source_analysis_dir: Path | None = None,
) -> list[Path]:
    analysed_files = [
        created_file.with_suffix("")
        for created_file in sorted((analysis_dir / "lin_files").glob("*_bunchID*.linx"))
    ]
    if analysed_files:
        return analysed_files

    # Fallback for cached analyses that do not use the bunchID suffix in file names.
    analysed_files = [
        created_file.with_suffix("")
        for created_file in sorted((analysis_dir / "lin_files").glob("*.linx"))
    ]
    if analysed_files:
        return analysed_files

    # Optics-only reruns reuse existing analysed stems from the saved ini instead of
    # regenerating lin_files under the new output directory.
    if configured_files:
        return _normalise_analysed_file_paths(configured_files, source_analysis_dir)

    raise FileNotFoundError(
        f"No analysed linx files found in {analysis_dir / 'lin_files'} and no input files were configured"
    )


def _normalise_analysed_file_paths(
    configured_files: list[Path | str],
    source_analysis_dir: Path | None,
) -> list[Path]:
    resolved_files: list[Path] = []
    base_dir = source_analysis_dir.parent if source_analysis_dir is not None else None
    for configured_file in configured_files:
        file_path = Path(configured_file)
        if not file_path.is_absolute() and base_dir is not None:
            file_path = base_dir / file_path
        if file_path.suffix in {".linx", ".liny"}:
            file_path = file_path.with_suffix("")
        resolved_files.append(file_path)
    return resolved_files


def _collect_bad_bpms(analysed_files: list[Path]) -> list[str]:
    bad_bpms: set[str] = set()
    for file in analysed_files:
        for suffix in (".bad_bpms_x", ".bad_bpms_y"):
            summary_file = file.parent / f"{file.name}{suffix}"
            if summary_file.exists():
                with summary_file.open("r") as handle:
                    bad_bpms.update(line.split(" ")[0] for line in handle.readlines())
    return sorted(bad_bpms)
