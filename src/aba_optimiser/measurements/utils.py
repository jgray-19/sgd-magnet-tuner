"""Utilities for measurements."""

from __future__ import annotations

import ast
import configparser
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def find_all_bad_bpms_from_analysis(optics_folder: Path) -> set[str]:
    """Find all bad BPMs from the analysis ini file and associated measurement folders.

    Args:
        optics_folder: Path to the optics/analysis directory

    Returns:
        Set of bad BPM names
    """
    # Find the analysis ini file
    ini_files = list(optics_folder.glob("analysis.*.ini"))
    if not ini_files:
        LOGGER.warning(
            f"No analysis.*.ini file found in {optics_folder}, using empty bad BPMs list"
        )
        return set()

    ini_file = ini_files[0]  # Take the first one if multiple
    LOGGER.info(f"Found analysis ini file: {ini_file}")

    # Parse the ini file
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Get the files list
    if "DEFAULT" not in config or "files" not in config["DEFAULT"]:
        LOGGER.warning(f"No 'files' entry in {ini_file}, using empty bad BPMs list")
        return set()

    # Parse the files list using ast.literal_eval (safe evaluation of Python literals)
    files_str = config["DEFAULT"]["files"]
    try:
        file_paths = ast.literal_eval(files_str)
    except (ValueError, SyntaxError) as e:
        LOGGER.warning(f"Failed to parse files list in {ini_file}: {e}, using empty bad BPMs list")
        return set()

    # Extract unique folders from file paths
    measurement_folders = set()
    for file_path in file_paths:
        folder = Path(file_path).parent
        measurement_folders.add(folder)

    LOGGER.info(f"Found {len(measurement_folders)} unique measurement folders")

    # Find bad BPMs in each folder
    all_bad_bpms = set()
    for folder in measurement_folders:
        if folder.exists():
            folder_bad_bpms = find_all_bad_bpms(folder)
            all_bad_bpms.update(folder_bad_bpms)
            LOGGER.debug(f"Found {len(folder_bad_bpms)} bad BPMs in {folder}")
        else:
            LOGGER.warning(f"Measurement folder does not exist: {folder}")

    LOGGER.info(f"Total unique bad BPMs found: {len(all_bad_bpms)}")
    return all_bad_bpms


def find_all_bad_bpms(measurement_dir: Path) -> set[str]:
    """Find all bad BPMs from .bad_bpms_x|y files in the measurement directory."""
    bad_bpms = set()
    for filepath in measurement_dir.glob("*.bad_bpms_*"):
        with filepath.open("r") as file:
            bad_bpms.update(line.split(" ")[0] for line in file.readlines())
    return bad_bpms
