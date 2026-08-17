"""Helpers for OMC3 bad-BPM summary files."""

from __future__ import annotations

import ast
import configparser
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def find_all_bad_bpms(measurement_dir: Path) -> set[str]:
    """Find all bad BPMs from ``*.bad_bpms_*`` files in a measurement directory."""
    bad_bpms: set[str] = set()
    for filepath in measurement_dir.glob("*.bad_bpms_*"):
        with filepath.open("r") as file:
            bad_bpms.update(line.split(" ")[0] for line in file.readlines())
    return bad_bpms


def find_all_bad_bpms_from_analysis(optics_folder: Path) -> set[str]:
    """Find bad BPMs from an OMC3 analysis ini and its measurement folders."""
    ini_files = list(optics_folder.glob("analysis*.ini"))
    if not ini_files:
        LOGGER.warning(
            "No analysis*.ini file found in %s, using empty bad BPMs list", optics_folder
        )
        return set()

    ini_file = ini_files[0]
    LOGGER.info("Found analysis ini file: %s", ini_file)

    config = configparser.ConfigParser()
    config.read(ini_file)

    if "DEFAULT" not in config or "files" not in config["DEFAULT"]:
        LOGGER.warning("No 'files' entry in %s, using empty bad BPMs list", ini_file)
        return set()

    try:
        file_paths = ast.literal_eval(config["DEFAULT"]["files"])
    except (ValueError, SyntaxError) as error:
        LOGGER.warning(
            "Failed to parse files list in %s: %s, using empty bad BPMs list", ini_file, error
        )
        return set()

    all_bad_bpms: set[str] = set()
    measurement_folders = {Path(file_path).parent for file_path in file_paths}
    LOGGER.info("Found %d unique measurement folders", len(measurement_folders))

    for folder in measurement_folders:
        if folder.exists():
            folder_bad_bpms = find_all_bad_bpms(folder)
            all_bad_bpms.update(folder_bad_bpms)
            LOGGER.debug("Found %d bad BPMs in %s", len(folder_bad_bpms), folder)
        else:
            LOGGER.warning("Measurement folder does not exist: %s", folder)

    LOGGER.info("Total unique bad BPMs found: %d", len(all_bad_bpms))
    return all_bad_bpms
