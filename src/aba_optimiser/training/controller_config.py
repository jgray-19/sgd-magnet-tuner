"""Configuration dataclasses for controller initialization.

These dataclasses group related parameters to reduce the number of
individual arguments passed to controller constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceConfig:
    """Configuration for the sequence segment used during optimisation.

    The fields define the magnet range to expose to MAD-NG, the optional BPM
    used as the sequence start, and any BPMs that should be ignored.
    """

    magnet_range: str
    first_bpm: str | None = None
    bad_bpms: list[str] | None = None


@dataclass
class MeasurementConfig:
    """Measurement inputs and associated machine-state files.

    Single file or scalar inputs are normalised to lists in ``__post_init__``
    so the rest of the training stack can treat multi-file and single-file
    runs uniformly.
    """

    measurement_files: list[Path] | Path
    corrector_files: list[Path | None] | None | Path = None
    tune_knobs_files: list[Path | None] | None | Path = None
    machine_deltaps: list[float] | float = 0.0
    bunches_per_file: int = 3
    flattop_turns: int = 6600

    # create a post-init method to ensure single values are converted to lists
    def __post_init__(self) -> None:
        """Ensure attributes are lists."""
        if isinstance(self.measurement_files, Path):
            self.measurement_files = [self.measurement_files]
        if self.corrector_files is None or isinstance(self.corrector_files, Path):
            self.corrector_files = [self.corrector_files]
        if self.tune_knobs_files is None or isinstance(self.tune_knobs_files, Path):
            self.tune_knobs_files = [self.tune_knobs_files]
        if isinstance(self.machine_deltaps, float | int):
            self.machine_deltaps = [self.machine_deltaps]
