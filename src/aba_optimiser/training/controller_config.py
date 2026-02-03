"""Configuration dataclasses for controller initialization.

These dataclasses group related parameters to reduce the number of
individual arguments passed to controller constructors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceConfig:
    """Configuration for MAD-NG sequence and beam parameters.

    Attributes:
        magnet_range: Range of magnets to optimize (e.g., "BPM.9R2.B1/BPM.9L3.B1")
        first_bpm: First BPM in the sequence (None for auto-detection)
        bad_bpms: List of BPM names to exclude from analysis
    """

    magnet_range: str
    first_bpm: str | None = None
    bad_bpms: list[str] | None = None


@dataclass
class MeasurementConfig:
    """Configuration for measurement data files.

    Attributes:
        measurement_files: Measurement data file(s) - can be single file or list
        corrector_files: Corrector strength file(s) - can be single file or list
        tune_knobs_files: Tune knob file(s) - can be single file or list
        machine_deltaps: Machine momentum offset(s) - can be single value or list
        num_tracks: Number of particle tracks per measurement file
        flattop_turns: Number of turns recorded on the flat top
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
