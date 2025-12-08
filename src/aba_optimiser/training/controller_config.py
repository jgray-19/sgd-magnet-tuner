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
        sequence_file_path: Path to the MAD-NG sequence file
        magnet_range: Range of magnets to optimize (e.g., "BPM.9R2.B1/BPM.9L3.B1")
        first_bpm: First BPM in the sequence (None for auto-detection)
        seq_name: Sequence name in MAD-NG file (None for auto-detection)
        beam_energy: Beam energy in GeV
        bad_bpms: List of BPM names to exclude from analysis
    """

    sequence_file_path: str | Path
    magnet_range: str
    first_bpm: str | None = None
    seq_name: str | None = None
    beam_energy: float = 6800.0
    bad_bpms: list[str] | None = None

    @classmethod
    def for_lhc_beam(
        cls,
        beam: int,
        magnet_range: str,
        sequence_path: Path | None = None,
        beam_energy: float = 6800.0,
        bad_bpms: list[str] | None = None,
    ) -> SequenceConfig:
        """Create configuration for LHC beam 1 or 2.

        Args:
            beam: Beam number (1 or 2)
            magnet_range: Range of magnets to optimize
            sequence_path: Optional custom sequence file path
            beam_energy: Beam energy in GeV
            bad_bpms: List of bad BPMs to exclude

        Returns:
            SequenceConfig configured for the specified LHC beam
        """
        from aba_optimiser.io.utils import get_lhc_file_path

        sequence_file = sequence_path if sequence_path is not None else get_lhc_file_path(beam)
        first_bpm = "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"
        seq_name = f"lhcb{beam}"

        return cls(
            sequence_file_path=sequence_file,
            magnet_range=magnet_range,
            first_bpm=first_bpm,
            seq_name=seq_name,
            beam_energy=beam_energy,
            bad_bpms=bad_bpms,
        )


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

    measurement_files: list[Path]
    corrector_files: list[Path] | None = None
    tune_knobs_files: list[Path] | None = None
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
        if isinstance(self.machine_deltaps, float):
            self.machine_deltaps = [self.machine_deltaps]


@dataclass
class BPMConfig:
    """BPM range configuration.

    Attributes:
        start_points: List of starting BPM names for each range
        end_points: List of ending BPM names for each range
    """

    start_points: list[str]
    end_points: list[str]

    def validate(self) -> None:
        """Validate BPM configuration."""
        if len(self.start_points) != len(self.end_points):
            raise ValueError(
                f"Number of start_points ({len(self.start_points)}) must match "
                f"end_points ({len(self.end_points)})"
            )


@dataclass
class ControllerConfig:
    """Complete controller configuration combining all parameter groups.

    This is a convenience class that bundles all configuration objects together.
    Using this reduces a 15+ parameter constructor to just a few grouped configs.
    """

    sequence_config: SequenceConfig
    bpm_config: BPMConfig
    measurement_config: MeasurementConfig | None = None
    optics_folder: str | Path | None = None
    initial_knob_strengths: dict[str, float] | None = None
    show_plots: bool = True

    def validate(self) -> None:
        """Validate the complete configuration."""
        self.bpm_config.validate()

        if self.measurement_config is None and self.optics_folder is None:
            raise ValueError("Either measurement_config or optics_folder must be provided")

        if self.measurement_config is not None and self.optics_folder is not None:
            raise ValueError("Cannot specify both measurement_config and optics_folder")
