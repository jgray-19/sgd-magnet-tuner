"""Configuration dataclasses for beta function matcher.

These dataclasses group related parameters for the matcher constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pymadng_utils.io.utils import read_knobs


@dataclass
class MatcherConfig:
    """Configuration for beta function matching.

    The matcher takes:
    - A target model twiss file (the betas we want to achieve)
    - Estimated quadrupole strengths from the controller
    - A list of knobs to adjust to minimise the beta difference

    Attributes:
        model_twiss_file: Path to TFS file containing target model twiss (with BETX, BETY)
        estimated_strengths: Either a dict of strengths or Path to JSON file
        knobs_list: List of knob names to adjust for beta correction
        tune_knobs: Dictionary of tune knob names to initial values
        sequence_file_path: Path to MAD-NG sequence file
        magnet_range: Range of magnets for matching (e.g., "BPM.9R2.B1/BPM.9L3.B1")
        beam_energy: Beam energy in GeV
        output_dir: Directory to save matching results
        seq_name: Sequence name in MAD-NG file
    """
        # knob_limits: Optional dict mapping knob names to (min, max) tuples

    model_twiss_file: Path
    estimated_strengths: dict[str, float] | Path
    knobs_list: list[str]
    tune_knobs: dict[str, float]
    sequence_file_path: Path
    magnet_range: str
    seq_name: str
    beam_energy: float = 6800.0
    output_dir: Path | None = None
    # knob_limits: dict[str, tuple[float, float]] | None = None

    def validate(self) -> None:
        """Validate that required files exist."""
        if not Path(self.model_twiss_file).exists():
            raise FileNotFoundError(f"Model twiss file not found: {self.model_twiss_file}")
        if isinstance(self.estimated_strengths, Path) and not self.estimated_strengths.exists():
            raise FileNotFoundError(
                f"Estimated strengths file not found: {self.estimated_strengths}"
            )
        if not Path(self.sequence_file_path).exists():
            raise FileNotFoundError(f"Sequence file not found: {self.sequence_file_path}")

    def get_estimated_strengths(self) -> dict[str, float]:
        """Get estimated strengths, loading from file if necessary.

        Returns:
            Dictionary mapping magnet names to their estimated strengths.
        """
        if isinstance(self.estimated_strengths, dict):
            return self.estimated_strengths.copy()

        return read_knobs(self.estimated_strengths)
