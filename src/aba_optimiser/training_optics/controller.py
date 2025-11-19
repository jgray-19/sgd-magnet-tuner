"""Controller for optics optimisation (beta functions using quadrupole strengths)."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tfs
from tensorboardX import SummaryWriter

from aba_optimiser.io.utils import get_lhc_file_path, read_knobs
from aba_optimiser.mad.optimising_mad_interface import OptimisationMadInterface
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager

if TYPE_CHECKING:
    from aba_optimiser.config import OptSettings

logger = logging.getLogger(__name__)

# omc3 file naming constants
BETA_NAME = "beta_phase_"
EXT = ".tfs"


class OpticsController:
    """
    Orchestrates optics optimisation using MAD-NG.

    This controller is specialized for beta function optimization using quadrupole
    strengths. It reads beta function measurements from TFS files and uses a single
    worker to optimize quadrupole strengths.
    """

    def __init__(
        self,
        opt_settings: OptSettings,
        sequence_file_path: str | Path,
        optics_folder: str | Path,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        first_bpm: str | None = None,
        seq_name: str | None = None,
        beam_energy: float = 6800.0,
    ):
        """
        Initialise the optics controller.

        Args:
            opt_settings (OptSettings): Optimisation settings (must have optimise_quadrupoles=True)
            sequence_file_path (str | Path): Path to the sequence file
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            bad_bpms (list[str] | None, optional): List of bad BPMs
            first_bpm (str | None, optional): First BPM
            seq_name (str | None, optional): Sequence name
            beam_energy (float, optional): Beam energy in GeV. Defaults to 6800.0
        """
        # Validate settings
        if not opt_settings.optimise_quadrupoles:
            raise ValueError(
                "OpticsController requires optimise_quadrupoles=True in opt_settings"
            )
        if opt_settings.optimise_bends or opt_settings.optimise_energy:
            logger.warning("OpticsController only supports quadrupole optimisation.")

        logger.info("Optimising quadrupoles for beta functions")
        self.opt_settings = opt_settings

        # Load beta function measurements from optics folder
        optics_path = Path(optics_folder)
        beta_x_file = optics_path / f"{BETA_NAME}x{EXT}"
        beta_y_file = optics_path / f"{BETA_NAME}y{EXT}"

        if not beta_x_file.exists():
            raise FileNotFoundError(f"Beta X file not found: {beta_x_file}")
        if not beta_y_file.exists():
            raise FileNotFoundError(f"Beta Y file not found: {beta_y_file}")

        logger.info(f"Loading beta measurements from {optics_path}")
        beta_x = tfs.read(beta_x_file, index="NAME")
        beta_y = tfs.read(beta_y_file, index="NAME")

        # Setup MAD interface
        self.mad_iface = OptimisationMadInterface(
            sequence_file_path,
            seq_name=seq_name,
            magnet_range=None,  # Whole ring
            opt_settings=opt_settings,
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            bad_bpms=bad_bpms,
            start_bpm=first_bpm,
            beam_energy=beam_energy,
        )

        # Get BPM names and filter bad ones
        common_bpms = (
            set(beta_x.index) & set(beta_y.index) & set(self.mad_iface.all_bpms)
        )
        if bad_bpms:
            common_bpms = common_bpms - set(bad_bpms)
        self.bpms = sorted(common_bpms)

        # Extract measurements
        self.beta_x_meas = beta_x.loc[self.bpms, "BETX"].values
        self.beta_y_meas = beta_y.loc[self.bpms, "BETY"].values
        self.beta_x_err = beta_x.loc[self.bpms, "ERRBETX"].values
        self.beta_y_err = beta_y.loc[self.bpms, "ERRBETY"].values

        logger.info(f"Loaded beta measurements for {len(self.bpms)} BPMs")

        # Handle true strengths
        if true_strengths is None:
            true_strengths = {}
        elif isinstance(true_strengths, Path):
            true_strengths = read_knobs(true_strengths)
        elif isinstance(true_strengths, dict):
            true_strengths = true_strengths.copy()

        # Initialize knobs
        if initial_knob_strengths is not None:
            self.mad_iface.update_knob_values(initial_knob_strengths)

        initial_strengths = self.mad_iface.receive_knob_values()
        knob_names = self.mad_iface.knob_names
        self.initial_knobs = dict(zip(knob_names, initial_strengths))

        # Filter true strengths
        self.filtered_true_strengths = (
            {k: true_strengths[k] for k in knob_names if k in true_strengths}
            if true_strengths
            else self.initial_knobs.copy()
        )

        # Setup optimisation
        self.optimisation_loop = OptimisationLoop(
            initial_strengths,
            knob_names,
            self.filtered_true_strengths,
            opt_settings,
            optimiser_type=opt_settings.optimiser_type,
        )

        self.result_manager = ResultManager(
            knob_names,
            self.mad_iface.elem_spos,
            show_plots=show_plots,
            opt_settings=opt_settings,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process using a single optics worker."""
        writer = self._setup_logging()

        # TODO: Start single optics worker here
        # The worker will compute beta functions from MAD and compare to measurements
        # For now, this is a placeholder until you implement the optics_worker

        logger.warning(
            "Optics worker not yet implemented. "
            "Optimization loop will not run until optics_worker is created."
        )

        # Placeholder: return initial knobs
        self.final_knobs = self.initial_knobs
        uncertainties = np.zeros(len(self.initial_knobs))

        writer.close()
        logger.info(
            "Optics controller initialized. Waiting for optics_worker implementation."
        )

        return self.final_knobs, dict(zip(self.final_knobs.keys(), uncertainties))

    def _setup_logging(self) -> SummaryWriter:
        """Sets up TensorBoard logging."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{ts}_optics_opt")


class LHCOpticsController(OpticsController):
    """
    LHC-specific optics controller that automatically sets sequence file path
    and first BPM based on beam number.
    """

    def __init__(
        self,
        beam: int,
        opt_settings: OptSettings,
        optics_folder: str | Path,
        sequence_path: Path | None = None,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        beam_energy: float = 6800.0,
    ):
        """
        Initialise the LHC optics controller.

        Args:
            beam (int): The beam number (1 or 2)
            opt_settings (OptSettings): Optimisation settings
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            sequence_path (Path | None, optional): Path to sequence file. If None, uses default
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            bad_bpms (list[str] | None, optional): List of bad BPMs
            beam_energy (float, optional): Beam energy in GeV. Defaults to 6800.0
        """
        if sequence_path is None:
            sequence_file_path = get_lhc_file_path(beam)
        else:
            sequence_file_path = sequence_path
        first_bpm = "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"
        seq_name = f"lhcb{beam}"

        super().__init__(
            opt_settings=opt_settings,
            sequence_file_path=sequence_file_path,
            optics_folder=optics_folder,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            corrector_file=corrector_file,
            tune_knobs_file=tune_knobs_file,
            true_strengths=true_strengths,
            bad_bpms=bad_bpms,
            first_bpm=first_bpm,
            seq_name=seq_name,
            beam_energy=beam_energy,
        )
