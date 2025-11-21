"""Controller for optics optimisation (beta functions using quadrupole strengths)."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from omc3.optics_measurements.constants import AMP_BETA_NAME, BETA_NAME, DISPERSION_NAME, ORBIT_NAME

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.training.base_controller import BaseController, LHCControllerMixin
from aba_optimiser.training.utils import extract_bpm_range_data, find_common_bpms, load_tfs_files
from aba_optimiser.training.worker_lifecycle import WorkerLifecycleManager
from aba_optimiser.workers import OpticsData, OpticsWorker, WorkerConfig

if TYPE_CHECKING:
    from aba_optimiser.training.controller_config import BPMConfig, SequenceConfig

X = "x"
Y = "y"

logger = logging.getLogger(__name__)


class OpticsController(BaseController):
    """
    Orchestrates optics optimisation using MAD-NG.

    This controller is specialised for beta function optimisation using quadrupole
    strengths. It reads beta function measurements from TFS files and uses a single
    worker to optimise quadrupole strengths.
    """

    def __init__(
        self,
        sequence_config: SequenceConfig,
        optics_folder: str | Path,
        bpm_config: BPMConfig,
        optimiser_config: OptimiserConfig,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
    ):
        """
        Initialise the optics controller.

        Args:
            sequence_config (SequenceConfig): Sequence and beam configuration
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_config (BPMConfig): BPM range configuration
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            use_errors (bool, optional): Whether to use measurement errors in optimisation. Defaults to True
        """
        logger.info("Optimising quadrupoles for beta functions")

        # Create optics-specific simulation config
        simulation_config = SimulationConfig(
            tracks_per_worker=1,
            num_workers=1,
            num_batches=1,
            optimise_energy=False,
            optimise_quadrupoles=True,
            optimise_bends=False,
        )

        # Initialize base controller
        super().__init__(
            optimiser_config=optimiser_config,
            simulation_config=simulation_config,
            sequence_file_path=sequence_config.sequence_file_path,
            magnet_range=sequence_config.magnet_range,
            bpm_start_points=bpm_config.start_points,
            bpm_end_points=bpm_config.end_points,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            bad_bpms=sequence_config.bad_bpms,
            first_bpm=sequence_config.first_bpm,
            seq_name=sequence_config.seq_name,
            beam_energy=sequence_config.beam_energy,
        )

        # Store optics-specific attributes
        self.optics_folder = Path(optics_folder)
        self.corrector_file = corrector_file
        self.tune_knobs_file = tune_knobs_file
        self.use_errors = use_errors

        # Create optics-specific worker payloads
        optics_path = Path(optics_folder)
        template_config = WorkerConfig(
            start_bpm="TEMP",
            end_bpm="TEMP",
            magnet_range=sequence_config.magnet_range,
            sequence_file_path=str(sequence_config.sequence_file_path),
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            beam_energy=sequence_config.beam_energy,
            sdir=np.nan,
            bad_bpms=sequence_config.bad_bpms,
            seq_name=sequence_config.seq_name,
        )

        self.worker_payloads = create_worker_payloads(
            optics_path,
            self.config_manager.bpm_ranges,
            sequence_config.bad_bpms,
            template_config,
            self.use_errors,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process using optics workers."""
        writer = self.setup_logging("optics_opt")

        # Create and start workers
        worker_manager = WorkerLifecycleManager(OpticsWorker)
        worker_manager.create_and_start_workers(
            [(data, config, self.simulation_config) for config, data in self.worker_payloads],
            send_handshake=True,
        )

        # Run optimisation
        self.final_knobs = self.optimisation_loop.run_optimisation(
            self.initial_knobs,
            worker_manager.parent_conns,
            writer,
            run_start=time.time(),
            total_turns=1,
        )

        # Terminate workers
        worker_manager.terminate_workers()

        # Generate results
        uncertainties = np.zeros(len(self.initial_knobs))
        self.result_manager.generate_plots(
            self.final_knobs,
            self.config_manager.initial_strengths,
            self.filtered_true_strengths,
            uncertainties,
        )

        writer.close()
        return self.final_knobs, dict(zip(self.final_knobs.keys(), uncertainties))


def create_worker_payloads(
    optics_dir: Path,
    bpms_ranges: list[str],
    bad_bpms: list[str] | None,
    template_config: WorkerConfig,
    use_errors: bool = True,
) -> list[tuple[WorkerConfig, OpticsData]]:
    """Create worker payloads for optics optimisation."""

    logger.info(f"Loading beta measurements from {optics_dir}")

    # Load all TFS files
    file_specs = {
        "beta_x": (AMP_BETA_NAME, X),
        "beta_y": (AMP_BETA_NAME, Y),
        "alfa_x": (BETA_NAME, X),
        "alfa_y": (BETA_NAME, Y),
        "disp_x": (DISPERSION_NAME, X),
        "disp_y": (DISPERSION_NAME, Y),
        "orbit_x": (ORBIT_NAME, X),
        "orbit_y": (ORBIT_NAME, Y),
    }
    tfs_data = load_tfs_files(optics_dir, file_specs)
    beta_x = tfs_data["beta_x"]
    beta_y = tfs_data["beta_y"]
    alfa_x = tfs_data["alfa_x"]
    alfa_y = tfs_data["alfa_y"]
    disp_x = tfs_data["disp_x"]
    disp_y = tfs_data["disp_y"]
    orbit_x = tfs_data["orbit_x"]
    orbit_y = tfs_data["orbit_y"]

    # Find common BPMs across all files
    common_bpms = find_common_bpms(beta_x, beta_y, disp_x, disp_y, orbit_x, orbit_y, alfa_x, alfa_y)
    logger.info(f"Found {len(common_bpms)} common BPMs for optics worker payloads.")

    beta_x = beta_x.loc[common_bpms]
    beta_y = beta_y.loc[common_bpms]
    alfa_x = alfa_x.loc[common_bpms]
    alfa_y = alfa_y.loc[common_bpms]
    disp_x = disp_x.loc[common_bpms]
    disp_y = disp_y.loc[common_bpms]
    orbit_x = orbit_x.loc[common_bpms]
    orbit_y = orbit_y.loc[common_bpms]
    for df in [beta_x, beta_y, disp_x, disp_y, orbit_x, orbit_y]:
        if any(bpm in df.index for bpm in (bad_bpms or [])):
            raise ValueError("Bad BPMs found in optics measurement data.")

    worker_payloads = []
    for sdir in [1, -1]:
        for bpm_range in bpms_ranges:
            start_bpm, end_bpm = bpm_range.split("/")
            init_bpm = start_bpm if sdir == 1 else end_bpm

            # Generate initial conditions
            init_cond = {}
            init_cond["beta11"] = beta_x.loc[init_bpm, "BETX"]
            init_cond["beta22"] = beta_y.loc[init_bpm, "BETY"]
            init_cond["alfa11"] = alfa_x.loc[init_bpm, "ALFX"]
            init_cond["alfa22"] = alfa_y.loc[init_bpm, "ALFY"]
            init_cond["dx"] = disp_x.loc[init_bpm, "DX"]
            init_cond["dpx"] = disp_x.loc[init_bpm, "DPX"]
            init_cond["dy"] = disp_y.loc[init_bpm, "DY"]
            init_cond["dpy"] = disp_y.loc[init_bpm, "DPY"]
            init_cond["x"] = orbit_x.loc[init_bpm, "X"]
            init_cond["y"] = orbit_y.loc[init_bpm, "Y"]

            # Extract comparison data using utility function
            # We need to extract from both beta_x and beta_y separately and then combine
            betx_data = extract_bpm_range_data(beta_x, start_bpm, end_bpm, sdir, ["BETX"])
            bety_data = extract_bpm_range_data(beta_y, start_bpm, end_bpm, sdir, ["BETY"])
            # alfx_data = extract_bpm_range_data(beta_x, start_bpm, end_bpm, sdir, ["ALFX"])
            # alfy_data = extract_bpm_range_data(beta_y, start_bpm, end_bpm, sdir, ["ALFY"])
            comp = np.hstack([betx_data, bety_data])  # , alfx_data, alfy_data])

            errbetx_data = extract_bpm_range_data(beta_x, start_bpm, end_bpm, sdir, ["ERRBETX"])
            errbety_data = extract_bpm_range_data(beta_y, start_bpm, end_bpm, sdir, ["ERRBETY"])
            # erralfx_data = extract_bpm_range_data(beta_x, start_bpm, end_bpm, sdir, ["ERRALFX"])
            # erralfy_data = extract_bpm_range_data(beta_y, start_bpm, end_bpm, sdir, ["ERRALFY"])
            err_comp = np.hstack([errbetx_data, errbety_data])  # , erralfx_data, erralfy_data])

            if not use_errors or np.all(err_comp == 0):
                logger.warning(
                    f"No valid errors found for BPM range {start_bpm} to {end_bpm}. "
                    "Using uniform errors of 1.0."
                )
                err_comp = np.ones_like(err_comp)

            config = replace(
                template_config,
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                sdir=sdir,
            )

            data = OpticsData(
                comparisons=comp,
                variances=err_comp**2,
                init_coords=init_cond,
            )

            worker_payloads.append((config, data))

    return worker_payloads


class LHCOpticsController(LHCControllerMixin, OpticsController):
    """
    LHC-specific optics controller that automatically sets sequence file path
    and first BPM based on beam number.
    """

    def __init__(
        self,
        beam: int,
        optics_folder: str | Path,
        bpm_config: BPMConfig,
        magnet_range: str,
        optimiser_config: OptimiserConfig,
        sequence_path: Path | None = None,
        show_plots: bool = True,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_file: Path | None = None,
        tune_knobs_file: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        bad_bpms: list[str] | None = None,
        beam_energy: float = 6800.0,
        use_errors: bool = True,
    ):
        """
        Initialise the LHC optics controller.

        Args:
            beam (int): The beam number (1 or 2)
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_config (BPMConfig): BPM range configuration
            magnet_range (str): Magnet range specification
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration
            sequence_path (Path | None, optional): Path to sequence file. If None, uses default
            show_plots (bool, optional): Whether to show plots. Defaults to True
            initial_knob_strengths (dict[str, float] | None, optional): Initial knob strengths
            corrector_file (Path | None, optional): Corrector strength file
            tune_knobs_file (Path | None, optional): Tune knob file
            true_strengths (Path | dict[str, float] | None, optional): True strengths
            bad_bpms (list[str] | None, optional): List of bad BPMs
            beam_energy (float, optional): Beam energy in GeV. Defaults to 6800.0
            use_errors (bool, optional): Whether to use measurement errors in optimisation. Defaults to True
        """
        # Create SequenceConfig using mixin helper
        sequence_config = self.create_sequence_config(
            beam=beam,
            magnet_range=magnet_range,
            sequence_path=sequence_path,
            bad_bpms=bad_bpms,
            beam_energy=beam_energy,
        )

        super().__init__(
            sequence_config=sequence_config,
            optics_folder=optics_folder,
            bpm_config=bpm_config,
            optimiser_config=optimiser_config,
            show_plots=show_plots,
            initial_knob_strengths=initial_knob_strengths,
            corrector_file=corrector_file,
            tune_knobs_file=tune_knobs_file,
            true_strengths=true_strengths,
            use_errors=use_errors,
        )
