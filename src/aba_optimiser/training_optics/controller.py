"""Controller for optics optimisation (beta functions using quadrupole strengths)."""

from __future__ import annotations

import datetime
import logging
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import tfs
from omc3.optics_measurements.constants import BETA_NAME, DISPERSION_NAME, EXT, ORBIT_NAME
from tensorboardX import SummaryWriter

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.io.utils import get_lhc_file_path, read_knobs
from aba_optimiser.training.configuration_manager import ConfigurationManager
from aba_optimiser.training.optimisation_loop import OptimisationLoop
from aba_optimiser.training.result_manager import ResultManager
from aba_optimiser.workers import OpticsData, OpticsWorker, WorkerConfig

X = "x"
Y = "y"

logger = logging.getLogger(__name__)


class OpticsController:
    """
    Orchestrates optics optimisation using MAD-NG.

    This controller is specialised for beta function optimisation using quadrupole
    strengths. It reads beta function measurements from TFS files and uses a single
    worker to optimise quadrupole strengths.
    """

    def __init__(
        self,
        sequence_file_path: str | Path,
        optics_folder: str | Path,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
        magnet_range: str,
        optimiser_config: OptimiserConfig,
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
            sequence_file_path (str | Path): Path to the sequence file
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_start_points (list[str]): Start BPMs for each range
            bpm_end_points (list[str]): End BPMs for each range
            magnet_range (str): Magnet range specification
            optimiser_config (OptimiserConfig): Gradient descent optimiser configuration
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
        logger.info("Optimising quadrupoles for beta functions")
        self.optimiser_config = optimiser_config
        simulation_config = SimulationConfig(
            tracks_per_worker=1,
            num_workers=1,
            num_batches=1,
            optimise_energy=False,
            optimise_quadrupoles=True,
            optimise_bends=False,
        )
        self.simulation_config = simulation_config

        # Load beta function measurements from optics folder
        optics_path = Path(optics_folder)

        # Remove all the bad bpms from the start and end points
        if bad_bpms is not None:
            for bpm in bad_bpms:
                if bpm in bpm_start_points:
                    bpm_start_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from start points")
                if bpm in bpm_end_points:
                    bpm_end_points.remove(bpm)
                    logger.warning(f"Removed bad BPM {bpm} from end points")

        # Initialise managers
        self.config_manager = ConfigurationManager(
            simulation_config, magnet_range, bpm_start_points, bpm_end_points
        )
        self.config_manager.setup_mad_interface(
            sequence_file_path,
            first_bpm,
            bad_bpms,
            seq_name,
            beam_energy,
        )
        self.config_manager.check_worker_and_bpms()

        template_config = WorkerConfig(
            start_bpm="TEMP",
            end_bpm="TEMP",
            magnet_range=magnet_range,
            sequence_file_path=sequence_file_path,
            corrector_strengths=corrector_file,
            tune_knobs_file=tune_knobs_file,
            beam_energy=beam_energy,
            sdir=np.nan,
            bad_bpms=bad_bpms,
            seq_name=seq_name,
        )

        self.worker_payloads = create_worker_payloads(
            optics_path,
            self.config_manager.bpm_ranges,
            bad_bpms,
            template_config,
        )

        # Handle true strengths
        if true_strengths is None:
            true_strengths = {}
        elif isinstance(true_strengths, Path):
            true_strengths = read_knobs(true_strengths)
        elif isinstance(true_strengths, dict):
            true_strengths = true_strengths.copy()

        # Initialise knobs
        self.initial_knobs, self.filtered_true_strengths = (
            self.config_manager.initialise_knob_strengths(true_strengths, initial_knob_strengths)
        )

        if not true_strengths:
            # Set the true strengths to the initial strengths if not provided
            self.filtered_true_strengths = self.initial_knobs.copy()

        # Setup optimisation
        self.optimisation_loop = OptimisationLoop(
            self.config_manager.initial_strengths,
            self.config_manager.knob_names,
            self.filtered_true_strengths,
            self.optimiser_config,
            simulation_config,
        )

        self.result_manager = ResultManager(
            self.config_manager.knob_names,
            self.config_manager.elem_spos,
            show_plots=show_plots,
            simulation_config=simulation_config,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the optimisation process using a single optics worker."""
        writer = self._setup_logging()
        parent_conns = []
        workers = []
        for worker_id, (config, data) in enumerate(self.worker_payloads):
            parent, child = mp.Pipe()
            w = OpticsWorker(child, worker_id, data, config, self.simulation_config)
            w.start()
            parent_conns.append(parent)
            workers.append(w)
            parent.send(None)

        self.final_knobs = self.optimisation_loop.run_optimisation(
            self.initial_knobs,
            parent_conns,
            writer,
            run_start=time.time(),
            total_turns=1,
        )
        uncertainties = np.zeros(len(self.initial_knobs))

        # Terminate workers
        logger.info("Terminating workers...")
        for conn in parent_conns:
            conn.send((None, None))
        for w in workers:
            w.join()

        # Generate plots if requested
        self.result_manager.generate_plots(
            self.final_knobs,
            self.config_manager.initial_strengths,
            self.filtered_true_strengths,
            uncertainties,
        )

        writer.close()
        return self.final_knobs, dict(zip(self.final_knobs.keys(), uncertainties))

    def _setup_logging(self) -> SummaryWriter:
        """Sets up TensorBoard logging."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return SummaryWriter(log_dir=f"runs/{ts}_optics_opt")


def create_worker_payloads(
    optics_dir: Path,
    bpms_ranges: list[str],
    bad_bpms: list[str] | None,
    template_config: WorkerConfig,
) -> list[tuple[WorkerConfig, OpticsData]]:
    """Create worker payloads for optics optimisation."""

    logger.info(f"Loading beta measurements from {optics_dir}")
    beta_x_file = optics_dir / f"{BETA_NAME}{X}{EXT}"
    beta_y_file = optics_dir / f"{BETA_NAME}{Y}{EXT}"
    disp_x_file = optics_dir / f"{DISPERSION_NAME}{X}{EXT}"
    disp_y_file = optics_dir / f"{DISPERSION_NAME}{Y}{EXT}"
    orbit_x_file = optics_dir / f"{ORBIT_NAME}{X}{EXT}"
    orbit_y_file = optics_dir / f"{ORBIT_NAME}{Y}{EXT}"

    for file in [beta_x_file, beta_y_file, disp_x_file, disp_y_file, orbit_x_file, orbit_y_file]:
        if not file.exists():
            raise FileNotFoundError(f"Required optics measurement file not found: {file}")

    beta_x = tfs.read(beta_x_file, index="NAME")
    beta_y = tfs.read(beta_y_file, index="NAME")
    disp_x = tfs.read(disp_x_file, index="NAME")
    disp_y = tfs.read(disp_y_file, index="NAME")
    orbit_x = tfs.read(orbit_x_file, index="NAME")
    orbit_y = tfs.read(orbit_y_file, index="NAME")

    common_bpms = (
        set(beta_x.index)
        & set(beta_y.index)
        & set(disp_x.index)
        & set(disp_y.index)
        & set(orbit_x.index)
        & set(orbit_y.index)
    )
    # Sort the BPMs in the order they appear in the beta_x file
    common_bpms = [bpm for bpm in beta_x.index if bpm in common_bpms]
    logger.info(f"Found {len(common_bpms)} common BPMs for optics worker payloads.")

    beta_x = beta_x.loc[common_bpms]
    beta_y = beta_y.loc[common_bpms]
    disp_x = disp_x.loc[common_bpms]
    disp_y = disp_y.loc[common_bpms]
    orbit_x = orbit_x.loc[common_bpms]
    orbit_y = orbit_y.loc[common_bpms]
    for df in [beta_x, beta_y, disp_x, disp_y, orbit_x, orbit_y]:
        if any(bpm in df.index for bpm in (bad_bpms or [])):
            raise ValueError("Bad BPMs found in optics measurement data.")

    worker_payloads = []
    for sdir in [1]:
        for bpm_range in bpms_ranges:
            start_bpm, end_bpm = bpm_range.split("/")
            init_bpm = start_bpm if sdir == 1 else end_bpm

            # Generate initial conditions
            init_cond = {}
            init_cond["beta11"] = beta_x.loc[init_bpm, "BETX"]
            init_cond["beta22"] = beta_y.loc[init_bpm, "BETY"]
            init_cond["alfa11"] = beta_x.loc[init_bpm, "ALFX"]
            init_cond["alfa22"] = beta_y.loc[init_bpm, "ALFY"]
            init_cond["dx"] = disp_x.loc[init_bpm, "DX"]
            init_cond["dpx"] = disp_x.loc[init_bpm, "DPX"]
            init_cond["dy"] = disp_y.loc[init_bpm, "DY"]
            init_cond["dpy"] = disp_y.loc[init_bpm, "DPY"]
            init_cond["x"] = orbit_x.loc[init_bpm, "X"]
            init_cond["y"] = orbit_y.loc[init_bpm, "Y"]

            # Comparison data
            # retrieve the measurements between start and end bpm or the final bpm in the list
            start_pos = beta_x.index.get_loc(start_bpm)
            end_pos = beta_x.index.get_loc(end_bpm) + 1
            beta_x_arr = beta_x["BETX"].to_numpy()
            beta_y_arr = beta_y["BETY"].to_numpy()
            err_beta_x_arr = beta_x["ERRBETX"].to_numpy()
            err_beta_y_arr = beta_y["ERRBETY"].to_numpy()
            if end_pos <= start_pos:
                beta_x_comp = np.concatenate((beta_x_arr[start_pos:], beta_x_arr[:end_pos]))
                beta_y_comp = np.concatenate((beta_y_arr[start_pos:], beta_y_arr[:end_pos]))
                err_beta_x_comp = np.concatenate(
                    (err_beta_x_arr[start_pos:], err_beta_x_arr[:end_pos])
                )
                err_beta_y_comp = np.concatenate(
                    (err_beta_y_arr[start_pos:], err_beta_y_arr[:end_pos])
                )
            else:
                beta_x_comp = beta_x_arr[start_pos:end_pos]
                beta_y_comp = beta_y_arr[start_pos:end_pos]
                err_beta_x_comp = err_beta_x_arr[start_pos:end_pos]
                err_beta_y_comp = err_beta_y_arr[start_pos:end_pos]
            # Reverse the order for negative direction
            if sdir == -1:
                beta_x_comp = beta_x_comp[::-1]
                beta_y_comp = beta_y_comp[::-1]
                err_beta_x_comp = err_beta_x_comp[::-1]
                err_beta_y_comp = err_beta_y_comp[::-1]
            # Now stack everything so is is (nbpms, 2)
            beta_comp = np.vstack((beta_x_comp, beta_y_comp)).T
            err_beta_comp = np.vstack((err_beta_x_comp, err_beta_y_comp)).T

            config = replace(
                template_config,
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                sdir=sdir,
            )

            data = OpticsData(
                beta_comparisons=beta_comp,
                beta_variances=err_beta_comp**2,
                init_coords=init_cond,
            )

            worker_payloads.append((config, data))

    return worker_payloads


class LHCOpticsController(OpticsController):
    """
    LHC-specific optics controller that automatically sets sequence file path
    and first BPM based on beam number.
    """

    def __init__(
        self,
        beam: int,
        optics_folder: str | Path,
        bpm_start_points: list[str],
        bpm_end_points: list[str],
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
    ):
        """
        Initialise the LHC optics controller.

        Args:
            beam (int): The beam number (1 or 2)
            optics_folder (str | Path): Path to folder containing beta_phase_x.tfs and beta_phase_y.tfs
            bpm_start_points (list[str]): Start BPMs for each range
            bpm_end_points (list[str]): End BPMs for each range
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
        """
        sequence_file_path = get_lhc_file_path(beam) if sequence_path is None else sequence_path
        first_bpm = "BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2"
        seq_name = f"lhcb{beam}"

        super().__init__(
            sequence_file_path=sequence_file_path,
            optics_folder=optics_folder,
            bpm_start_points=bpm_start_points,
            bpm_end_points=bpm_end_points,
            magnet_range=magnet_range,
            optimiser_config=optimiser_config,
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
