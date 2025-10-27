"""
Noise Effect Analysis for SGD Magnet Tuner

This script analyzes the effects of various noise sources on beam tracking:
- Quadrupole strength errors
- Initial condition perturbations
- Combined effects

The analysis uses parallel processing to compute multiple samples and
generates comparison plots of the standard deviations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from pymadng import MAD
from tqdm.contrib.concurrent import process_map

from aba_optimiser.config import (
    REL_K1_STD_DEV,
    SEQ_NAME,
    SEQUENCE_FILE,
)
from aba_optimiser.momentum_recon.transverse import calculate_pz
from scripts.plot_functions import (
    plot_error_bars_bpm_range,
    plot_std_log_comparison,
    show_plots,
)

if TYPE_CHECKING:
    import tfs

# Configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
rng = np.random.default_rng()


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for the noise effect simulation."""

    nturns: int = 10
    nangles: int = 5  # default number of initial angles
    num_error_samples: int = 70
    action0: float = 6e-9  # Initial action
    # bpm_range: str = "BPM.13R3.B1/BPM.12L4.B1" # Arc34
    # bpm_range: str = "BPMYB.4L2.B1/BPMYB.5L2.B1"
    bpm_range: str = "BPMYB.5L2.B1/BPMR.6L2.B1"
    # bpm_range: str = "BPMR.6L2.B1/BPM.7L2.B1"
    beam_energy: float = 6800


class TrackingResult(NamedTuple):
    """Container for tracking simulation results."""

    x_positions: pd.Series
    y_positions: pd.Series


class NoiseAnalysisResults(NamedTuple):
    """Container for noise analysis results."""

    std_x: np.ndarray
    std_y: np.ndarray

    def __str__(self):
        return f"NoiseAnalysisResults(std_x={self.std_x}, std_y={self.std_y})"


def build_track_command(
    df_twiss: tfs.TfsDataFrame,
    action: float,
    angle: float,
    config: SimulationConfig,
    npbms: int,
    initial_coords: tuple[float, float, float, float] | None = None,
) -> str:
    """
    Construct the MAD-NG track command string with given initial conditions and settings.

    Args:
        df_twiss: Twiss dataframe containing optical functions
        action: Initial action value
        angle: Initial angle for action
        nturns: Number of tracking turns
        config: Simulation configuration parameters

    Returns:
        MAD-NG track command string
    """
    # Apply random perturbations if requested
    # If explicit initial coordinates are provided, use them directly.
    # initial_coords is (x0, px0, y0, py0)
    if initial_coords is not None:
        x_init, px_init, y_init, py_init = initial_coords
        return (
            f"trk, mflw = track{{sequence=MADX['{SEQ_NAME}'], "
            f"X0={{x={x_init:.16e}, px={px_init:.16e}, y={y_init:.16e}, py={py_init:.16e}, t=0, pt=0}}, "
            f"nturn={config.nturns}}}"
        )
    start_bpm = config.bpm_range.split("/")[0]

    # Get optical functions at starting BPM
    beta11 = df_twiss.loc[start_bpm, "beta11"]
    beta22 = df_twiss.loc[start_bpm, "beta22"]
    alfa11 = df_twiss.loc[start_bpm, "alfa11"]
    alfa22 = df_twiss.loc[start_bpm, "alfa22"]

    # Calculate initial coordinates from action and angle
    cos0 = np.cos(angle)
    sin0 = np.sin(angle)
    x0 = np.sqrt(action * beta11) * cos0
    px0 = -np.sqrt(action / beta11) * (sin0 + alfa11 * cos0)
    y0 = np.sqrt(action * beta22) * cos0
    py0 = -np.sqrt(action / beta22) * (sin0 + alfa22 * cos0)

    return (
        f"trk, _ = track{{sequence=MADX['{SEQ_NAME}'], "
        f"X0={{x={x0:.16e}, px={px0:.16e}, y={y0:.16e}, py={py0:.16e}, t=0, pt=0}}, "
        f"nturn={config.nturns}}}\n"
    )


class MADSimulator:
    """Handles MAD-NG simulation setup and operations."""

    def __init__(
        self,
        config: SimulationConfig,
        matched_tunes: dict[str, float] | None = None,
        df_twiss: pd.DataFrame | None = None,
    ):
        self.matched_tunes = matched_tunes
        self.df_twiss = df_twiss
        self.config = config
        self.mad = self._setup_mad()
        self._match_tunes()
        self._get_quad_names()
        self.nbpms = len(self.df_twiss.index)

    def _setup_mad(self) -> MAD:
        """
        Initialize a MAD-NG process with sequence loaded, beam set, and BPMs selected.

        Args:
            config: Simulation configuration parameters

        Returns:
            Configured MAD instance
        """
        mad = MAD(stdout="mad_stdout.log", redirect_stderr=True, debug=True)
        init_string = f"""
MADX:load("{SEQUENCE_FILE.absolute()}")
local {SEQ_NAME} in MADX
{SEQ_NAME}.beam = beam {{ particle = 'proton', energy = {self.config.beam_energy} }}
local observed in MAD.element.flags
{SEQ_NAME}:deselect(observed)
{SEQ_NAME}:select(observed, {{pattern="BPM"}})
"""
        # Install marker at start BPM and cycle sequence
        bpm_name = self.config.bpm_range.split("/")[0]
        marker_name = mad.quote_strings(f"{bpm_name}_marker")
        install_string = f"""
{SEQ_NAME}:install{{
MAD.element.marker {marker_name} {{ at=-1e-10, from="{bpm_name}" }} ! 1e-10 is too small for a drift but ensures we cycle to before the BPM
}}
{SEQ_NAME}:cycle({marker_name})
"""
        mad.send(init_string + install_string)
        return mad

    def _match_tunes(self):
        """
        Match the working point tunes and return the matched knob values and Twiss at start.

        Returns:
            Tuple of (matched tune parameters, Twiss dataframe)
        """
        if self.matched_tunes is not None and self.df_twiss is not None:
            return

        self.mad["SEQ_NAME"] = SEQ_NAME
        self.mad["knob_range"] = self.config.bpm_range

        # Desired fractional tunes
        tunes = [0.28, 0.31]
        self.mad["result"] = self.mad.match(
            command=r"\ -> twiss{sequence=MADX[SEQ_NAME]}",
            variables=[
                {"var": "'MADX.dqx_b1_op'", "name": "'dqx_b1_op'"},
                {"var": "'MADX.dqy_b1_op'", "name": "'dqy_b1_op'"},
            ],
            equalities=[
                {"expr": f"\\t -> math.abs(t.q1)-(62+{tunes[0]})", "name": "'q1'"},
                {"expr": f"\\t -> math.abs(t.q2)-(60+{tunes[1]})", "name": "'q2'"},
            ],
            objective={"fmin": 1e-18},
            info=2,
        )

        self.matched_tunes = {
            key: self.mad[f"MADX['{key}']"] for key in ("dqx_b1_op", "dqy_b1_op")
        }

        # Get Twiss at start of sequence
        tw = self.mad.twiss(sequence=self.mad.MADX[SEQ_NAME], observe=1)[0]
        self.df_twiss: tfs.TfsDataFrame = tw.to_df()
        self.df_twiss.set_index("name", inplace=True)

    def _get_quad_names(self):
        self.mad.send(f"""
local knob_range = "{self.config.bpm_range}"
local elem_names = {{}}
for i, elm, s, ds in MADX.{SEQ_NAME}:iter(knob_range) do
    if elm.k1 and elm.k1 ~= 0 and elm.name:match("MQ%.") then
        -- Check if the element is a main quadrupole
        table.insert(elem_names, elm.name)
    end
end
py:send(elem_names, true)
""")
        self.quad_names = self.mad.recv()

    def get_last_turn_data(self) -> pd.DataFrame:
        """
        Retrieve the last-turn BPM data ordered by the longitudinal position 's'.

        Args:
            mad: MAD instance with tracking results

        Returns:
            DataFrame with last turn BPM data ordered by s
        """
        start, end = self.config.bpm_range.split("/")

        # Get the final turn's tracking data
        df_last = self.mad["trk"].to_df(columns=["name", "turn", "x", "y"])
        df_last = df_last[df_last["turn"] == df_last["turn"].max()].set_index("name")

        # Get BPM names between start and end based on s position
        bpm_names = self.df_twiss.loc[start:end].index.tolist()

        # Reindex to physical order by s and drop any missing BPMs
        return df_last.reindex(bpm_names).dropna(how="all")

    def get_track_end_positions(
        self,
        angle: float = 0.0,
        action: float | None = None,
        initial_coords: tuple[float, float, float, float] | None = None,
    ) -> TrackingResult:
        """
        Run a single-turn track for the given range and return x and y at the final turn for each BPM.

        Args:
            mad: MAD instance
            matched_tunes: Dictionary of matched tune parameters
            df_twiss: Twiss dataframe
            config: Simulation configuration
            angle: Initial angle
            action: Initial action (defaults to config.action0)
            initial_coords: Initial coordinates (x, px, y, py) for the start BPM

        Returns:
            TrackingResult with x and y positions
        """
        if action is None:
            action = self.config.action0

        # Launch tracking
        trk_command = build_track_command(
            self.df_twiss,
            action,
            angle,
            self.config,
            self.nbpms,
            initial_coords=initial_coords,
        )
        self.mad.send(trk_command)

        # Retrieve only the last-turn data
        df_last = self.get_last_turn_data()
        return TrackingResult(df_last["x"], df_last["y"])


class NoiseAnalyser:
    """Analyzes noise effects in beam tracking."""

    def __init__(
        self,
        config: SimulationConfig,
        matched_tunes: dict[str, float],
        df_twiss: pd.DataFrame,
    ):
        self.config = config
        self.simulator = MADSimulator(config, matched_tunes, df_twiss)
        self.matched_tunes = matched_tunes

        # Precompute arc boundaries and quadrupole names for efficiency
        self.arc_start = df_twiss.loc[config.bpm_range.split("/")[0], "s"]
        self.arc_end = df_twiss.loc[config.bpm_range.split("/")[1], "s"]

        # Use cached quadrupole names if available, otherwise compute and cache
        self.quad_names = self.simulator.quad_names

    def compute_baseline(self, angle: float) -> tuple[float, pd.Series, pd.Series]:
        """
        Compute baseline tracking for a given angle.

        Args:
            angle: Initial angle
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe

        Returns:
            Tuple of (angle, x_positions, y_positions)
        """
        result = self.simulator.get_track_end_positions(angle=angle)
        return angle, result.x_positions, result.y_positions

    def _add_quadrupole_errors(self):
        """
        Apply random errors to the quadrupole strengths.
        """
        modifier_string = ""
        for name in self.quad_names:
            noise = rng.normal(scale=REL_K1_STD_DEV)
            modifier_string += f"""
MADX['{name}'].k1 = MADX['{name}'].k1 + {noise:-.16e} * math.abs(MADX['{name}'].k1)
            """
        self.simulator.mad.send(modifier_string)

    def compute_error_sample(
        self,
        angle: float,
    ) -> TrackingResult:
        """
        Run a single-turn track with random quadrupole errors and return positions.

        Args:
            sample_idx: Sample index (for reproducibility)
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe
            angle: Initial angle

        Returns:
            TrackingResult with x and y positions
        """
        self._add_quadrupole_errors()
        return self.simulator.get_track_end_positions(
            angle=angle,
        )

    def compute_ic_sample(
        self,
        angle: float,
        magnet_errors: bool = False,
    ) -> TrackingResult:
        """
        Run a single-turn track with IC perturbations and return positions.

        Args:
            sample_idx: Sample index
            matched_tunes: Matched tune parameters
            df_twiss: Twiss dataframe
            angle: Initial angle

        Returns:
            TrackingResult with x and y positions
        """

        # 1) Run a 1-turn backwards tracking using MADSimulator: build a track command
        # that requests 2 turns and then read the turn==2 data.
        # We will temporarily send a custom track command and then parse last two turns.

        # Helper: perform a k-turn track and return DataFrame of that tracking
        def run_single_turn() -> pd.DataFrame:
            # Build a command similar to build_track_command but with explicit nturn=k
            trk_cmd = build_track_command(
                self.simulator.df_twiss,
                self.config.action0,
                angle,
                self.config,
                self.simulator.nbpms,
            )
            # Select only the two turn part.
            trk_cmd = trk_cmd.replace(
                f"nturn={self.config.nturns}",
                "nturn=1",
            )
            self.simulator.mad.send(trk_cmd)

            # Retrieve tracking dataframe
            return self.simulator.mad["trk"].to_df(
                columns=["name", "s", "turn", "x", "y"]  # , "px", "py"
            )

        if magnet_errors:
            self._add_quadrupole_errors()

        df = run_single_turn()

        recon_from_prev, recon_from_next = calculate_pz(
            df, inject_noise=True, tws=self.simulator.df_twiss, info=False, rng=rng
        )
        assert (recon_from_next.iloc[0] == recon_from_prev.iloc[0]).all(), (
            "Reconstruction from previous and next turns should match at the start."
        )

        initial_coords = tuple(recon_from_prev[["x", "px", "y", "py"]].iloc[0])

        # Print the error on px and py:
        # print("Error on px:", initial_coords[1] - df.iloc[0]["px"])
        # print("Error on py:", initial_coords[3] - df.iloc[0]["py"])

        # Use the simulator to run the track with explicit initial_coords
        # We call get_track_end_positions which will use build_track_command with initial_coords
        return self.simulator.get_track_end_positions(
            angle=angle, initial_coords=initial_coords
        )

    def compute_standard_deviations(
        self,
        baseline: dict[float, tuple[pd.Series, pd.Series]],
        results: dict[float, list[TrackingResult]],
        bpm_names: list[str],
    ) -> NoiseAnalysisResults:
        """
        Compute standard deviations from baseline for all angles and samples.

        Args:
            baseline: Baseline results for each angle
            results: Noise results for each angle
            bpm_names: Ordered list of BPM names

        Returns:
            NoiseAnalysisResults with standard deviations
        """
        # Flatten all differences across angles and samples
        diffs_x = []
        diffs_y = []

        for angle in baseline:
            bx, by = baseline[angle]
            for result in results[angle]:
                diffs_x.append(
                    (result.x_positions[bpm_names] - bx[bpm_names]).to_numpy()
                )
                diffs_y.append(
                    (result.y_positions[bpm_names] - by[bpm_names]).to_numpy()
                )

        diffs_x = np.stack(diffs_x, axis=0)
        diffs_y = np.stack(diffs_y, axis=0)

        return NoiseAnalysisResults(
            std_x=np.std(diffs_x, axis=0), std_y=np.std(diffs_y, axis=0)
        )


# Multiprocessing wrapper functions
def _compute_baseline_wrapper(args):
    """Wrapper for multiprocessing baseline computation."""
    angle, matched_tunes, df_twiss, config = args
    analyzer = NoiseAnalyser(config, matched_tunes, df_twiss)
    return analyzer.compute_baseline(angle)


def _compute_error_sample_wrapper(args):
    """Wrapper for multiprocessing error sample computation."""
    sample_idx, matched_tunes, df_twiss, angle, config = args
    analyzer = NoiseAnalyser(config, matched_tunes, df_twiss)
    result = analyzer.compute_error_sample(angle)
    return (sample_idx, angle), result


def _compute_ic_sample_wrapper(args):
    """Wrapper for multiprocessing IC sample computation."""
    sample_idx, matched_tunes, df_twiss, angle, config, magnet_errs = args
    analyzer = NoiseAnalyser(config, matched_tunes, df_twiss)
    result = analyzer.compute_ic_sample(angle, magnet_errs)
    return (sample_idx, angle), result


def save_error_bar_plot(
    s_positions: np.ndarray,
    baseline_x: pd.Series,
    baseline_y: pd.Series,
    std_x: np.ndarray,
    std_y: np.ndarray,
    title: str,
    filename: str,
    config: SimulationConfig,
    y_lim: tuple[float, float] | None = None,
) -> None:
    """
    Save an error bar plot with given parameters.

    Args:
        s_positions: BPM positions along the accelerator
        baseline_x: Baseline x positions
        baseline_y: Baseline y positions
        std_x: Standard deviation in x
        std_y: Standard deviation in y
        title: Plot title
        filename: Output filename
    """
    fig = plot_error_bars_bpm_range(
        s_positions, baseline_x, std_x, baseline_y, std_y, config.bpm_range, y_lim=y_lim
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(filename, dpi=300, bbox_inches="tight")


def main():
    """Main analysis function."""
    # Parse command-line arguments

    logger.setLevel(logging.INFO)

    # Create configuration
    config = SimulationConfig()

    logger.info(
        f"Starting analysis: nturns={config.nturns}, "
        f"nangles={config.nangles}, samples={config.num_error_samples}"
    )

    # Initialize temporary simulator to get baseline parameters
    temp_simulator = MADSimulator(config)
    matched_tunes = temp_simulator.matched_tunes
    df_twiss = temp_simulator.df_twiss

    # Initialize analyzer with df_twiss for optimisation
    analyzer = NoiseAnalyser(config, matched_tunes, df_twiss)

    # Sample angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, config.nangles, endpoint=False)

    # Compute baseline in parallel
    logger.info("Computing baseline tracks")
    baseline_args = [(angle, matched_tunes, df_twiss, config) for angle in angles]
    baseline_results = process_map(
        _compute_baseline_wrapper, baseline_args, desc="Baseline computation"
    )
    baseline = {ang: (xi, yi) for ang, xi, yi in baseline_results}

    # Use baseline of first angle for plotting reference
    baseline_x, baseline_y = baseline[angles[0]]
    bpm_names = baseline_x.index.tolist()

    # Get ordered BPM positions
    s_positions = df_twiss.loc[bpm_names, "s"].to_numpy()
    order = np.argsort(s_positions)
    bpm_names_ordered = [bpm_names[i] for i in order]
    s_positions = s_positions[order]

    # Compute quadrupole error samples in parallel
    logger.info("Computing quadrupole error samples")
    quad_tasks = list(product(range(config.num_error_samples), angles))
    quad_args = [
        (idx, matched_tunes, df_twiss, angle, config) for idx, angle in quad_tasks
    ]
    quad_flat = process_map(
        _compute_error_sample_wrapper, quad_args, desc="Quadrupole error samples"
    )

    # Reorganize results by angle
    results_quad = {angle: [] for angle in angles}
    for (idx, angle), result in quad_flat:
        results_quad[angle].append(result)

    # Compute IC perturbation samples in parallel
    logger.info("Computing IC perturbation samples")
    ic_args = [
        (idx, matched_tunes, df_twiss, angle, config, False)
        for idx, angle in quad_tasks
    ]
    ic_flat = process_map(
        _compute_ic_sample_wrapper, ic_args, desc="IC perturbation samples"
    )

    results_ic = {angle: [] for angle in angles}
    for (idx, angle), result in ic_flat:
        results_ic[angle].append(result)

    # Compute combined error samples in parallel
    logger.info("Computing combined error samples")
    combined_args = [
        (idx, matched_tunes, df_twiss, angle, config, True) for idx, angle in quad_tasks
    ]
    combined_flat = process_map(
        _compute_ic_sample_wrapper, combined_args, desc="Combined error samples"
    )

    results_combined = {angle: [] for angle in angles}
    for (idx, angle), result in combined_flat:
        results_combined[angle].append(result)

    # Compute standard deviations for each noise type
    logger.info("Computing standard deviations")
    quad_analysis = analyzer.compute_standard_deviations(
        baseline, results_quad, bpm_names_ordered
    )
    ic_analysis = analyzer.compute_standard_deviations(
        baseline, results_ic, bpm_names_ordered
    )
    combined_analysis = analyzer.compute_standard_deviations(
        baseline, results_combined, bpm_names_ordered
    )

    # Generate and save plots
    logger.info("Generating plots")
    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        quad_analysis.std_x,
        quad_analysis.std_y,
        "Quadrupole Error Bars",
        "plots/error_bars_bpm_range.png",
        config=config,
    )

    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        ic_analysis.std_x,
        ic_analysis.std_y,
        "IC Perturbation Error Bars",
        "plots/errorbar_comparison_ic.png",
        config=config,
    )

    save_error_bar_plot(
        s_positions,
        baseline_x[bpm_names_ordered],
        baseline_y[bpm_names_ordered],
        combined_analysis.std_x,
        combined_analysis.std_y,
        "Combined Errors (Quadrupole + IC) Error Bars",
        "plots/errorbar_comparison_combined.png",
        config=config,
    )

    # Plot standard deviations on logarithmic scale
    plot_std_log_comparison(
        s_positions,
        quad_analysis.std_x,
        quad_analysis.std_y,
        ic_analysis.std_x,
        ic_analysis.std_y,
        combined_analysis.std_x,
        combined_analysis.std_y,
        config.bpm_range,
        config.nturns,
    )

    show_plots()
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
