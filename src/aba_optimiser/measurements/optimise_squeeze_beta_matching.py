"""Run squeeze beta-matching studies from optics measurement directories."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import tfs

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT, OptimiserConfig

# Import helper functions from the coordinate optimisation script to stay DRY
from aba_optimiser.measurements.optimise_squeeze_quads import (
    get_beam_paths,
    save_arc_estimates,
)
from aba_optimiser.measurements.squeeze_config import MODEL_DIRS, PC
from aba_optimiser.measurements.squeeze_helpers import (
    get_analysis_dir,
    get_or_make_sequence,
)
from aba_optimiser.measurements.utils import find_all_bad_bpms_from_analysis
from aba_optimiser.training.controller_config import OutputConfig, SequenceConfig
from aba_optimiser.training_optics.controller import OpticsController

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
SQUEEZE_STEPS = [
    "1.2m",
    "1.2m_agc",
    "1.05m",
    "0.93m",
    "0.725m",
    "0.6m",
    "0.45m",
    "0.3m",
    "0.25m",
    "0.18m",
]


def get_default_optimiser_config() -> OptimiserConfig:
    """Get default optimiser configuration for optics optimisation."""
    return OptimiserConfig(
        max_epochs=600,
        warmup_epochs=30,
        warmup_lr_start=1e-10,
        max_lr=1,
        min_lr=1,
        gradient_converged_value=1e-3,
        optimiser_type="lbfgs",  # or 'adam'
    )

def get_bpm_points(arc_num: int, beam: int) -> tuple[str, list[str], list[str]]:
    """Get magnet range and BPM points for an arc."""
    next_arc = arc_num % 8 + 1
    suffix = f".B{beam}"
    start_bpm = f"BPM.9R{arc_num}{suffix}"
    # end_bpm = f"BPM.15R{arc_num}{suffix}" # Testing with BPM.15R
    end_bpm = f"BPM.9R{next_arc}{suffix}"
    # end_bpm = f"BPM.9L{next_arc}{suffix}"
    magnet_range = f"{start_bpm}/{end_bpm}"
    bpm_start_points = [f"BPM.{i}R{arc_num}{suffix}" for i in range(9, 30, 3)]
    bpm_end_points = [f"BPM.{i}R{next_arc}{suffix}" for i in range(9, 30, 3)]

    return magnet_range, bpm_start_points, bpm_end_points


def get_bpm_ranges_from_model(model_dir: str, beam: int) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Extract BPM ranges from twiss.dat file for IR optimisation.

    Args:
        model_dir: Path to the model directory containing twiss_elements.dat
        beam: Beam number (1 or 2)

    Returns:
        Tuple of (magnet_ranges, bpm_starts, bpm_end_points)
    """
    twiss_file = Path(model_dir) / "twiss_elements.dat"
    twiss_df = tfs.read(twiss_file, index="NAME")

    # Filter BPMs for this beam
    bpm_mask = twiss_df.index.str.startswith("BPM") & twiss_df.index.str.endswith(f".B{beam}")
    bpm_names = twiss_df.index[bpm_mask].tolist()

    # Regex to match BPM names: BPM.*.(IP)(L|R).*.B(beam)
    bpm_pattern = re.compile(r'BPM[A-Z]*\.(\d)([LR])(\d)\.B(\d+)')

    # Collect all matching BPMs with ip and side
    matches = [
        (bpm, int(match.group(3)), match.group(2), int(match.group(1)))
        for bpm in bpm_names
        if (match := bpm_pattern.match(bpm))
    ]

    magnet_ranges = []
    bpm_starts = []
    bpm_end_points = []

    ip_range = range(8, 0, -1) if beam == 2 else range(1, 9)
    before_side = 'L' if beam == 1 else 'R'
    after_side = 'R' if beam == 1 else 'L'
    for ip in ip_range:
        # Include BPMs from position 4 onwards to get more measurement points
        before_bpms = [bpm for bpm, ip_num, side, from_ip in matches if ip_num == ip and side == before_side and from_ip >= 4]
        after_bpms = [bpm for bpm, ip_num, side, from_ip in matches if ip_num == ip and side == after_side and from_ip >= 4]

        # Remove all bpms with W in their names
        # before_bpms = [bpm for bpm in before_bpms if "W" not in bpm]
        # after_bpms = [bpm for bpm in after_bpms if "W" not in bpm]

        bpm_starts.append(before_bpms)
        bpm_end_points.append(after_bpms)

        if beam == 1:
            magnet_ranges.append(f"BPM.9L{ip}.B1/BPM.9R{ip}.B1")
        else:
            magnet_ranges.append(f"BPM.9R{ip}.B2/BPM.9L{ip}.B2")
    return magnet_ranges, bpm_starts, bpm_end_points


def run_optics_optimisation_for_squeeze_step(
    beam: int,
    squeeze_step: str,
    show_plots: bool = False,
) -> None:
    """Run optics optimisation for a single squeeze step.

    Uses OpticsController to optimize quadrupole strengths directly from
    measured beta functions. Optimizes only the beta correction knobs.

    Args:
        beam: Beam number (1 or 2)
        squeeze_step: Squeeze step identifier (e.g., "1.2m")
        show_plots: Whether to show optimisation plots
    """
    logger.info(f"Running optics optimisation for beam {beam}, squeeze step {squeeze_step}")

    # Setup paths
    _, model_base_dir = get_beam_paths(beam, squeeze_step)
    model_dir = model_base_dir / MODEL_DIRS[beam][squeeze_step]
    sequence_path = get_or_make_sequence(beam, model_dir)

    # Results directory
    results_dir = PROJECT_ROOT / f"b{beam}_optics_results"
    results_dir.mkdir(exist_ok=True)

    # Get optics folder with measurements
    optics_folder = get_analysis_dir(beam, squeeze_step)
    logger.info(f"Using optics measurements from: {optics_folder}")

    # Load bad BPMs from the analysis ini file and associated measurement folders
    bad_bpms = find_all_bad_bpms_from_analysis(optics_folder)
    logger.info(f"Found {len(bad_bpms)} bad BPMs from analysis")

    tune_knobs_file = results_dir / "tune_knobs.txt"
    if not tune_knobs_file.exists():
        tune_knobs_file = None  # Don't use tune knobs if file not found
    logger.info(f"Using tune knobs file: {tune_knobs_file}")


    # Get beta correction knobs to optimize
    optimiser_config = get_default_optimiser_config()
    ir_magnet_ranges, ir_bpm_starts, ir_bpm_end_points = get_bpm_ranges_from_model(str(model_dir), beam)

    for arc in range(1, 9):
        magnet_range, start_points, end_points = get_bpm_points(arc, beam)

        sequence_config = SequenceConfig(
            magnet_range=magnet_range,
            bad_bpms=list(bad_bpms),
            first_bpm="BPM.33L2.B1" if beam == 1 else "BPM.34R8.B2",
        )

        accelerator = LHC(
            beam=beam,
            sequence_file=sequence_path,
            pc=PC,
            optimise_quadrupoles=True,
        )
        # Create and run controller
        controller = OpticsController(
            accelerator=accelerator,
            sequence_config=sequence_config,
            optimiser_config=optimiser_config,
            optics_folder=optics_folder,
            bpm_start_points=start_points,
            bpm_end_points=end_points,
            initial_knob_strengths=None,  # Start from sequence values or previous arc results
            corrector_file=None,  # Don't use correctors
            tune_knobs_file=None,  # Use tune knobs if available
            true_strengths=None,
            use_errors=True,  # Use measurement errors for weighting
            output_config=OutputConfig(show_plots=show_plots),
        )

        # Run optimisation
        final_knobs_arc, uncertainties = controller.run()
        # save_arc_estimates(results_dir, squeeze_step, arc, final_knobs)

        # Now optimize IRs
        # magnet_range = ir_magnet_ranges[arc - 1]
        # bpm_config = BPMConfig(
        #     start_points=ir_bpm_starts[arc - 1],
        #     end_points=ir_bpm_end_points[arc - 1],
        # )
        # # Create and run controller
        # controller = LHCOpticsController(
        #     beam=beam,
        #     optics_folder=optics_folder,
        #     bpm_config=bpm_config,
        #     magnet_range=magnet_range,
        #     optimiser_config=optimiser_config,
        #     sequence_path=sequence_path,
        #     show_plots=show_plots,
        #     initial_knob_strengths=None,  # Start from sequence values or previous results
        #     corrector_file=None,  # Don't use correctors
        #     tune_knobs_file=None,  # Use tune knobs if available
        #     true_strengths=None,
        #     bad_bpms=list(bad_bpms) if bad_bpms else None,
        #     pc=pc,
        #     use_errors=True,  # Use measurement errors for weighting
        # )

        # # Run optimisation
        # final_knobs_ir, uncertainties = controller.run()
        final_knobs_ir = {}
        save_arc_estimates(results_dir, squeeze_step, arc, {**final_knobs_arc, **final_knobs_ir})

    logger.info(f"Optics optimisation completed for {squeeze_step}")


def main():
    """Main entry point for optics optimisation."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Optimize LHC squeeze quadrupoles using optics measurements (beta functions)"
    )
    parser.add_argument(
        "--beam", type=int, choices=[1, 2], required=True, help="Beam number (1 or 2)"
    )
    parser.add_argument(
        "--squeeze-step", type=str, required=True, help="Squeeze step (e.g., '1.2m', '0.6m')"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show optimisation progress plots",
    )
    args = parser.parse_args()

    # Validate squeeze step
    if args.squeeze_step not in SQUEEZE_STEPS:
        raise ValueError(
            f"Unknown squeeze step '{args.squeeze_step}'. "
            f"Available steps: {', '.join(SQUEEZE_STEPS)}"
        )

    run_optics_optimisation_for_squeeze_step(
        beam=args.beam,
        squeeze_step=args.squeeze_step,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
