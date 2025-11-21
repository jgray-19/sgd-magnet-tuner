"""
Example script for estimating dispersion at corrector magnets.

This script demonstrates how to use the dispersion estimation module to
calculate horizontal and vertical dispersion at corrector locations using
optics analysis data from OMC3 and a sequence file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import tfs

from aba_optimiser.dispersion.dispersion_estimation import estimate_corrector_dispersions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Estimate dispersion at correctors and save results."""
    # Configuration
    optics_dir = Path(
        "path/to/optics/analysis"
    )  # Directory with beta_phase_x.tfs, dispersion_x.tfs, etc.
    sequence_file = Path("path/to/sequence.seq")  # MAD-X sequence file
    model_dir = Path("path/to/model")  # Directory with twiss_elements.dat
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Estimate horizontal dispersion
    logger.info("Estimating horizontal dispersion at correctors")
    disp_x_df, stats_x_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=sequence_file,
        model_dir=model_dir,
        seq_name="lhcb1",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=10,
        plane="x",
    )

    # Save results
    logger.info("Saving horizontal dispersion results")
    tfs.write(output_dir / "corrector_dispersion_x.tfs", disp_x_df)
    tfs.write(output_dir / "corrector_dispersion_x_detailed.tfs", stats_x_df)

    # Estimate vertical dispersion (if needed)
    logger.info("Estimating vertical dispersion at correctors")
    disp_y_df, stats_y_df = estimate_corrector_dispersions(
        optics_dir=optics_dir,
        sequence_file=sequence_file,
        model_dir=model_dir,
        seq_name="lhcb1",
        beam_energy_gev=6800,
        particle="proton",
        num_closest_bpms=10,
        plane="y",
    )

    # Save results
    logger.info("Saving vertical dispersion results")
    tfs.write(output_dir / "corrector_dispersion_y.tfs", disp_y_df)
    tfs.write(output_dir / "corrector_dispersion_y_detailed.tfs", stats_y_df)

    # Print summary statistics
    logger.info("\nHorizontal Dispersion Summary:")
    logger.info(f"  Mean: {disp_x_df['DISPERSION'].mean():.6e} m")
    logger.info(f"  Std:  {disp_x_df['DISPERSION'].std():.6e} m")
    logger.info(f"  Min:  {disp_x_df['DISPERSION'].min():.6e} m")
    logger.info(f"  Max:  {disp_x_df['DISPERSION'].max():.6e} m")

    logger.info("\nVertical Dispersion Summary:")
    logger.info(f"  Mean: {disp_y_df['DISPERSION'].mean():.6e} m")
    logger.info(f"  Std:  {disp_y_df['DISPERSION'].std():.6e} m")
    logger.info(f"  Min:  {disp_y_df['DISPERSION'].min():.6e} m")
    logger.info(f"  Max:  {disp_y_df['DISPERSION'].max():.6e} m")

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
