"""Plot the effect of SVD cleaning on phase space coordinates for a single BPM.

This script reads an SDDS file, processes it to compute px and py,
applies SVD cleaning, and plots x vs px and y vs py for all turns and bunches
at a single BPM (BPM.9R2.B1), showing both the original and cleaned data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tfs
from tmom_recon.physics.transverse import calculate_pz
from tmom_recon.svd import svd_clean_measurements
from turn_by_turn import read_tbt

from aba_optimiser.measurements.create_datafile import compute_vars_from_known_noise

logger = logging.getLogger(__name__)


def load_single_file(file_path: str | Path) -> pd.DataFrame:
    """Load a single SDDS file and convert to DataFrame format.

    Args:
        file_path: Path to the SDDS file

    Returns:
        DataFrame with columns: name, turn, x, y, kick_plane
    """
    logger.info(f"Loading data from {file_path}")
    meas_tbt = read_tbt(file_path, datatype="lhc")

    # Convert to DataFrame format
    all_data = []
    turn_offset = 1

    for bunch in meas_tbt.matrices:
        df_x = bunch.X.copy()
        df_y = bunch.Y.copy()
        df_x.index.name = "name"
        df_y.index.name = "name"
        df_x.columns = df_x.columns + turn_offset
        df_y.columns = df_y.columns + turn_offset

        df_combined = df_x.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
        df_combined["y"] = df_y.reset_index().melt(id_vars="name", var_name="turn", value_name="y")[
            "y"
        ]

        # Convert from mm to m
        df_combined["x"] = df_combined["x"] / 1000
        df_combined["y"] = df_combined["y"] / 1000
        df_combined["kick_plane"] = "xy"

        # Reorder rows based on BPM names
        original_order = df_x.index.tolist()
        df_combined["name"] = pd.Categorical(df_combined["name"], categories=original_order)
        df_combined = df_combined.sort_values(["turn", "name"]).reset_index(drop=True)

        all_data.append(df_combined)
        turn_offset += df_x.shape[1]

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df["name"] = combined_df["name"].astype("category")
    combined_df["turn"] = combined_df["turn"].astype("int32")

    logger.info(f"Loaded {len(combined_df)} data points from {len(all_data)} bunches")
    return combined_df


def add_momenta(
    df: pd.DataFrame, model_dir: str | Path, bad_bpms: list[str] | None = None
) -> pd.DataFrame:
    """Add px and py columns using the transverse momentum calculation.

    Args:
        df: DataFrame with x, y positions
        model_dir: Path to the model directory containing twiss_ac.dat
        bad_bpms: List of bad BPM names to assign infinite variance

    Returns:
        DataFrame with added px, py, var_x, var_y, var_px, var_py columns
    """
    logger.info("Computing transverse momenta (px, py)")

    if bad_bpms is None:
        bad_bpms = []

    # Load twiss parameters
    tws = tfs.read(Path(model_dir) / "twiss_ac.dat")
    tws.columns = [col.lower() for col in tws.columns]
    tws = tws.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    tws.headers = {k.lower(): v for k, v in tws.headers.items()}
    tws = tws.set_index("name")

    # Add variance columns first (required by calculate_pz)
    df_with_var = compute_vars_from_known_noise(df, bad_bpms)

    # Calculate px and py
    df_with_p = calculate_pz(df_with_var, tws=tws, inject_noise=False)

    # Drop NaN values
    if df_with_p["px"].isna().any() or df_with_p["py"].isna().any():
        logger.warning("NaN values found in px or py after calculation, dropping rows")
        df_with_p = df_with_p.dropna(subset=["px", "py"])

    logger.info("Momentum calculation complete")
    return df_with_p


def plot_phase_space(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    bpm_name: str,
    output_file: str | Path | None = None,
) -> None:
    """Plot x vs px and y vs py for a single BPM, showing SVD cleaning effect.

    Args:
        df_original: DataFrame with original data (before SVD cleaning)
        df_cleaned: DataFrame with cleaned data (after SVD cleaning)
        bpm_name: Name of the BPM to plot
        output_file: Optional path to save the figure
    """
    logger.info(f"Creating phase space plots for {bpm_name}")

    # Filter data for the specific BPM
    df_orig_bpm = df_original[df_original["name"] == bpm_name]
    df_clean_bpm = df_cleaned[df_cleaned["name"] == bpm_name]

    if len(df_orig_bpm) == 0:
        logger.error(f"No data found for BPM {bpm_name}")
        return

    logger.info(f"Found {len(df_orig_bpm)} data points for {bpm_name}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot x vs px
    axes[0].scatter(
        df_orig_bpm["x"] * 1000,  # Convert back to mm for plotting
        df_orig_bpm["px"] * 1000,
        alpha=0.3,
        s=2,
        label="Original",
        color="blue",
    )
    axes[0].scatter(
        df_clean_bpm["x"] * 1000,
        df_clean_bpm["px"] * 1000,
        alpha=0.5,
        s=2,
        label="SVD Cleaned",
        color="red",
    )
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("px [mrad]")
    axes[0].set_title(f"Horizontal Phase Space at {bpm_name}")
    axes[0].legend()
    axes[0].grid(visible=True, alpha=0.3)

    # Plot y vs py
    axes[1].scatter(
        df_orig_bpm["y"] * 1000,
        df_orig_bpm["py"] * 1000,
        alpha=0.3,
        s=2,
        label="Original",
        color="blue",
    )
    axes[1].scatter(
        df_clean_bpm["y"] * 1000,
        df_clean_bpm["py"] * 1000,
        alpha=0.5,
        s=2,
        label="SVD Cleaned",
        color="red",
    )
    axes[1].set_xlabel("y [mm]")
    axes[1].set_ylabel("py [mrad]")
    axes[1].set_title(f"Vertical Phase Space at {bpm_name}")
    axes[1].legend()
    axes[1].grid(visible=True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved to {output_file}")

    plt.show()
    logger.info("Plot complete")


def main():
    """Main function to process and plot SVD cleaning effect."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configuration
    sdds_file = "/user/slops/data/LHC_DATA/OP_DATA/FILL_DATA/11259/BPM/Beam1@BunchTurn@2025_11_07@07_53_05_820.sdds"
    model_dir = "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-11-07/LHCB1/Models/2025-11-07_B1_12cm_right_knobs/"
    bpm_name = "BPM.9R2.B1"
    output_file = "bpm_9r2_b1_svd_effect.png"

    logger.info("=" * 80)
    logger.info("Starting SVD cleaning effect visualization")
    logger.info("=" * 80)

    # Load the data
    df = load_single_file(sdds_file)

    # Add momenta to original data
    df_original = add_momenta(df.copy(), model_dir)

    # Apply SVD cleaning and add momenta to cleaned data
    logger.info("Applying SVD cleaning")
    df_cleaned_pos = svd_clean_measurements(df.copy())
    df_cleaned = add_momenta(df_cleaned_pos, model_dir)

    # Plot the results
    plot_phase_space(df_original, df_cleaned, bpm_name, output_file)

    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
