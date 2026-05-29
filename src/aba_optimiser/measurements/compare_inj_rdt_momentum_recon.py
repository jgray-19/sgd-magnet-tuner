"""Compare injection-region momentum reconstruction pipelines on measured data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tmom_recon import calculate_pz_measurement
from tmom_recon.kalman import LhcMadDriver, ModelProvider, reconstruct_px_py
from tmom_recon.svd import svd_clean_measurements
from turn_by_turn import read_tbt

from aba_optimiser.accelerators import LHC
from aba_optimiser.config import PROJECT_ROOT
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.measurements.optimise_squeeze_quads import MEAS_TIMES, ZEROHZ
from aba_optimiser.measurements.squeeze_config import BETABEAT_DIR, get_measurement_date
from aba_optimiser.measurements.squeeze_helpers import (
    get_analysis_dir,
    get_model_dir,
    get_or_make_sequence,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run the comparison command-line entry point."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Compare momentum reconstruction on Beam 1 inj_rdt data: "
            "SVD cleaning + legacy reconstruction + Kalman reconstruction."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "temp_analysis_inj_rdt_b1_compare",
        help="Directory for outputs (parquet + plots + MAD log).",
    )
    parser.add_argument(
        "--bpm",
        type=str,
        default="BPM.14R3.B1",
        help="BPM for phase-space plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving files.",
    )
    args = parser.parse_args()

    beam = 1
    squeeze_step = "inj_rdt"
    measurement_date = get_measurement_date(squeeze_step).replace("-", "_")
    measurement_time = MEAS_TIMES[beam][squeeze_step][ZEROHZ][0]  # Take first measurement
    measurement_folder = f"Beam{beam}@BunchTurn@{measurement_date}@{measurement_time}"
    measurement_file = (
        BETABEAT_DIR
        / get_measurement_date(squeeze_step)
        / f"LHCB{beam}"
        / "Measurements"
        / measurement_folder
        / f"{measurement_folder}.sdds"
    )

    model_dir = get_model_dir(beam, squeeze_step)
    analysis_dir = get_analysis_dir(beam, squeeze_step)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Measurement file: %s", measurement_file)
    LOGGER.info("Model directory: %s", model_dir)
    LOGGER.info("Analysis directory: %s", analysis_dir)
    LOGGER.info("Output directory: %s", args.output_dir)

    if not measurement_file.exists():
        raise FileNotFoundError(f"Measurement file not found: {measurement_file}")

    meas_tbt = read_tbt(measurement_file, datatype="lhc")
    all_data: list[pd.DataFrame] = []
    turn_offset = 1
    for bunch in meas_tbt.matrices:
        df_x = bunch.X.copy()
        df_y = bunch.Y.copy()
        df_x.index.name = "name"
        df_y.index.name = "name"
        df_x.columns = df_x.columns + turn_offset
        df_y.columns = df_y.columns + turn_offset

        df = df_x.reset_index().melt(id_vars="name", var_name="turn", value_name="x")
        df["y"] = df_y.reset_index().melt(id_vars="name", var_name="turn", value_name="y")["y"]
        df["x"] = df["x"] / 1000.0
        df["y"] = df["y"] / 1000.0

        bpm_order = df_x.index.tolist()
        df["name"] = pd.Categorical(df["name"], categories=bpm_order)
        df = df.sort_values(["turn", "name"]).reset_index(drop=True)

        all_data.append(df)
        turn_offset += df_x.shape[1]

    noisy_df = all_data[0] # cannot combine
    noisy_df["name"] = noisy_df["name"].astype(str)
    noisy_df["turn"] = noisy_df["turn"].astype(int)

    twiss_file = model_dir / "twiss.dat"
    if not twiss_file.exists():
        raise FileNotFoundError(f"Expected Twiss file not found: {twiss_file}")

    sequence_path = get_or_make_sequence(1, model_dir)
    accel = LHC(beam=beam, sequence_file=sequence_path, kinetic_energy=450)
    mad_iface = GradientDescentMadInterface(accel)
    tws = mad_iface.run_twiss()

    noisy_df = noisy_df[noisy_df["name"].isin(tws.index)].copy()

    LOGGER.info("Applying SVD cleaning")
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_df["var_x"] = (1e-4) ** 2
    cleaned_df["var_y"] = (1e-4) ** 2
    cleaned_df["var_px"] = (3e-6) ** 2
    cleaned_df["var_py"] = (3e-6) ** 2

    LOGGER.info("Running legacy momentum reconstruction (calculate_pz_measurement)")
    legacy_df = calculate_pz_measurement(
        cleaned_df.copy(),
        analysis_dir,
        model_tws=tws,
        reverse_meas_tws=False,
        info=True,
        include_errors=True,
        include_optics_errors=True,
        dpp_override=0.0,
    )
    # legacy_df["var_x"] = legacy_df["var_x"] / 100.0
    # legacy_df["var_y"] = legacy_df["var_y"] / 100.0

    # Any row that has NaNs in x/y/px/py in the legacy reconstruction should be dropped from all the DataFrames
    valid_mask = legacy_df[["x", "y", "px", "py"]].notna().all(axis=1)
    legacy_df = legacy_df[valid_mask].copy()
    # Since cleaned_df and legacy_df have different indexes, we need to filter cleaned_df based on the 'name' and 'turn' columns that match the valid rows in legacy_df
    cleaned_df = cleaned_df.merge(
        legacy_df[["name", "turn"]],
        on=["name", "turn"],
        how="inner",        suffixes=("", "_legacy"),
    )
    cleaned_df = cleaned_df.drop(columns=[col for col in cleaned_df.columns if col.endswith("_legacy")])
    print(len(cleaned_df), len(legacy_df))

    LOGGER.info("Running Kalman momentum reconstruction")
    mad_driver = LhcMadDriver(
        sequence_file=sequence_path,
        beam=1,
        beam_energy=6800.0,
        deltap=0.0,
        bpm_pattern="BPM",
        bad_bpms=None,
        corrector_strengths=None,
        tune_knobs_file=None,
        start_bpm="BPM.33L2.B1",
        mad_logfile=args.output_dir / "kalman_mad.log",
        used_ac_dipole=True,
        augment_params=True,
    )
    try:
        model = ModelProvider(mad_driver)
        kalman_out = reconstruct_px_py(
            cleaned_df.copy(),
            model=model,
            tws=tws,
            use_smoother=False,
            options={"svd_cleaning": False},
        )
    finally:
        if hasattr(mad_driver, "close"):
            mad_driver.close()

    kalman_df = kalman_out["df"] if isinstance(kalman_out, dict) else kalman_out

    cleaned_df.to_parquet(args.output_dir / "inj_rdt_b1_cleaned.parquet")
    legacy_df.to_parquet(args.output_dir / "inj_rdt_b1_legacy_reco.parquet")
    kalman_df.to_parquet(args.output_dir / "inj_rdt_b1_kalman_reco.parquet")

    legacy_bpm = legacy_df[legacy_df["name"] == args.bpm].copy()
    kalman_bpm = kalman_df[kalman_df["name"] == args.bpm].copy()

    if legacy_bpm.empty:
        raise ValueError(f"No legacy data found for BPM {args.bpm}")
    if kalman_bpm.empty:
        raise ValueError(f"No Kalman data found for BPM {args.bpm}")

    kalman_x_col = "x_clean" if "x_clean" in kalman_bpm.columns else "x"
    kalman_y_col = "y_clean" if "y_clean" in kalman_bpm.columns else "y"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].scatter(legacy_bpm["x"], legacy_bpm["px"], s=6, alpha=0.7)
    axes[0, 0].set_title(f"Legacy: x-px @ {args.bpm}")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("px [rad]")
    axes[0, 0].grid(visible=True, alpha=0.3)

    axes[0, 1].scatter(legacy_bpm["y"], legacy_bpm["py"], s=6, alpha=0.7)
    axes[0, 1].set_title(f"Legacy: y-py @ {args.bpm}")
    axes[0, 1].set_xlabel("y [m]")
    axes[0, 1].set_ylabel("py [rad]")
    axes[0, 1].grid(visible=True, alpha=0.3)

    axes[1, 0].scatter(kalman_bpm[kalman_x_col], kalman_bpm["px"], s=6, alpha=0.7)
    axes[1, 0].set_title(f"Kalman: {kalman_x_col}-px @ {args.bpm}")
    axes[1, 0].set_xlabel(f"{kalman_x_col} [m]")
    axes[1, 0].set_ylabel("px [rad]")
    axes[1, 0].grid(visible=True, alpha=0.3)

    axes[1, 1].scatter(kalman_bpm[kalman_y_col], kalman_bpm["py"], s=6, alpha=0.7)
    axes[1, 1].set_title(f"Kalman: {kalman_y_col}-py @ {args.bpm}")
    axes[1, 1].set_xlabel(f"{kalman_y_col} [m]")
    axes[1, 1].set_ylabel("py [rad]")
    axes[1, 1].grid(visible=True, alpha=0.3)

    fig.suptitle("Beam 1 inj_rdt phase-space comparison")
    fig.tight_layout()

    plot_path = args.output_dir / f"phase_space_{args.bpm.replace('.', '_')}.png"
    fig.savefig(plot_path, dpi=200)
    LOGGER.info("Saved plot: %s", plot_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
