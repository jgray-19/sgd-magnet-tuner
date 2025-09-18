from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gd_optimal_threshold(singular_values, rows, cols):
    """
    Gavish-Donoho optimal hard threshold for white-noise denoising.
    singular_values: singular values (descending)
    Returns an integer rank r to keep.
    """
    beta = rows / cols if rows <= cols else cols / rows
    # omega(beta) from Gavish & Donoho (2014), approximated:
    # omega ≈ median(singular values of pure noise)/sigma, but use closed form:
    # Use empirical fit (good across beta in (0,1]):
    # tau* = sigma * ( (4/√3) * sqrt(beta) )   -> implemented as multiplier on median
    # We'll use the recommended universal threshold on singular values:
    #     tau = median(s) * (0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43)
    # This simple polynomial fit is commonly used for practicality.
    c = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = np.median(singular_values) * c
    r = int(np.sum(singular_values > tau))
    logger.debug(f"GD optimal threshold: keeping {max(r, 1)} modes")
    return max(r, 1)


def _fill_small_nans(matrix, max_gap=5):
    """
    Fill short NaN gaps along time for each BPM column with linear interpolation,
    leaving long missing spans as NaN.
    """
    matrix_df = pd.DataFrame(matrix)
    interpolated_df = matrix_df.copy()
    for col in matrix_df.columns:
        series = matrix_df[col]
        # mark gaps
        is_na = series.isna().to_numpy()
        # linear interpolate everywhere first
        interpolated_series = series.interpolate(limit_direction="both")
        # now restore long gaps (longer than max_gap)
        if is_na.any():
            nan_groups = []
            gap_start = None
            for i, v in enumerate(is_na):
                if v and gap_start is None:
                    gap_start = i
                if (not v or i == len(is_na) - 1) and gap_start is not None:
                    gap_end = i if not v else i
                    if (gap_end - gap_start + 1) > max_gap:
                        nan_groups.append((gap_start, gap_end))
                    gap_start = None
            for start_idx, end_idx in nan_groups:
                interpolated_series.iloc[start_idx : end_idx + 1] = np.nan
        interpolated_df[col] = interpolated_series
    logger.debug(f"Filled NaN gaps in {len(matrix_df.columns)} columns")
    return interpolated_df.to_numpy()


def svd_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    center: str = "bpm",  # "bpm" (demean each BPM), "global" (demean all), or None
    rank: int | str = "auto",  # e.g. 4, or "auto" for GD threshold
    max_nan_gap: int = 5,  # interpolate only short gaps
) -> pd.DataFrame:
    """
    Returns a DataFrame with cleaned x,y using SVD modal truncation.
    """
    logger.info("Starting SVD cleaning of measurements")
    if bpm_list is None:
        bpm_list = meas_df["name"].unique().tolist()

    turn_range = np.arange(int(meas_df["turn"].min()), int(meas_df["turn"].max()) + 1)
    logger.debug(f"Processing {len(bpm_list)} BPMs over {len(turn_range)} turns")

    def _pivot_component(component: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        pivoted_df = (
            meas_df.pivot(index="turn", columns="name", values=component)
            .reindex(index=turn_range)
            .reindex(columns=bpm_list)
        )
        return (
            pivoted_df.values.astype(float),
            pivoted_df.index.to_numpy(),
            pivoted_df.columns.to_list(),
        )

    x_matrix, _, _ = _pivot_component("x")
    y_matrix, _, _ = _pivot_component("y")

    def svd_denoise(matrix):
        # Fill short NaN gaps for SVD; keep a mask to restore NaNs afterwards
        nan_mask = np.isnan(matrix)
        filled_matrix = _fill_small_nans(matrix, max_gap=max_nan_gap)

        # Centering
        if center == "bpm":
            mean = np.nanmean(filled_matrix, axis=0, keepdims=True)
        elif center == "global":
            mean = np.nanmean(filled_matrix, keepdims=True)
        else:
            mean = 0.0
        centered_matrix = filled_matrix - mean

        # Replace any remaining NaNs with 0 for SVD stability
        centered_matrix = np.nan_to_num(centered_matrix, copy=False)

        # SVD
        u_matrix, singular_values, vt_matrix = np.linalg.svd(
            centered_matrix, full_matrices=False
        )
        rows, cols = centered_matrix.shape
        logger.debug(
            f"SVD completed: shape {rows}x{cols}, singular values {len(singular_values)}"
        )

        chosen_rank = (
            _gd_optimal_threshold(singular_values, rows, cols)
            if rank == "auto"
            else int(rank)
        )
        logger.debug(f"Chosen rank: {chosen_rank}")

        # Reconstruct with top-r modes
        u_reduced = u_matrix[:, :chosen_rank]
        singular_values_reduced = singular_values[:chosen_rank]
        vt_reduced = vt_matrix[:chosen_rank, :]
        reconstructed_matrix = u_reduced @ np.diag(singular_values_reduced) @ vt_reduced

        # Add mean back
        reconstructed_matrix += mean

        # Restore the long NaN spans (we don't invent data there)
        reconstructed_matrix[nan_mask] = np.nan
        return reconstructed_matrix, chosen_rank, singular_values

    x_cleaned, rank_x, singular_values_x = svd_denoise(x_matrix)
    y_cleaned, rank_y, singular_values_y = svd_denoise(y_matrix)

    # Pack cleaned x and y back to long format
    cleaned_df = pd.DataFrame(
        {
            "turn": np.repeat(turn_range, len(bpm_list)),
            "name": np.tile(bpm_list, len(turn_range)),
            "x": x_cleaned.reshape(-1),
            "y": y_cleaned.reshape(-1),
        }
    )

    # Merge cleaned x and y back into the original meas_df, preserving all other columns
    # First, ensure meas_df has the same turn and name order
    indexed_meas_df = meas_df.set_index(["turn", "name"])
    indexed_cleaned_df = cleaned_df.set_index(["turn", "name"])

    # Update x and y in meas_df with cleaned values
    indexed_meas_df["x"] = indexed_cleaned_df["x"]
    indexed_meas_df["y"] = indexed_cleaned_df["y"]

    # Optional: attach metadata about chosen ranks
    indexed_meas_df.attrs["svd_rank_x"] = rank_x
    indexed_meas_df.attrs["svd_rank_y"] = rank_y
    indexed_meas_df.attrs["svd_singular_values_x"] = singular_values_x
    indexed_meas_df.attrs["svd_singular_values_y"] = singular_values_y
    indexed_meas_df.attrs["center"] = center
    logger.info("SVD cleaning completed successfully")
    return indexed_meas_df.reset_index()
