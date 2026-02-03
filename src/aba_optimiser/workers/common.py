"""Common data structures and utilities for all worker types.

This module defines shared data structures, configurations, and utility functions
used across different worker implementations (tracking and optics modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator


@dataclass
class WorkerConfig:
    """Configuration parameters common to all worker types.

    Instead of passing many individual parameters (sequence_file, beam_energy,
    seq_name, etc.), we pass an Accelerator instance that encapsulates all
    machine-specific configuration and can create MAD interfaces.

    Attributes:
        accelerator: Accelerator instance (e.g., LHC) with machine parameters
        start_bpm: Name of the starting BPM in the range
        end_bpm: Name of the ending BPM in the range
        magnet_range: MAD-NG range specification for magnets
        corrector_strengths: Path to corrector strength configuration
        tune_knobs_file: Path to tune knob definitions
        sdir: Direction of propagation (+1 forward, -1 backward)
        bad_bpms: List of BPM names to exclude from analysis
        debug: Enable debug mode for MAD interface
        mad_logfile: Path to MAD log file (can be None)
        optimise_knobs: List of global knob names to optimise
    """

    accelerator: Accelerator
    start_bpm: str
    end_bpm: str
    magnet_range: str
    corrector_strengths: Path | None
    tune_knobs_file: Path | None
    sdir: int = 1
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None


@dataclass
class PrecomputedTrackingWeights:
    """Precomputed weights shared across tracking workers.

    Attributes:
        x: Normalised weights for horizontal position
        y: Normalised weights for vertical position
        px: Normalised weights for horizontal momentum
        py: Normalised weights for vertical momentum
        hessian_x: Aggregated Hessian weights for x (unnormalised inverse-variance)
        hessian_y: Aggregated Hessian weights for y (unnormalised inverse-variance)
        hessian_px: Aggregated Hessian weights for px (unnormalised inverse-variance)
        hessian_py: Aggregated Hessian weights for py (unnormalised inverse-variance)
    """

    x: np.ndarray
    y: np.ndarray
    px: np.ndarray
    py: np.ndarray
    hessian_x: np.ndarray
    hessian_y: np.ndarray
    hessian_px: np.ndarray
    hessian_py: np.ndarray


@dataclass
class TrackingData:
    """Data container for particle tracking simulations.

    All comparison and variance arrays have shape (n_particles, n_data_points, 2)
    where the last dimension is [x, y] or [px, py].

    Attributes:
        position_comparisons: Reference position data [x, y] per particle and data point
        momentum_comparisons: Reference momentum data [px, py] per particle and data point
        position_variances: Measurement uncertainties for positions
        momentum_variances: Measurement uncertainties for momenta
        init_coords: Initial particle coordinates for tracking
        init_pts: Initial transverse momentum values per particle
        precomputed_weights: Optional globally precomputed weights shared by all workers
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)
    precomputed_weights: PrecomputedTrackingWeights | None


@dataclass
class OpticsData:
    """Data container for optics function measurements.

    Attributes:
        comparisons: Reference phase advance values [phase_adv_x, phase_adv_y] between consecutive BPMs
        variances: Measurement uncertainties for phase advances
        beta_comparisons: Reference beta function values [beta_x, beta_y] at each BPM
        beta_variances: Measurement uncertainties for beta functions
        init_coords: Initial Twiss parameters and dispersion

    Note: Phase advances are measured between consecutive BPMs, so there are (n_bpms - 1) measurements.
          Beta functions are measured at each BPM, so there are n_bpms measurements.
    """

    comparisons: np.ndarray  # Shape: (n_bpms-1, 2) - [phase_adv_x, phase_adv_y]
    variances: np.ndarray  # Shape: (n_bpms-1, 2) - [var_phase_adv_x, var_phase_adv_y]
    beta_comparisons: np.ndarray  # Shape: (n_bpms, 2) - [beta_x, beta_y]
    beta_variances: np.ndarray  # Shape: (n_bpms, 2) - [var_beta_x, var_beta_y]
    init_coords: dict[str, float]  # beta11, beta22, alfa11, alfa22, dx, dpx, dy, dpy


class WeightProcessor:
    """Utility class for processing and normalizing measurement weights.

    Provides static methods for converting variances to weights, normalizing,
    and aggregating weights for use in loss functions and Hessian approximations.
    """

    @staticmethod
    def variance_to_weight(variances: np.ndarray) -> np.ndarray:
        """Convert variances to inverse-variance weights.

        Invalid or non-positive variances are set to zero weight.

        Args:
            variances: Array of variance values

        Returns:
            Array of weights (1/variance for valid entries, 0 for invalid)
        """
        weights = np.zeros_like(variances, dtype=np.float64)
        valid = np.isfinite(variances) & (variances > 0.0)
        np.divide(1.0, variances, out=weights, where=valid)
        return weights

    @staticmethod
    def normalise_weights_globally(*weights_arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        """Normalize multiple weight arrays globally so that the maximum across all is 1.

        Args:
            weights_arrays: Multiple arrays of weight values

        Returns:
            Tuple of normalized weight arrays
        """
        global_max = max(np.max(weights) for weights in weights_arrays)
        if global_max > 0:
            return tuple(weights / global_max for weights in weights_arrays)
        return weights_arrays

    @staticmethod
    def aggregate_hessian_weights(weights: np.ndarray) -> np.ndarray:
        """Aggregate per-particle weights into per-BPM weights for Hessian.

        Computes mean weight across particles for each BPM, used in
        approximate Hessian calculations.

        Args:
            weights: Array of shape (n_particles, n_bpms)

        Returns:
            Array of shape (n_bpms,) with aggregated weights
        """
        if weights.size == 0:
            return np.array([], dtype=np.float64)

        sums = np.sum(weights, axis=0)
        counts = np.count_nonzero(weights, axis=0)
        aggregated = np.zeros_like(sums, dtype=np.float64)
        np.divide(sums, counts, out=aggregated, where=counts > 0)
        return aggregated

    @staticmethod
    def compute_variance_floor(
        variances: np.ndarray, percentile: float = 5, factor: float = 1.0
    ) -> float | None:
        """Compute a percentile-based variance floor value.

        Args:
            variances: Array of variance values
            percentile: Percentile (0-100) used to define the reference variance
            factor: Multiplicative factor applied to the percentile value

        Returns:
            Scalar floor value, or None if no valid variances are present.
        """
        v = np.asarray(variances, dtype=np.float64)
        valid = np.isfinite(v) & (v > 0.0)
        if not np.any(valid):
            return None
        ref = np.percentile(v[valid], percentile)
        return factor * ref

    @staticmethod
    def floor_variances(
        variances: np.ndarray,
        percentile: float = 10,
        factor: float = 1.0,
        floor_value: float | None = None,
    ) -> np.ndarray:
        """
        Floor unrealistically small variance values using a robust percentile-based rule.

        This function is intended to protect inverse-variance weighting from domination
        by a small number of pathologically tiny variances (e.g. due to numerical noise,
        quantisation, or failed uncertainty estimates).

        The floor is computed as:
            floor = factor * P_percentile(valid variances)

        where P_percentile is taken over finite, strictly positive variances only.

        Invalid (non-finite or non-positive) variances are left unchanged and are expected
        to be handled downstream (typically by assigning zero weight).

        Args:
            variances:
                Array of variance values. Can be any shape.
            percentile:
                Percentile (0-100) used to define the reference variance.
                Typical values: 0.5-2.0. Default is 1.0.
            factor:
                Multiplicative factor applied to the percentile value to obtain the floor.
                Values < 1 only floor extreme outliers; values ≈ 1 enforce a stricter floor.
            floor_value:
                Optional precomputed floor value to apply instead of computing from the
                provided variances.

        Returns:
            A new array with the same shape as `variances`, where valid entries smaller
            than the computed floor have been raised to the floor value.
        """
        v = np.asarray(variances, dtype=np.float64)

        # Identify valid variances
        valid = np.isfinite(v) & (v > 0.0)
        if not np.any(valid):
            return v.copy()

        # Compute robust floor
        var_floor = (
            WeightProcessor.compute_variance_floor(v, percentile=percentile, factor=factor)
            if floor_value is None
            else floor_value
        )

        # Apply floor
        v_out = v.copy()
        v_out[valid] = np.maximum(v_out[valid], var_floor)  # ty:ignore[no-matching-overload]

        return v_out


def split_array_to_batches(array: np.ndarray, num_batches: int, axis: int = 0) -> list[np.ndarray]:
    """Split an array into equal batches along specified axis.

    Args:
        array: Input array to split
        num_batches: Number of batches to create
        axis: Axis along which to split

    Returns:
        List of array batches
    """
    return np.array_split(array, num_batches, axis=axis)
