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


@dataclass
class WorkerConfig:
    """Configuration parameters common to all worker types.

    Attributes:
        start_bpm: Name of the starting BPM in the range
        end_bpm: Name of the ending BPM in the range
        magnet_range: MAD-NG range specification for magnets
        sequence_file_path: Path to the accelerator sequence file
        corrector_strengths: Path to corrector strength configuration
        tune_knobs_file: Path to tune knob definitions
        beam_energy: Beam energy in GeV
        sdir: Direction of propagation (+1 forward, -1 backward)
        bad_bpms: List of BPM names to exclude from analysis
        seq_name: Name of the sequence in MAD-NG (if not default)
        debug: Enable debug mode for MAD interface
        mad_logfile: Path to MAD log file (can be None)
        optimise_knobs: List of global knob names to optimise
    """

    start_bpm: str
    end_bpm: str
    magnet_range: str
    sequence_file_path: Path
    corrector_strengths: Path
    tune_knobs_file: Path
    beam_energy: float
    sdir: int = 1
    bad_bpms: list[str] | None = None
    seq_name: str | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    optimise_knobs: list[str] | None = None

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
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)


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
    init_coords: dict[str, np.ndarray]  # beta11, beta22, alfa11, alfa22, dx, dpx, dy, dpy


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
            Array of weights (1/variance for valid entries, 0 otherwise)
        """
        weights = np.zeros_like(variances, dtype=np.float64)
        valid = np.isfinite(variances) & (variances > 0.0)
        np.divide(1.0, variances, out=weights, where=valid)
        return weights

    @staticmethod
    def normalise_weights(weights: np.ndarray) -> np.ndarray:
        """Normalize weights so maximum weight is 1.

        Args:
            weights: Array of weight values

        Returns:
            Normalized weights with max value of 1
        """
        max_weight = np.max(weights)
        if max_weight > 0:
            return weights / max_weight
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
