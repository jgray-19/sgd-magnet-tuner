"""Common data structures and utilities for all worker types.

This module defines shared data structures, configurations, and utility functions
used across different worker implementations (tracking and optics modes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator

logger = logging.getLogger(__name__)

# Eigenvalues of the normal matrix below this floor are treated as unconstrained
# directions: flooring them keeps the inverted covariance finite and non-negative
# instead of exploding (or going negative through numerical noise).
HESSIAN_MIN_EIGENVALUE = 1e-35

class KickPlane(str, Enum):
    """Kick-plane options for worker routing and payload selection."""

    X = "x"
    Y = "y"
    XY = "xy"


@dataclass
class WorkerConfig:
    """Configuration shared by all worker processes.

    The accelerator object bundles machine-specific setup, while the remaining
    fields describe the local BPM range, tracking direction, and optional input
    files needed by the worker.

    Note:
        The old `start_bpm` / `end_bpm` / `observation_start_bpm` /
        `init_marker` names were removed in favour of explicit tracking and
        initial-condition terminology.
    """

    accelerator: Accelerator
    tracking_start_bpm: str
    tracking_end_bpm: str
    magnet_range: str
    # Per-measurement keyword arguments forwarded to the MAD-NG interface, e.g.
    # corrector_knobs, tune_knobs, b2_errors.
    interface_options: dict[str, Any] = field(default_factory=dict)
    observation_range_start_bpm: str | None = None
    initial_condition_marker: str | None = None
    # When False the worker leaves the sequence at its natural ``$start`` instead of
    # cycling to ``tracking_start_bpm``. Cycling to a BPM places that BPM at both the
    # ring start and the wrap, so full-ring multi-turn tracking observes it twice per
    # turn and overflows the result vectors; full-ring workers therefore track from the
    # fixed turn-increment start (``$start``) and only compare against their BPM range.
    cycle_sequence: bool = True
    sdir: int = 1
    kick_plane: KickPlane = KickPlane.XY
    bad_bpms: list[str] | None = None
    debug: bool = False
    mad_logfile: Path | None = None
    python_logfile: Path | None = None
    tracking_anchor_mode: str | None = None
    tracking_anchor_sources: list[str] | None = None
    observed_tracking_anchor_markers: list[str] | None = None
    cycle_marker: str | None = None


@dataclass
class PrecomputedTrackingWeights:
    """Per-observable weights reused by multiple tracking workers.

    The normalised arrays are used in the loss and gradient calculation, while
    the Hessian arrays keep the unfloored aggregate weights needed for the
    approximate second-order terms.
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
    """Reference data for a tracking-loss evaluation.

    Position and momentum comparison arrays use shape
    ``(n_particles, n_data_points, 2)``, with the last axis storing the two
    transverse components for each observable family.
    """

    position_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_comparisons: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    position_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    momentum_variances: np.ndarray  # Shape: (n_particles, n_data_points, 2)
    init_coords: np.ndarray  # Shape: (n_particles, 6)
    init_pts: np.ndarray  # Shape: (n_particles,)
    precomputed_weights: PrecomputedTrackingWeights | None


class ObservableKind(str, Enum):
    """How a per-BPM observable is turned into a residual.

    ``POINTWISE``
        The model value at each BPM is compared directly against the measurement:
        closed orbit, beta, alpha, dispersion.
    ``ADVANCE``
        Consecutive BPM values are differenced first, so a cumulative model
        quantity is compared against a measured *advance*. Used for phase, where
        only the BPM-to-BPM advance is measurable and the absolute value carries
        an arbitrary origin.
    """

    POINTWISE = "pointwise"
    ADVANCE = "advance"


#: Observables the MAD-NG side knows how to evaluate on the closed twiss, mapped
#: to how their residual is formed. Names are ``gphys.optfun`` function names,
#: except the closed-orbit coordinates which are read off the map's constant part.
#:
#: ``beta11``/``beta22`` are the uncoupled Ripken betas (the ``beta11``/``beta22``
#: twiss columns). The coupled Edwards-Teng ``betx``/``bety`` are deliberately not
#: offered: ``gphys`` routes those through ``nf_cplg``, which needs the beam's
#: relativistic beta on the map and is unavailable on a bare saved map.
#:
#: ``dx``/``dy`` are ``d(x)/d(pt)`` and ``d(y)/d(pt)`` - the MAD-X ``DX`` convention.
#: ``dpx``/``dpy`` are available but should normally be left out when fitting
#: omc3 output: omc3 derives its ``DPY`` from ``DY`` through the *model* transfer
#: matrix, so including both double-counts one measurement.
OBSERVABLE_KINDS: dict[str, ObservableKind] = {
    "x": ObservableKind.POINTWISE,
    "y": ObservableKind.POINTWISE,
    "px": ObservableKind.POINTWISE,
    "py": ObservableKind.POINTWISE,
    "beta11": ObservableKind.POINTWISE,
    "beta22": ObservableKind.POINTWISE,
    "alfa11": ObservableKind.POINTWISE,
    "alfa22": ObservableKind.POINTWISE,
    "dx": ObservableKind.POINTWISE,
    "dy": ObservableKind.POINTWISE,
    "dpx": ObservableKind.POINTWISE,
    "dpy": ObservableKind.POINTWISE,
    "mu1": ObservableKind.ADVANCE,
    "mu2": ObservableKind.ADVANCE,
}


@dataclass
class Observable:
    """One measured observable family, aligned to ``ClosedTwissData.bpm_names``.

    ``targets`` and ``variances`` have one entry per BPM for a ``POINTWISE``
    observable and one per *interval* (``n_bpms - 1``) for an ``ADVANCE`` one.
    A non-finite or non-positive variance drops that point from the fit, which is
    how partially-measured planes are handled without special-casing.
    """

    name: str
    targets: np.ndarray
    variances: np.ndarray

    @property
    def kind(self) -> ObservableKind:
        """Residual form for this observable."""
        try:
            return OBSERVABLE_KINDS[self.name]
        except KeyError:
            raise ValueError(
                f"Unknown observable '{self.name}'. Known: {sorted(OBSERVABLE_KINDS)}"
            ) from None


@dataclass
class ClosedTwissData:
    """Reference closed-twiss measurements for one momentum, for one worker.

    Arrays are ordered to match the model BPM ordering (the order the sequence's
    monitors are observed by twiss). ``bpm_names`` records that order so the
    worker can align the twiss output to these comparisons by name.

    ``pt`` is the known MAD-NG momentum coordinate of this measurement.
    It is a fixed input to twiss (``x0map.pt``), never an optimisation knob, so
    both the off-momentum bend response and the dispersive orbit come from the
    physics. Fitting several ``pt`` values jointly makes the per-magnet
    Jacobians independent and lifts the single-measurement rank deficiency.

    ``weight_scale`` and ``total_points`` are the *global* loss normalisation and
    must be identical across every worker in a fit. Each worker divides its
    inverse-variance weights by ``weight_scale`` and its loss/gradient/Hessian by
    ``total_points``. Were these derived per worker - from its own largest weight
    and its own point count - the optimiser would minimise
    ``sum_w (1/(max_w * N_w)) * chi2_w`` rather than the pooled ``sum_w chi2_w``,
    so the most precisely measured momentum would be silently down-weighted for
    being precise, and the reported 1-sigma (built from the un-normalised
    ``JᵀWJ``) would describe an estimator different from the one that was
    actually minimised. They are computed once, over every worker's
    observables, by :func:`create_worker_payloads`.
    """

    bpm_names: list[str]
    observables: list[Observable]
    pt: float = 0.0
    weight_scale: float = 1.0
    total_points: int = 1


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


def hessian_uncertainties(
    normal_matrix: np.ndarray,
    *,
    min_eigenvalue: float = HESSIAN_MIN_EIGENVALUE,
) -> np.ndarray:
    """1-sigma parameter uncertainties from a Gauss-Newton normal matrix.

    ``normal_matrix`` must be the weighted normal matrix ``A = JᵀWJ`` built with
    *physical* inverse-variance weights ``W = 1/σ²`` (units ``1/measurement²``).
    Its inverse is then the parameter covariance and ``sqrt(diag(A⁻¹))`` gives the
    1-sigma uncertainties in real parameter units - so callers must pass the
    un-normalised, true-variance Hessian, never one rescaled by an arbitrary
    loss-normalisation (which would put the result in a meaningless space).

    Note the factor-of-2 convention: for a chi-square ``Σ w r²`` the second
    derivative is ``2 JᵀWJ``; pass ``A = JᵀWJ`` here (half that), i.e. the normal
    matrix / Fisher information, not the raw chi-square Hessian.

    The matrix is symmetrised and its eigenvalues floored to ``min_eigenvalue`` so
    weakly-constrained or rank-deficient directions yield large-but-finite,
    non-negative uncertainties rather than blowing up or turning negative through
    accumulated numerical noise.
    """
    matrix = np.asarray(normal_matrix, dtype=np.float64)
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.maximum(eigenvalues, min_eigenvalue)

    n_clipped = int(np.count_nonzero(eigenvalues < min_eigenvalue))
    if n_clipped:
        logger.warning(
            "Normal matrix had %d eigenvalue(s) below %.3e; using the floor to keep "
            "uncertainties finite and non-negative.",
            n_clipped,
            min_eigenvalue,
        )

    covariance = (eigenvectors / clipped) @ eigenvectors.T
    variances = np.clip(np.diag(covariance), 0.0, None)
    return np.sqrt(variances)


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
