"""Fitter for closed-twiss optimisation (match a measured periodic optics solution)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aba_optimiser.config import OptimiserConfig, SimulationConfig
from aba_optimiser.optimisers.levenberg_marquardt import (
    LevenbergMarquardtConfig,
    LevenbergMarquardtOptimiser,
)
from aba_optimiser.training.base_fitter import BaseFitter
from aba_optimiser.training.workers.lifecycle import WorkerLifecycleManager
from aba_optimiser.workers import ClosedTwissData, ClosedTwissWorker, Observable, WorkerConfig
from aba_optimiser.workers.common import (
    ObservableKind,
    WeightProcessor,
    hessian_uncertainties,
)

if TYPE_CHECKING:
    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.config.models import OutputConfig, SequenceConfig

logger = logging.getLogger(__name__)


#: Measurement columns backing each fittable observable, as
#: ``observable -> (value column, error column)`` in the frame produced by
#: ``tmom_recon.build_twiss_from_measurements``.
#:
#: ``mu1``/``mu2`` are special: the measured frame carries the *cumulative* phase
#: and its cumulative variance, which are differenced into per-interval advances
#: (with the variance difference) by :func:`_advance_targets`.
MEASUREMENT_COLUMNS: dict[str, tuple[str, str]] = {
    "x": ("X", "ERRX"),
    "y": ("Y", "ERRY"),
    "beta11": ("BETX", "ERRBETX"),
    "beta22": ("BETY", "ERRBETY"),
    "alfa11": ("ALFX", "ERRALFX"),
    "alfa22": ("ALFY", "ERRALFY"),
    "dx": ("DX", "ERRDX"),
    "dy": ("DY", "ERRDY"),
    "dpx": ("DPX", "ERRDPX"),
    "dpy": ("DPY", "ERRDPY"),
    # ``MUX``/``MUY`` are what ``build_twiss_from_measurements`` names the
    # cumulative phase it accumulates from the measured adjacent advances, paired
    # with the cumulative ``mu1_var``/``mu2_var``. Both are differenced back into
    # per-interval advances by :func:`_advance_targets`.
    "mu1": ("MUX", "mu1_var"),
    "mu2": ("MUY", "mu2_var"),
}

#: Observables fitted when the caller does not choose: everything the closed
#: twiss and an omc3 optics measurement independently have in common. Fitting all
#: of them is the default because each family constrains a different combination
#: of the knobs, and inverse-variance weighting means a noisy family costs
#: nothing - it simply carries little weight. Narrow this only to isolate an
#: effect deliberately, as the tests do.
#:
#: ``dpx``/``dpy`` are the one exclusion, and it is a correctness one rather than
#: conservatism: omc3 does not measure them independently but *derives* them from
#: ``DX``/``DY`` through the model transfer matrix, so including them counts one
#: measurement twice. In the vertical it is worse than redundant - the derivation
#: assumes no vertical dispersion source exists anywhere, which is exactly the
#: hypothesis a ``Dy`` fit is testing. They remain available for callers whose
#: dispersion errors come from somewhere else.
DEFAULT_OBSERVABLES: tuple[str, ...] = (
    "x",
    "y",
    "beta11",
    "beta22",
    "alfa11",
    "alfa22",
    "mu1",
    "mu2",
    "dx",
    "dy",
)


class ClosedTwissFitter(BaseFitter):
    """Optimise knobs so the model's closed twiss matches measured optics.

    Each measured momentum is fitted by one full-ring worker that computes the
    *periodic* solution with the knobs as TPSA parameters, and reads the closed
    orbit, beta, phase and dispersion - and their per-knob derivatives - off the
    resulting normal-form maps. The optimiser drives the shared knobs to minimise
    the summed weighted residual over every enabled observable.

    Nothing is seeded from the measurement. There is no starting point to
    propagate from, so measurement noise never enters as an initial condition and
    no error can be silently absorbed by an assumed anchor.

    Passing several measurements taken at different momenta (``measurements``
    keyed by the known MAD-NG momentum coordinate ``pt``) fits them jointly: ``pt`` is
    a fixed per-worker input to twiss, not a knob, so the off-momentum bend
    response and the dispersive orbit come from the physics. The differing-
    ``pt`` Jacobians are independent, which lifts the single-measurement rank
    deficiency.

    Observables in different units (beta in m, phase in turns, orbit in m) are
    made commensurable purely by inverse-variance weighting from the measured
    errors; there is deliberately no per-family scale factor to tune.
    """

    _defer_managers = True

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        measurements: dict[float, str | Path | pd.DataFrame],
        observables: tuple[str, ...] = DEFAULT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_knobs: Path | None = None,
        tune_knobs: Path | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strength: float = 0.0,
        output_config: OutputConfig | None = None,
    ):
        """Initialise the closed-twiss fitter.

        Args:
            accelerator: Accelerator instance defining machine configuration.
            sequence_config: Sequence configuration (``magnet_range`` selects the
                optimised elements; use ``"$start/$end"`` for the whole ring).
            measurements: Measured optics keyed by their known momentum offset
                MAD-NG ``pt``. Each value is either a path to an omc3 measurement
                folder or a DataFrame indexed by BPM name carrying the columns in
                :data:`MEASUREMENT_COLUMNS` for the enabled observables. A single
                on-momentum measurement is ``{0.0: measurement}``.
            observables: Observable families to fit. See :data:`DEFAULT_OBSERVABLES`.
            lm_config: Levenberg-Marquardt solve configuration.
            initial_knob_strengths: Initial knob strengths in optimisation space.
            corrector_knobs: Optional corrector strengths file.
            tune_knobs: Optional tune knobs file.
            true_strengths: Optional true strengths for reporting.
            use_errors: Weight the residual by measurement errors when available.
                When ``False``, each observable family is weighted by the inverse
                of its own mean square target, which keeps families in different
                units commensurable without any measured error bars.
            prior_strength: Dimensionless strength of an isotropic Gaussian prior
                on the knobs, centred on ``initial_knob_strengths`` (the nominal
                model). ``0`` disables it. When set, a fixed Tikhonov term
                ``alpha·I`` is added to the Gauss-Newton Hessian and
                ``alpha·(theta - theta0)`` to the gradient, with
                ``alpha = prior_strength · median(diag H)`` fixed from the first
                iteration's data Hessian. This is a MAP prior, not Marquardt
                damping: it is isotropic, fixed, and shrinks toward the nominal
                knobs rather than the current point - so it regularises the
                noise-dominated weak directions without biasing the
                well-determined ones. Start small (``1e-6``-``1e-3``) and increase
                until the recovered knobs stop chasing noise.
            output_config: Output and logging configuration.
        """
        if not measurements:
            raise ValueError("measurements must contain at least one measured optics set")
        unknown = [name for name in observables if name not in MEASUREMENT_COLUMNS]
        if unknown:
            raise ValueError(
                f"Unknown observables {unknown}; known: {sorted(MEASUREMENT_COLUMNS)}"
            )
        if not observables:
            raise ValueError("At least one observable must be fitted")
        if accelerator.optimise_energy:
            # ``pt`` is an input here, not a knob: each worker pins its own
            # measured momentum on the parametric map so cofind returns that
            # worker's off-momentum closed solution. Fitting a global energy knob
            # on top would be both unidentifiable against the per-worker pin and
            # inconsistent - the knob list MAD-NG sees would be one shorter than
            # the one the fitter aggregates over.
            raise ValueError(
                "ClosedTwissFitter does not support accelerator.optimise_energy: the "
                "momentum of each measurement is a fixed input, supplied as the key of "
                "`measurements` (as pt, not dp/p). Set optimise_energy=False."
            )

        self.lm_config = lm_config or LevenbergMarquardtConfig()
        self.diagnostics: dict[str, object] = {}
        self.observable_names = tuple(observables)
        logger.info("LevenbergMarquardtConfig: %s", self.lm_config)
        logger.info(
            "Optimising knobs to match %d measured closed twiss set(s), observables: %s",
            len(measurements),
            ", ".join(self.observable_names),
        )

        simulation_config = SimulationConfig(
            num_workers=len(measurements),
            num_batches=1,
            use_fixed_bpm=True,
        )

        super().__init__(
            accelerator=accelerator,
            optimiser_config=_base_optimiser_config(self.lm_config),
            simulation_config=simulation_config,
            sequence_config=sequence_config,
            bpm_start_points=["$start"],
            bpm_end_points=["$end"],
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            output_config=output_config,
        )

        self.use_errors = use_errors
        self.prior_strength = float(prior_strength)
        if self.prior_strength < 0.0:
            raise ValueError("prior_strength must be >= 0")
        self.measurements = {
            float(pt): load_measurement(source, self.observable_names)
            for pt, source in measurements.items()
        }

        interface_options = {
            key: value
            for key, value in (
                ("corrector_knobs", corrector_knobs),
                ("tune_knobs", tune_knobs),
            )
            if value is not None
        }
        self.worker_payloads = create_worker_payloads(
            self.measurements,
            self.observable_names,
            self.config_manager.all_bpms,
            sequence_config.magnet_range,
            sequence_config.bad_bpms,
            accelerator,
            interface_options,
            self.use_errors,
            self.mad_logfile,
            self.python_logfile,
        )

    def run(self) -> tuple[dict[str, float], dict[str, float]]:
        """Execute the closed-twiss optimisation with a Gauss-Newton solve."""
        writer = self.setup_logging("closed_twiss_opt")
        worker_manager = WorkerLifecycleManager(ClosedTwissWorker)
        self.final_knobs = None

        try:
            worker_manager.create_and_start_workers(
                [(data, config, self.simulation_config) for config, data in self.worker_payloads],
                send_handshake=False,
            )
            channels = worker_manager.channels
            if channels is None:
                raise RuntimeError("Worker channels are not initialised")

            self.final_knobs, hessian, knob_names = self._gauss_newton(channels, writer)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt: terminating closed-twiss optimisation early.")
            self.final_knobs = getattr(self, "final_knobs", None) or dict(self.initial_knobs)
            hessian, knob_names = None, list(self.config_manager.knob_names)
        finally:
            worker_manager.terminate_workers()

        if writer is not None:
            writer.close()
        logger.info("Closed-twiss optimisation complete.")
        uncertainties = _hessian_uncertainties(hessian, knob_names)
        return self.final_knobs, uncertainties

    def _gauss_newton(
        self, channels, writer
    ) -> tuple[dict[str, float], np.ndarray | None, list[str]]:
        """Levenberg-Marquardt Gauss-Newton solve over the shared knobs.

        The closed twiss is close to linear in the knob strengths over the range
        a fit explores, so a curvature-preconditioned step (solve
        ``(H + lam·diag H) delta = -g`` with the summed Gauss-Newton Hessian
        ``H``) converges in a handful of iterations and, unlike plain gradient
        descent, actually resolves the weakly-conditioned directions the
        multi-delta measurements make identifiable.
        """
        knob_names = list(self.config_manager.knob_names)
        current = np.array([float(self.initial_knobs[name]) for name in knob_names], dtype=float)
        prior_mean = current.copy()
        prior_alpha: float | None = None
        run_start = time.time()

        optimiser = LevenbergMarquardtOptimiser(self.lm_config, initial_params=current)
        last_update = None
        completed_iterations = 0

        for iteration in range(self.lm_config.max_iterations):
            completed_iterations = iteration + 1
            current_knobs = dict(zip(knob_names, (float(v) for v in current), strict=False))
            loss, grad, hessian, _hessian_phys, particle_loss = self._collect_gn(
                channels, current_knobs, knob_names
            )
            if self.prior_strength > 0.0 and not particle_loss:
                if prior_alpha is None:
                    prior_alpha = _prior_alpha(self.prior_strength, hessian)
                loss, grad, hessian = _apply_prior(
                    loss, grad, hessian, current, prior_mean, prior_alpha
                )
            update = optimiser.update(current, loss, grad, hessian, particle_loss)
            last_update = update
            current = update.next_params

            # A rejected step is not a no-op: the optimiser has already retried
            # from its best point with more damping, so ``current`` is a new,
            # shorter step that still needs evaluating. Only the terminal
            # reasons (no curvature to retry from, or damping driven to the
            # ceiling without improvement) end the solve.
            if update.converged and not update.accepted:
                logger.warning(
                    "Levenberg-Marquardt stopped at iter %d (%s, lam=%.1e)",
                    iteration,
                    update.reason,
                    update.damping,
                )
                break
            if update.reason == "failed":
                logger.warning(
                    "Iter %d: closed orbit lost; retrying from best (lam=%.1e)",
                    iteration,
                    update.damping,
                )
                continue
            if not update.accepted:
                logger.info(
                    "Iter %d: step rejected (loss %.6e >= best %.6e); retrying from best "
                    "(lam=%.1e)",
                    iteration,
                    update.loss,
                    optimiser.best_loss,
                    update.damping,
                )
                continue

            self._log_gn_iteration(
                writer, iteration, update.loss, update.grad_norm, update.damping, run_start
            )

            if update.converged:
                logger.info(
                    "Levenberg-Marquardt converged (%s) at iter %d", update.reason, iteration
                )
                break

        best_knobs = dict(zip(knob_names, (float(v) for v in optimiser.best_params), strict=False))
        self.diagnostics = {
            "converged": bool(last_update is not None and last_update.converged),
            "reason": None if last_update is None else last_update.reason,
            "iterations": completed_iterations,
            "best_loss": float(optimiser.best_loss),
            "gradient_norm": (
                None if last_update is None else float(last_update.grad_norm)
            ),
            "damping": None if last_update is None else float(last_update.damping),
        }
        # The optimiser's Hessian is in the worker's normalised weight space, so
        # its inverse is not a covariance in physical knob units. Re-evaluate the
        # physical normal matrix ``JᵀWJ`` (true inverse-variance weights) at the
        # solution and regularise it with the same isotropic prior the fit used,
        # so the reported 1-sigma is the MAP posterior width in real units.
        _l, _g, _h, normal_matrix, particle_loss = self._collect_gn(
            channels, best_knobs, knob_names
        )
        if not particle_loss and self.prior_strength > 0.0:
            normal_matrix = normal_matrix + _prior_alpha(
                self.prior_strength, normal_matrix
            ) * np.eye(normal_matrix.shape[0])
        return best_knobs, normal_matrix, knob_names

    def _collect_gn(
        self, channels, knobs: dict[str, float], knob_names: list[str]
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Send knobs to every worker and sum their loss, gradient and Hessians.

        Returns both the normalised Hessian the optimiser steps with and the
        physical (un-normalised, true inverse-variance) Hessian whose inverse is
        the parameter covariance in real knob units.
        """
        channels.send_all((knobs, 0))
        results = channels.recv_all()
        if not results:
            raise RuntimeError("No closed-twiss workers returned results")

        n = len(knob_names)
        total_loss = 0.0
        agg_grad = np.zeros(n)
        agg_hess = np.zeros((n, n))
        agg_hess_phys = np.zeros((n, n))
        particle_loss = False
        for result in results:
            if not isinstance(result, tuple) or len(result) != 5:
                raise RuntimeError(f"Unexpected closed-twiss worker payload: {result!r}")
            _, grad, loss, hessian, hessian_phys = result
            if loss == float("inf"):
                raise RuntimeError("Worker error detected during closed-twiss optimisation")
            if np.isnan(loss):
                particle_loss = True
                continue
            agg_grad += np.asarray(grad, dtype=float)
            agg_hess += np.asarray(hessian, dtype=float)
            agg_hess_phys += np.asarray(hessian_phys, dtype=float)
            total_loss += float(loss)

        return total_loss, agg_grad, agg_hess, agg_hess_phys, particle_loss

    def _log_gn_iteration(
        self, writer, iteration: int, loss: float, grad_norm: float, lam: float, run_start: float
    ) -> None:
        """Log one Gauss-Newton iteration to the console and TensorBoard."""
        logger.info(
            "GN iter %d: loss=%.3e, |g|=%.3e, lam=%.1e, tt=%.1fs",
            iteration,
            loss,
            grad_norm,
            lam,
            time.time() - run_start,
        )
        if writer is not None:
            writer.add_scalar("loss", loss, iteration)
            writer.add_scalar("grad_norm", grad_norm, iteration)
            writer.add_scalar("lm_lambda", lam, iteration)
            writer.flush()


def _base_optimiser_config(lm_config: LevenbergMarquardtConfig) -> OptimiserConfig:
    """Build the minimal config required by BaseFitter setup.

    ClosedTwissFitter defers the shared optimisation-loop manager and runs its own
    LM solve, so only the logging/configuration-manager path observes this.
    """
    return OptimiserConfig(
        max_epochs=lm_config.max_iterations,
        warmup_epochs=0,
        warmup_lr_start=1.0,
        max_lr=1.0,
        min_lr=1.0,
        gradient_converged_value=lm_config.gradient_converged_value,
        optimiser_type="lbfgs",
    )


def _prior_alpha(prior_strength: float, data_hessian: np.ndarray) -> float:
    """Fix the isotropic Tikhonov coefficient from the first data Hessian.

    Scaling to ``median(diag H)`` makes ``prior_strength`` dimensionless and
    invariant to the worker-side weight normalisation, so the same value
    regularises regardless of the absolute residual scale.
    """
    diag = np.abs(np.diag(np.asarray(data_hessian, dtype=float)))
    positive = diag[diag > 0.0]
    scale = float(np.median(positive)) if positive.size else 0.0
    alpha = prior_strength * scale
    logger.info(
        "Knob prior enabled: alpha=%.3e (strength=%.3e x median diag H=%.3e)",
        alpha,
        prior_strength,
        scale,
    )
    return alpha


def _apply_prior(
    loss: float,
    grad: np.ndarray,
    hessian: np.ndarray,
    params: np.ndarray,
    prior_mean: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Add an isotropic Gaussian knob prior to the loss, gradient and Hessian.

    Implements the MAP term ``0.5·alpha·||theta - theta0||²`` consistently with
    the worker convention (``grad = dL/dtheta``, ``hessian = d²L/dtheta²``): the
    gradient gains ``alpha·(theta - theta0)`` and the Hessian ``alpha·I``. Because
    ``alpha·I`` is isotropic and fixed (unlike Marquardt's annealed
    ``lam·diag H``), it floors every eigenvalue of ``H`` at ``alpha``, damping the
    noise-amplifying weak directions while shrinking toward the nominal knobs
    rather than the current point.
    """
    delta = params - prior_mean
    grad = grad + alpha * delta
    hessian = hessian + alpha * np.eye(hessian.shape[0])
    loss = loss + 0.5 * alpha * float(delta @ delta)
    return loss, grad, hessian


def _hessian_uncertainties(
    normal_matrix: np.ndarray | None, knob_names: list[str]
) -> dict[str, float]:
    """1-sigma knob uncertainties from the physical normal matrix ``JᵀWJ``.

    Delegates the covariance numerics to the shared :func:`hessian_uncertainties`
    (symmetrise + eigenvalue floor) so weakly-constrained directions yield finite,
    non-negative uncertainties in real knob units. Returns NaN for an absent
    matrix (e.g. an interrupted fit).
    """
    if normal_matrix is None:
        return {name: float("nan") for name in knob_names}
    sigmas = hessian_uncertainties(normal_matrix)
    return dict(zip(knob_names, (float(s) for s in sigmas), strict=False))


def load_measurement(
    source: str | Path | pd.DataFrame, observables: tuple[str, ...]
) -> pd.DataFrame:
    """Load a measured optics set as a DataFrame indexed by BPM name.

    Args:
        source: Either an omc3 measurement folder or a DataFrame already carrying
            the columns listed in :data:`MEASUREMENT_COLUMNS`.
        observables: Observable names that must be present.

    Returns:
        DataFrame indexed by BPM name. Missing error columns are filled with NaN,
        which drops those points unless ``use_errors=False``.
    """
    if isinstance(source, pd.DataFrame):
        measurement = source.copy()
    else:
        from tmom_recon import build_twiss_from_measurements

        logger.info("Loading optics from measurement folder %s", source)
        measurement, has_dispersion = build_twiss_from_measurements(
            Path(source), include_errors=True
        )
        if not has_dispersion and any(name in ("dx", "dy") for name in observables):
            raise ValueError(
                f"Dispersion observables requested but {source} carries no dispersion data"
            )

    missing = [
        MEASUREMENT_COLUMNS[name][0]
        for name in observables
        if MEASUREMENT_COLUMNS[name][0] not in measurement.columns
    ]
    if missing:
        raise ValueError(f"Measurement is missing required columns: {sorted(missing)}")

    for name in observables:
        err_column = MEASUREMENT_COLUMNS[name][1]
        if err_column not in measurement.columns:
            measurement[err_column] = np.nan
    return measurement


def _advance_targets(
    measurement: pd.DataFrame, bpms: list[str], value_column: str, var_column: str
) -> tuple[np.ndarray, np.ndarray]:
    """Per-interval phase advance and its variance from a cumulative measurement.

    ``build_twiss_from_measurements`` accumulates the measured adjacent advances
    into a cumulative phase (and a cumulative variance), so differencing
    consecutive BPMs recovers exactly the independent advances that were measured
    - and their independent variances. The modulo keeps the result in ``[0, 1)``
    to match the wrapped model advance.
    """
    phase = measurement.loc[bpms, value_column].to_numpy(dtype=float)
    cumulative_var = measurement.loc[bpms, var_column].to_numpy(dtype=float)
    return np.mod(np.diff(phase), 1.0), np.diff(cumulative_var)


def create_worker_payloads(
    measurements: dict[float, pd.DataFrame],
    observables: tuple[str, ...],
    all_bpms: list[str],
    magnet_range: str,
    bad_bpms: list[str] | None,
    accelerator: Accelerator,
    interface_options: dict,
    use_errors: bool,
    mad_logfile: Path | None,
    python_logfile: Path | None,
) -> list[tuple[WorkerConfig, ClosedTwissData]]:
    """Build one full-ring closed-twiss worker payload per measured momentum.

    Every worker shares the same knobs and full-ring config; they differ only in
    their measurement and its fixed MAD-NG momentum coordinate ``pt``.

    The loss normalisation is stamped on afterwards, from every worker's
    observables at once: it has to be one common constant, or the workers are not
    minimising the same objective. See :func:`_stamp_global_normalisation`.
    """
    payloads = [
        _create_worker_payload(
            pt,
            measurement,
            observables,
            all_bpms,
            magnet_range,
            bad_bpms,
            accelerator,
            interface_options,
            use_errors,
            mad_logfile,
            python_logfile,
        )
        for pt, measurement in measurements.items()
    ]
    _stamp_global_normalisation(payloads)
    return payloads


def _stamp_global_normalisation(payloads: list[tuple[WorkerConfig, ClosedTwissData]]) -> None:
    """Give every worker the same weight scale and point count.

    The workers' losses are summed by the fitter, so any per-worker scaling of a
    worker's own loss is a re-weighting of that momentum in the joint fit. Both
    normalisations therefore have to be global constants:

    ``weight_scale``
        The largest inverse-variance weight anywhere in the fit. Dividing by it
        keeps the numbers near unity without changing any *relative* weight, so
        the argmin is exactly that of the pooled chi-square. Taking each worker's
        own maximum instead would divide each momentum by a different number.
    ``total_points``
        The total number of weighted points in the fit. This converts the summed
        chi-square into a mean, which is what the Levenberg-Marquardt damping
        scale is tuned against; dividing by each worker's own count would make a
        momentum measured at fewer BPMs count for more per point.

    Both cancel out of the Gauss-Newton step ``H^-1 g`` when applied uniformly,
    which is precisely why they must be uniform: the fit is then invariant to
    them, and only the physical inverse-variance weights decide the answer.
    """
    weights = [
        WeightProcessor.variance_to_weight(np.asarray(observable.variances, dtype=float))
        for _config, data in payloads
        for observable in data.observables
    ]
    largest = max((float(np.max(w)) for w in weights if w.size), default=0.0)
    scale = largest if largest > 0.0 else 1.0
    points = max(1, sum(int(np.count_nonzero(w)) for w in weights))
    logger.info(
        "Global loss normalisation over %d worker(s): weight scale %.6e, %d weighted points",
        len(payloads),
        scale,
        points,
    )
    for _config, data in payloads:
        data.weight_scale = scale
        data.total_points = points


def _create_worker_payload(
    pt: float,
    measurement: pd.DataFrame,
    observables: tuple[str, ...],
    all_bpms: list[str],
    magnet_range: str,
    bad_bpms: list[str] | None,
    accelerator: Accelerator,
    interface_options: dict,
    use_errors: bool,
    mad_logfile: Path | None,
    python_logfile: Path | None,
) -> tuple[WorkerConfig, ClosedTwissData]:
    """Build a single full-ring closed-twiss worker payload for one momentum.

    BPMs without a measurement are added to ``bad_bpms`` so twiss does not observe
    them, keeping the model observables aligned with the measured targets.
    """
    measured_bpms = [bpm for bpm in all_bpms if bpm in measurement.index]
    if len(measured_bpms) < 2:
        raise ValueError(f"Fewer than two model BPMs have a measurement (pt={pt}).")

    unmeasured = [bpm for bpm in all_bpms if bpm not in measurement.index]
    logger.info(
        "Closed-twiss fit (pt=%g) over %d measured BPMs (%d model BPMs unobserved)",
        pt,
        len(measured_bpms),
        len(unmeasured),
    )

    observable_data = [
        _build_observable(name, measurement, measured_bpms, use_errors, pt)
        for name in observables
    ]

    worker_bad_bpms = list(unmeasured) + (list(bad_bpms) if bad_bpms else [])
    config = WorkerConfig(
        accelerator=accelerator,
        tracking_start_bpm="$start",
        tracking_end_bpm="$end",
        magnet_range=magnet_range,
        interface_options=interface_options,
        cycle_sequence=False,
        sdir=1,
        bad_bpms=worker_bad_bpms,
        mad_logfile=mad_logfile,
        python_logfile=python_logfile,
    )
    data = ClosedTwissData(
        bpm_names=measured_bpms,
        observables=observable_data,
        pt=pt,
    )
    return (config, data)


def _build_observable(
    name: str,
    measurement: pd.DataFrame,
    bpms: list[str],
    use_errors: bool,
    pt: float,
) -> Observable:
    """Extract one observable's targets and variances from a measurement frame."""
    value_column, err_column = MEASUREMENT_COLUMNS[name]
    observable = Observable(name=name, targets=np.empty(0), variances=np.empty(0))

    if observable.kind is ObservableKind.ADVANCE:
        # The phase "error" column is already a cumulative *variance*, so
        # differencing gives the per-interval variance directly.
        targets, variances = _advance_targets(measurement, bpms, value_column, err_column)
    else:
        targets = measurement.loc[bpms, value_column].to_numpy(dtype=float)
        errors = measurement.loc[bpms, err_column].to_numpy(dtype=float)
        variances = errors**2

    if not use_errors or not np.any(np.isfinite(variances) & (variances > 0)):
        # Weight each family by the inverse of its own mean square target. Without
        # this an unweighted fit would be dominated by whichever family happens to
        # carry the largest numbers (beta in metres against orbit in millimetres),
        # which is a units artefact rather than a statement about information.
        scale = float(np.mean(targets[np.isfinite(targets)] ** 2)) if targets.size else 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        logger.warning(
            "No usable errors for '%s' (pt=%g); weighting by 1/<target^2> = %.3e.",
            name,
            pt,
            1.0 / scale,
        )
        variances = np.full_like(targets, scale, dtype=float)

    return Observable(name=name, targets=targets, variances=variances)
