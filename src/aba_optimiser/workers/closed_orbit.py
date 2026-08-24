"""Closed-orbit worker for absolute and reference-subtracted orbit series."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aba_optimiser.workers.closed_twiss import (
    ClosedTwissWorker,
    _align_observables,
    _weighted_loss_gradient_hessian,
)
from aba_optimiser.workers.common import ClosedTwissData, Observable

if TYPE_CHECKING:
    from pymadng import MAD

LOGGER = logging.getLogger(__name__)
ORBIT_OBSERVABLES = ("x", "y")


@dataclass
class ClosedOrbitMeasurementData:
    """Internal prepared measurement and the model states it compares."""

    observables: list[Observable]
    pt: float = 0.0
    reference_pt: float = 0.0


@dataclass
class ClosedOrbitSeriesData:
    """Internal worker payload for measurements sharing a control setting."""

    bpm_names: list[str]
    measurements: list[ClosedOrbitMeasurementData]
    control_knob: str | None = None
    control_nominal: float = 0.0
    control_delta: float = 0.0
    absolute_planes: tuple[str, ...] = ()
    weight_scale: float = 1.0
    total_points: int = 1

    @property
    def all_observables(self) -> list[Observable]:
        """Observable blocks contributing to this worker's loss."""
        return [
            observable
            for measurement in self.measurements
            for observable in measurement.observables
        ]


class ClosedOrbitWorker(ClosedTwissWorker):
    """Fit several closed-orbit measurements in one MAD-NG process.

    Every measurement retains its own target, signal momentum, and reference
    momentum. Repeated model states are cached only within an iteration; this
    makes a shared global reference cheap without conflating the signal closed
    orbits at different momenta.
    """

    def prepare_data(self, data: ClosedOrbitSeriesData) -> None:
        if not data.measurements:
            raise ValueError("ClosedOrbitSeriesData needs at least one measurement")
        first = data.measurements[0]
        if not first.observables:
            raise ValueError("A closed-orbit measurement needs at least one observable")

        names = tuple(observable.name for observable in first.observables)
        unsupported = [name for name in names if name not in ORBIT_OBSERVABLES]
        if unsupported:
            raise ValueError(f"ClosedOrbitWorker supports x/y only, got {unsupported}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate closed-orbit observables: {names}")
        for measurement in data.measurements[1:]:
            other = tuple(observable.name for observable in measurement.observables)
            if other != names:
                raise ValueError(
                    "Every measurement in a closed-orbit series must use the same "
                    f"observable order, got {other} after {names}"
                )

        absolute = tuple(dict.fromkeys(data.absolute_planes))
        unknown = set(absolute) - set(names)
        if unknown:
            raise ValueError(f"Unknown absolute plane(s) {sorted(unknown)}")
        has_relative = any(name not in absolute for name in names)
        has_momentum_signal = any(
            measurement.pt != measurement.reference_pt for measurement in data.measurements
        )
        if (
            data.control_delta == 0.0
            and not has_momentum_signal
            and has_relative
            and not absolute
        ):
            raise ValueError(
                "A closed-orbit series needs a control delta, a momentum change, "
                "or an absolute plane"
            )

        proxy = ClosedTwissData(
            bpm_names=data.bpm_names,
            observables=first.observables,
            pt=first.pt,
            weight_scale=data.weight_scale,
            total_points=data.total_points,
        )
        super().prepare_data(proxy)
        self.series_measurements = list(data.measurements)
        self.control_knob = data.control_knob
        self.control_nominal = float(data.control_nominal)
        self.control_delta = float(data.control_delta)
        self.absolute_planes = absolute
        self._subtract = np.array(
            [float(name not in absolute) for name in self.orbit_coords], dtype=float
        )
        self._measurement_alignment = None

    def _set_control(self, mad: MAD, value: float) -> None:
        if self.control_knob is not None:
            mad.send(f"MADX['{self.control_knob}'] = {value:.15e}")

    @staticmethod
    def _set_pt(mad: MAD, value: float) -> None:
        mad.send(f"x0map.pt:set0({value:.15e})")

    def _align_measurements(self, twiss_names: list[str]) -> None:
        alignments = []
        for measurement in self.series_measurements:
            targets, raw_weights, weights = _align_observables(
                measurement.observables,
                self._measured_index,
                twiss_names,
                self.weight_scale,
                worker_id=self.worker_id,
            )
            alignments.append((measurement.observables, targets, weights, raw_weights))
        self._measurement_alignment = alignments
        self._twiss_bpm_order = twiss_names

    def _model_and_jacobian(self, mad: MAD):
        mad.send("compute_closed_twiss()")
        if not mad.recv():
            LOGGER.warning(
                "Worker %s: closed orbit not found; flagging step for backtrack",
                self.worker_id,
            )
            return None

        frame = mad.twiss_tbl.to_df(columns=["name", *self.orbit_coords])
        twiss_names = list(frame["name"])
        if self._twiss_bpm_order != twiss_names:
            self._align_measurements(twiss_names)

        n_bpms = len(twiss_names)
        values = frame[list(self.orbit_coords)].to_numpy(dtype=float).T
        jacobian = np.empty((len(self.orbit_coords), n_bpms, self.n_knobs))
        mad.send("send_orbit_jacobian()")
        for index in range(len(self.orbit_coords)):
            jacobian[index] = np.asarray(mad.recv(), dtype=float).reshape(
                n_bpms, self.n_knobs
            )
        return values, jacobian

    def _compare_to_reference(self, signal, reference):
        """Subtract the reference only in the relative observable planes."""
        model = signal[0] - reference[0] * self._subtract[:, None]
        jacobian = signal[1] - reference[1] * self._subtract[:, None, None]
        return model, jacobian

    def compute_gradients_and_loss(
        self, mad: MAD, knob_updates: dict[str, float], batch: int
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        del batch
        commands = [
            f"loaded_sequence['{name}']:set0({value:.15e})"
            for name, value in knob_updates.items()
            if name in self.knob_name_set
        ]
        if commands:
            mad.send("\n".join(commands))

        failure = (
            np.zeros(self.n_knobs),
            float("nan"),
            np.zeros((self.n_knobs, self.n_knobs)),
            np.zeros((self.n_knobs, self.n_knobs)),
        )
        cache = {}

        def evaluate(control: float, pt: float):
            key = (control, pt)
            if key not in cache:
                self._set_control(mad, control)
                self._set_pt(mad, pt)
                cache[key] = self._model_and_jacobian(mad)
            return cache[key]

        gradient = np.zeros(self.n_knobs)
        loss = 0.0
        hessian = np.zeros((self.n_knobs, self.n_knobs))
        normal_matrix = np.zeros((self.n_knobs, self.n_knobs))
        signal_control = self.control_nominal + self.control_delta

        try:
            for index, measurement in enumerate(self.series_measurements):
                signal = evaluate(signal_control, float(measurement.pt))
                if signal is None:
                    return failure

                if np.any(self._subtract):
                    reference = evaluate(
                        self.control_nominal, float(measurement.reference_pt)
                    )
                    if reference is None:
                        return failure
                    model, jacobian = self._compare_to_reference(signal, reference)
                else:
                    model, jacobian = signal

                observables, targets, weights, raw_weights = self._measurement_alignment[index]
                part_grad, part_loss, part_hessian, part_normal = (
                    _weighted_loss_gradient_hessian(
                        model,
                        jacobian,
                        observables,
                        targets,
                        weights,
                        raw_weights,
                    )
                )
                gradient += part_grad
                loss += part_loss
                hessian += part_hessian
                normal_matrix += part_normal
        finally:
            self._set_control(mad, self.control_nominal)

        return gradient, loss, hessian, normal_matrix
