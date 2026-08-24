"""Public closed-orbit fitter for LOCO-style machine-state comparisons."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aba_optimiser.training_closed_twiss.fitter import (
    LevenbergMarquardtConfig,
    _create_worker_payload,
    _GaussNewtonFitter,
    _stamp_global_normalisation,
    load_measurement,
)
from aba_optimiser.workers.closed_orbit import (
    ClosedOrbitMeasurementData,
    ClosedOrbitSeriesData,
    ClosedOrbitWorker,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.training.config.models import OutputConfig, SequenceConfig

LOGGER = logging.getLogger(__name__)
CLOSED_ORBIT_OBSERVABLES = ("x", "y")


@dataclass(frozen=True)
class ClosedOrbitMeasurement:
    """One measured orbit at a signal momentum and its reference momentum."""

    orbit: pd.DataFrame
    pt: float = 0.0
    reference_pt: float = 0.0


@dataclass(frozen=True)
class ClosedOrbitSeries:
    """Measurements evaluated in one process under one control-knob trim.

    Each measurement keeps its own target and momentum. Combining measurements
    here changes process layout only; their residuals and normal equations are
    still evaluated and summed independently.
    """

    measurements: tuple[ClosedOrbitMeasurement, ...]
    control_knob: str | None = None
    control_nominal: float = 0.0
    control_delta: float = 0.0
    absolute_planes: tuple[str, ...] = ()
    label: str = ""


class ClosedOrbitFitter(_GaussNewtonFitter):
    """Fit magnet knobs to absolute or reference-subtracted closed orbits.

    A :class:`ClosedOrbitSeries` is the unit assigned to one MAD-NG process.
    Putting several measurements in one series changes only process layout:
    each keeps its own target, ``pt`` and ``reference_pt``, and contributes an
    independent residual/Jacobian block to the same global objective.
    """

    worker_class = ClosedOrbitWorker
    log_suffix = "closed_orbit_opt"
    fit_label = "Closed-orbit"

    def __init__(
        self,
        accelerator: Accelerator,
        sequence_config: SequenceConfig,
        series: list[ClosedOrbitSeries] | tuple[ClosedOrbitSeries, ...],
        observables: tuple[str, ...] = CLOSED_ORBIT_OBSERVABLES,
        lm_config: LevenbergMarquardtConfig | None = None,
        initial_knob_strengths: dict[str, float] | None = None,
        corrector_knobs: Path | Mapping[str, float] | None = None,
        tune_knobs: Path | Mapping[str, float] | None = None,
        true_strengths: Path | dict[str, float] | None = None,
        use_errors: bool = True,
        prior_strengths: Mapping[str, float] | None = None,
        output_config: OutputConfig | None = None,
    ) -> None:
        if not series:
            raise ValueError("series must contain at least one closed-orbit series")
        observables = tuple(observables)
        if not observables:
            raise ValueError("At least one closed-orbit observable must be fitted")
        unsupported = [name for name in observables if name not in CLOSED_ORBIT_OBSERVABLES]
        if unsupported:
            raise ValueError(
                f"ClosedOrbitFitter supports {CLOSED_ORBIT_OBSERVABLES}, got {unsupported}"
            )
        for item in series:
            if not item.measurements:
                raise ValueError(
                    f"Closed-orbit series {item.label or '<unnamed>'!r} has no measurements"
                )

        self.observable_names = observables
        self.series = tuple(series)
        super().__init__(
            accelerator,
            sequence_config,
            num_workers=len(self.series),
            lm_config=lm_config,
            initial_knob_strengths=initial_knob_strengths,
            true_strengths=true_strengths,
            use_errors=use_errors,
            prior_strengths=prior_strengths,
            output_config=output_config,
        )

        interface_options = {
            key: value
            for key, value in (
                ("corrector_knobs", corrector_knobs),
                ("tune_knobs", tune_knobs),
            )
            if value is not None
        }
        self.worker_payloads = self._create_series_payloads(
            sequence_config, accelerator, interface_options
        )

    def _create_series_payloads(self, sequence_config, accelerator, interface_options):
        payloads = []
        for item in self.series:
            frames = [
                load_measurement(measurement.orbit, self.observable_names)
                for measurement in item.measurements
            ]
            common_bpms = [
                bpm
                for bpm in self.config_manager.all_bpms
                if all(bpm in frame.index for frame in frames)
            ]
            if len(common_bpms) < 2:
                raise ValueError(
                    f"Fewer than two common model BPMs in closed-orbit series "
                    f"{item.label or '<unnamed>'!r}"
                )

            measurement_data = []
            config = None
            for measurement, frame in zip(item.measurements, frames, strict=True):
                worker_config, base_data = _create_worker_payload(
                    float(measurement.pt),
                    frame.loc[common_bpms],
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
                config = config or worker_config
                measurement_data.append(
                    ClosedOrbitMeasurementData(
                        observables=base_data.observables,
                        pt=float(measurement.pt),
                        reference_pt=float(measurement.reference_pt),
                    )
                )

            payloads.append(
                (
                    config,
                    ClosedOrbitSeriesData(
                        bpm_names=common_bpms,
                        measurements=measurement_data,
                        control_knob=item.control_knob,
                        control_nominal=float(item.control_nominal),
                        control_delta=float(item.control_delta),
                        absolute_planes=tuple(item.absolute_planes),
                    ),
                )
            )
        _stamp_global_normalisation(payloads)
        LOGGER.info(
            "Closed-orbit fit: %d worker series, %d independent measurements",
            len(payloads),
            sum(len(item.measurements) for item in self.series),
        )
        return payloads
