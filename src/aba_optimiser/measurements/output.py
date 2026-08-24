"""Output configuration helpers for measurement optimisation workflows."""

from __future__ import annotations

from pathlib import Path

from aba_optimiser.training.config.models import OutputConfig


def measurement_output_config(
    analysis_dir: Path,
    stage: str | None = None,
    *,
    include_uncertainty: bool = True,
    write_tensorboard_logs: bool = True,
    parallel_hessian: bool | int = True,
) -> OutputConfig:
    """Create a PSB-style output config rooted in a measurement analysis directory.

    When ``stage`` is provided, logs are written below ``analysis_dir / stage``.
    When omitted, ``analysis_dir`` itself is treated as the stage directory, matching
    the explicit ``OutputConfig`` pattern used in ``../psb_md``.
    """
    stage_dir = analysis_dir / stage if stage is not None else analysis_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    return OutputConfig(
        write_tensorboard_logs=write_tensorboard_logs,
        include_uncertainty=include_uncertainty,
        parallel_hessian=parallel_hessian,
        tensorboard_root=stage_dir / "tensorboard",
        mad_logfile=stage_dir / "mad_log.txt",
        python_logfile=stage_dir / "python_worker_log.txt",
    )
