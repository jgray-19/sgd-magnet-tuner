from __future__ import annotations

from aba_optimiser.measurements.output import measurement_output_config


def test_measurement_output_config_uses_stage_directory(tmp_path):
    output_config = measurement_output_config(tmp_path, "arc_1")

    stage_dir = tmp_path / "arc_1"
    assert stage_dir.is_dir()
    assert output_config.tensorboard_root == stage_dir / "tensorboard"
    assert output_config.mad_logfile == stage_dir / "mad_log.txt"
    assert output_config.python_logfile == stage_dir / "python_worker_log.txt"


def test_measurement_output_config_accepts_concrete_analysis_dir(tmp_path):
    analysis_dir = tmp_path / "single_stage"

    output_config = measurement_output_config(
        analysis_dir,
        include_uncertainty=False,
        write_tensorboard_logs=False,
        parallel_hessian=2,
    )

    assert analysis_dir.is_dir()
    assert output_config.tensorboard_root == analysis_dir / "tensorboard"
    assert output_config.mad_logfile == analysis_dir / "mad_log.txt"
    assert output_config.python_logfile == analysis_dir / "python_worker_log.txt"
    assert output_config.include_uncertainty is False
    assert output_config.write_tensorboard_logs is False
    assert output_config.parallel_hessian == 2
