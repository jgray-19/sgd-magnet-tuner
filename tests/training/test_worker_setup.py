from __future__ import annotations

from pathlib import Path

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.worker_setup import WorkerRangeSpec, WorkerSetupHelper


def _make_helper(tmp_path: Path) -> WorkerSetupHelper:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    accelerator = SPS(sequence_file=seq_file, beam_energy=450.0)
    accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    return WorkerSetupHelper(
        accelerator=accelerator,
        all_bpms=["BPV.13108", "BPH.13208", "BPV.13308", "BPV.20108"],
        fixed_start="BPH.13208",
        fixed_end="BPV.20108",
        use_fixed_bpm=True,
        bad_bpms=None,
        file_kick_planes={0: "x", 1: "xy"},
        magnet_range="$start/$end",
        corrector_strengths_files=[tmp_path / "corr0.tfs", tmp_path / "corr1.tfs"],
        tune_knobs_files=[tmp_path / "knobs0.txt", tmp_path / "knobs1.txt"],
        debug=False,
        mad_logfile=None,
    )


def test_build_range_specs_for_multi_turn_creates_forward_and_backward_ranges(tmp_path: Path) -> None:
    helper = _make_helper(tmp_path)

    specs = helper.build_range_specs(
        start_bpms=["BPH.13208"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=2,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
    )

    assert specs == [
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=-1),
    ]


def test_build_observation_plan_filters_single_plane_bpms_and_requires_measurable_start(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    forward_plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        file_idx=0,
    )
    assert len(forward_plans) == 1
    forward_plan = forward_plans[0]
    assert forward_plan.kick_plane == "x"
    assert forward_plan.bpm_names == ["BPH.13208"]

    backward_plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=-1),
        file_idx=0,
    )
    assert backward_plans == []


def test_build_observation_plans_keeps_dual_plane_worker_for_dual_plane_bpms(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)
    helper.accelerator.infer_monitor_plane = lambda bpm: "HV"  # type: ignore[method-assign]

    plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        file_idx=1,
    )

    assert len(plans) == 1
    assert plans[0].kick_plane == "xy"
    assert plans[0].bpm_names == ["BPH.13208", "BPV.13308", "BPV.20108", "BPV.13108"]


def test_build_observation_plans_split_dual_plane_data_across_single_plane_bpms(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    forward_plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        file_idx=1,
    )
    backward_plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=-1),
        file_idx=1,
    )

    assert [(plan.kick_plane, plan.bpm_names) for plan in forward_plans] == [
        ("x", ["BPH.13208"]),
    ]
    assert [(plan.kick_plane, plan.bpm_names) for plan in backward_plans] == [
        ("y", ["BPV.13108", "BPV.20108", "BPV.13308"]),
    ]
    assert forward_plans[0].bad_bpms == ["BPV.13308", "BPV.20108", "BPV.13108"]
    assert backward_plans[0].bad_bpms == ["BPH.13208"]


def test_make_worker_config_uses_file_specific_artifacts(tmp_path: Path) -> None:
    helper = _make_helper(tmp_path)
    helper.accelerator.infer_monitor_plane = lambda bpm: "HV"  # type: ignore[method-assign]
    plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        file_idx=1,
    )

    assert len(plans) == 1
    config = helper.make_worker_config(plans[0])

    assert config.corrector_strengths == tmp_path / "corr1.tfs"
    assert config.tune_knobs_file == tmp_path / "knobs1.txt"
    assert config.kick_plane == "xy"
