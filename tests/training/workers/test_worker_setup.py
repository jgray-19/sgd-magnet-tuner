from __future__ import annotations

from pathlib import Path

from aba_optimiser.accelerators import SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.config.tracking import FullRingBpmTrackingPlan
from aba_optimiser.training.workers.setup import WorkerRangeSpec, WorkerSetupHelper


def _make_helper(tmp_path: Path) -> WorkerSetupHelper:
    seq_file = tmp_path / "sps.seq"
    seq_file.write_text("! Dummy SPS sequence file\n")
    accelerator = SPS(sequence_file=seq_file, kinetic_energy=450.0)
    accelerator.infer_monitor_plane = lambda bpm: "H" if "BPH" in bpm else "V"  # type: ignore[method-assign]
    return WorkerSetupHelper(
        accelerator=accelerator,
        all_bpms=["BPH.13008", "BPV.13108", "BPH.13208", "BPV.13308", "BPV.20108"],
        fixed_start="BPH.13208",
        fixed_end="BPV.20108",
        use_fixed_bpm=True,
        bad_bpms=None,
        file_kick_planes={0: "x", 1: "xy"},
        magnet_range="$start/$end",
        interface_options_per_file=[
            {"corrector_strengths": tmp_path / "corr0.tfs", "tune_knobs_file": tmp_path / "knobs0.txt"},
            {"corrector_strengths": tmp_path / "corr1.tfs", "tune_knobs_file": tmp_path / "knobs1.txt"},
        ],
        debug=False,
        mad_logfile=None,
        python_logfile=None,
        tracking_plan=FullRingBpmTrackingPlan(),
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

    # Full-ring workers anchor at the plane's first BPM ($start order), not the
    # user-supplied start BPM, so the sequence is never cycled to a BPM.
    assert specs == [
        WorkerRangeSpec(start_bpm="BPH.13008", end_bpm="BPH.13208", sdir=1),
        WorkerRangeSpec(start_bpm="BPH.13008", end_bpm="BPH.13208", sdir=-1),
    ]


def test_build_observation_plan_filters_single_plane_bpms_and_requires_measurable_start(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    range_specs = helper.build_range_specs(
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
    forward_plans = helper.build_observation_plans(range_specs[0], file_idx=0)
    assert len(forward_plans) == 1
    forward_plan = forward_plans[0]
    assert forward_plan.kick_plane == "x"
    assert forward_plan.bpm_names == ["BPH.13008", "BPH.13208"]

    backward_plans = helper.build_observation_plans(range_specs[1], file_idx=0)
    assert len(backward_plans) == 1
    backward_plan = backward_plans[0]
    assert backward_plan.kick_plane == "x"
    assert backward_plan.bpm_names == ["BPH.13208", "BPH.13008"]
    assert backward_plan.range_spec == range_specs[1]


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
    assert plans[0].bpm_names == [
        "BPH.13208",
        "BPV.13308",
        "BPV.20108",
        "BPH.13008",
        "BPV.13108",
    ]


def test_build_observation_plans_split_dual_plane_data_across_single_plane_bpms(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    range_specs = helper.build_range_specs(
        start_bpms=["BPH.13208", "BPV.13308"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=4,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
    )
    forward_plans = helper.build_observation_plans(range_specs[0], file_idx=1)
    backward_plans = helper.build_observation_plans(range_specs[1], file_idx=1)
    y_forward_plans = helper.build_observation_plans(range_specs[2], file_idx=1)

    assert [
        (plan.kick_plane, plan.bpm_names, plan.range_spec)
        for plan in forward_plans
    ] == [
        (
            "x",
            ["BPH.13008", "BPH.13208"],
            WorkerRangeSpec(start_bpm="BPH.13008", end_bpm="BPH.13208", sdir=1),
        ),
    ]
    assert [
        (plan.kick_plane, plan.bpm_names, plan.range_spec)
        for plan in backward_plans
    ] == [
        (
            "x",
            ["BPH.13208", "BPH.13008"],
            WorkerRangeSpec(start_bpm="BPH.13008", end_bpm="BPH.13208", sdir=-1),
        ),
    ]
    assert [
        (plan.kick_plane, plan.bpm_names, plan.range_spec)
        for plan in y_forward_plans
    ] == [
        (
            "y",
            ["BPV.13108", "BPV.13308", "BPV.20108"],
            WorkerRangeSpec(start_bpm="BPV.13108", end_bpm="BPV.20108", sdir=1),
        ),
    ]
    # Off-plane BPMs are unobserved across the whole ring, not just inside the named
    # range: a full-ring multi-turn track traverses every element (incl. the wrap),
    # so an off-plane BPM left observed outside the range overflows the result vectors.
    assert forward_plans[0].bad_bpms == ["BPV.13108", "BPV.13308", "BPV.20108"]
    assert y_forward_plans[0].bad_bpms == ["BPH.13008", "BPH.13208"]


def test_make_worker_config_uses_file_specific_artifacts(tmp_path: Path) -> None:
    helper = _make_helper(tmp_path)
    helper.accelerator.infer_monitor_plane = lambda bpm: "HV"  # type: ignore[method-assign]
    plans = helper.build_observation_plans(
        WorkerRangeSpec(start_bpm="BPH.13208", end_bpm="BPV.13108", sdir=1),
        file_idx=1,
    )

    assert len(plans) == 1
    config = helper.make_worker_config(plans[0])

    assert config.interface_options == {
        "corrector_strengths": tmp_path / "corr1.tfs",
        "tune_knobs_file": tmp_path / "knobs1.txt",
    }
    assert config.kick_plane == "xy"


def test_observation_plan_uses_plane_compatible_range_for_split_full_ring_workers(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    range_specs = helper.build_range_specs(
        start_bpms=["BPH.13208", "BPV.13308"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            tracks_per_worker=1,
            num_workers=4,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
    )
    plans = helper.build_observation_plans(range_specs[2], file_idx=1)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.kick_plane == "y"
    assert plan.bpm_names == ["BPV.13108", "BPV.13308", "BPV.20108"]
    assert plan.range_spec == WorkerRangeSpec(
        start_bpm="BPV.13108",
        end_bpm="BPV.20108",
        sdir=1,
    )

    config = helper.make_worker_config(plan)

    assert config.tracking_start_bpm == "BPV.13108"
    assert config.tracking_end_bpm == "BPV.20108"
