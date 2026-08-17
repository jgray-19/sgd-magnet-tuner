from __future__ import annotations

from pathlib import Path

from aba_optimiser.accelerators import PSB, SPS
from aba_optimiser.config import SimulationConfig
from aba_optimiser.training.config.tracking import ACDTrackingPlan, FullRingBpmTrackingPlan
from aba_optimiser.training.workers.setup import WorkerRangeSpec, WorkerSetupHelper
from aba_optimiser.workers.common import KickPlane


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
            {"corrector_knobs": tmp_path / "corr0.tfs", "tune_knobs": tmp_path / "knobs0.txt"},
            {"corrector_knobs": tmp_path / "corr1.tfs", "tune_knobs": tmp_path / "knobs1.txt"},
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
        "corrector_knobs": tmp_path / "corr1.tfs",
        "tune_knobs": tmp_path / "knobs1.txt",
    }
    assert config.kick_plane == "xy"


def test_acd_marker_plan_uses_markers_as_initial_conditions_not_targets(tmp_path: Path) -> None:
    seq_file = tmp_path / "psb.seq"
    seq_file.write_text("! Dummy PSB sequence file\n")
    accelerator = PSB(ring=3, sequence_file=seq_file, kinetic_energy=0.160)
    accelerator.infer_monitor_plane = lambda bpm: "HV"  # type: ignore[method-assign]
    plan = ACDTrackingPlan(acd_name="BR3.DES3L1")
    all_bpms = [
        "BR3.BPM1L3",
        "BR3.BPM2L3",
        "BR3.DES3L1_before",
        "BR3.DES3L1_after",
        "BR3.BPM3L3",
        "BR3.BPM4L3",
    ]
    helper = WorkerSetupHelper(
        accelerator=accelerator,
        all_bpms=all_bpms,
        fixed_start="$start",
        fixed_end="$end",
        use_fixed_bpm=False,
        bad_bpms=None,
        file_kick_planes={0: "xy"},
        magnet_range="$start/$end",
        interface_options_per_file=[{}],
        debug=False,
        mad_logfile=None,
        python_logfile=None,
        tracking_plan=plan,
    )

    specs = helper.build_range_specs(
        start_bpms=["BR3.DES3L1_after", "BR3.DES3L1_before"],
        end_bpms=[],
        simulation_config=SimulationConfig(
            num_workers=2,
            num_batches=1,
            run_arc_by_arc=False,
            n_run_turns=1,
        ),
    )

    forward = helper.get_worker_bpm_names(
        specs[0].start_bpm, specs[0].end_bpm, specs[0].sdir, KickPlane.XY
    )
    backward = helper.get_worker_bpm_names(
        specs[1].start_bpm, specs[1].end_bpm, specs[1].sdir, KickPlane.XY
    )

    assert specs == [
        WorkerRangeSpec("BR3.DES3L1_after", "BR3.DES3L1_before", 1),
        WorkerRangeSpec("BR3.DES3L1_after", "BR3.DES3L1_before", -1),
    ]
    assert forward == ["BR3.BPM3L3", "BR3.BPM4L3", "BR3.BPM1L3", "BR3.BPM2L3"]
    assert backward == ["BR3.BPM2L3", "BR3.BPM1L3", "BR3.BPM4L3", "BR3.BPM3L3"]
    assert all("DES3L1" not in bpm for bpm in forward + backward)

    forward_plan = helper.build_observation_plans(
        specs[0], file_idx=0, available_bpms=set(all_bpms)
    )[0]
    backward_plan = helper.build_observation_plans(
        specs[1], file_idx=0, available_bpms=set(all_bpms)
    )[0]
    assert forward_plan.init_marker == "BR3.DES3L1_after"
    assert backward_plan.init_marker == "BR3.DES3L1_before"
    assert helper.make_worker_config(forward_plan).initial_condition_marker == "BR3.DES3L1_after"
    assert helper.make_worker_config(backward_plan).initial_condition_marker == "BR3.DES3L1_before"


def test_observation_plan_uses_plane_compatible_range_for_split_full_ring_workers(
    tmp_path: Path,
) -> None:
    helper = _make_helper(tmp_path)

    range_specs = helper.build_range_specs(
        start_bpms=["BPH.13208", "BPV.13308"],
        end_bpms=[],
        simulation_config=SimulationConfig(
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
