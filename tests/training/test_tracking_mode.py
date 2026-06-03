from __future__ import annotations

from aba_optimiser.training.config.tracking import KickerTrackingPlan


def test_kicker_tracking_plan_keeps_observation_range_anchored_at_marker() -> None:
    plan = KickerTrackingPlan(kicker_name="BI3.KSW1L4")

    # The worker BPM range must start at the kicker marker itself so MAD returns
    # observations in the same order as the kicker measurement payload.
    assert plan.observation_start_bpm(["BR3.BPM2L3", "BR3.BPM1L3"]) is None
