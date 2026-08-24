from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from aba_optimiser.measurements.squeeze.io import (
    get_central_measurement_time,
    prepare_frequency_metadata,
)


def test_get_central_measurement_time_returns_chronological_middle() -> None:
    meas_times = {
        "0Hz": ["16_52_08_398"],
        "+50Hz": ["17_00_27_504"],
        "-50Hz": ["17_53_50_407"],
        "+100Hz": ["17_05_53_444"],
        "-100Hz": ["17_59_32_491"],
    }

    central = get_central_measurement_time(meas_times, "inj")

    assert central.isoformat() == "2025-04-20T17:05:53+00:00"
    assert central.tzinfo == ZoneInfo("UTC")


def test_prepare_frequency_metadata_uses_supplied_energy(
    monkeypatch, tmp_path: Path
) -> None:
    meas_base_dir = tmp_path / "Measurements"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    folder = meas_base_dir / "Beam1@BunchTurn@2025_04_20@16_52_08_398"
    folder.mkdir(parents=True)
    (folder / "Beam1@BunchTurn@2025_04_20@16_52_08_398.sdds").write_text("")

    monkeypatch.setattr(
        "aba_optimiser.measurements.squeeze.io.find_all_bad_bpms",
        lambda _: {"BPM.1L1.B1"},
    )

    captured: dict[str, float] = {}

    def fake_save_online_knobs(*args, **kwargs) -> float:
        captured["energy"] = kwargs["energy"]
        return float(kwargs["energy"])

    monkeypatch.setattr(
        "aba_optimiser.measurements.squeeze.io.save_online_knobs",
        fake_save_online_knobs,
    )

    _, _, _, bad_bpms, energy = prepare_frequency_metadata(
        freq="0Hz",
        times=["16_52_08_398"],
        beam=1,
        meas_base_dir=meas_base_dir,
        results_dir=results_dir,
        squeeze_step="inj",
        energy=6800.0,
    )

    assert bad_bpms == {"BPM.1L1.B1"}
    assert energy == 6800.0
    assert captured["energy"] == 6800.0
