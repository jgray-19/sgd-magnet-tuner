"""Tests for the pure config/logic helpers in aba_optimiser.measurements.squeeze.optimisation.

``optimise_arc``/``process_squeeze_step`` themselves are real-measurement-data
orchestration entry points (require raw SDDS files and AFS model directories)
and aren't practical to exercise end to end here; this covers the
config-building and checkpoint-resolution logic they depend on.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

from aba_optimiser.measurements.ac_dipole import ACDipoleOptimisationWindow
from aba_optimiser.measurements.squeeze.optimisation import (
    create_configs,
    get_ac_dipole_bpm_points,
    get_default_simulation_config,
    resolve_restore_resume,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetAcDipoleBpmPoints:
    def test_returns_range_and_start_end_points(self) -> None:
        window = ACDipoleOptimisationWindow(
            bpm_upstream="BPM.10L4.B1", bpm_downstream="BPM.9L4.B1"
        )

        magnet_range, start_points, end_points = get_ac_dipole_bpm_points(1, window)

        assert magnet_range == "BPM.9L4.B1/BPM.10L4.B1"
        assert start_points == ["BPM.9L4.B1"]
        assert end_points == ["BPM.10L4.B1"]

    def test_rejects_upstream_bpm_from_wrong_beam(self) -> None:
        window = ACDipoleOptimisationWindow(
            bpm_upstream="BPM.10L4.B2", bpm_downstream="BPM.9L4.B1"
        )
        with pytest.raises(ValueError, match="Upstream BPM"):
            get_ac_dipole_bpm_points(1, window)

    def test_rejects_downstream_bpm_from_wrong_beam(self) -> None:
        window = ACDipoleOptimisationWindow(
            bpm_upstream="BPM.10L4.B1", bpm_downstream="BPM.9L4.B2"
        )
        with pytest.raises(ValueError, match="Downstream BPM"):
            get_ac_dipole_bpm_points(1, window)


class TestCreateConfigs:
    def test_builds_sequence_and_measurement_configs(self, tmp_path: Path) -> None:
        window = ACDipoleOptimisationWindow(
            bpm_upstream="BPM.10L4.B1", bpm_downstream="BPM.9L4.B1"
        )
        measurements = [
            {"file": tmp_path / "m0.parquet", "corrector_file": tmp_path / "c0.tfs",
             "tune_knobs": tmp_path / "t0.txt"},
            {"file": tmp_path / "m1.parquet", "corrector_file": tmp_path / "c1.tfs",
             "tune_knobs": tmp_path / "t1.txt"},
        ]
        b2_errors = tmp_path / "b2.tfs"

        sequence_config, start_points, end_points, measurement_config = create_configs(
            beam=1,
            all_bad_bpms={"BPM.5L4.B1"},
            measurements=measurements,
            window=window,
            b2_errors=b2_errors,
        )

        assert sequence_config.magnet_range == "BPM.9L4.B1/BPM.10L4.B1"
        assert sequence_config.bad_bpms == ["BPM.5L4.B1"]
        assert start_points == ["BPM.9L4.B1"]
        assert end_points == ["BPM.10L4.B1"]

        assert set(measurement_config.measurements) == {m["file"] for m in measurements}
        for m in measurements:
            details = measurement_config.measurements[m["file"]]
            assert details.interface_options["corrector_knobs"] == m["corrector_file"]
            assert details.interface_options["tune_knobs"] == m["tune_knobs"]
            assert details.interface_options["b2_errors"] == b2_errors
            assert details.first_bpm == "BPM.33L2.B1"

    def test_beam_2_uses_beam_2_first_bpm(self, tmp_path: Path) -> None:
        window = ACDipoleOptimisationWindow(
            bpm_upstream="BPM.10L4.B2", bpm_downstream="BPM.9L4.B2"
        )
        measurements = [
            {"file": tmp_path / "m0.parquet", "corrector_file": None, "tune_knobs": None}
        ]

        _, _, _, measurement_config = create_configs(
            beam=2,
            all_bad_bpms=set(),
            measurements=measurements,
            window=window,
            b2_errors=tmp_path / "b2.tfs",
        )

        details = measurement_config.measurements[measurements[0]["file"]]
        assert details.first_bpm == "BPM.34R8.B2"


def test_get_default_simulation_config_defaults() -> None:
    config = get_default_simulation_config()

    assert config.data_fraction == 1.0
    assert config.num_batches == 20
    assert config.num_workers == 60
    assert config.use_fixed_bpm is True
    assert config.run_arc_by_arc is True
    assert config.n_run_turns == 1
    assert config.optimise_momenta is False


def test_get_default_simulation_config_overrides() -> None:
    config = get_default_simulation_config(data_fraction=0.5, num_batches=5)

    assert config.data_fraction == 0.5
    assert config.num_batches == 5


class TestResolveRestoreResume:
    def test_no_restore_requested_returns_inputs_unchanged(self, tmp_path: Path) -> None:
        arc_numbers, restore_bends, restore_quads, restore_arc = resolve_restore_resume(
            arc_numbers=[1, 2, 3],
            checkpoint_dir=tmp_path,
            beam=1,
            squeeze_step="1.2m",
            restore_bends_opt=False,
            restore_quads_opt=False,
        )

        assert arc_numbers == [1, 2, 3]
        assert restore_bends is False
        assert restore_quads is False
        assert restore_arc is None

    def test_no_matching_checkpoints_falls_back_to_no_restore(self, tmp_path: Path) -> None:
        arc_numbers, restore_bends, restore_quads, restore_arc = resolve_restore_resume(
            arc_numbers=[1, 2],
            checkpoint_dir=tmp_path,
            beam=1,
            squeeze_step="1.2m",
            restore_bends_opt=True,
            restore_quads_opt=False,
        )

        assert arc_numbers == [1, 2]
        assert restore_bends is False
        assert restore_quads is False
        assert restore_arc is None

    def test_finds_most_recent_matching_checkpoint_and_trims_arc_list(
        self, tmp_path: Path
    ) -> None:
        older = tmp_path / "checkpoint_b1_1_2m_arc1_quads.json"
        newer = tmp_path / "checkpoint_b1_1_2m_arc2_quads.json"
        older.write_text("{}")
        newer.write_text("{}")

        now = time.time()
        os.utime(older, (now - 10, now - 10))
        os.utime(newer, (now, now))

        arc_numbers, restore_bends, restore_quads, restore_arc = resolve_restore_resume(
            arc_numbers=[1, 2, 3],
            checkpoint_dir=tmp_path,
            beam=1,
            squeeze_step="1.2m",
            restore_bends_opt=False,
            restore_quads_opt=True,
        )

        assert restore_arc == 2
        assert arc_numbers == [2, 3]
        assert restore_quads is True
        assert restore_bends is False

    def test_restore_arc_incompatible_with_requested_arcs_falls_back(
        self, tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "checkpoint_b1_1_2m_arc5_quads.json"
        checkpoint.write_text("{}")

        arc_numbers, restore_bends, restore_quads, restore_arc = resolve_restore_resume(
            arc_numbers=[1, 2, 3],
            checkpoint_dir=tmp_path,
            beam=1,
            squeeze_step="1.2m",
            restore_bends_opt=False,
            restore_quads_opt=True,
        )

        assert arc_numbers == [1, 2, 3]
        assert restore_bends is False
        assert restore_quads is False
        assert restore_arc is None
