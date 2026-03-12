"""Worker setup helpers for tracking optimisation.

This module owns the logic that decides which workers should exist for a given
set of BPM ranges and measurement files. It is intentionally separate from
payload construction and multiprocessing lifecycle code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aba_optimiser.training.utils import create_bpm_range_specs, extract_bpm_range_names
from aba_optimiser.workers import WorkerConfig

if TYPE_CHECKING:
    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig


@dataclass(frozen=True)
class WorkerRangeSpec:
    """Logical BPM range assigned to a worker before file-specific filtering."""

    start_bpm: str
    end_bpm: str
    sdir: int

    @property
    def init_bpm(self) -> str:
        """Return the BPM used to initialise tracking for this direction."""
        return self.start_bpm if self.sdir > 0 else self.end_bpm


@dataclass(frozen=True)
class WorkerObservationPlan:
    """Per-file observation settings after kick-plane filtering."""

    range_spec: WorkerRangeSpec
    file_idx: int
    kick_plane: str
    bpm_names: list[str]
    bad_bpms: list[str] | None

    @property
    def init_bpm(self) -> str:
        """Return the BPM used to initialise tracking for this plan."""
        return self.range_spec.init_bpm


@dataclass(frozen=True)
class WorkerRuntimeMetadata:
    """Controller-side metadata retained for screening and diagnostics."""

    worker_id: int
    start_bpm: str
    end_bpm: str
    sdir: int
    kick_plane: str
    n_run_turns: int
    bpm_names: list[str]


class WorkerSetupHelper:
    """Build worker ranges, observation plans, and worker configs.

    Plane routing follows four rules:
    - dual-plane data + dual-plane BPMs -> one dual-plane worker
    - single-plane data + dual-plane BPMs -> one single-plane worker
    - dual-plane data + single-plane BPMs -> split into plane-specific workers
    - single-plane data + single-plane BPMs -> one single-plane worker
    """

    def __init__(
        self,
        accelerator: Accelerator,
        all_bpms: list[str],
        fixed_start: str,
        fixed_end: str,
        use_fixed_bpm: bool,
        bad_bpms: list[str] | None,
        file_kick_planes: dict[int, str],
        magnet_range: str,
        corrector_strengths_files: list[Path],
        tune_knobs_files: list[Path],
        debug: bool,
        mad_logfile: Path | None,
    ) -> None:
        self.accelerator = accelerator
        self.all_bpms = all_bpms
        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self.use_fixed_bpm = use_fixed_bpm
        self.bad_bpms = bad_bpms
        self.file_kick_planes = file_kick_planes
        self.magnet_range = magnet_range
        self.corrector_strengths_files = corrector_strengths_files
        self.tune_knobs_files = tune_knobs_files
        self.debug = debug
        self.mad_logfile = mad_logfile

    @staticmethod
    def merge_bad_bpms(*bad_bpm_lists: list[str] | None) -> list[str] | None:
        """Merge bad-BPM lists while preserving the first occurrence order."""
        merged: list[str] = []
        for bpm_list in bad_bpm_lists:
            if bpm_list is None:
                continue
            for bpm in bpm_list:
                if bpm not in merged:
                    merged.append(bpm)
        return merged or None

    def bpm_supports_plane(self, bpm: str, kick_plane: str) -> bool:
        """Return whether `bpm` can measure the requested kick plane."""
        plane = self.accelerator.infer_monitor_plane(bpm)
        if kick_plane == "x":
            return "H" in plane
        if kick_plane == "y":
            return "V" in plane
        if kick_plane == "xy":
            return ("H" in plane) or ("V" in plane)
        raise ValueError(f"Unsupported kick plane {kick_plane!r}")

    def bpm_supports_both_planes(self, bpm: str) -> bool:
        """Return whether `bpm` can measure both transverse planes."""
        return self.bpm_supports_plane(bpm, "x") and self.bpm_supports_plane(bpm, "y")

    def get_range_bpm_names(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None = None,
    ) -> list[str]:
        """Return the raw BPM range after applying explicit exclusions."""
        bpm_names = extract_bpm_range_names(self.all_bpms, start_bpm, end_bpm, sdir)
        excluded = set(bad_bpms or [])
        return [bpm for bpm in bpm_names if bpm not in excluded]

    def get_worker_bpm_names(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        kick_plane: str,
        bad_bpms: list[str] | None = None,
    ) -> list[str]:
        """Return the BPMs a worker should observe, in tracking order."""
        bpm_names = self.get_range_bpm_names(start_bpm, end_bpm, sdir, bad_bpms)
        if kick_plane == "xy":
            return bpm_names
        return [bpm for bpm in bpm_names if self.bpm_supports_plane(bpm, kick_plane)]

    def get_worker_bad_bpms(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        kick_plane: str,
    ) -> list[str] | None:
        """Return per-worker BPM exclusions needed by the MAD interface."""
        if kick_plane == "xy":
            return self.bad_bpms

        range_bpms = self.get_range_bpm_names(start_bpm, end_bpm, sdir)
        plane_filtered = [
            bpm for bpm in range_bpms if not self.bpm_supports_plane(bpm, kick_plane)
        ]
        return self.merge_bad_bpms(self.bad_bpms, plane_filtered)

    def build_range_specs(
        self,
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
    ) -> list[WorkerRangeSpec]:
        """Return logical worker ranges before file-specific plane filtering."""
        if simulation_config.run_arc_by_arc:
            return [
                WorkerRangeSpec(start_bpm, end_bpm, sdir)
                for start_bpm, end_bpm, sdir in create_bpm_range_specs(
                    start_bpms,
                    end_bpms,
                    self.use_fixed_bpm,
                    self.fixed_start,
                    self.fixed_end,
                )
            ]

        return [
            WorkerRangeSpec(
                start_bpm=start_bpm,
                end_bpm=self.all_bpms[self.all_bpms.index(start_bpm) - 1],
                sdir=sdir,
            )
            for start_bpm in start_bpms
            for sdir in (1, -1)
        ]

    @staticmethod
    def get_primary_file_idx(turn_batch: list[int], file_turn_map: dict[int, int]) -> int:
        """Return the unique measurement file serving a worker batch."""
        primary_file_idx = file_turn_map[turn_batch[0]]
        if any(file_turn_map[turn] != primary_file_idx for turn in turn_batch):
            raise ValueError("Worker batch contains turns from multiple measurement files")
        return primary_file_idx

    def get_worker_planes(self, data_plane: str, range_bpms: list[str]) -> tuple[str, ...]:
        """Return the worker plane(s) required for one file/range combination."""
        if data_plane == "xy" and all(self.bpm_supports_both_planes(bpm) for bpm in range_bpms):
            return ("xy",)
        if data_plane == "xy":
            return ("x", "y")
        return (data_plane,)

    def make_observation_plan(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
        worker_plane: str,
    ) -> WorkerObservationPlan | None:
        """Build one worker plan, or return `None` when the range is incompatible."""
        bad_bpms = self.get_worker_bad_bpms(
            range_spec.start_bpm,
            range_spec.end_bpm,
            range_spec.sdir,
            worker_plane,
        )
        bpm_names = self.get_worker_bpm_names(
            range_spec.start_bpm,
            range_spec.end_bpm,
            range_spec.sdir,
            worker_plane,
            bad_bpms,
        )
        if not bpm_names or range_spec.init_bpm not in bpm_names:
            return None
        return WorkerObservationPlan(
            range_spec=range_spec,
            file_idx=file_idx,
            kick_plane=worker_plane,
            bpm_names=bpm_names,
            bad_bpms=bad_bpms,
        )

    def build_observation_plans(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
    ) -> list[WorkerObservationPlan]:
        """Return the per-file worker plan(s) for a range and measurement file.

        Dual-plane files are kept as dual-plane workers only when every BPM in
        the range can measure both planes. Otherwise the range is split into x
        and y workers, and each worker only keeps BPMs that can observe its
        plane and initialise from its direction-specific start BPM.
        """
        data_plane = self.file_kick_planes.get(file_idx, "xy")
        range_bpms = self.get_range_bpm_names(
            range_spec.start_bpm,
            range_spec.end_bpm,
            range_spec.sdir,
            self.bad_bpms,
        )
        if not range_bpms:
            return []

        plans: list[WorkerObservationPlan] = []
        for worker_plane in self.get_worker_planes(data_plane, range_bpms):
            plan = self.make_observation_plan(range_spec, file_idx, worker_plane)
            if plan is not None:
                plans.append(plan)

        return plans

    def make_worker_config(self, plan: WorkerObservationPlan) -> WorkerConfig:
        """Build the worker configuration object for one plan."""
        return WorkerConfig(
            accelerator=self.accelerator,
            start_bpm=plan.range_spec.start_bpm,
            end_bpm=plan.range_spec.end_bpm,
            magnet_range=self.magnet_range,
            corrector_strengths=self.corrector_strengths_files[plan.file_idx],
            tune_knobs_file=self.tune_knobs_files[plan.file_idx],
            sdir=plan.range_spec.sdir,
            kick_plane=plan.kick_plane,
            bad_bpms=plan.bad_bpms,
            debug=self.debug,
            mad_logfile=self.mad_logfile,
        )

    @staticmethod
    def make_runtime_metadata(
        worker_id: int,
        config: WorkerConfig,
        bpm_names: list[str],
        n_run_turns: int,
    ) -> WorkerRuntimeMetadata:
        """Return the metadata needed after a worker has started."""
        return WorkerRuntimeMetadata(
            worker_id=worker_id,
            start_bpm=config.start_bpm,
            end_bpm=config.end_bpm,
            sdir=config.sdir,
            kick_plane=config.kick_plane,
            n_run_turns=n_run_turns,
            bpm_names=bpm_names,
        )
