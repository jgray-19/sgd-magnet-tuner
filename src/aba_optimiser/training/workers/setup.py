"""Worker setup helpers for tracking optimisation.

This module owns the logic that decides which workers should exist for a given
set of BPM ranges and measurement files. It is intentionally separate from
payload construction and multiprocessing lifecycle code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aba_optimiser.training.config.tracking import TrackingPlan, WorkerRangeSpec
from aba_optimiser.training.utils import bpm_supports_both_planes, bpm_supports_plane
from aba_optimiser.workers import WorkerConfig
from aba_optimiser.workers.common import KickPlane

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.accelerators import Accelerator
    from aba_optimiser.config import SimulationConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerObservationPlan:
    """Per-file observation settings after kick-plane filtering."""

    range_spec: WorkerRangeSpec
    file_idx: int
    kick_plane: KickPlane
    bpm_names: list[str]
    bad_bpms: list[str] | None
    init_marker: str | None = None

    @property
    def init_bpm(self) -> str:
        """Return the BPM used to initialise tracking for this plan."""
        return self.init_marker or self.range_spec.init_bpm


@dataclass(frozen=True)
class WorkerRuntimeMetadata:
    """Controller-side metadata retained for screening and diagnostics."""

    worker_id: int
    file_idx: int
    start_bpm: str
    end_bpm: str
    sdir: int
    kick_plane: KickPlane
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
        file_kick_planes: dict[int, str | KickPlane],
        magnet_range: str,
        interface_options_per_file: list[dict],
        debug: bool,
        mad_logfile: Path | None,
        python_logfile: Path | None,
        tracking_plan: TrackingPlan,
    ) -> None:
        self.accelerator = accelerator
        self.all_bpms = all_bpms
        self.fixed_start = fixed_start
        self.fixed_end = fixed_end
        self.use_fixed_bpm = use_fixed_bpm
        self.bad_bpms = bad_bpms
        self.file_kick_planes = file_kick_planes
        self.magnet_range = magnet_range
        self.interface_options_per_file = interface_options_per_file
        self.debug = debug
        self.mad_logfile = mad_logfile
        self.python_logfile = python_logfile
        self.tracking_plan = tracking_plan

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

    def bpm_supports_plane(self, bpm: str, kick_plane: KickPlane) -> bool:
        """Return whether `bpm` can measure the requested kick plane."""
        return bpm_supports_plane(self.accelerator, bpm, kick_plane.value)

    def bpm_supports_both_planes(self, bpm: str) -> bool:
        """Return whether `bpm` can measure both transverse planes."""
        return bpm_supports_both_planes(self.accelerator, bpm)

    def get_range_bpm_names(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None = None,
    ) -> list[str]:
        """Return the raw BPM range after applying explicit exclusions."""
        start_plane = self._plane_for_bpm(start_bpm)
        end_plane = self._plane_for_bpm(end_bpm)
        if start_plane is not None and start_plane == end_plane:
            return self.tracking_plan.get_range_bpm_names(
                all_bpms=self._bpms_for_plane(start_plane),
                start_bpm=start_bpm,
                end_bpm=end_bpm,
                sdir=sdir,
                bad_bpms=bad_bpms,
            )
        return self.tracking_plan.get_range_bpm_names(
            all_bpms=self.all_bpms,
            start_bpm=start_bpm,
            end_bpm=end_bpm,
            sdir=sdir,
            bad_bpms=bad_bpms,
        )

    def get_worker_bpm_names(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        kick_plane: KickPlane,
        bad_bpms: list[str] | None = None,
    ) -> list[str]:
        """Return the BPMs a worker should observe, in tracking order."""
        bpm_names = self.get_range_bpm_names(start_bpm, end_bpm, sdir, bad_bpms)
        if kick_plane == KickPlane.XY:
            return bpm_names
        return [bpm for bpm in bpm_names if self.bpm_supports_plane(bpm, kick_plane)]

    def get_worker_bad_bpms(
        self,
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        kick_plane: KickPlane,
    ) -> list[str] | None:
        """Return per-worker BPM exclusions needed by the MAD interface."""
        if kick_plane == KickPlane.XY:
            return self.bad_bpms

        # Unobserve every off-plane BPM in the *whole ring*, not just those inside
        # the named range. A full-ring multi-turn track traverses the entire ring
        # (and the wrap between the range end and start each turn), so any off-plane
        # BPM left observed outside the range still fires save_data and overflows the
        # nbpms * n_run_turns result vectors (MAD seti "index out of bounds").
        del start_bpm, end_bpm, sdir
        plane_filtered = [
            bpm for bpm in self.all_bpms if not self.bpm_supports_plane(bpm, kick_plane)
        ]
        return self.merge_bad_bpms(self.bad_bpms, plane_filtered)

    def build_range_specs(
        self,
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
    ) -> list[WorkerRangeSpec]:
        """Return logical worker ranges before file-specific plane filtering."""
        if self.tracking_plan.init_marker is not None or not any(
            not self.bpm_supports_both_planes(bpm) for bpm in self.all_bpms
        ):
            return self.tracking_plan.build_range_specs(
                start_bpms=start_bpms,
                end_bpms=end_bpms,
                all_bpms=self.all_bpms,
                simulation_config=simulation_config,
                use_fixed_bpm=self.use_fixed_bpm,
                fixed_start=self.fixed_start,
                fixed_end=self.fixed_end,
            )

        return self._build_single_plane_range_specs(
            start_bpms=start_bpms,
            end_bpms=end_bpms,
            simulation_config=simulation_config,
        )

    def _plane_for_bpm(self, bpm: str) -> KickPlane | None:
        """Return the single transverse plane measured by a BPM, if any."""
        try:
            supports_x = self.bpm_supports_plane(bpm, KickPlane.X)
            supports_y = self.bpm_supports_plane(bpm, KickPlane.Y)
        except ValueError:
            return None
        if supports_x and not supports_y:
            return KickPlane.X
        if supports_y and not supports_x:
            return KickPlane.Y
        return None

    def _bpms_for_plane(self, plane: KickPlane) -> list[str]:
        """Return model BPMs that can observe one transverse plane."""
        return [bpm for bpm in self.all_bpms if self.bpm_supports_plane(bpm, plane)]

    def _bpm_behind_in_plane(self, start_bpm: str, plane_bpms: list[str]) -> str:
        """Return the previous BPM in the same plane as ``start_bpm``."""
        if start_bpm not in plane_bpms:
            raise ValueError(
                f"Start BPM '{start_bpm}' is not in the {plane_bpms} list for its plane"
            )
        return plane_bpms[plane_bpms.index(start_bpm) - 1]

    def _single_plane_user_bpms(
        self,
        bpms: list[str],
        *,
        label: str,
    ) -> dict[KickPlane, list[str]]:
        """Split user-selected BPMs by their real monitor plane."""
        by_plane = {KickPlane.X: [], KickPlane.Y: []}
        for bpm in bpms:
            plane = self._plane_for_bpm(bpm)
            if plane is None:
                LOGGER.warning(
                    "Single-plane range planning keeps dual-plane %s BPM %s in both planes",
                    label,
                    bpm,
                )
                by_plane[KickPlane.X].append(bpm)
                by_plane[KickPlane.Y].append(bpm)
            else:
                by_plane[plane].append(bpm)
        return by_plane

    def _build_single_plane_range_specs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        simulation_config: SimulationConfig,
    ) -> list[WorkerRangeSpec]:
        """Build ranges from same-plane BPM boundaries for single-plane machines."""
        starts_by_plane = self._single_plane_user_bpms(start_bpms, label="start")
        ends_by_plane = self._single_plane_user_bpms(end_bpms, label="end")
        range_specs: list[WorkerRangeSpec] = []

        for plane in (KickPlane.X, KickPlane.Y):
            plane_bpms = self._bpms_for_plane(plane)
            if not plane_bpms:
                continue

            starts = starts_by_plane[plane]
            ends = ends_by_plane[plane]
            if not starts and not ends:
                LOGGER.warning(
                    "No %s-plane BPM boundaries were provided for single-plane range planning",
                    plane.value,
                )
                continue
            if simulation_config.run_arc_by_arc and bool(starts) != bool(ends):
                raise ValueError(
                    f"Single-plane arc-by-arc ranges need both start and end BPMs for "
                    f"the {plane.value}-plane; got {len(starts)} starts and {len(ends)} ends"
                )

            if not simulation_config.run_arc_by_arc:
                # Full-ring workers track from the fixed turn-increment start ($start),
                # so anchor every worker at the plane's first BPM rather than cycling
                # to each user start BPM (which would be double-observed at the wrap).
                plane_starts = (
                    starts if self.tracking_plan.cycle_to_init_bpm else [plane_bpms[0]]
                )
                for start_bpm in plane_starts:
                    end_bpm = self._bpm_behind_in_plane(start_bpm, plane_bpms)
                    range_specs.extend(
                        WorkerRangeSpec(start_bpm=start_bpm, end_bpm=end_bpm, sdir=sdir)
                        for sdir in (1, -1)
                    )
                continue

            range_specs.extend(
                self.tracking_plan.build_range_specs(
                    start_bpms=starts,
                    end_bpms=ends,
                    all_bpms=plane_bpms,
                    simulation_config=simulation_config,
                    use_fixed_bpm=self.use_fixed_bpm,
                    fixed_start=starts[0] if self.use_fixed_bpm else self.fixed_start,
                    fixed_end=ends[0] if self.use_fixed_bpm else self.fixed_end,
                )
            )
        return range_specs

    @staticmethod
    def get_primary_file_idx(turn_batch: list[int], file_turn_map: dict[int, int]) -> int:
        """Return the unique measurement file serving a worker batch."""
        primary_file_idx = file_turn_map[turn_batch[0]]
        if any(file_turn_map[turn] != primary_file_idx for turn in turn_batch):
            raise ValueError("Worker batch contains turns from multiple measurement files")
        return primary_file_idx

    def get_worker_planes(
        self, data_plane: KickPlane, range_bpms: list[str]
    ) -> tuple[KickPlane, ...]:
        """Return the worker plane(s) required for one file/range combination."""
        if data_plane == KickPlane.XY and all(
            self.bpm_supports_both_planes(bpm) for bpm in range_bpms
        ):
            return (KickPlane.XY,)
        if data_plane == KickPlane.XY:
            planes = [
                plane
                for plane in (KickPlane.X, KickPlane.Y)
                if any(self.bpm_supports_plane(bpm, plane) for bpm in range_bpms)
            ]
            return tuple(planes)
        return (data_plane,)

    def make_observation_plan(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
        worker_plane: KickPlane,
        available_bpms: set[str] | None = None,
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
        init_marker = self.tracking_plan.init_marker
        if available_bpms is not None:
            missing_bpms = [bpm for bpm in bpm_names if bpm not in available_bpms]
            if missing_bpms:
                LOGGER.warning(
                    "File %d range %s/%s sdir=%d: %d BPMs missing from measurement data; adding to bad BPMs",
                    file_idx,
                    range_spec.start_bpm,
                    range_spec.end_bpm,
                    range_spec.sdir,
                    len(missing_bpms),
                )
                bad_bpms = self.merge_bad_bpms(bad_bpms, missing_bpms)
                bpm_names = [bpm for bpm in bpm_names if bpm in available_bpms]
            if init_marker is not None and init_marker not in available_bpms:
                LOGGER.warning(
                    "File %d range %s/%s sdir=%d: init marker %s missing from measurement data",
                    file_idx,
                    range_spec.start_bpm,
                    range_spec.end_bpm,
                    range_spec.sdir,
                    init_marker,
                )
                return None

        if init_marker is None and range_spec.init_bpm not in bpm_names:
            return None
        if not bpm_names:
            return None

        return WorkerObservationPlan(
            range_spec=range_spec,
            file_idx=file_idx,
            kick_plane=worker_plane,
            bpm_names=bpm_names,
            bad_bpms=bad_bpms,
            init_marker=init_marker,
        )

    def build_observation_plans(
        self,
        range_spec: WorkerRangeSpec,
        file_idx: int,
        available_bpms: set[str] | None = None,
    ) -> list[WorkerObservationPlan]:
        """Return the per-file worker plan(s) for a range and measurement file.

        Dual-plane files are kept as dual-plane workers only when every BPM in
        the range can measure both planes. Otherwise the range is split into x
        and y workers, and each worker only keeps BPMs that can observe its
        plane and initialise from its direction-specific start BPM.
        """
        data_plane_raw = self.file_kick_planes.get(file_idx, KickPlane.XY)
        data_plane = KickPlane(data_plane_raw)
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
            plan = self.make_observation_plan(
                range_spec,
                file_idx,
                worker_plane,
                available_bpms=available_bpms,
            )
            if plan is not None:
                plans.append(plan)

        return plans

    def make_worker_config(self, plan: WorkerObservationPlan) -> WorkerConfig:
        """Build the worker configuration object for one plan."""
        return WorkerConfig(
            accelerator=self.accelerator,
            tracking_start_bpm=plan.range_spec.start_bpm,
            tracking_end_bpm=plan.range_spec.end_bpm,
            magnet_range=self.magnet_range,
            interface_options=self.interface_options_per_file[plan.file_idx],
            observation_range_start_bpm=self.tracking_plan.observation_start_bpm(self.all_bpms),
            initial_condition_marker=plan.init_marker,
            cycle_sequence=self.tracking_plan.cycle_to_init_bpm,
            sdir=plan.range_spec.sdir,
            kick_plane=plan.kick_plane,
            bad_bpms=plan.bad_bpms,
            debug=self.debug,
            mad_logfile=self.mad_logfile,
            python_logfile=self.python_logfile,
            install_acd_markers=self.tracking_plan.requires_acd_markers(),
        )

    @staticmethod
    def make_runtime_metadata(
        worker_id: int,
        file_idx: int,
        config: WorkerConfig,
        bpm_names: list[str],
        n_run_turns: int,
    ) -> WorkerRuntimeMetadata:
        """Return the metadata needed after a worker has started."""
        return WorkerRuntimeMetadata(
            worker_id=worker_id,
            file_idx=file_idx,
            start_bpm=config.tracking_start_bpm,
            end_bpm=config.tracking_end_bpm,
            sdir=config.sdir,
            kick_plane=config.kick_plane,
            n_run_turns=n_run_turns,
            bpm_names=bpm_names,
        )
