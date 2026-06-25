"""Tracking-mode planning for standard and kicker-initialised runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aba_optimiser.training.utils import create_bpm_range_specs, extract_bpm_range_names

if TYPE_CHECKING:
    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.models import KickerConfig

# Imported lazily inside methods to avoid a circular import at module load time.
# Both helpers are stable, side-effect-free utilities.
def _extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir, *, allow_missing_start=False):
    return extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir, allow_missing_start)


def _create_bpm_range_specs(start_bpms, end_bpms, use_fixed_bpm, fixed_start=None, fixed_end=None):
    return create_bpm_range_specs(start_bpms, end_bpms, use_fixed_bpm, fixed_start, fixed_end)


def _boundary_turns_for_track(track_turns: list, margin: int) -> list:
    """Return the boundary turns (first/last ``margin`` entries) for one track."""
    if len(track_turns) <= 2 * margin:
        return track_turns
    return track_turns[:margin] + track_turns[-margin:]


def _bpm_behind(all_bpms: list[str], bpm: str) -> str:
    """Return the BPM immediately behind ``bpm`` in ring order."""
    if bpm not in all_bpms:
        raise ValueError(f"Start BPM '{bpm}' not found in model BPM list")
    return all_bpms[all_bpms.index(bpm) - 1]


class TrackingPlan(ABC):
    """Encapsulate mode-specific tracking behavior."""

    @property
    @abstractmethod
    def init_marker(self) -> str | None:
        """Return the element used to initialise tracking, if any."""

    @property
    @abstractmethod
    def allow_missing_start(self) -> bool:
        """Return whether the start element may be absent from the BPM list."""

    @property
    @abstractmethod
    def force_forward_tracking(self) -> bool:
        """Return whether only the forward direction should be tracked."""

    @property
    def enable_validation(self) -> bool:
        """Return whether validation workers should be enabled."""
        return True

    @property
    def cycle_to_init_bpm(self) -> bool:
        """Return whether workers cycle the sequence to their init/start BPM.

        Cycling to a BPM places it at both the ring start and the wrap, so a worker
        that tracks the full ring observes it twice per turn. Plans that track the
        whole ring keep the natural ``$start`` (the fixed turn-increment start) and
        return ``False`` here; range-limited plans (arc-by-arc, kicker) cycle to their
        own init point and return ``True``.
        """
        return True

    def tracking_anchor_mode(self) -> str | None:
        """Return the MAD-interface preparation mode for marker-anchored tracking."""
        return None

    def uses_fixed_bpm_window(self) -> bool:
        """Return whether fixed BPM start/end derivation applies to this plan."""
        return True

    def format_range_for_log(self, bpm_range: str) -> str:
        """Return a human-readable range label for logs."""
        if bpm_range == "$start/$end":
            return "full cycled sequence ($start/$end)"
        return bpm_range

    def log_filtered_tracking_points(
        self,
        logger,
        start_points: list[str],
        end_points: list[str],
    ) -> None:
        """Log post-filter tracking boundaries using this plan's terminology."""
        logger.info(
            "After filtering bad BPMs, BPM tracking start points: %s; end points: %s",
            start_points,
            end_points,
        )

    def log_fixed_bpm_derivation_skipped(self, logger, start_points: list[str]) -> None:
        """Log why fixed BPM derivation was skipped for this plan."""
        logger.info(
            "Skipping fixed BPM derivation for this tracking plan; start points: %s",
            start_points,
        )

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Return the BPMs compared against measurements."""
        return bpms_in_range

    def extra_markers(self) -> list[str]:
        """Return non-BPM markers that must be kept in measurement data."""
        return []

    def tracking_anchor_markers(self) -> list[str]:
        """Return non-BPM tracking anchors that may not appear in the BPM range."""
        return self.extra_markers()

    def tracking_anchor_sources(self) -> list[str]:
        """Return source elements needed to prepare marker-anchored tracking."""
        return self.tracking_anchor_markers()

    def cycle_marker(self) -> str | None:
        """Return an optional unobserved element used only for sequence cycling."""
        return None

    def observation_start_bpm(self, all_bpms: list[str]) -> str | None:
        """Return the BPM used as the worker observation-range start."""
        return None

    def range_specs_per_batch(
        self,
        *,
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        """Return the range-spec count used for worker planning."""
        if not run_arc_by_arc:
            return num_starts * 2, f"2 directions x {num_starts} start BPMs"
        if use_fixed_bpm:
            return num_starts + num_ends, f"fixed pairs ({num_starts} starts + {num_ends} ends)"
        return num_starts * num_ends * 2, f"2 directions x {num_starts} starts x {num_ends} ends"

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Return boundary turns and the filtered list of usable start turns.

        A start turn is unusable when the multi-turn track it seeds would cross a
        bunch boundary. Each bunch's first/last turns are therefore removed, using
        the per-file ``bunch_number`` grouping read from the measurement data.
        """
        turns_per_sample = (
            1 if simulation_config.run_arc_by_arc else simulation_config.n_run_turns
        )
        boundary_margin = max(1, turns_per_sample)
        turns_to_remove = set()
        boundary_turns_by_file: dict[int, set[int]] = {}

        for file_idx, bunches in bunch_turns_by_file.items():
            for bunch_turns in bunches.values():
                boundary_turns = _boundary_turns_for_track(sorted(bunch_turns), boundary_margin)
                boundary_turns_by_file.setdefault(file_idx, set()).update(boundary_turns)
                turns_to_remove.update(boundary_turns)

        return boundary_turns_by_file, [t for t in available_turns if t not in turns_to_remove]

    def bpm_pairs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[tuple[str, str]]:
        """Return logical controller-side BPM pairs."""
        if not run_arc_by_arc:
            return [(start, _bpm_behind(all_bpms, start)) for start in start_bpms]
        if use_fixed_bpm:
            return [(s, fixed_end) for s in start_bpms] + [(fixed_start, e) for e in end_bpms]
        return [(s, e) for s in start_bpms for e in end_bpms]

    def build_range_specs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[WorkerRangeSpec]:
        """Return worker range specs before file-specific plane filtering."""
        if simulation_config.run_arc_by_arc:
            return [
                WorkerRangeSpec(start_bpm, end_bpm, sdir)
                for start_bpm, end_bpm, sdir in _create_bpm_range_specs(
                    start_bpms,
                    end_bpms,
                    use_fixed_bpm,
                    fixed_start,
                    fixed_end,
                )
            ]

        range_specs: list[WorkerRangeSpec] = []
        for start_bpm in start_bpms:
            end_bpm = _bpm_behind(all_bpms, start_bpm)
            range_specs.extend(
                WorkerRangeSpec(start_bpm=start_bpm, end_bpm=end_bpm, sdir=sdir)
                for sdir in (1, -1)
            )
        return range_specs

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        """Return the BPMs in one logical observation range."""
        bpm_names = _extract_bpm_range_names(
            all_bpms,
            start_bpm,
            end_bpm,
            sdir,
            allow_missing_start=self.allow_missing_start,
        )
        excluded = set(bad_bpms or [])
        return [bpm for bpm in bpm_names if bpm not in excluded]

    def n_data_points(
        self,
        *,
        all_bpms: list[str],
        mad_iface,
        bpm_pairs: list[tuple[str, str]],
        n_turns: int,
    ) -> dict[tuple[str, str], int]:
        """Return expected worker payload sizes for each logical range."""
        from aba_optimiser.workers import TrackingWorker

        counts: dict[tuple[str, str], int] = {}
        for start, end in bpm_pairs:
            _, n_bpms, _ = mad_iface.count_bpms(f"{start}/{end}")
            counts[(start, end)] = TrackingWorker.get_n_data_points(n_bpms, n_turns=n_turns)
        return counts


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
class ArcByArcTrackingPlan(TrackingPlan):
    """BPM-initialised tracking over explicit arc/window ranges."""

    @property
    def init_marker(self) -> str | None:
        return None

    @property
    def allow_missing_start(self) -> bool:
        return False

    @property
    def force_forward_tracking(self) -> bool:
        return False


@dataclass(frozen=True)
class FullRingBpmTrackingPlan(TrackingPlan):
    """Multi-turn tracking around the full ring from the fixed turn-increment start.

    Every worker tracks the whole ring from the natural ``$start`` (where each
    measured turn begins) rather than cycling to a per-worker BPM. Anchoring all
    workers at the same point keeps the simulated and measured turn boundaries
    aligned and stops the start BPM being observed twice per turn at the wrap.
    """

    @property
    def init_marker(self) -> str | None:
        return None

    @property
    def allow_missing_start(self) -> bool:
        return False

    @property
    def force_forward_tracking(self) -> bool:
        return False

    @property
    def cycle_to_init_bpm(self) -> bool:
        return False

    def build_range_specs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[WorkerRangeSpec]:
        """Anchor every worker at ``$start`` (``all_bpms[0]``) for the full ring.

        The per-worker ``start_bpms`` only chose where the sequence was cycled before;
        anchoring them all here removes that choice (and the wrap double-count) while
        still covering both tracking directions.
        """
        del start_bpms, end_bpms, use_fixed_bpm, fixed_start, fixed_end, simulation_config
        anchor = all_bpms[0]
        wrap_end = _bpm_behind(all_bpms, anchor)
        return [
            WorkerRangeSpec(start_bpm=anchor, end_bpm=wrap_end, sdir=sdir) for sdir in (1, -1)
        ]


@dataclass(frozen=True)
class KickerTrackingPlan(TrackingPlan):
    """Forward-only tracking starting from a kicker initial-condition marker."""

    kicker_name: str

    @property
    def init_marker(self) -> str | None:
        return self.kicker_name

    @property
    def allow_missing_start(self) -> bool:
        return True

    @property
    def force_forward_tracking(self) -> bool:
        return True

    @property
    def enable_validation(self) -> bool:
        return False

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        return all_bpms

    def extra_markers(self) -> list[str]:
        return [self.kicker_name]

    def tracking_anchor_markers(self) -> list[str]:
        return []

    def tracking_anchor_sources(self) -> list[str]:
        return [self.kicker_name]

    def cycle_marker(self) -> str | None:
        return f"{self.kicker_name}_centre"

    def tracking_anchor_mode(self) -> str | None:
        return "kicker"

    def observation_start_bpm(self, all_bpms: list[str]) -> str | None:
        # Kicker mode compares only real BPMs. The measured kicker marker supplies
        # the initial coordinates, while the MAD sequence is cycled to an unobserved
        # centre marker so the observed BPM order begins just after the kicker.
        del all_bpms
        return None

    def range_specs_per_batch(
        self,
        *,
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        return num_starts, f"kicker forward-only x {num_starts} start marker(s)"

    def log_filtered_tracking_points(
        self,
        logger,
        start_points: list[str],
        end_points: list[str],
    ) -> None:
        logger.info(
            "After filtering bad BPMs, kicker tracking start marker(s): %s; end points: %s",
            start_points,
            end_points,
        )

    def select_available_turns(
        self,
        *,
        bunch_turns_by_file: dict[int, dict[int, list[int]]],
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        del simulation_config, available_turns
        boundary_turns_by_file = {file_idx: set() for file_idx in bunch_turns_by_file}
        kicker_start_turns: list[int] = []
        for bunches in bunch_turns_by_file.values():
            for bunch_turns in bunches.values():
                kicker_start_turns.append(min(bunch_turns))
        return boundary_turns_by_file, sorted(kicker_start_turns)

    def bpm_pairs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[tuple[str, str]]:
        del start_bpms, end_bpms, run_arc_by_arc, use_fixed_bpm, fixed_start, fixed_end
        start = all_bpms[0]
        return [(start, _bpm_behind(all_bpms, start))]

    def build_range_specs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[WorkerRangeSpec]:
        del start_bpms, end_bpms, simulation_config, use_fixed_bpm, fixed_start, fixed_end
        start = all_bpms[0]
        return [WorkerRangeSpec(start_bpm=start, end_bpm=_bpm_behind(all_bpms, start), sdir=1)]

    def get_range_bpm_names(
        self,
        *,
        all_bpms: list[str],
        start_bpm: str,
        end_bpm: str,
        sdir: int,
        bad_bpms: list[str] | None,
    ) -> list[str]:
        excluded = set(bad_bpms or [])
        if start_bpm not in all_bpms:
            raise ValueError(f"Kicker tracking start BPM '{start_bpm}' not found in model BPM list")
        start_idx = all_bpms.index(start_bpm)
        marker_order = all_bpms[start_idx:] + all_bpms[:start_idx]
        return [bpm for bpm in marker_order if bpm not in excluded]

    def n_data_points(
        self,
        *,
        all_bpms: list[str],
        mad_iface,
        bpm_pairs: list[tuple[str, str]],
        n_turns: int,
    ) -> dict[tuple[str, str], int]:
        from aba_optimiser.workers import TrackingWorker

        n_bpms = len(all_bpms)
        return {
            (start, end): TrackingWorker.get_n_data_points(n_bpms, n_turns=n_turns)
            for start, end in bpm_pairs
        }


@dataclass(frozen=True)
class ACDTrackingPlan(ArcByArcTrackingPlan):
    """Bidirectional tracking initialised at AC-dipole markers."""

    acd_name: str

    @property
    def acd_after(self) -> str:
        return f"{self.acd_name}_after"

    @property
    def acd_before(self) -> str:
        return f"{self.acd_name}_before"

    @property
    def allow_missing_start(self) -> bool:
        return False

    def tracking_anchor_mode(self) -> str | None:
        return "acd"

    def uses_fixed_bpm_window(self) -> bool:
        return False

    def extra_markers(self) -> list[str]:
        return [self.acd_after, self.acd_before]

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        return all_bpms

    def observation_start_bpm(self, all_bpms: list[str]) -> str | None:
        del all_bpms
        return None

    def range_specs_per_batch(
        self,
        *,
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        num_starts: int,
        num_ends: int,
    ) -> tuple[int, str]:
        return 2, "ACD bidirectional (forward + backward)"

    def log_filtered_tracking_points(
        self,
        logger,
        start_points: list[str],
        end_points: list[str],
    ) -> None:
        del start_points, end_points
        logger.info(
            "After filtering bad BPMs, ACD tracking start markers: forward=%s, backward=%s",
            self.acd_after,
            self.acd_before,
        )

    def log_fixed_bpm_derivation_skipped(self, logger, start_points: list[str]) -> None:
        logger.info(
            "Skipping fixed BPM derivation for ACD marker tracking; "
            "initial conditions use tracking markers %s",
            start_points,
        )

    def bpm_pairs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        run_arc_by_arc: bool,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[tuple[str, str]]:
        return [(self.acd_after, self.acd_before)]

    def build_range_specs(
        self,
        *,
        start_bpms: list[str],
        end_bpms: list[str],
        all_bpms: list[str],
        simulation_config: SimulationConfig,
        use_fixed_bpm: bool,
        fixed_start: str,
        fixed_end: str,
    ) -> list[WorkerRangeSpec]:
        return [
            WorkerRangeSpec(start_bpm=self.acd_after, end_bpm=self.acd_before, sdir=1),
            WorkerRangeSpec(start_bpm=self.acd_after, end_bpm=self.acd_before, sdir=-1),
        ]

    # def n_data_points(
    #     self,
    #     *,
    #     all_bpms: list[str],
    #     mad_iface,
    #     bpm_pairs: list[tuple[str, str]],
    #     n_turns: int,
    # ) -> dict[tuple[str, str], int]:
    #     from aba_optimiser.workers import TrackingWorker

    #     n_bpms = len(all_bpms)
    #     return {
    #         (self.acd_after, self.acd_before): TrackingWorker.get_n_data_points(
    #             n_bpms, n_turns=n_turns
    #         )
    #     }


def build_tracking_plan(
    *,
    kicker_config: KickerConfig | None,
    simulation_config: SimulationConfig,
    acd_name: str | None,
) -> TrackingPlan:
    """Return the tracking plan for the current controller run."""
    if acd_name is not None:
        return ACDTrackingPlan(acd_name=acd_name)
    if kicker_config is None:
        if simulation_config.run_arc_by_arc:
            return ArcByArcTrackingPlan()
        return FullRingBpmTrackingPlan()
    return KickerTrackingPlan(kicker_name=kicker_config.kicker_name)
