"""Tracking-mode planning for standard and kicker-initialised runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from aba_optimiser.config import SimulationConfig
    from aba_optimiser.training.config.models import KickerConfig

# Imported lazily inside methods to avoid a circular import at module load time.
# Both helpers are stable, side-effect-free utilities.
def _extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir, *, allow_missing_start=False):
    from aba_optimiser.training.utils import extract_bpm_range_names
    return extract_bpm_range_names(all_bpms, start_bpm, end_bpm, sdir, allow_missing_start)


def _create_bpm_range_specs(start_bpms, end_bpms, use_fixed_bpm, fixed_start=None, fixed_end=None):
    from aba_optimiser.training.utils import create_bpm_range_specs
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

    def observed_bpms(self, bpms_in_range: list[str], all_bpms: list[str]) -> list[str]:
        """Return the BPMs compared against measurements."""
        return bpms_in_range

    def extra_markers(self) -> list[str]:
        """Return non-BPM markers that must be kept in measurement data."""
        return []

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
        track_data: dict[int, pd.DataFrame],
        flattop_turns: int,
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        """Return boundary turns and the filtered list of usable start turns."""
        turns_per_sample = (
            1 if simulation_config.run_arc_by_arc else simulation_config.n_run_turns
        )
        turns_to_remove = set()
        boundary_turns_by_file: dict[int, set[int]] = {}

        for file_idx, df in track_data.items():
            file_turns = sorted(df.index.get_level_values("turn").unique())
            for track_idx in range(0, len(file_turns), flattop_turns):
                track_turns = file_turns[track_idx : track_idx + flattop_turns]
                boundary_margin = max(1, turns_per_sample)
                boundary_turns = _boundary_turns_for_track(track_turns, boundary_margin)
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
    """BPM-initialised multi-turn tracking around the full ring."""

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

    def observation_start_bpm(self, all_bpms: list[str]) -> str | None:
        # In kicker mode the worker must keep the tracking range anchored at the
        # kicker marker itself. Starting the MAD BPM range at the first real BPM
        # shifts the returned observation order by one point relative to the
        # measurement payload, which makes the loss nearly insensitive to the
        # true quadrupole errors.
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

    def select_available_turns(
        self,
        *,
        track_data: dict[int, pd.DataFrame],
        flattop_turns: int,
        simulation_config: SimulationConfig,
        available_turns: list[int],
    ) -> tuple[dict[int, set[int]], list[int]]:
        boundary_turns_by_file = {file_idx: set() for file_idx in track_data}
        kicker_start_turns: list[int] = []
        for df in track_data.values():
            file_turns = sorted(df.index.get_level_values("turn").unique())
            kicker_start_turns.extend(file_turns[::flattop_turns])
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
        ring_end = all_bpms[-1]
        return [(start, ring_end) for start in start_bpms]

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
        ring_end = all_bpms[-1]
        return [
            WorkerRangeSpec(start_bpm=start_bpm, end_bpm=ring_end, sdir=1)
            for start_bpm in start_bpms
        ]

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
        return [bpm for bpm in all_bpms if bpm not in excluded]

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


def build_tracking_plan(
    *,
    kicker_config: KickerConfig | None,
    simulation_config: SimulationConfig,
) -> TrackingPlan:
    """Return the tracking plan for the current controller run."""
    if kicker_config is None:
        if simulation_config.run_arc_by_arc:
            return ArcByArcTrackingPlan()
        return FullRingBpmTrackingPlan()
    return KickerTrackingPlan(kicker_name=kicker_config.kicker_name)
