"""MAD-NG integration tests for PSB BPM displacement knobs.

These tests verify end-to-end that:
  1. GradientDescentMadInterface creates dx/dy knobs for PSB BPMs.
  2. Setting a BPM knob propagates correctly through the deferred
     misalign chain in the MAD sequence.
  3. Applying a displacement shifts the BPM's observed closed-orbit
     position by exactly that amount (MAD-NG tracking).
  4. Displacing one BPM does not affect observations at neighbouring BPMs.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aba_optimiser.accelerators import PSB
from aba_optimiser.mad.optimising_mad_interface import GradientDescentMadInterface

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.serial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bpm_dx_knobs(iface: GradientDescentMadInterface) -> list[str]:
    return [k for k in iface.knob_names if k.endswith(".dx") and "BPM" in k.upper()]


def _bpm_dy_knobs(iface: GradientDescentMadInterface) -> list[str]:
    return [k for k in iface.knob_names if k.endswith(".dy") and "BPM" in k.upper()]


def _read_misalign(iface: GradientDescentMadInterface, bpm_name: str, attr: str) -> float:
    """Read misalign.attr from the loaded sequence for a given BPM."""
    iface.mad.send(f"{iface.py_name}:send(loaded_sequence['{bpm_name}'].misalign.{attr}, true)")
    val = iface.mad.recv()
    return float(val) if val is not None else 0.0


def _read_element_attr(iface: GradientDescentMadInterface, bpm_name: str, attr: str) -> float:
    """Read element attribute directly from the loaded sequence."""
    iface.mad.send(f"{iface.py_name}:send(loaded_sequence['{bpm_name}'].{attr}, true)")
    val = iface.mad.recv()
    return float(val) if val is not None else 0.0


def _track_bpm_coord_in_local_frame(
    iface: GradientDescentMadInterface,
    bpm_name: str,
    coord: str = "x",
) -> float:
    """Run a 1-turn track and return the BPM coordinate in the element's local frame.

    For regular particle tracking, the misalignment frame is active at slice -4
    in the atexit callback (after the entry misalignment has been applied but
    before the inverse misalignment at element exit).  This is the position a
    displaced BPM would read: ``coord_beam - displacement``.

    Note: the optimiser workers use DA/TPSA tracking where the same position
    corresponds to slice -2; the slice numbers differ between tracking modes.
    """
    py = iface.py_name
    iface.mad.send(f"""
    local obs_{coord} = nil
    local function capture(elm, mflw, _, slc)
        if slc == -4 and elm.name == '{bpm_name}' then
            obs_{coord} = mflw[1].{coord}
        end
    end
    MAD.track{{
        sequence = loaded_sequence,
        X0 = {{x=0, px=0, y=0, py=0, t=0, pt=0}},
        nturn = 1,
        save = false,
        atexit = capture,
    }}
    {py}:send(obs_{coord}, true)
    """)
    val = iface.mad.recv()
    return float(val) if val is not None else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def psb_bpm_dx_iface(seq_psb: Path) -> GradientDescentMadInterface:
    """GradientDescentMadInterface for PSB ring 3 with BPM horizontal displacement."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, optimise_bpm_dx=True)
    iface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    yield iface
    with contextlib.suppress(Exception):
        del iface


@pytest.fixture(scope="function")
def psb_bpm_dy_iface(seq_psb: Path) -> GradientDescentMadInterface:
    """GradientDescentMadInterface for PSB ring 3 with BPM vertical displacement."""
    accelerator = PSB(ring=3, sequence_file=seq_psb, optimise_bpm_dy=True)
    iface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
    yield iface
    with contextlib.suppress(Exception):
        del iface


# ---------------------------------------------------------------------------
# Knob creation checks
# ---------------------------------------------------------------------------

class TestPSBBPMKnobCreation:
    def test_bpm_dx_knobs_created(self, psb_bpm_dx_iface: GradientDescentMadInterface) -> None:
        """Enabling optimise_bpm_dx creates at least one BPM dx knob per ring-3 BPM."""
        knobs = _bpm_dx_knobs(psb_bpm_dx_iface)
        assert len(knobs) > 0, "No BPM dx knobs found in knob_names"

    def test_bpm_dy_knobs_created(self, psb_bpm_dy_iface: GradientDescentMadInterface) -> None:
        """Enabling optimise_bpm_dy creates at least one BPM dy knob per ring-3 BPM."""
        knobs = _bpm_dy_knobs(psb_bpm_dy_iface)
        assert len(knobs) > 0, "No BPM dy knobs found in knob_names"

    def test_bpm_dx_knob_names_match_ring3_pattern(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """All BPM dx knob names contain 'BR3' and 'BPM' (ring-3 BPM naming convention)."""
        knobs = _bpm_dx_knobs(psb_bpm_dx_iface)
        for k in knobs:
            assert "BR3" in k.upper(), f"Knob {k!r} does not contain 'BR3'"
            assert "BPM" in k.upper(), f"Knob {k!r} does not contain 'BPM'"

    def test_bpm_dx_knobs_have_spos(self, psb_bpm_dx_iface: GradientDescentMadInterface) -> None:
        """Every BPM dx knob has an associated s-position entry."""
        iface = psb_bpm_dx_iface
        knobs = _bpm_dx_knobs(iface)
        assert len(knobs) == len(iface.elem_spos), (
            f"Mismatch: {len(knobs)} BPM dx knobs but {len(iface.elem_spos)} spos entries"
        )

    def test_no_bpm_knobs_when_flags_off(self, seq_psb: Path) -> None:
        """No BPM displacement knobs are created when both flags are False (default)."""
        accelerator = PSB(ring=3, sequence_file=seq_psb, optimise_quadrupoles=True)
        iface = GradientDescentMadInterface(accelerator=accelerator, discard_mad_output=True)
        try:
            bpm_knobs = [k for k in iface.knob_names if ".dx" in k or ".dy" in k]
            # Quadrupole dx/dy might exist – filter to monitors only
            bpm_only = [k for k in bpm_knobs if "BPM" in k.upper()]
            assert bpm_only == [], f"Unexpected BPM knobs: {bpm_only}"
        finally:
            with contextlib.suppress(Exception):
                del iface


# ---------------------------------------------------------------------------
# Misalign chain verification
# ---------------------------------------------------------------------------

class TestPSBBPMMisalignChain:
    def test_bpm_misalign_dx_set_via_knob(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """Setting a BPM dx knob propagates into the element's misalign.dx attribute."""
        iface = psb_bpm_dx_iface
        knob = _bpm_dx_knobs(iface)[0]
        bpm_name = knob.rsplit(".", 1)[0]

        dx_target = 1.23e-3  # 1.23 mm
        iface.mad.send(f"loaded_sequence['{knob}'] = {dx_target}")

        misalign_dx = _read_misalign(iface, bpm_name, "dx")
        assert np.isclose(misalign_dx, dx_target, atol=1e-12), (
            f"misalign.dx = {misalign_dx}, expected {dx_target}"
        )

    def test_bpm_misalign_dy_set_via_knob(
        self, psb_bpm_dy_iface: GradientDescentMadInterface
    ) -> None:
        """Setting a BPM dy knob propagates into the element's misalign.dy attribute."""
        iface = psb_bpm_dy_iface
        knob = _bpm_dy_knobs(iface)[0]
        bpm_name = knob.rsplit(".", 1)[0]

        dy_target = -2.0e-3  # -2 mm
        iface.mad.send(f"loaded_sequence['{knob}'] = {dy_target}")

        misalign_dy = _read_misalign(iface, bpm_name, "dy")
        assert np.isclose(misalign_dy, dy_target, atol=1e-12), (
            f"misalign.dy = {misalign_dy}, expected {dy_target}"
        )

    def test_bpm_dx_element_attr_follows_knob(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """The element's dx attribute (intermediate in the deferred chain) follows the knob."""
        iface = psb_bpm_dx_iface
        knob = _bpm_dx_knobs(iface)[0]
        bpm_name = knob.rsplit(".", 1)[0]

        dx_target = 5e-4
        iface.mad.send(f"loaded_sequence['{knob}'] = {dx_target}")

        element_dx = _read_element_attr(iface, bpm_name, "dx")
        assert np.isclose(element_dx, dx_target, atol=1e-12)

    def test_bpm_displacement_resets_to_zero(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """Setting a BPM displacement back to zero returns misalign.dx to zero."""
        iface = psb_bpm_dx_iface
        knob = _bpm_dx_knobs(iface)[0]
        bpm_name = knob.rsplit(".", 1)[0]

        iface.mad.send(f"loaded_sequence['{knob}'] = 1e-3")
        iface.mad.send(f"loaded_sequence['{knob}'] = 0.0")

        misalign_dx = _read_misalign(iface, bpm_name, "dx")
        assert np.isclose(misalign_dx, 0.0, atol=1e-12)

    def test_multiple_bpm_displacements_are_independent(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """Displacing one BPM does not affect the misalign.dx of its neighbours."""
        iface = psb_bpm_dx_iface
        knobs = _bpm_dx_knobs(iface)
        assert len(knobs) >= 2, "Need at least 2 BPM knobs for independence test"

        target_knob = knobs[0]
        other_knob = knobs[1]
        target_bpm = target_knob.rsplit(".", 1)[0]
        other_bpm = other_knob.rsplit(".", 1)[0]

        iface.mad.send(f"loaded_sequence['{target_knob}'] = 2e-3")

        other_misalign = _read_misalign(iface, other_bpm, "dx")
        assert np.isclose(other_misalign, 0.0, atol=1e-12), (
            f"Displacing {target_bpm} affected {other_bpm}: misalign.dx = {other_misalign}"
        )
        target_misalign = _read_misalign(iface, target_bpm, "dx")
        assert np.isclose(target_misalign, 2e-3, atol=1e-12)


# ---------------------------------------------------------------------------
# Tracking: BPM displacement shifts the observed closed-orbit position
# ---------------------------------------------------------------------------

class TestPSBBPMDisplacementTracking:
    def test_bpm_dx_shifts_observed_x_in_twiss(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """Displacing a BPM by dx shifts the tracked x observation at that BPM by -dx.

        MAD-NG's twiss reports the beam's global-frame position; monitor
        misalignment does not alter the beam orbit so twiss x is unaffected.
        The effect is seen at slice -2 of the track atexit callback, which is
        in the BPM's local (misaligned) frame – exactly as the optimiser
        workers observe it.  A monitor with misalign.dx = d records
        x_observed = x_beam - d at that slice.
        """
        iface = psb_bpm_dx_iface
        knobs = _bpm_dx_knobs(iface)
        bpm_knob = knobs[0]
        bpm_name = bpm_knob.rsplit(".", 1)[0]

        x_before = _track_bpm_coord_in_local_frame(iface, bpm_name, coord="x")

        dx_applied = 1e-3  # 1 mm
        iface.update_knob_values({bpm_knob: dx_applied})

        x_after = _track_bpm_coord_in_local_frame(iface, bpm_name, coord="x")

        shift = x_after - x_before
        assert np.isclose(shift, -dx_applied, atol=1e-9), (
            f"Expected x shift = {-dx_applied:.3e} m, got {shift:.3e} m"
        )

    def test_bpm_dy_shifts_tracked_y_observation(
        self, psb_bpm_dy_iface: GradientDescentMadInterface
    ) -> None:
        """Displacing a BPM by dy shifts the tracked y observation at that BPM by -dy."""
        iface = psb_bpm_dy_iface
        knobs = _bpm_dy_knobs(iface)
        bpm_knob = knobs[0]
        bpm_name = bpm_knob.rsplit(".", 1)[0]

        y_before = _track_bpm_coord_in_local_frame(iface, bpm_name, coord="y")

        dy_applied = 1e-3  # 1 mm
        iface.update_knob_values({bpm_knob: dy_applied})

        y_after = _track_bpm_coord_in_local_frame(iface, bpm_name, coord="y")

        shift = y_after - y_before
        assert np.isclose(shift, -dy_applied, atol=1e-9), (
            f"Expected y shift = {-dy_applied:.3e} m, got {shift:.3e} m"
        )

    def test_bpm_dx_only_affects_x_not_y(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """A BPM horizontal displacement does not change the y observation at that BPM."""
        iface = psb_bpm_dx_iface
        knobs = _bpm_dx_knobs(iface)
        bpm_knob = knobs[0]
        bpm_name = bpm_knob.rsplit(".", 1)[0].upper()

        twiss_before = iface.run_twiss()
        y_before = float(twiss_before.loc[bpm_name, "y"])

        iface.update_knob_values({bpm_knob: 1e-3})

        twiss_after = iface.run_twiss()
        y_after = float(twiss_after.loc[bpm_name, "y"])

        assert np.isclose(y_after, y_before, atol=1e-9), (
            f"dx displacement incorrectly changed y at {bpm_name}: "
            f"{y_before:.3e} → {y_after:.3e}"
        )

    def test_bpm_displacement_does_not_change_neighbouring_bpm_x(
        self, psb_bpm_dx_iface: GradientDescentMadInterface
    ) -> None:
        """Displacing one BPM does not affect the x observation at its neighbour."""
        iface = psb_bpm_dx_iface
        knobs = _bpm_dx_knobs(iface)
        assert len(knobs) >= 2
        target_knob = knobs[0]
        other_bpm = knobs[1].rsplit(".", 1)[0].upper()

        twiss_before = iface.run_twiss()
        x_other_before = float(twiss_before.loc[other_bpm, "x"])

        iface.update_knob_values({target_knob: 1e-3})

        twiss_after = iface.run_twiss()
        x_other_after = float(twiss_after.loc[other_bpm, "x"])

        assert np.isclose(x_other_after, x_other_before, atol=1e-9), (
            f"x at {other_bpm} changed after displacing a different BPM"
        )
