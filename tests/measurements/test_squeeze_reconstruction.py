"""Real (no-mock) integration test for the squeeze AC-dipole reconstruction path.

Tracks a genuine AC-dipole excitation through the LHC beam-1 model with xtrack,
writes the resulting BPM turn-by-turn data to a real SDDS file, and feeds that
file through :func:`reconstruct_ac_dipole_measurements` exactly as production
does. No collaborators are mocked: this exercises the real MAD-NG-backed
``calculate_pz`` reconstruction end to end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("tmom_recon")
pytest.importorskip("xtrack_tools")

import turn_by_turn as tbt
from tmom_recon import ReconstructionFrame
from turn_by_turn.structures import TbtData, TransverseData
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.monitors import line_to_dataframes

from aba_optimiser.measurements.squeeze.reconstruction import reconstruct_ac_dipole_measurements
from aba_optimiser.noise.noise import build_bpm_variance_maps
from tests.training.helpers import generate_xsuite_env_with_errors

if TYPE_CHECKING:
    from pathlib import Path

    from aba_optimiser.mad.aba_mad_interface import AbaMadInterface

pytestmark = pytest.mark.serial

DRIVEN_TUNES = [0.27, 0.322]
LHC_HORIZONTAL_EXCITATION = 0.000371554879506
LHC_VERTICAL_EXCITATION = 0.000415765635123


@pytest.mark.slow
def test_reconstruct_ac_dipole_measurements_end_to_end(
    tmp_path: Path,
    seq_b1: Path,
    model_dir_b1: Path,
    loaded_interface: AbaMadInterface,
) -> None:
    """A real ACD-tracked LHC measurement file reconstructs to non-trivial px/py."""
    ramp_turns = 1000
    flattop_turns = 100

    env, _magnet_strengths, _matched_tunes, _corrector_table = generate_xsuite_env_with_errors(
        loaded_interface,
        dpp_value=0,
        corrector_file=None,
        perturb_quads=False,
        perturb_bends=False,
        apply_orbit_correction=False,
    )
    line = env["lhcb1"]
    tws = line.twiss(method="4d", delta0=0)

    monitored_line = run_ac_dipole_tracking(
        line=line,
        tws=tws,
        acd_marker="mkqa.6l4.b1",
        sequence_name="lhcb1",
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=DRIVEN_TUNES,
        bpm_pattern="bpm.*[^k]",
        deltap=0.0,
        horizontal_excitation=LHC_HORIZONTAL_EXCITATION,
        vertical_excitation=LHC_VERTICAL_EXCITATION,
    )

    track_df = line_to_dataframes(monitored_line)[0]
    track_df = track_df[
        (track_df["turn"] > ramp_turns) & (track_df["turn"] < ramp_turns + flattop_turns)
    ].copy()
    track_df["turn"] = track_df["turn"] - ramp_turns - 1
    track_df = track_df[~track_df["name"].str.contains(r"bpmcs\.", case=False, regex=True)].copy()
    track_df["name"] = track_df["name"].str.upper()

    # Real measurement files only ever contain genuinely instrumented BPMs (the
    # ones the noise table has entries for); the full-ring xtrack monitor
    # pattern also matches special/non-acquisition BPM markers in the model.
    known_bpms, _ = build_bpm_variance_maps("lhc")
    track_df = track_df[track_df["name"].isin(known_bpms)].copy()

    bpm_names = sorted(track_df["name"].unique())
    turns = sorted(track_df["turn"].unique())
    x_pivot = track_df.pivot(index="name", columns="turn", values="x").loc[bpm_names, turns]
    y_pivot = track_df.pivot(index="name", columns="turn", values="y").loc[bpm_names, turns]
    # SDDS/LHC turn-by-turn files store positions in millimetres.
    x_pivot = x_pivot * 1000.0
    y_pivot = y_pivot * 1000.0
    x_pivot.columns = range(len(turns))
    y_pivot.columns = range(len(turns))

    tbt_data = TbtData(
        matrices=[TransverseData(X=x_pivot, Y=y_pivot)],
        nturns=len(turns),
        bunch_ids=[0],
    )
    measurement_file = tmp_path / "measurement.sdds"
    tbt.write(measurement_file, tbt_data, datatype="lhc")

    results = reconstruct_ac_dipole_measurements(
        measurement_files=[measurement_file],
        model_dir=model_dir_b1,
        sequence_path=seq_b1,
        beam=1,
        energy=6800.0,
        frame=ReconstructionFrame(
            orbit_zero=track_df.groupby("name")[["x", "y"]].mean(),
            dynamic_planes=("x", "y"),
        ),
        num_workers=1,
    )

    assert set(results) == {measurement_file.stem}
    bpm_table = results[measurement_file.stem]

    assert {"name", "turn", "x", "y", "px", "py", "var_x", "var_y", "bunch_number"} <= set(
        bpm_table.columns
    )
    assert (bpm_table["bunch_number"] == 0).all()

    upstream = bpm_table.attrs["ac_dipole_bpm_upstream"]
    downstream = bpm_table.attrs["ac_dipole_bpm_downstream"]
    for bpm_name in (upstream, downstream):
        momenta = bpm_table.loc[bpm_table["name"] == bpm_name, ["px", "py"]]
        assert not momenta.empty
        assert np.isfinite(momenta.to_numpy()).all()
        assert (momenta.to_numpy() != 0.0).any()

    marker_names = set(bpm_table["name"].astype(str))
    assert any(name.endswith("_before") for name in marker_names)
    assert any(name.endswith("_after") for name in marker_names)

    for key in ("DPP_EST", "PT_EST", "ac_dipole_marker"):
        assert key in bpm_table.attrs
