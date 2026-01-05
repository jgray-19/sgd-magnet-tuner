"""MAD-NG utility functions for LHC model creation and tune matching."""

from pathlib import Path

from pymadng import MAD

from .config import (
    AC_MARKER_OFFSET,
    AC_MARKER_PATTERN,
    ENERGY,
    MODEL_COLUMNS,
    MODEL_HEADER,
    MODEL_STRENGTHS,
    TUNE_MATCH_FMIN,
    TUNE_MATCH_RTOL,
    TUNE_MATCH_TOLERANCE,
    TUNE_Q1_OFFSET,
    TUNE_Q2_OFFSET,
)
from .tfs_utils import convert_multiple_tfs_files


def initialise_madng_model(
    mad: MAD,
    beam: int,
    model_dir: Path,
    *,
    tunes: list[float] | None = None,
    matching_knob: str = "_op",
) -> None:
    """
    Initialize the LHC accelerator model in MAD-NG.

    Loads the saved MAD-X sequence into MAD-NG, sets beam parameters,
    and optionally matches tunes to target values.

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    beam : int
        Beam number (1 or 2).
    model_dir : Path
        Directory containing the saved sequence files.
    tunes : list[float], optional
        Target fractional tunes [Q1, Q2]. If None, uses [0.28, 0.31].
    matching_knob : str, optional
        Suffix for the tune matching knobs. Default is "_op".
    """
    if tunes is None:
        tunes = [0.28, 0.31]

    saved_seq = model_dir / f"lhcb{beam}_saved.seq"
    saved_mad = model_dir / f"lhcb{beam}_saved.mad"

    if not saved_seq.exists():
        raise FileNotFoundError(
            f"Saved sequence file not found in {model_dir}. Run make_madx_sequence first."
        )

    # Load the saved sequence
    mad.MADX.load(f"'{saved_seq}'", f"'{saved_mad}'")

    # Set beam parameters
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy={ENERGY}}};
MADX.lhcb{beam}.beam = lhc_beam;
print("Initialized MAD-NG model for beam {beam}");
    """)

    # Match tunes
    match_model_tunes(mad, beam, tunes, matching_knob)


def get_current_tunes(mad: MAD, beam: int, label: str = "") -> tuple[float, float]:
    """
    Retrieve current tunes from the MAD-NG model.

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    beam : int
        Beam number.
    label : str, optional
        Label for logging (e.g., "Initial", "Final").

    Returns
    -------
    tuple[float, float]
        Fractional tunes (Q1, Q2).

    Raises
    ------
    AssertionError
        If received values are not floats.
    """
    mad.send(f"""
local tbl = twiss {{sequence=MADX.lhcb{beam}}};
py:send({{tbl.q1, tbl.q2}}, true)
    """)
    q1, q2 = mad.recv()

    if not isinstance(q1, float) or not isinstance(q2, float):
        raise TypeError(f"Expected float tunes, got {type(q1)} and {type(q2)}")

    log_msg = f"{label} tunes" if label else "Tunes"
    print(f"{log_msg}: Q1={q1:.6f}, Q2={q2:.6f}")

    return q1, q2


def match_model_tunes(
    mad: MAD,
    beam: int,
    target_tunes: list[float],
    matching_knob: str = "_op",
) -> None:
    """
    Match model tunes to target fractional values.

    Uses MAD-NG's matching algorithm to adjust tune knobs until the
    model tunes match the target values within tolerance.

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    beam : int
        Beam number.
    target_tunes : list[float]
        Target fractional tunes [Q1, Q2] (e.g., [0.28, 0.31]).
    matching_knob : str, optional
        Suffix for the tune matching knobs (e.g., "_op" for dQx.b1_op,
        "" for dQx.b1, "_sq" for dQx.b1_sq). Default is "_op".

    Notes
    -----
    Absolute tunes are computed as: Q1_abs = 62 + target_tunes[0],
    Q2_abs = 60 + target_tunes[1].
    """
    q1, q2 = get_current_tunes(mad, beam, "Initial")

    # Check if already matched
    q1_frac = q1 % 1
    q2_frac = q2 % 1

    if (
        abs(target_tunes[0] - q1_frac) < TUNE_MATCH_TOLERANCE
        and abs(target_tunes[1] - q2_frac) < TUNE_MATCH_TOLERANCE
    ):
        print("Tunes already matched within tolerance, skipping matching.")
        return

    # Target absolute tunes
    target_q1_abs = TUNE_Q1_OFFSET + target_tunes[0]
    target_q2_abs = TUNE_Q2_OFFSET + target_tunes[1]

    # Execute matching
    match_cmd = f"""
match {{
  command := twiss {{sequence=MADX.lhcb{beam}}},
  variables = {{
    rtol={TUNE_MATCH_RTOL},
    {{ var = 'MADX.dqx_b{beam}{matching_knob}', name='dQx.b{beam}{matching_knob}' }},
    {{ var = 'MADX.dqy_b{beam}{matching_knob}', name='dQy.b{beam}{matching_knob}' }},
  }},
  equalities = {{
    {{ expr = \\t -> math.abs(t.q1)-{target_q1_abs}, name='q1' }},
    {{ expr = \\t -> math.abs(t.q2)-{target_q2_abs}, name='q2' }},
  }},
  objective = {{ fmin={TUNE_MATCH_FMIN} }},
}};
    """
    mad.send(match_cmd)

    get_current_tunes(mad, beam, "Final")


def add_strength_columns(mad: MAD, table_name: str) -> None:
    """
    Add multipole strength columns to a MAD-NG twiss table.

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    table_name : str
        Name of the twiss table in MAD-NG.
    """
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({table_name}, strength_cols)
    """).send(MODEL_STRENGTHS)


def configure_bpm_observation(mad: MAD, beam: int) -> None:
    """
    Configure the model to observe BPM elements only.

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    beam : int
        Beam number.
    """
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:select(observed, {{pattern="BPM"}})
    """)


def compute_and_export_twiss_tables(
    mad: MAD,
    beam: int,
    model_dir: Path,
    *,
    tunes: list[float],
    drv_tunes: list[float],
) -> None:
    """
    Compute twiss tables with MAD-NG and export to TFS files.

    Generates three twiss tables:
    - twiss_elements.dat: All non-drift elements with strengths
    - twiss.dat: BPM observations with natural tunes
    - twiss_ac.dat: BPM observations with AC dipole excitation

    Parameters
    ----------
    mad : MAD
        Active MAD-NG instance.
    beam : int
        Beam number.
    model_dir : Path
        Output directory for TFS files.
    tunes : list[float]
        Natural fractional tunes [Q1, Q2].
    drv_tunes : list[float]
        Driven fractional tunes [Q1, Q2] for AC dipole.
    """
    # Setup table configuration
    mad.send(f"""
-- Receive table configuration
hnams = py:recv()
cols = py:recv()
str_cols = py:recv()

cols = MAD.utility.tblcat(cols, str_cols)

-- Compute twiss for all elements (coupling enabled for R-matrix)
twiss_elements = twiss {{ sequence=MADX.lhcb{beam}, coupling=true }}
twiss_elements:select(nil, \\ -> true)
twiss_elements:deselect{{pattern="drift"}}
""")
    mad.send(MODEL_HEADER).send(MODEL_COLUMNS).send(MODEL_STRENGTHS)

    add_strength_columns(mad, "twiss_elements")

    # Write elements table
    mad.send(f"""twiss_elements:write("{model_dir / "twiss_elements.dat"}", cols, hnams)""")

    # Configure BPM observation
    configure_bpm_observation(mad, beam)

    # Compute natural twiss
    mad.send(f"""
twiss_data = twiss {{sequence=MADX.lhcb{beam}, coupling=true, observe=1}}
""")

    # Install AC dipole kickers
    ac_marker = AC_MARKER_PATTERN.format(beam=beam)
    mad.send(f"""
local hackicker, vackicker in MAD.element
MADX.lhcb{beam}:install{{
    hackicker "hackicker" {{
        at = {AC_MARKER_OFFSET},
        from = "{ac_marker}",
        nat_q = {tunes[0]:.5e},
        drv_q = {drv_tunes[0]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta11,
    }},
    vackicker "vackicker" {{
        at = {AC_MARKER_OFFSET},
        from = "{ac_marker}",
        nat_q = {tunes[1]:.5e},
        drv_q = {drv_tunes[1]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta22,
    }}
}}
twiss_ac = twiss {{sequence=MADX.lhcb{beam}, coupling=true, observe=1}}
        """)

    # Add strengths to all tables
    add_strength_columns(mad, "twiss_ac")
    add_strength_columns(mad, "twiss_data")

    # Write all tables
    mad.send(f"""
twiss_ac:write("{model_dir / "twiss_ac.dat"}", cols, hnams)
twiss_data:write("{model_dir / "twiss.dat"}", cols, hnams)
print("Exported twiss tables")
py:send("export_complete")
""")

    result = mad.receive()
    if result != "export_complete":
        raise RuntimeError(f"Failed to export twiss tables: {result}")

    print(f"Successfully exported twiss tables to {model_dir}")


def update_model_with_madng(
    beam: int,
    model_dir: Path,
    *,
    tunes: list[float] | None = None,
    drv_tunes: list[float] | None = None,
    matching_knob: str = "_op",
) -> None:
    """
    Update LHC model using MAD-NG with tune matching and twiss computation.

    This is the main workflow function that:
    1. Initializes the MAD-NG model
    2. Matches tunes to target values
    3. Computes and exports twiss tables (elements, natural, AC dipole)
    4. Converts TFS files to MAD-X format

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    model_dir : Path
        Model directory containing saved sequences and for output files.
    tunes : list[float], optional
        Natural fractional tunes [Q1, Q2]. Defaults to [0.28, 0.31].
    drv_tunes : list[float], optional
        Driven fractional tunes [Q1, Q2]. Defaults to [0.0, 0.0].
    matching_knob : str, optional
        Suffix for the tune matching knobs. Default is "_op".
    """
    if tunes is None:
        tunes = [0.28, 0.31]
    if drv_tunes is None:
        drv_tunes = [0.0, 0.0]

    print(f"\n{'=' * 60}")
    print(f"Updating model for beam {beam} with MAD-NG")
    print(f"Natural tunes: {tunes}, Driven tunes: {drv_tunes}")
    print(f"{'=' * 60}\n")

    with MAD() as mad:
        # Initialize and match tunes
        initialise_madng_model(mad, beam, model_dir, tunes=tunes, matching_knob=matching_knob)

        # Compute and export twiss tables
        compute_and_export_twiss_tables(
            mad,
            beam,
            model_dir,
            tunes=tunes,
            drv_tunes=drv_tunes,
        )

    # Convert TFS files to MAD-X format
    tfs_files = [
        model_dir / "twiss_ac.dat",
        model_dir / "twiss_elements.dat",
        model_dir / "twiss.dat",
    ]
    convert_multiple_tfs_files(tfs_files)

    print(f"\nModel update complete for beam {beam}\n")
