#!/usr/bin/env python3
"""
Script to create LHC model directories for beam 1 and beam 2 at 18cm optics using omc3.
"""

import pathlib
from pathlib import Path

import tfs
from cpymad.madx import Madx
from omc3.model_creator import create_instance_and_model
from pymadng import MAD

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     pass

# Configuration
MODIFIER = "R2025aRP_A18cmC18cmA10mL200cm_Flat.madx"
NAT_TUNES = [0.28, 0.31]
DRV_TUNES = [0.27, 0.322]
ENERGY = 6800
YEAR = "2025"

# From model.py
MADX_FILENAME = "job.create_model_nominal.madx"


def make_madx_seq(
    beam: int,
    model_dir: pathlib.Path,
    beam4: bool = False,
) -> None:
    """
    Generate the MAD-X sequence file for the given beam.

    If beam4 is True, adjust the sequence for beam 4 settings (used for tracking).
    """
    madx_file = model_dir / MADX_FILENAME
    with madx_file.open("r") as f:
        lines = f.readlines()
    if beam4:
        assert beam == 2, "Beam 4 sequence can only be generated for beam 2"
        print("Generating beam4 sequence for tracking")

    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))
        for i, line in enumerate(lines):
            if beam4:
                if "define_nominal_beams" in line:
                    madx.input(
                        "beam, sequence=LHCB2, particle=proton, energy=6800, kbunch=1, npart=1.15E11, bv=1;"
                    )
                    continue
                if "acc-models-lhc/lhc.seq" in line:
                    line = line.replace("acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq")
            if "coupling_knob" in line:
                madx.input(line)
                break  # The coupling knob is the last line to be read
            madx.input(line)
        madx.input(
            f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq", noexpr=false;
            """
        )


def create_model(beam: int, output_dir: pathlib.Path):
    """Create and compress LHC model for the specified beam."""
    print(f"Creating LHC model for beam {beam} at {output_dir}...")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create the model
    create_instance_and_model(
        accel="lhc",
        fetch="afs",
        type="nominal",
        beam=beam,
        year=YEAR,
        driven_excitation="acd",
        energy=ENERGY,
        nat_tunes=NAT_TUNES,
        drv_tunes=DRV_TUNES,
        modifiers=[MODIFIER],
        outputdir=output_dir,
    )

    # Generate the MAD-X sequence
    make_madx_seq(beam, output_dir, beam4=beam == 2)
    update_model_with_ng(beam, output_dir, tunes=NAT_TUNES, drv_tunes=DRV_TUNES)
    print(f"Model for beam {beam} created and compressed successfully.")


MODEL_STRENGTHS = [
    "k1l",
    "k2l",
    "k3l",
    "k4l",
    "k5l",
    "k1sl",
    "k2sl",
    "k3sl",
    "k4sl",
    "k5sl",
]
MODEL_HEADER = [
    "name",
    "type",
    "title",
    "origin",
    "date",
    "time",
    "refcol",
    "direction",
    "observe",
    "energy",
    "deltap",
    "length",
    "q1",
    "q2",
    "q3",
    "dq1",
    "dq2",
    "dq3",
]

MODEL_COLUMNS = [
    "name",
    "kind",
    "s",
    "betx",
    "alfx",
    "bety",
    "alfy",
    "mu1",
    "mu2",
    "dx",
    "dpx",
    "dy",
    "dpy",
    "r11",
    "r12",
    "r21",
    "r22",
]


def start_madng(
    mad: MAD,
    beam: int,
    model_dir: Path,
    *,
    tunes: list[float] = [0.28, 0.31],
) -> None:
    """
    Initialise the accelerator model within MAD-NG.

    Loads the saved sequence and sets the beam parameters.
    """
    saved_seq = model_dir / f"lhcb{beam}_saved.seq"
    saved_mad = model_dir / f"lhcb{beam}_saved.mad"

    mad.MADX.load(f"'{saved_seq}'", f"'{saved_mad}'")
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy=6800}};
MADX.lhcb{beam}.beam = lhc_beam;
print("Initialising model with beam:", {beam});
    """)
    match_tunes(mad, beam, tunes)


def _print_tunes(mad: MAD, beam: int, label: str) -> tuple[float, float]:
    mad.send(f"""
local tbl = twiss {{sequence=MADX.lhcb{beam}}};
py:send({{tbl.q1, tbl.q2}}, true)
    """)
    q1, q2 = mad.recv()
    assert isinstance(q1, float) and isinstance(q2, float), "Received tunes are not floats"
    print(f"{label} tunes: ", q1, q2)
    return q1, q2


def match_tunes(mad: MAD, beam: int, tunes: list[float] = [0.28, 0.31]) -> None:
    """
    Match the tunes of the model to the desired values using MAD-NG.

    The target tunes are hardcoded (62.28 and 60.31 in absolute value) and the routine
    uses a matching command to adjust the optics accordingly.
    """
    q1, q2 = _print_tunes(mad, beam, "Initial")
    if abs(tunes[0] - q1 % 1) < 1e-6 and abs(tunes[1] - q2 % 1) < 1e-6:
        print("Tunes already matched, skipping matching.")
        return
    mad.send(f"""
match {{
  command := twiss {{sequence=MADX.lhcb{beam}}},
  variables = {{
    rtol=1e-6,
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{
    {{ expr = \\t -> math.abs(t.q1)-(62+{tunes[0]}), name='q1' }},
    {{ expr = \\t -> math.abs(t.q2)-(60+{tunes[1]}), name='q2' }},
  }},
  objective = {{ fmin=1e-7 }},
}};
    """)
    _print_tunes(mad, beam, "Final")


def add_strengths_to_twiss(mad: MAD, mtable_name: str) -> None:
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({mtable_name}, strength_cols)
    """).send(MODEL_STRENGTHS)


def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Convert the TFS DataFrame to a format compatible with MAD-X.

    This function performs the following steps:
      - Converts all column names and header keys to uppercase.
      - Renames the columns 'MU1' and 'MU2' to 'MUX' and 'MUY', respectively.
      - Renames drift element names to consecutive values starting at 'DRIFT_0'.
      - Removes rows containing 'vkicker' or 'hkicker' in the 'KIND' column.
      - Sets the 'NAME' column as the index and removes rows with '$start' or '$end'.

    Parameters
    ----------
    tfs_df : tfs.TfsDataFrame
        The input TFS DataFrame.

    Returns
    -------
    tfs.TfsDataFrame
        The converted TFS DataFrame.
    """
    # Convert all headers and column names to uppercase.
    tfs_df.columns = tfs_df.columns.str.upper()
    tfs_df.headers = {key.upper(): value for key, value in tfs_df.headers.items()}

    # Rename columns 'MU1' and 'MU2' to 'MUX' and 'MUY'
    tfs_df = tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"})

    # Replace drift names so that they are consecutive (starting from DRIFT_0)
    drifts = tfs_df[tfs_df["KIND"] == "drift"]
    replace_names = [f"DRIFT_{i}" for i in range(len(drifts))]
    tfs_df["NAME"] = tfs_df["NAME"].replace(drifts["NAME"].to_list(), replace_names)

    # Remove rows containing 'vkicker' or 'hkicker' in the 'KIND' column.
    # tfs_df = tfs_df[~tfs_df["KIND"].str.contains("vkicker|hkicker")]

    # Set the NAME column as index and remove unwanted rows
    tfs_df = tfs_df.set_index("NAME")
    return tfs_df.filter(regex=r"^(?!\$start|\$end).*$", axis="index")

    #  tfs_df


def export_tfs_to_madx(tfs_file: Path) -> None:
    """
    Read a TFS file, convert its contents to MAD-X format, and write it back.

    This function uses the `convert_tfs_to_madx` function to adjust the TFS file
    for compatibility with MAD-X and then writes the converted DataFrame back to disk.

    Parameters
    ----------
    tfs_file : Path
        Path to the TFS file.
    """
    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df)
    tfs.write(tfs_file, tfs_df, save_index="NAME")


def observe_bpms(mad: MAD, beam: int) -> None:
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:  select(observed, {{pattern="BPM"}})
    """)


def update_model_with_ng(
    beam: int,
    model_dir: Path,
    tunes: list[float] = [0.28, 0.31],
    drv_tunes: list[float] = [0.0, 0.0],
) -> None:
    """
    Update the accelerator model with MAD-NG and perform tune matching.

    This routine loads the saved sequence into MAD-NG, initialises the beam parameters,
    and then calls the tune matching routine.
    """
    with MAD() as mad:
        start_madng(mad, beam, model_dir, tunes=tunes)
        mad.send(f"""
-- Set the twiss table information needed for the model update
hnams = py:recv()
cols = py:recv()
str_cols = py:recv()

cols = MAD.utility.tblcat(cols, str_cols)

! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{ sequence=MADX.lhcb{beam}, coupling=true }}
twiss_elements:select(nil, \\ -> true) ! Everything
twiss_elements:deselect{{pattern="drift"}}
""")
        mad.send(MODEL_HEADER).send(MODEL_COLUMNS).send(MODEL_STRENGTHS)
        add_strengths_to_twiss(mad, "twiss_elements")
        mad.send(
            # True below is to make sure only selected rows are written
            f"""twiss_elements:write("{model_dir / "twiss_elements.dat"}", cols, hnams, true)"""
        )
        observe_bpms(mad, beam)
        ac_marker = f"MKQA.6L4.B{beam}"
        mad.send(f"""
twiss_data = twiss {{sequence=MADX.lhcb{beam}, coupling=true, observe=1 }}
""")
        mad.send(f"""
local hackicker, vackicker in MAD.element
MADX.lhcb{beam}:install{{
    hackicker "hackicker" {{
        at = 1.583/2,
        from = "{ac_marker}",

        -- quad part
        nat_q = {tunes[0]:.5e},
        drv_q = {drv_tunes[0]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta11,
    }},
    vackicker "vackicker" {{
        at = 1.583/2,
        from = "{ac_marker}",

        -- quad part
        nat_q = {tunes[1]:.5e},
        drv_q = {drv_tunes[1]:.5e},
        ac_bet = twiss_elements['{ac_marker}'].beta22,
    }}
}}
twiss_ac = twiss {{sequence=MADX.lhcb{beam}, coupling=true, observe=1 }}
        """)
        add_strengths_to_twiss(mad, "twiss_ac")
        add_strengths_to_twiss(mad, "twiss_data")
        mad.send(f"""
twiss_ac:write("{model_dir / "twiss_ac.dat"}", cols, hnams)
twiss_data:write("{model_dir / "twiss.dat"}", cols, hnams)
print("Replaced twiss data tables")
py:send("write complete")
""")
        assert mad.receive() == "write complete", "Error in writing twiss tables"

    # Read the twiss data tables and then convert all the headers to uppercase and column names to uppercase
    export_tfs_to_madx(model_dir / "twiss_ac.dat")
    export_tfs_to_madx(model_dir / "twiss_elements.dat")
    export_tfs_to_madx(model_dir / "twiss.dat")


if __name__ == "__main__":
    # Create models for both beams
    data_dir = pathlib.Path(__file__).parent / "data"

    # Beam 1
    model_dir_b1 = data_dir / "model_b1__t0.28_0.31_18cm"
    create_model(beam=1, output_dir=model_dir_b1)

    # Beam 2
    model_dir_b2 = data_dir / "model_b2__t0.28_0.31_18cm"
    create_model(beam=2, output_dir=model_dir_b2)

    print("All models created.")
