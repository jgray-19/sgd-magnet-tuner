#!/usr/bin/env python3
"""
Script to create LHC model directories for beam 1 and beam 2 at 18cm optics using omc3.
"""

import pathlib

from cpymad.madx import Madx
from omc3.model_creator import create_instance_and_model

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
                    line = line.replace(
                        "acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq"
                    )
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
    make_madx_seq(beam, output_dir)
    if beam == 2:
        make_madx_seq(beam, output_dir, beam4=True)

    print(f"Model for beam {beam} created and compressed successfully.")

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
