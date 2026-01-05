#!/usr/bin/env python3
"""
Script to create LHC model directories for beam 1 and beam 2 at 18cm optics using omc3.

This script orchestrates the complete model creation workflow:
1. Creates nominal model using omc3
2. Generates MAD-X sequences
3. Updates model with MAD-NG (tune matching, twiss computation)
4. Exports TFS files in MAD-X format
"""

import pathlib

from omc3.model_creator import create_instance_and_model

from .config import DRV_TUNES, ENERGY, MODIFIER, NAT_TUNES, YEAR
from .madng_utils import update_model_with_madng
from .madx_utils import make_madx_sequence


def create_lhc_model(
    beam: int,
    output_dir: pathlib.Path,
    *,
    nat_tunes: list[float] | None = None,
    drv_tunes: list[float] | None = None,
    energy: int | None = None,
    year: str | None = None,
    modifiers: str | list[str] | None = None,
    matching_knob: str = "_op",
) -> None:
    """
    Create a complete LHC model for the specified beam.

    This function performs the full workflow:
    1. Creates model instance using omc3
    2. Generates MAD-X sequence files (including beam4 for tracking if beam=2)
    3. Updates model with MAD-NG (tune matching and twiss computation)

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    output_dir : pathlib.Path
        Directory where model files will be created.
    nat_tunes : list[float], optional
        Natural fractional tunes [Q1, Q2]. Defaults to config values.
    drv_tunes : list[float], optional
        Driven fractional tunes [Q1, Q2]. Defaults to config values.
    energy : int, optional
        Beam energy in GeV. Defaults to config value.
    year : str, optional
        LHC year/era. Defaults to config value.
    modifier : str, optional
        Optics modifier file name. Defaults to config value.
    matching_knob : str, optional
        Suffix for the tune matching knobs (e.g., "_op" for dQx.b1_op,
        "" for dQx.b1, "_sq" for dQx.b1_sq). Default is "_op".

    Raises
    ------
    ValueError
        If beam is not 1 or 2.
    """
    if beam not in (1, 2):
        raise ValueError(f"Beam must be 1 or 2, got {beam}")

    # Use defaults from config if not specified
    nat_tunes = nat_tunes or NAT_TUNES
    drv_tunes = drv_tunes or DRV_TUNES
    energy = energy or ENERGY
    year = year or YEAR

    if not modifiers:
        modifiers = [MODIFIER]
    elif isinstance(modifiers, str):
        modifiers = [modifiers]

    print(f"\n{'=' * 70}")
    print(f"Creating LHC Model for Beam {beam}")
    print(f"{'=' * 70}")
    print(f"Output directory: {output_dir}")
    print(f"Natural tunes: {nat_tunes}")
    print(f"Driven tunes: {drv_tunes}")
    print(f"Energy: {energy} GeV")
    print(f"Year: {year}")
    print(f"Modifiers: {modifiers}")
    print(f"{'=' * 70}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Create base model with omc3
    print("Step 1: Creating base model with omc3...")
    create_instance_and_model(
        accel="lhc",
        fetch="afs",
        type="nominal",
        beam=beam,
        year=year,
        driven_excitation="acd",
        energy=energy,
        nat_tunes=nat_tunes,
        drv_tunes=drv_tunes,
        modifiers=modifiers,
        outputdir=output_dir,
    )
    print("✓ Base model created\n")

    # Step 2: Generate MAD-X sequences
    print("Step 2: Generating MAD-X sequences...")
    make_madx_sequence(beam, output_dir, beam4=(beam == 2), matching_knob=matching_knob)
    print("✓ MAD-X sequences generated\n")

    # Step 3: Update with MAD-NG
    print("Step 3: Updating model with MAD-NG...")
    update_model_with_madng(
        beam,
        output_dir,
        tunes=nat_tunes,
        drv_tunes=drv_tunes,
        matching_knob=matching_knob,
    )
    print("✓ Model update complete\n")

    print(f"{'=' * 70}")
    print(f"Model for beam {beam} created successfully!")
    print(f"Location: {output_dir}")
    print(f"{'=' * 70}\n")


def main() -> None:
    """Main entry point for creating LHC models."""
    # Determine output directory relative to this script
    data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Model naming convention: model_b{beam}__t{q1}_{q2}_{optics}
    nat_tunes = NAT_TUNES
    optics_label = "18cm"

    print("\n" + "=" * 70)
    print("LHC Model Creation Script")
    print("=" * 70 + "\n")

    # Create Beam 1 model
    model_dir_b1 = data_dir / f"model_b1__t{nat_tunes[0]}_{nat_tunes[1]}_{optics_label}"
    create_lhc_model(beam=1, output_dir=model_dir_b1)

    # Create Beam 2 model
    model_dir_b2 = data_dir / f"model_b2__t{nat_tunes[0]}_{nat_tunes[1]}_{optics_label}"
    create_lhc_model(beam=2, output_dir=model_dir_b2)

    print("\n" + "=" * 70)
    print("All models created successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
