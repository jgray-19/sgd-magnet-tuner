"""MAD-X utility functions for LHC model creation."""

import pathlib

from cpymad.madx import Madx

from .config import ENERGY, LHC_KBUNCH, LHC_NPART, MADX_FILENAME


def make_madx_sequence(
    beam: int,
    model_dir: pathlib.Path,
    *,
    beam4: bool = False,
    madx_filename: str | None = None,
) -> None:
    """
    Generate and save the MAD-X sequence file for the specified beam.

    This function reads the MAD-X job file, processes it line by line,
    and saves the sequence. For beam 4 mode (tracking), it adjusts beam
    settings and uses the lhcb4.seq file instead of lhc.seq.

    Parameters
    ----------
    beam : int
        Beam number (1 or 2).
    model_dir : pathlib.Path
        Directory containing the model files.
    beam4 : bool, optional
        If True, configure for beam 4 tracking mode (only valid for beam 2).
        Default is False.
    madx_filename : str, optional
        Name of the MAD-X job file. If None, uses default from config.

    Raises
    ------
    AssertionError
        If beam4 is True but beam is not 2.
    FileNotFoundError
        If the MAD-X job file doesn't exist.
    """
    if beam4 and beam != 2:
        raise ValueError("Beam 4 sequence can only be generated for beam 2")

    filename = madx_filename or MADX_FILENAME
    madx_file = model_dir / filename

    if not madx_file.exists():
        raise FileNotFoundError(f"MAD-X file not found: {madx_file}")

    with madx_file.open("r") as f:
        lines = f.readlines()

    if beam4:
        print(f"Generating beam4 sequence for tracking (beam {beam})")

    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))

        for line in lines:
            # Handle beam4 specific modifications
            if beam4:
                if "define_nominal_beams" in line:
                    # Override with beam4 settings
                    beam_cmd = (
                        f"beam, sequence=LHCB2, particle=proton, "
                        f"energy={ENERGY}, kbunch={LHC_KBUNCH}, "
                        f"npart={LHC_NPART:.2e}, bv=1;"
                    )
                    madx.input(beam_cmd)
                    continue

                if "acc-models-lhc/lhc.seq" in line:
                    # Use beam4 sequence file instead
                    line = line.replace("acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq")

            # Stop at coupling knob (last line to process)
            if "coupling_knob" in line:
                madx.input(line)
                break

            madx.input(line)

        # Save the sequence
        save_cmd = f"""
set, format="-16.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq", noexpr=false;
        """
        madx.input(save_cmd)

    print(f"Saved MAD-X sequence for beam {beam} to lhcb{beam}_saved.seq")
