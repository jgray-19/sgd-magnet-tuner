"""MAD-X utility functions for LHC model creation."""

import pathlib

from cpymad.madx import Madx

from .config import ENERGY, LHC_KBUNCH, LHC_NPART, MADX_FILENAME


def make_madx_sequence(
    beam: int,
    model_dir: pathlib.Path,
    *,
    seq_outdir: pathlib.Path | None = None,
    beam4: bool = False,
    madx_filename: str | None = None,
    matching_knob: str = "_op",
) -> pathlib.Path:
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
    seq_outdir : pathlib.Path, optional
        Output directory for the generated sequence file. If None,
        saves in model_dir.
    beam4 : bool, optional
        If True, configure for beam 4 tracking mode (only valid for beam 2).
        Default is False.
    madx_filename : str, optional
        Name of the MAD-X job file. If None, uses default from config.
    matching_knob : str, optional
        Suffix for the tune matching knobs (e.g., "_op" for dQx.b1_op,
        "" for dQx.b1, "_sq" for dQx.b1_sq). Default is "_op".

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

            # Replace match_tunes call with custom matching routine if non-default knob
            if matching_knob != "_op" and "exec, match_tunes(" in line:
                # Extract tune values and beam number from the line
                # Format: exec, match_tunes(0.28, 0.31, 1);
                import re

                match = re.search(r"match_tunes\(([\d.]+),\s*([\d.]+),\s*(\d+)\)", line)
                if match:
                    qx, qy, beam_num = match.groups()
                    qx_total = float(qx) - int(float(qx)) + 62
                    qy_total = float(qy) - int(float(qy)) + 60

                    # Use madx.input for custom matching
                    match_cmd = f"""
match, sequence=LHCB{beam_num};
vary, name=dQx.b{beam_num}{matching_knob};
vary, name=dQy.b{beam_num}{matching_knob};
constraint, range=#E, mux={qx_total}, muy={qy_total};
lmdif, tolerance=1E-10;
endmatch;
"""
                    madx.input(match_cmd)
                    continue

            # Stop at coupling knob (last line to process)
            if "coupling_knob" in line:
                madx.input(line)
                break

            madx.input(line)

        seq_name = f"lhcb{beam}_saved.seq"
        seq_path = seq_outdir / seq_name if seq_outdir else model_dir / seq_name
        # Save the sequence
        save_cmd = f"""
set, format="-16.16e";
save, sequence=lhcb{beam}, file="{seq_path.absolute()}", noexpr=false;
        """
        madx.input(save_cmd)

    print(f"Saved MAD-X sequence for beam {beam} to {seq_path}")
    return seq_path
