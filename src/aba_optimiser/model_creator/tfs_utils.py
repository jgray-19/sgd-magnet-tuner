"""TFS file format conversion utilities."""

from pathlib import Path

import tfs


def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """
    Convert TFS DataFrame from MAD-NG format to MAD-X format.

    This function performs the following transformations:
    - Converts all column names and header keys to uppercase
    - Renames phase advance columns: 'MU1' → 'MUX', 'MU2' → 'MUY'
    - Renames drift elements to consecutive numbering (DRIFT_0, DRIFT_1, ...)
    - Sets 'NAME' as index and removes special rows ('$start', '$end')

    Parameters
    ----------
    tfs_df : tfs.TfsDataFrame
        Input TFS DataFrame in MAD-NG format.

    Returns
    -------
    tfs.TfsDataFrame
        Converted TFS DataFrame in MAD-X format.
    """
    # Reset the index to manipulate columns
    tfs_df = tfs_df.reset_index()

    # Make the headers all upper case for consistency
    tfs_df.headers = {k.upper(): v for k, v in tfs_df.headers.items()}

    # Make the columns all upper case for consistency
    tfs_df.columns = [col.upper() for col in tfs_df.columns]

    # Change MU1 and MU2 from MUX and MUY column names
    tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"}, inplace=True)

    # Change disp1 and disp3 to DX and DY
    tfs_df.rename(
        columns={"DISP1": "DX", "DISP2": "DPX", "DISP3": "DY", "DISP4": "DPY"}, inplace=True
    )

    # Renumber drift elements consecutively
    drifts = tfs_df[
        (tfs_df["KIND"] == "drift") & (tfs_df["NAME"].str.lower().str.startswith("drift"))
    ]
    if not drifts.empty:
        new_drift_names = [f"DRIFT_{i}" for i in range(len(drifts))]
        tfs_df.loc[tfs_df["KIND"] == "drift", "NAME"] = new_drift_names

    # Set NAME as index and filter out special markers
    tfs_df = tfs_df.set_index("NAME")
    return tfs_df.filter(regex=r"^(?!\$start|\$end).*$", axis="index")


def export_tfs_to_madx(tfs_file: Path) -> None:
    """
    Read a TFS file, convert to MAD-X format, and overwrite the original.

    This is a convenience function that reads a TFS file, applies the
    MAD-NG to MAD-X conversion, and writes it back.

    Parameters
    ----------
    tfs_file : Path
        Path to the TFS file to convert.

    Raises
    ------
    FileNotFoundError
        If the TFS file doesn't exist.
    """
    if not tfs_file.exists():
        raise FileNotFoundError(f"TFS file not found: {tfs_file}")

    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df)
    tfs.write(tfs_file, tfs_df, save_index="NAME")

    print(f"Converted TFS file to MAD-X format: {tfs_file.name}")


def convert_multiple_tfs_files(tfs_files: list[Path]) -> None:
    """
    Convert multiple TFS files to MAD-X format.

    Parameters
    ----------
    tfs_files : list[Path]
        List of TFS file paths to convert.
    """
    for tfs_file in tfs_files:
        export_tfs_to_madx(tfs_file)
