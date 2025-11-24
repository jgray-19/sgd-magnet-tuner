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
    # Convert to uppercase for MAD-X compatibility
    tfs_df.columns = tfs_df.columns.str.upper()
    tfs_df.headers = {key.upper(): value for key, value in tfs_df.headers.items()}

    # Rename phase advance columns to MAD-X convention
    tfs_df = tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"})

    # Renumber drift elements consecutively
    drifts = tfs_df[tfs_df["KIND"] == "drift"]
    if not drifts.empty:
        drift_names = drifts["NAME"].tolist()
        new_drift_names = [f"DRIFT_{i}" for i in range(len(drift_names))]
        tfs_df["NAME"] = tfs_df["NAME"].replace(drift_names, new_drift_names)

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
