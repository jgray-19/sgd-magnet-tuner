"""
Script to recreate beta beating plots from saved TFS files.

This script reads the TFS files saved during the test and recreates
the beta beating plots showing the errors before and after correction.

Usage:
    python plot_beta_beating.py <tmp_dir>

where <tmp_dir> is the directory containing the TFS files from the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tfs


def plot_beta_beating(tmp_dir: Path | str) -> None:
    """
    Plot beta beating from TFS files.

    Args:
        tmp_dir: Directory containing the TFS files
    """
    tmp_dir = Path(tmp_dir)

    # Read the TFS files
    before_file = tmp_dir / "beta_beating_before_correction.tfs"
    after_file = tmp_dir / "beta_beating_after_correction.tfs"

    if not before_file.exists():
        raise FileNotFoundError(f"File not found: {before_file}")
    if not after_file.exists():
        raise FileNotFoundError(f"File not found: {after_file}")

    beta_before = tfs.read(before_file)
    beta_after = tfs.read(after_file)

    # Plot beta beating before corrections
    plt.figure(figsize=(12, 6))
    plt.plot(beta_before["s"], beta_before["betax_error_percent"], label="BetaX error (%)")
    plt.plot(beta_before["s"], beta_before["betay_error_percent"], label="BetaY error (%)")
    plt.xlabel("s (m)")
    plt.ylabel("Relative beta error (%)")
    plt.title("Beta function errors due to magnet perturbations")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Plot beta beating after corrections
    plt.figure(figsize=(12, 6))
    plt.plot(beta_after["s"], beta_after["betax_error_percent"], label="Estimated BetaX error (%)")
    plt.plot(beta_after["s"], beta_after["betay_error_percent"], label="Estimated BetaY error (%)")
    plt.xlabel("s (m)")
    plt.ylabel("Relative beta error (%)")
    plt.title("Estimated beta function errors after bend optimisation")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Print some statistics
    print("\nBeta beating statistics:")
    print("\nBefore correction:")
    print(f"  BetaX - Max: {beta_before['betax_error_percent'].abs().max():.2f}%, "
          f"RMS: {(beta_before['betax_error_percent']**2).mean()**0.5:.2f}%")
    print(f"  BetaY - Max: {beta_before['betay_error_percent'].abs().max():.2f}%, "
          f"RMS: {(beta_before['betay_error_percent']**2).mean()**0.5:.2f}%")

    print("\nAfter correction:")
    print(f"  BetaX - Max: {beta_after['betax_error_percent'].abs().max():.2f}%, "
          f"RMS: {(beta_after['betax_error_percent']**2).mean()**0.5:.2f}%")
    print(f"  BetaY - Max: {beta_after['betay_error_percent'].abs().max():.2f}%, "
          f"RMS: {(beta_after['betay_error_percent']**2).mean()**0.5:.2f}%")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_beta_beating.py <tmp_dir>")
        print("\nExample:")
        print("  python plot_beta_beating.py /tmp/pytest-of-user/pytest-current/test_controller_bend_opt_simple0/")
        sys.exit(1)

    plot_beta_beating(sys.argv[1])
