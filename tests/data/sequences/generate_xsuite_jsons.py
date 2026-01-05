#!/usr/bin/env python3
"""
Script to generate xsuite JSON files for seq_b1.
"""

from pathlib import Path

from aba_optimiser.xsuite.xsuite_tools import create_xsuite_environment

# Paths
data_dir = Path(__file__).parent

# For beam 1
seq_b1 = data_dir / "lhcb1.seq"

json_b1 = data_dir / "lhcb1.json"

print(f"Creating xsuite JSON for beam 1: {seq_b1}")
env_b1 = create_xsuite_environment(sequence_file=seq_b1, json_file=json_b1)
print(f"Saved to {json_b1}")

print("Done!")