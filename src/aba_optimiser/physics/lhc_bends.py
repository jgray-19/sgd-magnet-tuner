"""Utilities for bend magnet processing."""

from __future__ import annotations

import re


def normalise_lhcbend_magnets(true_strengths_dict: dict[str, float]) -> dict[str, float]:
    """Normalize bend magnet keys by removing [ABCD] and averaging."""
    # Normalize bend magnet keys (remove [ABCD] and average)
    pattern = r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.k0"
    normalised = {}
    for key, value in true_strengths_dict.items():
        match = re.match(pattern, key)
        if match:
            new_key = match.group(1) + match.group(3) + ".k0"
            if new_key not in normalised:
                normalised[new_key] = []
            normalised[new_key].append(value)
        else:
            normalised[key] = value

    lb = 1.4300000000000001e01  # Length of the dipole
    # ld = 1.3595053011569682  # Length of the drift in between dipoles

    def k_mean(k_list: list[float]):
        num_k = len(k_list)
        return lb * sum(k_list) / (num_k * lb)  # + (num_k - 1) * ld)

    return {k: k_mean(v) if isinstance(v, list) else v for k, v in normalised.items()}
