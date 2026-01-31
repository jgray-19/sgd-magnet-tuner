"""Utilities for bend magnet processing."""

from __future__ import annotations

import logging
import re


def normalise_lhcbend_magnets(
    true_strengths_dict: dict[str, float], magnet_lengths: dict[str, float]
) -> dict[str, float]:
    """Normalize bend magnet keys by removing [A-G] letters and length-weighted averaging.

    Patterns:
    - sbends: MB.[ABCD]<number><LR><1-8>.B[12].k0 - Combined regardless of sign
    - rbends: MB<RXWAL><...>.[A-G]<number><LR><1-8>...k0 - Only combined if same sign
    """
    patterns = [
        (r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.k0", "sbend"),
        (r"(MB[RXWAL]\w*\.)([A-G]?)([0-9]+[LR][1-8].*)\.k0", "rbend"),
    ]

    normalised = {}
    counts = {"sbend": 0, "rbend": 0}

    for key, value in true_strengths_dict.items():
        matched = False
        for pattern, bend_type in patterns:
            if match := re.match(pattern, key):
                base_key = f"{match.group(1)}{match.group(3)}.k0"

                # For rbends, separate by sign (positive/negative)
                if bend_type == "rbend":
                    sign_suffix = "_p" if value >= 0 else "_n"
                    new_key = base_key.replace(".k0", f"{sign_suffix}.k0")
                else:
                    # sbends are combined regardless of sign
                    new_key = base_key

                if new_key not in normalised:
                    normalised[new_key] = []
                normalised[new_key].append((value, magnet_lengths[key]))
                counts[bend_type] += 1
                matched = True
                break

        if not matched:
            normalised[key] = value

    logging.info(f"Normalised {counts['sbend']} sbends and {counts['rbend']} rbends.")

    def length_weighted_average(k_list: list[tuple[float, float]]) -> float:
        """Compute length-weighted average of k0 values."""
        total_length = sum(length for _, length in k_list)
        weighted_sum = sum(k * length for k, length in k_list)
        return weighted_sum / total_length if total_length != 0 else 0.0

    return {
        key: length_weighted_average(val) if isinstance(val, list) else val
        for key, val in normalised.items()
    }
