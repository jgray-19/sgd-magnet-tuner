"""Pure utilities for grouping accelerator magnets and their knob values."""

from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)

_GROUPED_QFO_KNOB_RE = re.compile(
    r"^(BR\.QFO)CELL(\d+)(\.(?:dk1l|dx|dy|tilt))$", re.IGNORECASE
)


def expand_psb_grouped_quadrupole_knobs(values: dict[str, float]) -> dict[str, float]:
    """Expand shared PSB QFO-cell values to physical per-element knob names."""
    expanded: dict[str, float] = {}
    for name, value in values.items():
        match = _GROUPED_QFO_KNOB_RE.fullmatch(name)
        if match is None:
            expanded[name] = float(value)
            continue
        prefix, cell, suffix = match.groups()
        expanded[f"{prefix}{cell}1{suffix}"] = float(value)
        expanded[f"{prefix}{cell}2{suffix}"] = float(value)
    return expanded


def normalise_lhcbend_magnets(
    true_strengths_dict: dict[str, float], magnet_lengths: dict[str, float]
) -> dict[str, float]:
    """Normalise LHC bend names and length-weight-average grouped strengths.

    S-bends are grouped regardless of sign. R-bends are grouped separately for
    positive and negative strengths.
    """
    patterns = [
        (r"(MB\.)([ABCD])([0-9]+[LR][1-8]\.B[12])\.dk0l", "sbend"),
        (r"(MB[RXWAL]\w*\.)([A-G]?)([0-9]+[LR][1-8].*)\.dk0l", "rbend"),
    ]

    normalised = {}
    counts = {"sbend": 0, "rbend": 0}

    for key, value in true_strengths_dict.items():
        matched = False
        for pattern, bend_type in patterns:
            if match := re.match(pattern, key):
                base_key = f"{match.group(1)}{match.group(3)}.dk0l"
                if bend_type == "rbend":
                    sign_suffix = "_p" if value >= 0 else "_n"
                    new_key = base_key.replace(".dk0l", f"{sign_suffix}.dk0l")
                else:
                    new_key = base_key

                if new_key not in normalised:
                    normalised[new_key] = []
                normalised[new_key].append((value, magnet_lengths[key]))
                counts[bend_type] += 1
                matched = True
                break

        if not matched:
            normalised[key] = value

    LOGGER.info("Normalised %d sbends and %d rbends.", counts["sbend"], counts["rbend"])

    def length_weighted_average(k_list: list[tuple[float, float]]) -> float:
        total_length = sum(length for _, length in k_list)
        weighted_sum = sum(k * length for k, length in k_list)
        return weighted_sum / total_length if total_length != 0 else 0.0

    return {
        key: length_weighted_average(value) if isinstance(value, list) else value
        for key, value in normalised.items()
    }
