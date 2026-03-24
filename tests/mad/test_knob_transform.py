from __future__ import annotations

import numpy as np
import pytest

from aba_optimiser.mad.knob_transform import KnobSpaceTransform


def _sample_transform() -> KnobSpaceTransform:
    return KnobSpaceTransform(
        dknl_knob_to_absolute={"MQ.1L1.B1.dk1": "MQ.1L1.B1.k1", "MB.1L1.B1.dk0": "MB.1L1.B1.k0"},
        absolute_to_dknl_knob={"MQ.1L1.B1.k1": "MQ.1L1.B1.dk1", "MB.1L1.B1.k0": "MB.1L1.B1.dk0"},
        dknl_knob_base_strength={"MQ.1L1.B1.dk1": 0.02, "MB.1L1.B1.dk0": 0.001},
        dknl_knob_length={"MQ.1L1.B1.dk1": 3.0, "MB.1L1.B1.dk0": 14.3},
    )


def test_absolute_to_optimisation_knobs_converts_dknl_targets() -> None:
    transform = _sample_transform()
    values = {
        "MQ.1L1.B1.k1": 0.023,
        "MB.1L1.B1.k0": 0.0017,
        "pt": 1.0e-4,
    }

    converted = transform.absolute_to_optimisation_knobs(values)

    assert converted["MQ.1L1.B1.dk1"] == pytest.approx((0.023 - 0.02) * 3.0)
    assert converted["MB.1L1.B1.dk0"] == pytest.approx((0.0017 - 0.001) * 14.3)
    assert converted["pt"] == pytest.approx(1.0e-4)


def test_optimisation_to_absolute_knobs_converts_back() -> None:
    transform = _sample_transform()
    values = {
        "MQ.1L1.B1.dk1": 0.009,
        "MB.1L1.B1.dk0": 0.01001,
        "pt": 1.25e-4,
    }

    converted = transform.optimisation_to_absolute_knobs(values)

    assert converted["MQ.1L1.B1.k1"] == pytest.approx(0.02 + 0.009 / 3.0)
    assert converted["MB.1L1.B1.k0"] == pytest.approx(0.001 + 0.01001 / 14.3)
    assert converted["pt"] == pytest.approx(1.25e-4)


def test_format_knob_names_for_output_maps_dknl_suffixes() -> None:
    transform = _sample_transform()
    names = ["MQ.1L1.B1.dk1", "MB.1L1.B1.dk0", "pt"]

    formatted = transform.format_knob_names_for_output(names)

    assert formatted == ["MQ.1L1.B1.k1", "MB.1L1.B1.k0", "pt"]


def test_convert_uncertainties_to_absolute_scales_by_length() -> None:
    transform = _sample_transform()
    names = ["MQ.1L1.B1.dk1", "MB.1L1.B1.dk0", "pt"]
    uncertainties = np.array([0.003, 0.0143, 2.0e-6], dtype=np.float64)

    converted = transform.convert_uncertainties_to_absolute(names, uncertainties)

    assert converted[0] == pytest.approx(0.003 / 3.0)
    assert converted[1] == pytest.approx(0.0143 / 14.3)
    assert converted[2] == pytest.approx(2.0e-6)


def test_zero_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="Zero length is invalid"):
        KnobSpaceTransform(
            dknl_knob_to_absolute={"MQ.1L1.B1.dk1": "MQ.1L1.B1.k1"},
            absolute_to_dknl_knob={"MQ.1L1.B1.k1": "MQ.1L1.B1.dk1"},
            dknl_knob_base_strength={"MQ.1L1.B1.dk1": 0.02},
            dknl_knob_length={"MQ.1L1.B1.dk1": 0.0},
        )
