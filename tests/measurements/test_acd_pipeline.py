from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.measurements.acd_pipeline import (
    ACDOpticsAnalysisConfig,
    build_mixed_closed_orbit_reference,
    long_frame_to_tbt_data,
    make_live_marker_momentum_callback,
    merge_reconstructed_momenta,
    run_driven_and_compensated_optics,
    subtract_closed_orbit,
)


def test_long_frame_to_tbt_data_preserves_name_and_turn_order() -> None:
    frame = pd.DataFrame(
        {
            "name": ["BPM2", "BPM1", "BPM2", "BPM1"],
            "turn": [1, 1, 0, 0],
            "x": [21.0, 11.0, 20.0, 10.0],
            "y": [-21.0, -11.0, -20.0, -10.0],
        }
    )
    result = long_frame_to_tbt_data(frame, source_file=Path("input.sdds"))

    assert result.nturns == 2
    assert result.meta["file"] == "input.sdds"
    assert result.matrices[0].X.index.tolist() == ["BPM2", "BPM1"]
    np.testing.assert_array_equal(result.matrices[0].X, [[20.0, 21.0], [10.0, 11.0]])


def test_mixed_reference_uses_measured_positions_and_model_angles() -> None:
    measured = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["BPM1", "BPM2"])
    fitted = pd.DataFrame(
        {
            "x": [10.0, 20.0],
            "y": [30.0, 40.0],
            "px": [5.0, 6.0],
            "py": [7.0, 8.0],
        },
        index=["BPM1", "BPM2"],
    )

    result = build_mixed_closed_orbit_reference(measured, fitted)

    np.testing.assert_array_equal(result["x"], [1.0, 2.0])
    np.testing.assert_array_equal(result["y"], [3.0, 4.0])
    np.testing.assert_array_equal(result["px"], [5.0, 6.0])
    np.testing.assert_array_equal(result["py"], [7.0, 8.0])


def test_run_driven_optics_rejects_harpy_cleaning(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0],
        }
    )

    with pytest.raises(ValueError, match="Harpy/OMC3 cleaning is disabled"):
        run_driven_and_compensated_optics(
            frame,
            source_file=Path("input.sdds"),
            output_dir=tmp_path,
            config=ACDOpticsAnalysisConfig(
                model_dir=tmp_path / "model",
                harpy_options={"clean": True},
                optics_options={},
            ),
        )


def test_merge_reconstructed_momenta_preserves_measurement_names() -> None:
    current = pd.DataFrame(
        {
            "turn": [4],
            "name": ["acd_before"],
            "x": [1.0],
            "px": [2.0],
            "y": [3.0],
            "py": [4.0],
            "var_px": [5.0],
            "var_py": [6.0],
        }
    ).set_index(["turn", "name"])
    reconstructed = pd.DataFrame(
        {
            "turn": [0],
            "name": ["ACD_BEFORE"],
            "px": [20.0],
            "py": [40.0],
            "var_px": [50.0],
            "var_py": [60.0],
        }
    )

    result = merge_reconstructed_momenta(current, reconstructed)

    assert result.index.tolist() == [(4, "acd_before")]
    assert result.loc[(4, "acd_before"), "px"] == 20.0
    assert result.loc[(4, "acd_before"), "py"] == 40.0


def test_subtract_closed_orbit_centres_only_matching_elements() -> None:
    frame = pd.DataFrame(
        {
            "name": ["bpm1", "acd_after"],
            "x": [3.0, 4.0],
            "px": [5.0, 6.0],
            "y": [7.0, 8.0],
            "py": [9.0, 10.0],
        }
    )
    reference = pd.DataFrame(
        {"x": [1.0], "px": [2.0], "y": [3.0], "py": [4.0]},
        index=["BPM1"],
    )

    result = subtract_closed_orbit(frame, reference)

    np.testing.assert_array_equal(result.loc[0, ["x", "px", "y", "py"]], [2, 3, 4, 5])
    np.testing.assert_array_equal(result.loc[1, ["x", "px", "y", "py"]], [4, 6, 8, 10])


def test_live_marker_callback_refreshes_only_at_requested_interval() -> None:
    current = pd.DataFrame(
        {
            "turn": [4],
            "name": ["acd_before"],
            "x": [1.0],
            "px": [2.0],
            "y": [3.0],
            "py": [4.0],
            "var_px": [5.0],
            "var_py": [6.0],
        }
    ).set_index(["turn", "name"])

    class Generator:
        calls: list[tuple[dict[str, float], float]] = []

        def update(self, *, magnet_strengths: dict[str, float], pt: float) -> pd.DataFrame:
            self.calls.append((magnet_strengths, pt))
            return pd.DataFrame(
                {
                    "turn": [0],
                    "name": ["ACD_BEFORE"],
                    "px": [20.0],
                    "py": [40.0],
                    "var_px": [50.0],
                    "var_py": [60.0],
                }
            )

    generator = Generator()
    worker_manager = SimpleNamespace(build_update_coords=lambda value: value)
    controller = SimpleNamespace(
        data_manager=SimpleNamespace(track_data={0: current}),
        worker_manager=worker_manager,
        optimisation_loop=SimpleNamespace(
            max_epochs=10,
            best_loss=1.0,
            best_knobs={"q1.dk1l": 1e-3},
        ),
    )
    callback = make_live_marker_momentum_callback(
        controller=controller,
        generators={0: generator},
        pts={0: 0.125},
        refresh_every=2,
    )

    assert callback({"q1.dk1l": 1e-3}, {}) is None
    refreshed = callback({"q1.dk1l": 2e-3}, {})

    assert generator.calls == [({"q1.dk1l": 2e-3}, 0.125)]
    assert controller.optimisation_loop.best_loss == float("inf")
    assert controller.optimisation_loop.best_knobs == {"q1.dk1l": 2e-3}
    assert refreshed[0].loc[(4, "acd_before"), "px"] == 20.0
    assert refreshed[0].loc[(4, "acd_before"), "py"] == 40.0


def test_live_marker_callback_skips_recoverable_reconstruction_error() -> None:
    class ReconstructionError(Exception):
        pass

    class Generator:
        def update(self, **_kwargs: object) -> pd.DataFrame:
            raise ReconstructionError

    optimisation_loop = SimpleNamespace(
        max_epochs=10,
        best_loss=1.0,
        best_knobs={"q1.dk1l": 1e-3},
    )
    controller = SimpleNamespace(
        data_manager=SimpleNamespace(track_data={0: pd.DataFrame()}),
        worker_manager=SimpleNamespace(build_update_coords=lambda value: value),
        optimisation_loop=optimisation_loop,
    )
    callback = make_live_marker_momentum_callback(
        controller=controller,
        generators={0: Generator()},
        pts={0: 0.0},
        recoverable_exceptions=(ReconstructionError,),
    )

    assert callback({"q1.dk1l": 2e-3}, {}) is None
    assert optimisation_loop.best_loss == 1.0
    assert optimisation_loop.best_knobs == {"q1.dk1l": 1e-3}
