"""Tests for measurement utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aba_optimiser.measurements.utils import merge_horizontal_vertical_bpms


class TestMergeHorizontalVerticalBPMs:
    """Tests for merging horizontal and vertical BPM dataframes."""

    def test_merge_all_dual_plane(self) -> None:
        """Test merging when all BPMs are dual-plane."""
        # Create horizontal data
        df_h = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [1, 2, 1, 2],
            "x": [0.001, 0.002, 0.003, 0.004],
            "var_x": [1e-8, 1e-8, 1e-8, 1e-8],
            "px": [1e-5, 2e-5, 3e-5, 4e-5],
            "var_px": [1e-10, 1e-10, 1e-10, 1e-10],
        })

        # Create vertical data
        df_v = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [1, 2, 1, 2],
            "y": [0.005, 0.006, 0.007, 0.008],
            "var_y": [2e-8, 2e-8, 2e-8, 2e-8],
            "py": [5e-5, 6e-5, 7e-5, 8e-5],
            "var_py": [2e-10, 2e-10, 2e-10, 2e-10],
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # Check all measurements are present
        assert len(merged) == 4
        assert list(merged.columns) == ["name", "turn", "x", "var_x", "px", "var_px",
                                         "y", "var_y", "py", "var_py", "kick_plane"]
        assert merged["kick_plane"].iloc[0] == "xy"
        assert not merged["x"].isna().any()
        assert not merged["y"].isna().any()

    def test_merge_horizontal_only(self) -> None:
        """Test merging when some BPMs only have horizontal measurements."""
        df_h = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [1, 2, 1, 2],
            "x": [0.001, 0.002, 0.003, 0.004],
            "var_x": [1e-8, 1e-8, 1e-8, 1e-8],
            "px": [1e-5, 2e-5, 3e-5, 4e-5],
            "var_px": [1e-10, 1e-10, 1e-10, 1e-10],
        })

        # BPM1 has vertical, BPM2 doesn't
        df_v = pd.DataFrame({
            "name": ["BPM1", "BPM1"],
            "turn": [1, 2],
            "y": [0.005, 0.006],
            "var_y": [2e-8, 2e-8],
            "py": [5e-5, 6e-5],
            "var_py": [2e-10, 2e-10],
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # BPM1 should have both planes
        bpm1_data = merged[merged["name"] == "BPM1"]
        assert not bpm1_data["x"].isna().any()
        assert not bpm1_data["y"].isna().any()
        assert (bpm1_data["kick_plane"] == "xy").all()

        # BPM2 should only have horizontal (y is NaN)
        bpm2_data = merged[merged["name"] == "BPM2"]
        assert not bpm2_data["x"].isna().any()
        assert bpm2_data["y"].isna().all()
        assert bpm2_data["var_y"].isna().all()
        assert (bpm2_data["kick_plane"] == "x").all()

    def test_merge_vertical_only(self) -> None:
        """Test merging when some BPMs only have vertical measurements."""
        # BPM1 has horizontal, BPM2 doesn't
        df_h = pd.DataFrame({
            "name": ["BPM1", "BPM1"],
            "turn": [1, 2],
            "x": [0.001, 0.002],
            "var_x": [1e-8, 1e-8],
            "px": [1e-5, 2e-5],
            "var_px": [1e-10, 1e-10],
        })

        df_v = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],
            "turn": [1, 2, 1, 2],
            "y": [0.005, 0.006, 0.007, 0.008],
            "var_y": [2e-8, 2e-8, 2e-8, 2e-8],
            "py": [5e-5, 6e-5, 7e-5, 8e-5],
            "var_py": [2e-10, 2e-10, 2e-10, 2e-10],
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # BPM1 should have both planes
        bpm1_data = merged[merged["name"] == "BPM1"]
        assert not bpm1_data["x"].isna().any()
        assert not bpm1_data["y"].isna().any()
        assert (bpm1_data["kick_plane"] == "xy").all()

        # BPM2 should only have vertical (x is NaN)
        bpm2_data = merged[merged["name"] == "BPM2"]
        assert bpm2_data["x"].isna().all()
        assert not bpm2_data["y"].isna().any()
        assert bpm2_data["var_x"].isna().all()
        assert (bpm2_data["kick_plane"] == "y").all()

    def test_merge_mixed_scenario(self) -> None:
        """Test complex scenario with dual-plane, H-only, and V-only BPMs."""
        df_h = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM2", "BPM2"],  # BPM1 dual, BPM2 H-only
            "turn": [1, 2, 1, 2],
            "x": [0.001, 0.002, 0.003, 0.004],
            "var_x": [1e-8, 1e-8, 1e-8, 1e-8],
            "px": [1e-5, 2e-5, 3e-5, 4e-5],
            "var_px": [1e-10, 1e-10, 1e-10, 1e-10],
        })

        df_v = pd.DataFrame({
            "name": ["BPM1", "BPM1", "BPM3", "BPM3"],  # BPM1 dual, BPM3 V-only
            "turn": [1, 2, 1, 2],
            "y": [0.005, 0.006, 0.009, 0.010],
            "var_y": [2e-8, 2e-8, 2e-8, 2e-8],
            "py": [5e-5, 6e-5, 9e-5, 10e-5],
            "var_py": [2e-10, 2e-10, 2e-10, 2e-10],
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # Check we have all 3 BPMs
        assert set(merged["name"].unique()) == {"BPM1", "BPM2", "BPM3"}

        # BPM1: dual-plane
        bpm1_data = merged[merged["name"] == "BPM1"]
        assert (bpm1_data["kick_plane"] == "xy").all()

        # BPM2: horizontal-only
        bpm2_data = merged[merged["name"] == "BPM2"]
        assert (bpm2_data["kick_plane"] == "x").all()
        assert bpm2_data["y"].isna().all()

        # BPM3: vertical-only
        bpm3_data = merged[merged["name"] == "BPM3"]
        assert (bpm3_data["kick_plane"] == "y").all()
        assert bpm3_data["x"].isna().all()

    def test_merge_preserves_turn_structure(self) -> None:
        """Test that turn structure is preserved correctly."""
        df_h = pd.DataFrame({
            "name": ["BPM1"] * 5,
            "turn": list(range(1, 6)),
            "x": [0.001 * i for i in range(1, 6)],
            "var_x": [1e-8] * 5,
            "px": [1e-5 * i for i in range(1, 6)],
            "var_px": [1e-10] * 5,
        })

        df_v = pd.DataFrame({
            "name": ["BPM1"] * 5,
            "turn": list(range(1, 6)),
            "y": [0.002 * i for i in range(1, 6)],
            "var_y": [2e-8] * 5,
            "py": [2e-5 * i for i in range(1, 6)],
            "var_py": [2e-10] * 5,
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # Check turns are sequential
        assert list(merged["turn"].unique()) == list(range(1, 6))
        assert len(merged) == 5

    def test_merge_dtype_preservation(self) -> None:
        """Test that data types are preserved correctly."""
        df_h = pd.DataFrame({
            "name": ["BPM1", "BPM2"],
            "turn": [1, 1],
            "x": [0.001, 0.002],
            "var_x": [1e-8, 1e-8],
            "px": [1e-5, 2e-5],
            "var_px": [1e-10, 1e-10],
        })

        df_v = pd.DataFrame({
            "name": ["BPM1", "BPM2"],
            "turn": [1, 1],
            "y": [0.005, 0.006],
            "var_y": [2e-8, 2e-8],
            "py": [5e-5, 6e-5],
            "var_py": [2e-10, 2e-10],
        })

        merged = merge_horizontal_vertical_bpms(df_h, df_v)

        # Check name is categorical and turn is int32
        assert merged["name"].dtype.name == "category"
        assert merged["turn"].dtype == np.int32

        # Check float columns are float64
        for col in ["x", "y", "var_x", "var_y", "px", "py", "var_px", "var_py"]:
            assert merged[col].dtype == np.float64
