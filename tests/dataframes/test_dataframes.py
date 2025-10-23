"""
Tests for aba_optimiser.dataframes.utils module.
"""

from __future__ import annotations

import pandas as pd
import pytest
import tfs

from aba_optimiser.dataframes.utils import (
    filter_out_marker,
    filter_out_markers,
    select_markers,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": ["BPM1", "MARKER1", "QUAD1", "MARKER2", "BPM2", "DIPOLE1"],
            "s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "beta_x": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "beta_y": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )


@pytest.fixture
def sample_tfs_dataframe(sample_dataframe: pd.DataFrame) -> tfs.TfsDataFrame:
    """Create a sample TFS DataFrame for testing."""
    return tfs.TfsDataFrame(sample_dataframe.set_index("name"))


def get_names(result: pd.DataFrame, use_index: bool) -> list[str]:
    """Helper to get names from result based on index usage."""
    return list(result.index) if use_index else list(result["name"])


class TestFilterOutMarker:
    """Tests for filter_out_marker function."""

    @pytest.mark.parametrize("use_index", [True, False])
    def test_filter_out_single_marker(
        self, sample_dataframe: pd.DataFrame, use_index: bool
    ) -> None:
        """Test filtering out a marker."""
        df = sample_dataframe.set_index("name") if use_index else sample_dataframe
        df = tfs.TfsDataFrame(df) if use_index else df
        result = filter_out_marker(df, "MARKER1")
        expected_names = ["BPM1", "QUAD1", "MARKER2", "BPM2", "DIPOLE1"]
        assert get_names(result, use_index) == expected_names
        assert len(result) == 5

    def test_filter_out_nonexistent_marker(
        self, sample_tfs_dataframe: tfs.TfsDataFrame
    ) -> None:
        """Test filtering out a marker that doesn't exist."""
        result = filter_out_marker(sample_tfs_dataframe, "NONEXISTENT")
        assert len(result) == len(sample_tfs_dataframe)
        assert list(result.index) == list(sample_tfs_dataframe.index)

    def test_filter_out_all_markers(
        self, sample_tfs_dataframe: tfs.TfsDataFrame
    ) -> None:
        """Test filtering out all markers."""
        result = filter_out_marker(
            filter_out_marker(sample_tfs_dataframe, "MARKER1"), "MARKER2"
        )
        expected_names = ["BPM1", "QUAD1", "BPM2", "DIPOLE1"]
        assert list(result.index) == expected_names
        assert len(result) == 4


class TestFilterOutMarkers:
    """Tests for filter_out_markers function."""

    @pytest.mark.parametrize("use_index", [True, False])
    def test_filter_out_multiple_markers(
        self, sample_dataframe: pd.DataFrame, use_index: bool
    ) -> None:
        """Test filtering out multiple markers."""
        df = sample_dataframe.set_index("name") if use_index else sample_dataframe
        df = tfs.TfsDataFrame(df) if use_index else df
        result = filter_out_markers(df, ["MARKER1", "MARKER2"])
        expected_names = ["BPM1", "QUAD1", "BPM2", "DIPOLE1"]
        assert get_names(result, use_index) == expected_names
        assert len(result) == 4

    @pytest.mark.parametrize("markers", [[], ["NONEXISTENT1", "NONEXISTENT2"]])
    def test_filter_out_edge_cases(
        self, sample_tfs_dataframe: tfs.TfsDataFrame, markers: list[str]
    ) -> None:
        """Test filtering out with edge cases."""
        result = filter_out_markers(sample_tfs_dataframe, markers)
        assert len(result) == 6  # Original length
        assert list(result.index) == list(sample_tfs_dataframe.index)


class TestSelectMarkers:
    """Tests for select_markers function."""

    @pytest.mark.parametrize("use_index", [True, False])
    @pytest.mark.parametrize(
        "marker_input,expected_names,expected_length",
        [
            ("MARKER1", ["MARKER1"], 1),
            (["MARKER1", "MARKER2"], ["MARKER1", "MARKER2"], 2),
        ],
    )
    def test_select_markers_basic(
        self,
        sample_dataframe: pd.DataFrame,
        use_index: bool,
        marker_input: str | list[str],
        expected_names: list[str],
        expected_length: int,
    ) -> None:
        """Test selecting markers with different inputs."""
        df = sample_dataframe.set_index("name") if use_index else sample_dataframe
        result = select_markers(df, marker_input)
        assert get_names(result, use_index) == expected_names
        assert len(result) == expected_length

    @pytest.mark.parametrize(
        "marker_input,expected_length,expected_names",
        [
            ("NONEXISTENT", 0, []),
            ([], 0, []),
            (["MARKER1", "NONEXISTENT", "MARKER2"], 2, ["MARKER1", "MARKER2"]),
        ],
    )
    def test_select_markers_edge_cases(
        self,
        sample_dataframe: pd.DataFrame,
        marker_input: str | list[str],
        expected_length: int,
        expected_names: list[str],
    ) -> None:
        """Test selecting markers with edge cases."""
        result = select_markers(sample_dataframe, marker_input)
        assert len(result) == expected_length
        if expected_names:
            assert list(result["name"]) == expected_names
