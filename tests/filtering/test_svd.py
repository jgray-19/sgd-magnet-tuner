"""
Tests for aba_optimiser.filtering.svd module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aba_optimiser.filtering.svd import svd_clean_measurements


class TestSvdCleanMeasurements:
    """Tests for svd_clean_measurements function."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality with simple data."""
        # Create simple test data
        turns = np.arange(10)
        bpms = ["BPM1", "BPM2"]
        data = []
        for turn in turns:
            for bpm in bpms:
                data.append(
                    {
                        "turn": turn,
                        "name": bpm,
                        "x": turn * 0.1,
                        "y": turn * 0.2,
                    }
                )
        meas_df = pd.DataFrame(data)

        result = svd_clean_measurements(meas_df, bpm_list=bpms, rank=2)

        # Check shape and columns
        assert len(result) == len(meas_df)
        assert set(result.columns) == set(meas_df.columns)
        assert result["turn"].equals(meas_df["turn"])
        assert result["name"].equals(meas_df["name"])

    def test_with_nans(self) -> None:
        """Test handling of NaN values."""
        turns = np.arange(10)
        bpms = ["BPM1", "BPM2"]
        data = []
        for turn in turns:
            for bpm in bpms:
                x_val = turn * 0.1 if turn != 5 else np.nan
                y_val = turn * 0.2 if turn != 5 else np.nan
                data.append(
                    {
                        "turn": turn,
                        "name": bpm,
                        "x": x_val,
                        "y": y_val,
                    }
                )
        meas_df = pd.DataFrame(data)

        result = svd_clean_measurements(meas_df, bpm_list=bpms, rank=2)

        # Check that NaN is preserved
        nan_row = result[(result["turn"] == 5)]
        assert pd.isna(nan_row["x"].iloc[0])
        assert pd.isna(nan_row["y"].iloc[0])

    def test_centering_options(self) -> None:
        """Test different centering options."""
        turns = np.arange(10)
        bpms = ["BPM1", "BPM2"]
        data = []
        for turn in turns:
            for bpm in bpms:
                data.append(
                    {
                        "turn": turn,
                        "name": bpm,
                        "x": turn * 0.1 + 1.0,  # add offset
                        "y": turn * 0.2 + 2.0,
                    }
                )
        meas_df = pd.DataFrame(data)

        # Test bpm centering
        result_bpm = svd_clean_measurements(
            meas_df, bpm_list=bpms, center="bpm", rank=1
        )
        # Test global centering
        result_global = svd_clean_measurements(
            meas_df, bpm_list=bpms, center="global", rank=1
        )
        # Test no centering
        result_none = svd_clean_measurements(
            meas_df, bpm_list=bpms, center=None, rank=1
        )

        # All should have same shape
        assert len(result_bpm) == len(result_global) == len(result_none) == len(meas_df)

    def test_rank_options(self) -> None:
        """Test different rank options."""
        rng = np.random.default_rng(42)
        turns = np.arange(20)
        bpms = ["BPM1", "BPM2", "BPM3"]
        data = []
        for turn in turns:
            for bpm in bpms:
                data.append(
                    {
                        "turn": turn,
                        "name": bpm,
                        "x": np.sin(turn * 0.1) + rng.normal(0, 0.1),
                        "y": np.cos(turn * 0.1) + rng.normal(0, 0.1),
                    }
                )
        meas_df = pd.DataFrame(data)

        # Test fixed rank
        result_fixed = svd_clean_measurements(meas_df, bpm_list=bpms, rank=2)
        # Test auto rank
        result_auto = svd_clean_measurements(meas_df, bpm_list=bpms, rank="auto")

        assert len(result_fixed) == len(result_auto) == len(meas_df)

    @pytest.mark.parametrize("rank", [1, 2, "auto"])
    def test_rank_parameter(self, rank) -> None:
        """Test rank parameter variations."""
        turns = np.arange(10)
        bpms = ["BPM1"]
        data = []
        for turn in turns:
            data.append(
                {
                    "turn": turn,
                    "name": "BPM1",
                    "x": turn * 0.1,
                    "y": turn * 0.2,
                }
            )
        meas_df = pd.DataFrame(data)

        result = svd_clean_measurements(meas_df, bpm_list=bpms, rank=rank)
        assert len(result) == len(meas_df)

    def test_noise_reduction(self) -> None:
        """Test that SVD cleaning reduces added noise."""
        rng = np.random.default_rng(42)  # For reproducibility
        turns = np.arange(100)
        bpms = ["BPM1", "BPM2", "BPM3"]

        # Create clean signal: low-rank (rank 2)
        clean_signal = []
        for turn in turns:
            for i, bpm in enumerate(bpms):
                x_clean = np.sin(turn * 0.1) + i * 0.5  # coherent motion
                y_clean = np.cos(turn * 0.1) + i * 0.3
                clean_signal.append((turn, bpm, x_clean, y_clean))

        clean_df = pd.DataFrame(clean_signal, columns=["turn", "name", "x", "y"])

        # Add noise
        noisy_df = clean_df.copy()
        noise_level = 0.5
        noisy_df["x"] += rng.normal(0, noise_level, len(noisy_df))
        noisy_df["y"] += rng.normal(0, noise_level, len(noisy_df))  # Clean with SVD
        cleaned_df = svd_clean_measurements(noisy_df, bpm_list=bpms, rank=2)

        # Calculate RMS error before and after cleaning
        x_error_before = np.sqrt(np.mean((noisy_df["x"] - clean_df["x"]) ** 2))
        x_error_after = np.sqrt(np.mean((cleaned_df["x"] - clean_df["x"]) ** 2))
        y_error_before = np.sqrt(np.mean((noisy_df["y"] - clean_df["y"]) ** 2))
        y_error_after = np.sqrt(np.mean((cleaned_df["y"] - clean_df["y"]) ** 2))

        # Assert that error is reduced
        assert x_error_after < x_error_before
        assert y_error_after < y_error_before

    def test_exact_low_rank_reconstruction(self) -> None:
        """If the data are exactly low-rank, SVD with correct rank should reconstruct it nearly exactly."""
        turns = np.arange(60)
        bpms = [f"BPM{i}" for i in range(5)]

        # Build rank-2 signals for x and y as linear combinations of two time modes and BPM coefficients
        a_t = np.sin(0.2 * turns)  # mode 1 over turns
        b_t = np.cos(0.15 * turns)  # mode 2 over turns
        alpha = np.linspace(0.5, 1.5, len(bpms))  # BPM weights for mode 1
        beta = np.linspace(-0.3, 0.7, len(bpms))  # BPM weights for mode 2

        rows = []
        for t in turns:
            for j, name in enumerate(bpms):
                x_val = a_t[t] * alpha[j] + b_t[t] * beta[j]
                y_val = 0.8 * a_t[t] * (alpha[j] - 0.2) + 1.1 * b_t[t] * (beta[j] + 0.1)
                rows.append({"turn": int(t), "name": name, "x": x_val, "y": y_val})

        df = pd.DataFrame(rows)

        # With rank=2 and no centering, reconstruction should match very closely
        cleaned_rank2 = svd_clean_measurements(df, bpm_list=bpms, center=None, rank=2)

        # RMS error should be near machine precision
        x_err2 = float(np.sqrt(np.mean((cleaned_rank2["x"] - df["x"]) ** 2)))
        y_err2 = float(np.sqrt(np.mean((cleaned_rank2["y"] - df["y"]) ** 2)))
        assert x_err2 < 1e-10
        assert y_err2 < 1e-10

    def test_rank_sensitivity_on_low_rank_data(self) -> None:
        """Using too-small rank should degrade fit relative to correct rank on low-rank data."""
        turns = np.arange(60)
        bpms = [f"BPM{i}" for i in range(6)]

        a_t = np.sin(0.22 * turns)
        b_t = np.cos(0.11 * turns)
        alpha = np.linspace(0.2, 1.2, len(bpms))
        beta = np.linspace(-0.5, 0.5, len(bpms))

        rows = []
        for t in turns:
            for j, name in enumerate(bpms):
                x_val = a_t[t] * alpha[j] + b_t[t] * beta[j]
                y_val = 1.3 * a_t[t] * (alpha[j] + 0.1) + 0.6 * b_t[t] * (beta[j] - 0.2)
                rows.append({"turn": int(t), "name": name, "x": x_val, "y": y_val})

        df = pd.DataFrame(rows)

        cleaned_rank1 = svd_clean_measurements(df, bpm_list=bpms, center=None, rank=1)
        cleaned_rank2 = svd_clean_measurements(df, bpm_list=bpms, center=None, rank=2)

        # Rank-2 should fit better (lower RMS error) than rank-1
        x_err1 = float(np.sqrt(np.mean((cleaned_rank1["x"] - df["x"]) ** 2)))
        x_err2 = float(np.sqrt(np.mean((cleaned_rank2["x"] - df["x"]) ** 2)))
        y_err1 = float(np.sqrt(np.mean((cleaned_rank1["y"] - df["y"]) ** 2)))
        y_err2 = float(np.sqrt(np.mean((cleaned_rank2["y"] - df["y"]) ** 2)))

        assert x_err2 < x_err1
        assert y_err2 < y_err1
