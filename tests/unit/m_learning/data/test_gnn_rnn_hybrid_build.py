"""Tests for m_learning.data.gnn_rnn_hybrid build module."""

import pytest

from src.m_learning.data.gnn_rnn_hybrid import build_gnn_data, GnnDataResult


class TestBuildGnnData:
    """Tests for build_gnn_data (synthetic path)."""

    def test_synthetic_returns_gnn_data_result(self):
        """build_gnn_data with use_synthetic=True returns GnnDataResult."""
        result = build_gnn_data(
            use_synthetic=True,
            n_trades=30,
            n_elementary=20,
            n_targets=5,
            n_samples=60,
            n_timesteps=10,
            train_ratio=0.6,
            val_ratio=0.2,
            projection_ratio=0.2,
            batch_size=8,
            seed=42,
        )
        assert isinstance(result, GnnDataResult)
        assert result.train_ds is not None
        assert result.val_ds is not None
        assert result.proj_ds is not None
        assert result.metadata["use_synthetic"] is True
        assert result.metadata["n_trades"] == 30
        assert result.metadata["n_samples"] == 60

    def test_splits_sum_to_one_raises(self):
        """Splits that do not sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            build_gnn_data(
                use_synthetic=True,
                n_samples=100,
                train_ratio=0.5,
                val_ratio=0.2,
                projection_ratio=0.2,
            )

    def test_dataset_elements_synthetic(self):
        """Synthetic train dataset yields (inputs_dict, targets) batches."""
        result = build_gnn_data(
            use_synthetic=True,
            n_trades=20,
            n_elementary=12,
            n_targets=3,
            n_samples=40,
            n_timesteps=6,
            batch_size=4,
            seed=0,
        )
        for inputs, targets in result.train_ds.take(1):
            assert "trade_features" in inputs
            assert "adjacency_matrix" in inputs
            assert "pnl_history" in inputs
            assert "target_indices" in inputs
            assert targets.shape[0] <= 4
            break
