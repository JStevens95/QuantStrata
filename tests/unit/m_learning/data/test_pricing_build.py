"""Tests for m_learning.data.pricing build module."""

import pytest

from src.m_learning.data.pricing import build_pricing_data, PricingDataResult


class TestBuildPricingData:
    """Tests for build_pricing_data."""

    def test_returns_pricing_data_result(self):
        """build_pricing_data returns PricingDataResult with tf.data.Dataset(s)."""
        result = build_pricing_data(
            n_samples=500,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=32,
            seed=42,
            normalize=True,
        )
        assert isinstance(result, PricingDataResult)
        assert result.train_ds is not None
        assert result.val_ds is not None
        assert result.test_ds is not None
        assert result.feature_stats is not None
        assert result.target_stats is not None
        assert result.metadata["n_samples"] == 500
        assert result.metadata["batch_size"] == 32

    def test_splits_sum_to_one_raises(self):
        """Splits that do not sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            build_pricing_data(
                n_samples=100,
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,  # sums to 0.9
            )

    def test_dataset_elements(self):
        """Train dataset yields (features, targets) batches."""
        result = build_pricing_data(
            n_samples=100,
            batch_size=16,
            seed=0,
        )
        for features, targets in result.train_ds.take(1):
            assert features.shape[0] <= 16
            assert features.shape[1] == 6  # spot, strike, vol, rate, expiry, is_call
            assert targets.shape[0] <= 16
            assert targets.shape[1] == 1
            break
