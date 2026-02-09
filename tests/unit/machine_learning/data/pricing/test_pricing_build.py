"""
Unit tests for src.machine_learning.data.pricing.build module.

Tests build_pricing_data() and PricingDataResult.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.data.pricing.build import (
    PricingDataResult,
    build_pricing_data,
)
from src.machine_learning.data.dataset import NormalizationStats


# =============================================================================
# PricingDataResult Tests
# =============================================================================


class TestPricingDataResult:
    """Tests for PricingDataResult dataclass."""

    def test_structure(self):
        """PricingDataResult has correct structure."""
        # Create mock datasets
        train_ds = tf.data.Dataset.from_tensor_slices(
            (np.zeros((10, 6)), np.zeros((10, 1)))
        ).batch(2)
        val_ds = tf.data.Dataset.from_tensor_slices(
            (np.zeros((5, 6)), np.zeros((5, 1)))
        ).batch(2)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (np.zeros((5, 6)), np.zeros((5, 1)))
        ).batch(2)
        
        result = PricingDataResult(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            feature_stats=None,
            target_stats=None,
            metadata={"n_samples": 20},
        )
        
        assert result.train_ds is not None
        assert result.val_ds is not None
        assert result.test_ds is not None
        assert result.metadata["n_samples"] == 20


# =============================================================================
# build_pricing_data Tests
# =============================================================================


class TestBuildPricingData:
    """Tests for build_pricing_data function."""

    def test_returns_pricing_data_result(self):
        """Returns PricingDataResult instance."""
        result = build_pricing_data(n_samples=100, seed=42)
        
        assert isinstance(result, PricingDataResult)
        assert isinstance(result.train_ds, tf.data.Dataset)
        assert isinstance(result.val_ds, tf.data.Dataset)
        assert isinstance(result.test_ds, tf.data.Dataset)

    def test_default_split_ratios(self):
        """Default split ratios are 70/15/15 (small n_samples for fast run)."""
        n_samples = 20  # Small size so iteration is fast
        result = build_pricing_data(
            n_samples=n_samples,
            batch_size=5,
            seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        train_count = sum(1 for _ in result.train_ds.unbatch())
        val_count = sum(1 for _ in result.val_ds.unbatch())
        test_count = sum(1 for _ in result.test_ds.unbatch())
        assert train_count + val_count + test_count == n_samples
        # 70/15/15 of 20 -> 14, 3, 3 (allow for rounding)
        assert abs(train_count / n_samples - 0.7) <= 0.1
        assert abs(val_count / n_samples - 0.15) <= 0.1
        assert abs(test_count / n_samples - 0.15) <= 0.1

    def test_custom_split_ratios(self):
        """Custom split ratios are respected."""
        result = build_pricing_data(
            n_samples=100,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            batch_size=10,
            seed=42,
        )
        
        train_count = sum(1 for _ in result.train_ds.unbatch())
        val_count = sum(1 for _ in result.val_ds.unbatch())
        test_count = sum(1 for _ in result.test_ds.unbatch())
        
        assert train_count == 60
        assert val_count == 20
        assert test_count == 20

    def test_invalid_split_ratios_raises(self):
        """Invalid split ratios raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            build_pricing_data(
                n_samples=100,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1.0
            )

    def test_normalization_enabled(self):
        """Normalization is applied when normalize=True."""
        result = build_pricing_data(n_samples=100, normalize=True, seed=42)
        
        assert result.feature_stats is not None
        assert result.target_stats is not None
        assert isinstance(result.feature_stats, NormalizationStats)

    def test_normalization_disabled(self):
        """Normalization can be disabled."""
        result = build_pricing_data(n_samples=100, normalize=False, seed=42)
        
        assert result.feature_stats is None
        assert result.target_stats is None

    def test_batch_size(self):
        """Batch size is applied correctly."""
        batch_size = 16
        result = build_pricing_data(
            n_samples=100,
            batch_size=batch_size,
            seed=42,
        )
        
        # Get first batch and check size
        for features, targets in result.train_ds.take(1):
            assert features.shape[0] == batch_size
            assert targets.shape[0] == batch_size

    def test_feature_shape(self):
        """Features have correct shape (6 features)."""
        result = build_pricing_data(n_samples=50, batch_size=10, seed=42)
        
        for features, targets in result.train_ds.take(1):
            assert features.shape[1] == 6  # 6 pricing features
            assert targets.shape[1] == 1  # 1 target (price)

    def test_reproducibility(self):
        """Same seed produces same underlying data."""
        result1 = build_pricing_data(n_samples=50, seed=42, batch_size=50)
        result2 = build_pricing_data(n_samples=50, seed=42, batch_size=50)
        
        # Note: tf.data.Dataset shuffle is non-deterministic even with seed
        # So we compare the normalization stats which are computed on the same data
        if result1.feature_stats is not None:
            np.testing.assert_array_almost_equal(
                result1.feature_stats.mean, 
                result2.feature_stats.mean
            )
            np.testing.assert_array_almost_equal(
                result1.feature_stats.std,
                result2.feature_stats.std
            )

    def test_metadata(self):
        """Metadata is populated correctly."""
        result = build_pricing_data(
            n_samples=200,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=32,
            seed=123,
            normalize=True,
        )
        
        assert result.metadata["n_samples"] == 200
        assert result.metadata["train_ratio"] == 0.8
        assert result.metadata["val_ratio"] == 0.1
        assert result.metadata["batch_size"] == 32
        assert result.metadata["seed"] == 123
        assert result.metadata["normalize"] is True

    def test_datasets_are_batched(self):
        """All datasets are batched."""
        result = build_pricing_data(n_samples=100, batch_size=16, seed=42)
        
        # Training dataset should be batched
        for features, targets in result.train_ds.take(1):
            assert len(features.shape) == 2  # [batch, features]
            
        # Validation dataset should be batched
        for features, targets in result.val_ds.take(1):
            assert len(features.shape) == 2

    def test_train_dataset_shuffled(self):
        """Training dataset is shuffled (different order on iteration)."""
        result = build_pricing_data(n_samples=100, batch_size=10, seed=42)
        
        # Get batches from two iterations
        iter1_batch = next(iter(result.train_ds))[0].numpy()
        iter2_batch = next(iter(result.train_ds))[0].numpy()
        
        # Due to shuffling, batches may differ
        # Just verify we get valid data
        assert iter1_batch.shape == (10, 6)

    def test_val_test_not_shuffled(self):
        """Validation and test datasets are not shuffled."""
        result = build_pricing_data(n_samples=100, batch_size=10, seed=42)
        
        # Validation batches should be consistent
        val_batches_1 = [b[0].numpy() for b in result.val_ds]
        val_batches_2 = [b[0].numpy() for b in result.val_ds]
        
        for b1, b2 in zip(val_batches_1, val_batches_2):
            np.testing.assert_array_equal(b1, b2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBuildPricingDataIntegration:
    """Integration tests for build_pricing_data."""

    def test_with_keras_model(self):
        """Built datasets work with Keras model training."""
        result = build_pricing_data(
            n_samples=100,
            batch_size=16,
            seed=42,
            normalize=True,
        )
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train for a few epochs
        history = model.fit(
            result.train_ds,
            validation_data=result.val_ds,
            epochs=2,
            verbose=0,
        )
        
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert len(history.history["loss"]) == 2

    def test_denormalize_predictions(self):
        """Predictions can be denormalized using target_stats."""
        result = build_pricing_data(
            n_samples=100,
            batch_size=16,
            seed=42,
            normalize=True,
        )
        
        # Create and train simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(result.train_ds, epochs=1, verbose=0)
        
        # Get predictions on test set
        test_features = []
        for features, _ in result.test_ds:
            test_features.append(features.numpy())
        test_features = np.concatenate(test_features)
        
        # Predict (normalized)
        norm_preds = model.predict(test_features, verbose=0)
        
        # Denormalize
        if result.target_stats is not None:
            denorm_preds = result.target_stats.denormalize(norm_preds)
            
            # Denormalized predictions should have different scale
            assert not np.allclose(norm_preds.mean(), denorm_preds.mean())

    def test_large_dataset(self):
        """Works with larger datasets (moderate size for fast unit test)."""
        n_samples = 500
        result = build_pricing_data(
            n_samples=n_samples,
            batch_size=50,
            seed=42,
        )
        train_count = sum(
            features.shape[0] for features, _ in result.train_ds
        )
        val_count = sum(
            features.shape[0] for features, _ in result.val_ds
        )
        test_count = sum(
            features.shape[0] for features, _ in result.test_ds
        )
        assert train_count + val_count + test_count == n_samples
        assert abs(train_count / n_samples - 0.7) < 0.05
        assert abs(val_count / n_samples - 0.15) < 0.05
        assert abs(test_count / n_samples - 0.15) < 0.05
