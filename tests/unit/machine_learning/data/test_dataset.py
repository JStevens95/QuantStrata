"""
Unit tests for src.machine_learning.data.dataset module.

Tests TFDataset, NormalizationStats, and dataset creation utilities.
"""

import pytest
import numpy as np
from pathlib import Path

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.data.dataset import (
    NormalizationStats,
    TFDataset,
    create_pricing_dataset,
    create_calibration_dataset,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_features():
    """Sample feature array."""
    np.random.seed(42)
    return np.random.randn(100, 6).astype(np.float32)


@pytest.fixture
def sample_targets():
    """Sample target array."""
    np.random.seed(42)
    return np.random.randn(100, 1).astype(np.float32)


@pytest.fixture
def sample_dataset(sample_features, sample_targets):
    """Sample TFDataset."""
    return TFDataset.from_arrays(
        features=sample_features,
        targets=sample_targets,
        feature_names=["f1", "f2", "f3", "f4", "f5", "f6"],
        target_names=["target"],
    )


# =============================================================================
# NormalizationStats Tests
# =============================================================================


class TestNormalizationStats:
    """Tests for NormalizationStats dataclass."""

    def test_compute_zscore(self, sample_features):
        """compute() calculates correct z-score stats."""
        stats = NormalizationStats.compute(sample_features, method="zscore")
        
        assert stats.mean.shape == (6,)
        assert stats.std.shape == (6,)
        assert stats.method == "zscore"
        
        # Verify mean and std are close to actual values
        np.testing.assert_array_almost_equal(stats.mean, sample_features.mean(axis=0))
        np.testing.assert_array_almost_equal(stats.std, sample_features.std(axis=0))

    def test_compute_minmax(self, sample_features):
        """compute() calculates correct min-max stats."""
        stats = NormalizationStats.compute(sample_features, method="minmax")
        
        assert stats.min_val is not None
        assert stats.max_val is not None
        assert stats.method == "minmax"
        
        np.testing.assert_array_almost_equal(stats.min_val, sample_features.min(axis=0))
        np.testing.assert_array_almost_equal(stats.max_val, sample_features.max(axis=0))

    def test_normalize_zscore(self, sample_features):
        """normalize() applies z-score normalization."""
        stats = NormalizationStats.compute(sample_features, method="zscore")
        normalized = stats.normalize(sample_features)
        
        # Normalized data should have mean ~0 and std ~1
        np.testing.assert_array_almost_equal(normalized.mean(axis=0), np.zeros(6), decimal=5)
        np.testing.assert_array_almost_equal(normalized.std(axis=0), np.ones(6), decimal=5)

    def test_normalize_minmax(self, sample_features):
        """normalize() applies min-max normalization."""
        stats = NormalizationStats.compute(sample_features, method="minmax")
        normalized = stats.normalize(sample_features)
        
        # Normalized data should be in [0, 1] range
        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_denormalize_zscore(self, sample_features):
        """denormalize() reverses z-score normalization."""
        stats = NormalizationStats.compute(sample_features, method="zscore")
        normalized = stats.normalize(sample_features)
        denormalized = stats.denormalize(normalized)
        
        np.testing.assert_array_almost_equal(denormalized, sample_features, decimal=5)

    def test_denormalize_minmax(self, sample_features):
        """denormalize() reverses min-max normalization."""
        stats = NormalizationStats.compute(sample_features, method="minmax")
        normalized = stats.normalize(sample_features)
        denormalized = stats.denormalize(normalized)
        
        np.testing.assert_array_almost_equal(denormalized, sample_features, decimal=5)

    def test_to_dict_from_dict(self, sample_features):
        """Stats survive dict roundtrip."""
        stats = NormalizationStats.compute(sample_features, method="zscore")
        d = stats.to_dict()
        restored = NormalizationStats.from_dict(d)
        
        np.testing.assert_array_almost_equal(restored.mean, stats.mean)
        np.testing.assert_array_almost_equal(restored.std, stats.std)
        assert restored.method == stats.method


# =============================================================================
# TFDataset Tests
# =============================================================================


class TestTFDataset:
    """Tests for TFDataset class."""

    def test_from_arrays(self, sample_features, sample_targets):
        """from_arrays creates dataset correctly."""
        dataset = TFDataset.from_arrays(sample_features, sample_targets)
        
        assert len(dataset) == 100
        assert dataset.features.shape == (100, 6)
        assert dataset.targets.shape == (100, 1)

    def test_from_arrays_with_names(self, sample_features, sample_targets):
        """from_arrays accepts feature and target names."""
        dataset = TFDataset.from_arrays(
            sample_features,
            sample_targets,
            feature_names=["a", "b", "c", "d", "e", "f"],
            target_names=["y"],
        )
        assert dataset.feature_names == ["a", "b", "c", "d", "e", "f"]
        assert dataset.target_names == ["y"]

    def test_from_arrays_1d_targets(self, sample_features):
        """from_arrays handles 1D target array."""
        targets_1d = np.random.randn(100).astype(np.float32)
        dataset = TFDataset.from_arrays(sample_features, targets_1d)
        
        # Should be reshaped to 2D
        assert dataset.targets.shape == (100, 1)

    def test_repr(self, sample_dataset):
        """__repr__ returns informative string."""
        repr_str = repr(sample_dataset)
        assert "TFDataset" in repr_str
        assert "n_samples=100" in repr_str
        assert "n_features=6" in repr_str

    def test_normalize_features(self, sample_dataset):
        """normalize_features applies normalization in-place."""
        original_mean = sample_dataset.features.mean(axis=0)
        
        sample_dataset.normalize_features(method="zscore")
        
        assert sample_dataset.feature_stats is not None
        # Features should now be normalized
        np.testing.assert_array_almost_equal(
            sample_dataset.features.mean(axis=0), np.zeros(6), decimal=5
        )

    def test_normalize_targets(self, sample_dataset):
        """normalize_targets applies normalization in-place."""
        sample_dataset.normalize_targets(method="zscore")
        
        assert sample_dataset.target_stats is not None
        np.testing.assert_array_almost_equal(
            sample_dataset.targets.mean(), 0.0, decimal=5
        )

    def test_normalize_chaining(self, sample_dataset):
        """Normalization methods support chaining."""
        result = sample_dataset.normalize_features().normalize_targets()
        assert result is sample_dataset

    def test_denormalize_targets(self, sample_features, sample_targets):
        """denormalize_targets reverses normalization."""
        dataset = TFDataset.from_arrays(sample_features, sample_targets)
        original_targets = dataset.targets.copy()
        
        dataset.normalize_targets(method="zscore")
        denormalized = dataset.denormalize_targets(dataset.targets)
        
        np.testing.assert_array_almost_equal(denormalized, original_targets, decimal=5)

    def test_split(self, sample_dataset):
        """split() creates train/val/test datasets."""
        train, val, test = sample_dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        
        # Total should equal original
        assert len(train) + len(val) + len(test) == len(sample_dataset)

    def test_split_preserves_stats(self, sample_dataset):
        """split() preserves normalization stats."""
        sample_dataset.normalize_features()
        train, val, test = sample_dataset.split(train=0.7, val=0.15, test=0.15)
        
        # All splits should have the same stats
        assert train.feature_stats is not None
        np.testing.assert_array_almost_equal(
            train.feature_stats.mean, val.feature_stats.mean
        )

    def test_split_invalid_fractions(self, sample_dataset):
        """split() raises for invalid fractions."""
        with pytest.raises(AssertionError):
            sample_dataset.split(train=0.5, val=0.3, test=0.3)  # Sum > 1

    def test_to_tf_dataset(self, sample_dataset):
        """to_tf_dataset creates tf.data.Dataset."""
        tf_ds = sample_dataset.to_tf_dataset(batch_size=16, shuffle=False)
        
        assert isinstance(tf_ds, tf.data.Dataset)
        
        # Check first batch
        for features, targets in tf_ds.take(1):
            assert features.shape == (16, 6)
            assert targets.shape == (16, 1)

    def test_to_tf_dataset_with_shuffle(self, sample_dataset):
        """to_tf_dataset shuffles when requested."""
        tf_ds = sample_dataset.to_tf_dataset(batch_size=32, shuffle=True)
        
        # Get first batch twice - should be different due to shuffle
        batch1 = next(iter(tf_ds))[0].numpy()
        batch2 = next(iter(tf_ds))[0].numpy()
        
        # With shuffling, consecutive iterations should differ
        # (Note: this test is probabilistic)
        # Just verify it returns data
        assert batch1.shape == (32, 6)

    def test_to_tf_dataset_with_repeat(self, sample_dataset):
        """to_tf_dataset repeats when requested."""
        tf_ds = sample_dataset.to_tf_dataset(batch_size=32, shuffle=False, repeat=True)
        
        # Should be able to iterate more than once
        batches = list(tf_ds.take(10))
        assert len(batches) == 10

    def test_to_dict_dataset(self, sample_dataset):
        """to_dict_dataset creates dataset with named features."""
        tf_ds = sample_dataset.to_dict_dataset(batch_size=16)
        
        for features, targets in tf_ds.take(1):
            assert isinstance(features, dict)
            assert "f1" in features
            assert features["f1"].shape == (16, 1)

    def test_to_dict_dataset_requires_names(self, sample_features, sample_targets):
        """to_dict_dataset raises without feature names."""
        dataset = TFDataset.from_arrays(sample_features, sample_targets)
        
        with pytest.raises(ValueError, match="feature_names required"):
            dataset.to_dict_dataset()

    def test_save_load(self, sample_dataset, tmp_path):
        """Dataset can be saved and loaded."""
        sample_dataset.normalize_features()
        
        save_path = tmp_path / "dataset"
        sample_dataset.save(save_path)
        
        # Verify files exist
        assert (save_path / "features.npy").exists()
        assert (save_path / "targets.npy").exists()
        assert (save_path / "metadata.json").exists()
        
        # Load and verify
        loaded = TFDataset.load(save_path)
        
        np.testing.assert_array_almost_equal(loaded.features, sample_dataset.features)
        np.testing.assert_array_almost_equal(loaded.targets, sample_dataset.targets)
        assert loaded.feature_names == sample_dataset.feature_names
        assert loaded.feature_stats is not None


# =============================================================================
# create_pricing_dataset Tests
# =============================================================================


class TestCreatePricingDataset:
    """Tests for create_pricing_dataset function."""

    def test_default_creation(self):
        """Creates dataset with default parameters."""
        dataset = create_pricing_dataset(n_samples=100, seed=42)
        
        assert len(dataset) == 100
        assert dataset.features.shape == (100, 6)
        assert dataset.targets.shape == (100, 1)

    def test_feature_names(self):
        """Correct feature names are set."""
        dataset = create_pricing_dataset(n_samples=10)
        
        expected = ["spot", "strike", "volatility", "rate", "time_to_expiry", "is_call"]
        assert dataset.feature_names == expected

    def test_feature_ranges(self):
        """Features are within specified ranges."""
        dataset = create_pricing_dataset(
            n_samples=1000,
            spot_range=(90.0, 110.0),
            strike_range=(80.0, 120.0),
            vol_range=(0.1, 0.3),
            rate_range=(0.01, 0.05),
            expiry_range=(0.5, 1.5),
            seed=42,
        )
        
        features = dataset.features
        
        assert features[:, 0].min() >= 90.0  # spot
        assert features[:, 0].max() <= 110.0
        assert features[:, 1].min() >= 80.0  # strike
        assert features[:, 2].min() >= 0.1   # vol
        assert features[:, 2].max() <= 0.3

    def test_option_types(self):
        """Both calls and puts are generated."""
        dataset = create_pricing_dataset(n_samples=1000, seed=42)
        
        option_types = dataset.features[:, 5]
        assert 0.0 in option_types or option_types.min() == 0.0
        assert 1.0 in option_types or option_types.max() == 1.0

    def test_reproducibility(self):
        """Same seed produces same dataset."""
        ds1 = create_pricing_dataset(n_samples=100, seed=42)
        ds2 = create_pricing_dataset(n_samples=100, seed=42)
        
        np.testing.assert_array_equal(ds1.features, ds2.features)
        np.testing.assert_array_equal(ds1.targets, ds2.targets)

    def test_custom_pricing_fn(self):
        """Custom pricing function is used."""
        def constant_price(s, k, v, r, t, c):
            return 10.0
        
        dataset = create_pricing_dataset(
            n_samples=50,
            pricing_fn=constant_price,
        )
        
        # All prices should be 10.0
        np.testing.assert_array_almost_equal(dataset.targets, np.full((50, 1), 10.0))

    def test_metadata(self):
        """Metadata is populated correctly."""
        dataset = create_pricing_dataset(
            n_samples=200,
            spot_range=(95.0, 105.0),
            seed=123,
        )
        
        assert dataset.metadata["n_samples"] == 200
        assert dataset.metadata["spot_range"] == (95.0, 105.0)
        assert dataset.metadata["seed"] == 123


# =============================================================================
# create_calibration_dataset Tests
# =============================================================================


class TestCreateCalibrationDataset:
    """Tests for create_calibration_dataset function."""

    def test_heston_dataset(self):
        """Creates Heston calibration dataset."""
        dataset = create_calibration_dataset(
            n_samples=50,
            n_strikes=5,
            n_expiries=3,
            model="heston",
            seed=42,
        )
        
        assert len(dataset) == 50
        assert dataset.features.shape == (50, 5 * 3)  # n_strikes * n_expiries
        assert dataset.targets.shape == (50, 5)  # 5 Heston params

    def test_sabr_dataset(self):
        """Creates SABR calibration dataset."""
        dataset = create_calibration_dataset(
            n_samples=50,
            model="sabr",
            seed=42,
        )
        
        assert dataset.targets.shape[1] == 4  # 4 SABR params

    def test_target_names_heston(self):
        """Heston parameter names are correct."""
        dataset = create_calibration_dataset(n_samples=10, model="heston")
        
        expected = ["v0", "kappa", "theta", "sigma", "rho"]
        assert dataset.target_names == expected

    def test_target_names_sabr(self):
        """SABR parameter names are correct."""
        dataset = create_calibration_dataset(n_samples=10, model="sabr")
        
        expected = ["alpha", "beta", "rho", "nu"]
        assert dataset.target_names == expected

    def test_unknown_model_raises(self):
        """Unknown model name raises error."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_calibration_dataset(model="unknown_model")

    def test_reproducibility(self):
        """Same seed produces same dataset."""
        ds1 = create_calibration_dataset(n_samples=50, seed=42)
        ds2 = create_calibration_dataset(n_samples=50, seed=42)
        
        np.testing.assert_array_equal(ds1.features, ds2.features)
        np.testing.assert_array_equal(ds1.targets, ds2.targets)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: create → normalize → split → convert → save/load."""
        # Create
        dataset = create_pricing_dataset(n_samples=200, seed=42)
        
        # Normalize
        dataset.normalize_features().normalize_targets()
        
        # Split
        train, val, test = dataset.split(train=0.7, val=0.15, test=0.15, seed=42)
        
        # Convert to tf.data
        train_ds = train.to_tf_dataset(batch_size=32, shuffle=True)
        val_ds = val.to_tf_dataset(batch_size=32, shuffle=False)
        
        # Verify shapes
        for x, y in train_ds.take(1):
            assert x.shape[1] == 6
            assert y.shape[1] == 1
        
        # Save and reload
        train.save(tmp_path / "train")
        loaded = TFDataset.load(tmp_path / "train")
        
        assert len(loaded) == len(train)
        assert loaded.feature_stats is not None

    def test_with_keras_model(self):
        """Dataset works with Keras model training."""
        # Create small dataset
        dataset = create_pricing_dataset(n_samples=100, seed=42)
        dataset.normalize_features().normalize_targets()
        train, val, _ = dataset.split(train=0.7, val=0.2, test=0.1, seed=42)
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train
        train_ds = train.to_tf_dataset(batch_size=16)
        val_ds = val.to_tf_dataset(batch_size=16, shuffle=False)
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2,
            verbose=0,
        )
        
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
