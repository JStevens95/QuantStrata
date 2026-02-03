"""Tests for m_learning.data.types."""

import numpy as np
import pytest

from src.m_learning.data.types import MLDataset, PricingFeatures, CalibrationFeatures


class TestMLDataset:
    """Tests for MLDataset."""

    def test_creation(self):
        """Test dataset creation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        dataset = MLDataset(features=X, targets=y)

        assert dataset.features.shape == (100, 5)
        assert dataset.targets.shape == (100,)
        assert len(dataset) == 100

    def test_with_names(self):
        """Test dataset with feature/target names."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        dataset = MLDataset(
            features=X,
            targets=y,
            feature_names=["a", "b", "c"],
            target_names=["price"],
        )

        assert dataset.feature_names == ["a", "b", "c"]
        assert dataset.target_names == ["price"]

    def test_split_basic(self):
        """Test basic train/test split."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)

        dataset = MLDataset(features=X, targets=y)
        train, test = dataset.split(train_ratio=0.8)

        assert len(train) == 80
        assert len(test) == 20

    def test_split_with_seed(self):
        """Test reproducible split with seed."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        dataset = MLDataset(features=X, targets=y)
        train1, _ = dataset.split(seed=42)
        train2, _ = dataset.split(seed=42)

        np.testing.assert_array_equal(train1.features, train2.features)

    def test_split_preserves_names(self):
        """Test that split preserves feature/target names."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        dataset = MLDataset(
            features=X,
            targets=y,
            feature_names=["a", "b", "c"],
        )
        train, test = dataset.split()

        assert train.feature_names == dataset.feature_names
        assert test.feature_names == dataset.feature_names


class TestPricingFeatures:
    """Tests for PricingFeatures."""

    def test_creation(self):
        """Test pricing features creation."""
        n = 10
        features = PricingFeatures(
            spot=np.ones(n) * 100,
            strike=np.ones(n) * 100,
            vol=np.ones(n) * 0.2,
            rate=np.ones(n) * 0.05,
            expiry=np.ones(n) * 1.0,
            option_type=np.ones(n),
        )

        assert len(features.spot) == n

    def test_to_array(self):
        """Test conversion to array."""
        n = 5
        features = PricingFeatures(
            spot=np.ones(n) * 100,
            strike=np.ones(n) * 100,
            vol=np.ones(n) * 0.2,
            rate=np.ones(n) * 0.05,
            expiry=np.ones(n) * 1.0,
            option_type=np.ones(n),
        )

        arr = features.to_array()

        assert arr.shape == (n, 6)

    def test_to_array_with_rate_foreign(self):
        """Test conversion to array with foreign rate."""
        n = 5
        features = PricingFeatures(
            spot=np.ones(n) * 100,
            strike=np.ones(n) * 100,
            vol=np.ones(n) * 0.2,
            rate=np.ones(n) * 0.05,
            expiry=np.ones(n) * 1.0,
            option_type=np.ones(n),
            rate_foreign=np.ones(n) * 0.03,
        )

        arr = features.to_array()

        assert arr.shape == (n, 7)

    def test_feature_names(self):
        """Test feature names."""
        names = PricingFeatures.feature_names()
        assert len(names) == 6
        assert "spot" in names

        names_fx = PricingFeatures.feature_names(include_rate_foreign=True)
        assert len(names_fx) == 7


class TestCalibrationFeatures:
    """Tests for CalibrationFeatures."""

    def test_creation(self):
        """Test calibration features creation."""
        n = 10
        n_quotes = 9
        features = CalibrationFeatures(
            market_quotes=np.random.randn(n, n_quotes),
            strikes=np.array([90, 100, 110]),
            expiries=np.array([0.25, 0.5, 1.0]),
            spot=np.ones(n) * 100,
        )

        assert features.market_quotes.shape == (n, n_quotes)

    def test_to_array(self):
        """Test conversion to array."""
        n = 5
        n_quotes = 9
        features = CalibrationFeatures(
            market_quotes=np.random.randn(n, n_quotes),
            strikes=np.array([90, 100, 110]),
            expiries=np.array([0.25, 0.5, 1.0]),
            spot=np.ones(n) * 100,
        )

        arr = features.to_array()

        # quotes + strikes + expiries + spot
        expected_cols = n_quotes + 3 + 3 + 1
        assert arr.shape == (n, expected_cols)
