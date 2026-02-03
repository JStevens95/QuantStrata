"""Tests for m_learning.data.pricing."""

import numpy as np
import pytest

from src.m_learning.data.pricing import (
    build_pricing_dataset_from_mc,
    build_pricing_dataset_from_analytic,
)
from src.m_learning.data.types import MLDataset


class TestBuildPricingDatasetFromMC:
    """Tests for build_pricing_dataset_from_mc."""

    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = build_pricing_dataset_from_mc(
            n_samples=100,
            n_paths=1000,
            seed=42,
        )

        assert isinstance(dataset, MLDataset)
        assert dataset.features.shape == (100, 6)
        assert dataset.targets.shape == (100,)

    def test_reproducibility(self):
        """Test reproducible generation with seed."""
        ds1 = build_pricing_dataset_from_mc(n_samples=50, n_paths=500, seed=42)
        ds2 = build_pricing_dataset_from_mc(n_samples=50, n_paths=500, seed=42)

        np.testing.assert_array_almost_equal(ds1.features, ds2.features)
        np.testing.assert_array_almost_equal(ds1.targets, ds2.targets)

    def test_custom_ranges(self):
        """Test custom parameter ranges."""
        dataset = build_pricing_dataset_from_mc(
            n_samples=50,
            n_paths=500,
            spot_range=(90, 110),
            strike_range=(95, 105),
            vol_range=(0.15, 0.25),
            seed=42,
        )

        # Check features are within range
        spots = dataset.features[:, 0]
        strikes = dataset.features[:, 1]
        vols = dataset.features[:, 2]

        assert np.all(spots >= 90) and np.all(spots <= 110)
        assert np.all(strikes >= 95) and np.all(strikes <= 105)
        assert np.all(vols >= 0.15) and np.all(vols <= 0.25)

    def test_prices_are_positive(self):
        """Test that generated prices are non-negative."""
        dataset = build_pricing_dataset_from_mc(
            n_samples=100,
            n_paths=5000,
            seed=42,
        )

        assert np.all(dataset.targets >= 0)

    def test_feature_names(self):
        """Test feature names are set."""
        dataset = build_pricing_dataset_from_mc(n_samples=10, n_paths=100)

        assert dataset.feature_names is not None
        assert "spot" in dataset.feature_names
        assert "strike" in dataset.feature_names

    def test_metadata(self):
        """Test metadata is set."""
        dataset = build_pricing_dataset_from_mc(n_samples=10, n_paths=1000)

        assert dataset.metadata["method"] == "mc"
        assert dataset.metadata["n_paths"] == 1000


class TestBuildPricingDatasetFromAnalytic:
    """Tests for build_pricing_dataset_from_analytic."""

    def test_basic_generation(self):
        """Test basic dataset generation with analytic pricer."""
        def dummy_pricer(spot, strike, vol, rate, expiry, option_type):
            # Simple intrinsic value approximation
            if option_type == 1:
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)

        dataset = build_pricing_dataset_from_analytic(
            n_samples=50,
            pricer_fn=dummy_pricer,
            seed=42,
        )

        assert isinstance(dataset, MLDataset)
        assert dataset.features.shape == (50, 6)
        assert dataset.targets.shape == (50,)

    def test_with_bsm_style_pricer(self):
        """Test with a BSM-style pricer function."""
        def bsm_approx(spot, strike, vol, rate, expiry, option_type):
            # Very rough approximation for testing
            d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * expiry) / (vol * np.sqrt(expiry) + 1e-8)
            # Use simplified approximation
            if option_type == 1:
                return max(spot - strike * np.exp(-rate * expiry), 0) * 0.5
            else:
                return max(strike * np.exp(-rate * expiry) - spot, 0) * 0.5

        dataset = build_pricing_dataset_from_analytic(
            n_samples=100,
            pricer_fn=bsm_approx,
            seed=42,
        )

        assert np.all(dataset.targets >= 0)

    def test_metadata(self):
        """Test metadata is set correctly."""
        def dummy_pricer(spot, strike, vol, rate, expiry, option_type):
            return 1.0

        dataset = build_pricing_dataset_from_analytic(
            n_samples=10,
            pricer_fn=dummy_pricer,
        )

        assert dataset.metadata["method"] == "analytic"
