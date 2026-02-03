"""Tests for m_learning.data.delta_hedging."""

import numpy as np
import pytest

from src.m_learning.data.delta_hedging import (
    DELTA_HEDGE_FEATURE_NAMES,
    HedgingPath,
    build_delta_hedging_dataset,
    generate_gbm_path,
    path_to_feature_target_arrays,
    simulate_hedging_path,
    simulate_hedging_paths,
)


class TestGenerateGbmPath:
    def test_shape_and_bounds(self):
        path = generate_gbm_path(100.0, 0.05, 0.2, 1.0, 10, seed=42)
        assert path.shape == (11,)
        assert path[0] == 100.0

    def test_reproducibility(self):
        p1 = generate_gbm_path(100.0, 0.05, 0.2, 1.0, 5, seed=1)
        p2 = generate_gbm_path(100.0, 0.05, 0.2, 1.0, 5, seed=1)
        np.testing.assert_array_almost_equal(p1, p2)


class TestSimulateHedgingPath:
    def test_path_structure(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 12, seed=42)
        assert isinstance(path, HedgingPath)
        assert len(path.times) == 13
        assert path.spot.shape == path.option_value.shape == path.delta.shape == (13,)
        assert path.spot[0] == 100.0
        assert path.times[-1] == 1.0

    def test_delta_bounds_call(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 20, seed=42)
        assert np.all(path.delta >= 0) and np.all(path.delta <= 1)

    def test_delta_bounds_put(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, -1, 20, seed=42)
        assert np.all(path.delta >= -1) and np.all(path.delta <= 0)


class TestBuildDeltaHedgingDataset:
    def test_shapes(self):
        features, targets = build_delta_hedging_dataset(
            n_paths=3, n_steps=5, seed=42
        )
        # 3 paths * 6 points each = 18 rows
        assert features.shape == (18, 5)
        assert targets.shape == (18,)
        assert list(features[0])  # all finite
        assert np.all(np.isfinite(targets))

    def test_feature_names(self):
        assert len(DELTA_HEDGE_FEATURE_NAMES) == 5
        assert "moneyness" in DELTA_HEDGE_FEATURE_NAMES
        assert "time_to_expiry" in DELTA_HEDGE_FEATURE_NAMES

    def test_path_to_feature_target_arrays(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 4, seed=42)
        features, targets = path_to_feature_target_arrays(
            path, 100.0, 1.0, 0.05, 0.2, 1
        )
        assert features.shape == (5, 5)
        assert targets.shape == (5,)
        np.testing.assert_array_almost_equal(targets, path.delta)
