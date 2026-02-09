"""
Unit tests for src.machine_learning.data.calibration.build module.

Tests build_calibration_dataset() and CalibrationDataResult.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
pytest.importorskip("tensorflow")

from src.machine_learning.data.calibration.build import (
    CalibrationDataResult,
    build_calibration_dataset,
)
from src.machine_learning.data.types import MLDataset


# =============================================================================
# CalibrationDataResult Tests
# =============================================================================


class TestCalibrationDataResult:
    """Tests for CalibrationDataResult dataclass."""

    def test_structure(self):
        """CalibrationDataResult has correct structure and to_ml_dataset()."""
        n = 20
        features = np.random.randn(n, 50).astype(np.float32)
        targets = np.random.randn(n, 5).astype(np.float32)
        result = CalibrationDataResult(
            features=features,
            targets=targets,
            feature_names=["iv_%d" % i for i in range(50)],
            target_names=["kappa", "theta", "sigma", "rho", "v0"],
            metadata={"model": "heston", "n_samples": n},
        )
        assert result.features.shape == (n, 50)
        assert result.targets.shape == (n, 5)
        assert len(result.feature_names) == 50
        assert len(result.target_names) == 5
        ml_ds = result.to_ml_dataset()
        assert isinstance(ml_ds, MLDataset)
        assert ml_ds.features.shape == result.features.shape
        assert ml_ds.targets.shape == result.targets.shape


# =============================================================================
# build_calibration_dataset Tests
# =============================================================================


class TestBuildCalibrationDataset:
    """Tests for build_calibration_dataset()."""

    def test_returns_ml_dataset(self):
        """build_calibration_dataset returns MLDataset."""
        ds = build_calibration_dataset(
            n_samples=100,
            n_strikes=5,
            n_expiries=3,
            model="heston",
            seed=42,
        )
        assert isinstance(ds, MLDataset)
        assert ds.features is not None
        assert ds.targets is not None
        assert ds.features.shape[0] == 100
        assert ds.targets.shape[0] == 100

    def test_feature_target_shapes(self):
        """Feature and target shapes match n_samples and surface/params."""
        n_strikes, n_expiries = 10, 5
        ds = build_calibration_dataset(
            n_samples=50,
            n_strikes=n_strikes,
            n_expiries=n_expiries,
            model="heston",
            seed=123,
        )
        # Features = flattened IV surface: n_samples x (n_strikes * n_expiries)
        assert ds.features.shape == (50, n_strikes * n_expiries)
        # Targets = model params (e.g. Heston has 5)
        assert ds.targets.shape[0] == 50
        assert ds.targets.ndim == 2

    def test_reproducibility_with_seed(self):
        """Same seed produces same dataset."""
        ds1 = build_calibration_dataset(n_samples=30, n_strikes=5, n_expiries=3, model="heston", seed=99)
        ds2 = build_calibration_dataset(n_samples=30, n_strikes=5, n_expiries=3, model="heston", seed=99)
        np.testing.assert_array_almost_equal(ds1.features, ds2.features)
        np.testing.assert_array_almost_equal(ds1.targets, ds2.targets)

    def test_sabr_model(self):
        """build_calibration_dataset accepts model='sabr'."""
        ds = build_calibration_dataset(
            n_samples=40,
            n_strikes=5,
            n_expiries=3,
            model="sabr",
            seed=1,
        )
        assert ds.features.shape[0] == 40
        assert ds.targets.shape[0] == 40
