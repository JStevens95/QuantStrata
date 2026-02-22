"""Unit tests for rade_ml.data.dataset -- build_tf_dataset."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.core.config import DataPipelineConfig
from src.rade_ml.data.dataset import build_tf_dataset


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    return X, y


@pytest.fixture
def config():
    return DataPipelineConfig(batch_size=16, shuffle=False, cache=False)


class TestBuildTfDatasetSimple:
    def test_returns_tf_dataset(self, simple_data, config):
        X, y = simple_data
        ds = build_tf_dataset(X, y, config)
        assert isinstance(ds, tf.data.Dataset)

    def test_correct_batch_size(self, simple_data, config):
        X, y = simple_data
        ds = build_tf_dataset(X, y, config)
        for batch_x, batch_y in ds.take(1):
            assert batch_x.shape[0] == 16

    def test_all_samples_covered(self, simple_data, config):
        X, y = simple_data
        ds = build_tf_dataset(X, y, config)
        total = sum(batch_y.shape[0] for _, batch_y in ds)
        assert total == 100

    def test_shape_mismatch_raises(self, config):
        X = np.random.randn(50, 5)
        y = np.random.randn(60, 1)
        with pytest.raises(ValueError, match="!="):
            build_tf_dataset(X, y, config)


class TestBuildTfDatasetDict:
    def test_dict_variable_inputs(self, config):
        X = {"feat_a": np.random.randn(40, 3).astype(np.float32),
             "feat_b": np.random.randn(40, 2).astype(np.float32)}
        y = np.random.randn(40, 1).astype(np.float32)
        ds = build_tf_dataset(X, y, config)
        for batch_x, batch_y in ds.take(1):
            assert "feat_a" in batch_x
            assert "feat_b" in batch_x


class TestBuildTfDatasetStatic:
    def test_static_inputs_injected(self, config):
        var = {"pnl": np.random.randn(30, 5).astype(np.float32)}
        tgt = np.random.randn(30, 1).astype(np.float32)
        static = {"features": np.random.randn(10, 4).astype(np.float32)}
        ds = build_tf_dataset(var, tgt, config, static_inputs=static)
        for batch_x, batch_y in ds.take(1):
            assert "features" in batch_x
            assert "pnl" in batch_x

    def test_static_shape_preserved(self, config):
        var = np.random.randn(20, 3).astype(np.float32)
        tgt = np.random.randn(20, 1).astype(np.float32)
        static = {"adj": np.random.randn(5, 5).astype(np.float32)}
        ds = build_tf_dataset(var, tgt, config, static_inputs=static)
        for batch_x, batch_y in ds.take(1):
            assert batch_x["adj"].shape == (5, 5)


class TestBuildTfDatasetOptions:
    def test_shuffle_enabled(self, simple_data):
        X, y = simple_data
        cfg = DataPipelineConfig(batch_size=100, shuffle=True)
        ds = build_tf_dataset(X, y, cfg)
        for batch_x, _ in ds.take(1):
            pass

    def test_cache_enabled(self, simple_data):
        X, y = simple_data
        cfg = DataPipelineConfig(batch_size=16, cache=True)
        ds = build_tf_dataset(X, y, cfg)
        assert isinstance(ds, tf.data.Dataset)

    def test_drop_remainder(self, simple_data):
        X, y = simple_data
        cfg = DataPipelineConfig(batch_size=16, drop_remainder=True)
        ds = build_tf_dataset(X, y, cfg)
        total = sum(batch_y.shape[0] for _, batch_y in ds)
        assert total == 96  # 100 // 16 * 16
