"""Unit tests for rade_ml_pt.data.dataset -- build_dataloader (PyTorch)."""
import numpy as np
import pytest
import torch

from torch.utils.data import DataLoader

from src.rade_ml_pt.core.config import DataPipelineConfig
from src.rade_ml_pt.data.dataset import build_dataloader


@pytest.fixture
def simple_data():
    """Generate simple random features and targets for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    return X, y


@pytest.fixture
def config():
    """Default data pipeline configuration for testing."""
    return DataPipelineConfig(batch_size=16, shuffle=False, cache=False)


class TestBuildDataloaderSimple:
    def test_returns_dataloader(self, simple_data, config):
        """Verify build_dataloader returns a PyTorch DataLoader."""
        X, y = simple_data
        dl = build_dataloader(X, y, config)
        assert isinstance(dl, DataLoader)

    def test_correct_batch_size(self, simple_data, config):
        """Verify batches have the configured batch size."""
        X, y = simple_data
        dl = build_dataloader(X, y, config)
        batch_x, batch_y = next(iter(dl))
        assert batch_x.shape[0] == 16

    def test_all_samples_covered(self, simple_data, config):
        """Verify all samples are covered across all batches."""
        X, y = simple_data
        dl = build_dataloader(X, y, config)
        total = sum(batch_y.shape[0] for _, batch_y in dl)
        assert total == 100

    def test_shape_mismatch_raises(self, config):
        """Verify ValueError when input and target sample counts differ."""
        X = np.random.randn(50, 5)
        y = np.random.randn(60, 1)
        with pytest.raises(ValueError, match="!="):
            build_dataloader(X, y, config)


class TestBuildDataloaderDict:
    def test_dict_variable_inputs(self, config):
        """Verify dict-mode variable inputs are preserved in batch output."""
        X = {"feat_a": np.random.randn(40, 3).astype(np.float32),
             "feat_b": np.random.randn(40, 2).astype(np.float32)}
        y = np.random.randn(40, 1).astype(np.float32)
        dl = build_dataloader(X, y, config)
        batch_x, batch_y = next(iter(dl))
        assert "feat_a" in batch_x
        assert "feat_b" in batch_x


class TestBuildDataloaderStatic:
    def test_static_inputs_injected(self, config):
        """Verify static inputs are merged into every batch."""
        var = {"pnl": np.random.randn(30, 5).astype(np.float32)}
        tgt = np.random.randn(30, 1).astype(np.float32)
        static = {"features": np.random.randn(10, 4).astype(np.float32)}
        dl = build_dataloader(var, tgt, config, static_inputs=static)
        batch_x, batch_y = next(iter(dl))
        assert "features" in batch_x
        assert "pnl" in batch_x

    def test_static_shape_preserved(self, config):
        """Verify static input tensors preserve their original shape (not batched)."""
        var = np.random.randn(20, 3).astype(np.float32)
        tgt = np.random.randn(20, 1).astype(np.float32)
        static = {"adj": np.random.randn(5, 5).astype(np.float32)}
        dl = build_dataloader(var, tgt, config, static_inputs=static)
        batch_x, batch_y = next(iter(dl))
        assert batch_x["adj"].shape == (5, 5)


class TestBuildDataloaderOptions:
    def test_shuffle_enabled(self, simple_data):
        """Verify shuffle mode runs without error."""
        X, y = simple_data
        cfg = DataPipelineConfig(batch_size=100, shuffle=True)
        dl = build_dataloader(X, y, cfg)
        batch_x, _ = next(iter(dl))
        assert batch_x is not None

    def test_drop_remainder(self, simple_data):
        """Verify drop_last removes the incomplete final batch."""
        X, y = simple_data
        cfg = DataPipelineConfig(batch_size=16, drop_remainder=True)
        dl = build_dataloader(X, y, cfg)
        total = sum(batch_y.shape[0] for _, batch_y in dl)
        # 100 // 16 * 16 = 96
        assert total == 96
