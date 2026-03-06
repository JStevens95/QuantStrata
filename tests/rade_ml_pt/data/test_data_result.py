"""Unit tests for rade_ml_pt.data.result -- DataBuildResult (PyTorch)."""
import pytest
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.data.result import DataBuildResult


class TestDataBuildResult:
    def test_defaults_are_none(self):
        """Verify default fields are None with empty metadata."""
        r = DataBuildResult()
        assert r.train_ds is None
        assert r.val_ds is None
        assert r.test_ds is None
        assert r.metadata == {}

    def test_accepts_dataloaders(self):
        """Verify DataBuildResult accepts a DataLoader and stores metadata."""
        ds = TensorDataset(torch.randn(10, 3), torch.randn(10, 1))
        dl = DataLoader(ds, batch_size=2)
        r = DataBuildResult(train_ds=dl, metadata={"n_train": 10})
        assert r.train_ds is not None
        assert r.metadata["n_train"] == 10
