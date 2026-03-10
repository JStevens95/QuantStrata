"""Shared fixtures for hybrid GNN-RNN pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.core.config import TrainingConfig
from src.rade_ml_pt.data.result import DataBuildResult
from src.rade_ml_pt.pipelines.config import PipelineConfig

BATCH, SEQ, ELEM, TARG = 4, 10, 12, 8
NUM_TRADES = ELEM + TARG


class DummyModel(nn.Module):
    """Minimal model that accepts [B, SEQ, ELEM] and outputs [B, TARG]."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(SEQ * ELEM, TARG)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


def make_loaders():
    """Build tiny train/val DataLoaders with shapes matching model expectations."""
    np.random.seed(0)
    X = np.random.randn(BATCH * 2, SEQ, ELEM).astype(np.float32)
    y = np.random.randn(BATCH * 2, TARG).astype(np.float32)

    train_ds = DataLoader(
        TensorDataset(torch.from_numpy(X[:BATCH]), torch.from_numpy(y[:BATCH])),
        batch_size=BATCH,
    )
    val_ds = DataLoader(
        TensorDataset(torch.from_numpy(X[BATCH:]), torch.from_numpy(y[BATCH:])),
        batch_size=BATCH,
    )
    return train_ds, val_ds


@pytest.fixture
def data_result():
    """Minimal DataBuildResult with train/val loaders."""
    train_ds, val_ds = make_loaders()
    return DataBuildResult(train_ds=train_ds, val_ds=val_ds)


@pytest.fixture
def data_result_with_test():
    """DataBuildResult with train/val/test loaders."""
    train_ds, val_ds = make_loaders()
    np.random.seed(1)
    X_test = np.random.randn(BATCH, SEQ, ELEM).astype(np.float32)
    y_test = np.random.randn(BATCH, TARG).astype(np.float32)
    test_ds = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH,
    )
    return DataBuildResult(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds)


@pytest.fixture
def dummy_model():
    """A simple nn.Module for pipeline testing."""
    return DummyModel()


@pytest.fixture
def pipeline_config():
    """Default PipelineConfig for testing (no registry/tracking/artifacts)."""
    return PipelineConfig(
        training_config=TrainingConfig(epochs=2, loss="mse", early_stopping=None, log_dir=None).to_dict(),
        model_config=None,
        metadata={"run_name": "test_run", "generate_training_report": False},
    )


@pytest.fixture
def pipeline_config_with_dirs(tmp_path):
    """PipelineConfig with registry, tracking, and artifacts directories."""
    return PipelineConfig(
        training_config=TrainingConfig(epochs=2, loss="mse", early_stopping=None, log_dir=None).to_dict(),
        model_config=None,
        registry_dir=str(tmp_path / "registry"),
        tracking_dir=str(tmp_path / "tracking"),
        artifacts_dir=str(tmp_path / "artifacts"),
        metadata={"run_name": "test_run", "generate_training_report": False},
    )
