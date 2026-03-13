"""Shared fixtures for ensemble pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.core.types import TrainingResult
from src.rade_ml_pt.registry.store import ModelRegistry


CLUSTER_0_TRADES = ["trade_A", "trade_B", "trade_C"]
CLUSTER_1_TRADES = ["trade_D", "trade_E"]
N_TARGETS_0 = len(CLUSTER_0_TRADES)
N_TARGETS_1 = len(CLUSTER_1_TRADES)


class PipelineTestModel(nn.Module):
    """Minimal model that takes (B, features) -> (B, n_targets)."""

    def __init__(self, in_features: int = 4, n_targets: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_features, n_targets)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs.get("features", next(iter(inputs.values())))
        elif isinstance(inputs, (tuple, list)):
            x = inputs[0]
        else:
            x = inputs
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.fc(x[:, :self.fc.in_features])


def _make_test_loader(n_samples: int, n_features: int, n_targets: int):
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, n_targets)
    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False)


@pytest.fixture
def cluster_mapping():
    return {
        "cluster_0": list(CLUSTER_0_TRADES),
        "cluster_1": list(CLUSTER_1_TRADES),
    }


@pytest.fixture
def registry_with_members(tmp_path, cluster_mapping):
    """Register two member models and return (registry, member_versions, config)."""
    registry = ModelRegistry(tmp_path / "registry")
    versions = {}

    for cid, trades in cluster_mapping.items():
        n_tgt = len(trades)
        model = PipelineTestModel(in_features=4, n_targets=n_tgt)
        result = TrainingResult(
            best_val_loss=0.05,
            best_train_loss=0.03,
            final_epoch=5,
            best_epoch=4,
        )
        entry = registry.register(model, result, tags=[f"{cid}_latest"])
        versions[cid] = entry.version

        # Save a test dataset for eval pipeline testing.
        ds_dir = registry.root_dir / entry.version / "datasets"
        ds_dir.mkdir(parents=True, exist_ok=True)
        test_loader = _make_test_loader(8, 4, n_tgt)
        torch.save(test_loader.dataset, str(ds_dir / "test.pt"))

    config = EnsembleConfig(
        cluster_mapping=cluster_mapping,
        aggregation="concat",
        member_configs={
            cid: {"training_config": {"epochs": 2, "loss": "mse"}}
            for cid in cluster_mapping
        },
        registry_dir=str(tmp_path / "registry"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    return registry, versions, config
