"""Shared fixtures for ensemble tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.rade_ml_pt.ensemble.config import EnsembleConfig


CLUSTER_0_TRADES = ["trade_A", "trade_B", "trade_C"]
CLUSTER_1_TRADES = ["trade_D", "trade_E"]
ALL_TRADES = CLUSTER_0_TRADES + CLUSTER_1_TRADES
N_SCENARIOS = 8
N_TARGETS_0 = len(CLUSTER_0_TRADES)
N_TARGETS_1 = len(CLUSTER_1_TRADES)
N_TOTAL = len(ALL_TRADES)


class SimpleMember(nn.Module):
    """Minimal model: linear from input_dim to n_targets."""

    def __init__(self, input_dim: int = 4, n_targets: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_targets)

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = inputs.get("features", inputs.get("trade_features"))
        else:
            x = inputs
        return self.fc(x)


@pytest.fixture
def cluster_mapping():
    return {
        "cluster_0": list(CLUSTER_0_TRADES),
        "cluster_1": list(CLUSTER_1_TRADES),
    }


@pytest.fixture
def ensemble_config(cluster_mapping):
    return EnsembleConfig(
        member_configs={
            "cluster_0": {
                "training_config": {"epochs": 2, "loss": "mse"},
                "data_config": {"batch_size": 4},
            },
            "cluster_1": {
                "training_config": {"epochs": 2, "loss": "mse"},
                "data_config": {"batch_size": 4},
            },
        },
        cluster_mapping=cluster_mapping,
        aggregation="concat",
        registry_dir=None,
        artifacts_dir=None,
    )


@pytest.fixture
def member_models():
    """Dict of simple member models matching cluster target counts."""
    return {
        "cluster_0": SimpleMember(input_dim=4, n_targets=N_TARGETS_0),
        "cluster_1": SimpleMember(input_dim=4, n_targets=N_TARGETS_1),
    }


@pytest.fixture
def member_inputs():
    """Dict of random inputs for each member."""
    torch.manual_seed(0)
    return {
        "cluster_0": {"features": torch.randn(N_SCENARIOS, 4)},
        "cluster_1": {"features": torch.randn(N_SCENARIOS, 4)},
    }


@pytest.fixture
def member_predictions():
    """Numpy prediction arrays per cluster."""
    np.random.seed(0)
    return {
        "cluster_0": np.random.randn(N_SCENARIOS, N_TARGETS_0).astype(np.float32),
        "cluster_1": np.random.randn(N_SCENARIOS, N_TARGETS_1).astype(np.float32),
    }


@pytest.fixture
def member_targets():
    """Numpy target arrays per cluster."""
    np.random.seed(1)
    return {
        "cluster_0": np.random.randn(N_SCENARIOS, N_TARGETS_0).astype(np.float32),
        "cluster_1": np.random.randn(N_SCENARIOS, N_TARGETS_1).astype(np.float32),
    }
