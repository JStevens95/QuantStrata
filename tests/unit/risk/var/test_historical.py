"""Unit tests for historical VaR."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.var.config import VarConfig, VarResult
from src.risk.var.historical import historical_var


def test_historical_var_basic() -> None:
    np.random.seed(42)
    pnl = np.random.randn(500) * 100  # daily P&L
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    result = historical_var(pnl, config)
    assert isinstance(result, VarResult)
    assert result.method == "historical"
    assert result.confidence == 0.99
    assert result.horizon_days == 1
    assert result.var > 0
    assert result.cvar is not None
    assert result.cvar >= result.var
    assert result.metadata["n_observations"] == 500


def test_historical_var_wrong_method_raises() -> None:
    pnl = np.array([1.0, -2.0, 3.0])
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    with pytest.raises(ValueError, match="method must be 'historical'"):
        historical_var(pnl, config)


def test_historical_var_empty_raises() -> None:
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    with pytest.raises(ValueError, match="must not be empty"):
        historical_var(np.array([]), config)


def test_historical_var_horizon_scaling() -> None:
    np.random.seed(1)
    pnl = np.random.randn(200) * 50
    config_1 = VarConfig(confidence=0.95, horizon_days=1, method="historical")
    config_10 = VarConfig(confidence=0.95, horizon_days=10, method="historical")
    r1 = historical_var(pnl, config_1)
    r10 = historical_var(pnl, config_10)
    assert r10.var > r1.var
    assert abs(r10.var / r1.var - np.sqrt(10)) < 0.5  # approximate sqrt scaling


def test_historical_var_no_cvar() -> None:
    pnl = np.array([1.0, -1.0, 2.0, -2.0, 0.0])
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    result = historical_var(pnl, config, include_cvar=False)
    assert result.cvar is None
