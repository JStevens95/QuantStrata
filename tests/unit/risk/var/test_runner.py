"""Unit tests for VaR runner (compute_var facade)."""

from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.risk.sensitivities.result import SensitivityKey, SensitivityRow, SensitivitiesReport
from src.risk.var.config import VarConfig
from src.risk.var.runner import compute_var


def test_compute_var_historical() -> None:
    np.random.seed(1)
    pnl = np.random.randn(100) * 50
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    result = compute_var(config, pnl_series=pnl)
    assert result.method == "historical"
    assert result.var > 0


def test_compute_var_parametric() -> None:
    report = SensitivitiesReport(rows=[
        SensitivityRow(
            key=SensitivityKey(greek="delta", market_id=MarketId.parse("FX.SPOT.EURUSD")),
            value=100.0,
            method="analytic",
            bump=None,
            units="",
        ),
    ])
    key = SensitivityKey(greek="delta", market_id=MarketId.parse("FX.SPOT.EURUSD"))
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    result = compute_var(
        config,
        sensitivities_report=report,
        factor_volatilities={key: 0.01},
    )
    assert result.method == "parametric"
    assert result.var > 0


def test_compute_var_historical_missing_pnl_raises() -> None:
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    with pytest.raises(ValueError, match="pnl_series is required"):
        compute_var(config)


def test_compute_var_parametric_missing_inputs_raises() -> None:
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    with pytest.raises(ValueError, match="sensitivities_report and factor_volatilities"):
        compute_var(config)
