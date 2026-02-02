"""Unit tests for parametric (delta-normal) VaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.marketdata.core.ids import MarketId
from src.risk.sensitivities.result import SensitivityKey, SensitivityRow, SensitivitiesReport
from src.risk.var.config import VarConfig, VarResult
from src.risk.var.parametric import parametric_var


def _make_report(rows: list[tuple[str, str | None, float]]) -> SensitivitiesReport:
    out = []
    for greek, mid_str, value in rows:
        mid = MarketId.parse(mid_str) if mid_str else None
        out.append(
            SensitivityRow(
                key=SensitivityKey(greek=greek, market_id=mid),
                value=value,
                method="analytic",
                bump=None,
                units="",
            )
        )
    return SensitivitiesReport(rows=out)


def test_parametric_var_basic() -> None:
    report = _make_report([
        ("delta", "FX.SPOT.EURUSD", 100.0),
        ("vega", "FX.VOL.EURUSD", 50.0),
    ])
    spot_key = SensitivityKey(greek="delta", market_id=MarketId.parse("FX.SPOT.EURUSD"))
    vol_key = SensitivityKey(greek="vega", market_id=MarketId.parse("FX.VOL.EURUSD"))
    factor_vols = {spot_key: 0.01, vol_key: 0.005}  # 1% daily spot, 0.5% daily vol
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    result = parametric_var(report, factor_vols, config)
    assert isinstance(result, VarResult)
    assert result.method == "parametric"
    assert result.var > 0
    assert result.cvar is not None
    # Delta-normal: VaR = z * sqrt(Gamma' Sigma Gamma); z(0.99) ~ 2.33
    expected_std = np.sqrt(100**2 * 0.01**2 + 50**2 * 0.005**2)
    expected_var = stats.norm.ppf(0.99) * expected_std
    assert abs(result.var - expected_var) < 0.01


def test_parametric_var_wrong_method_raises() -> None:
    report = _make_report([("delta", "FX.SPOT.EURUSD", 1.0)])
    key = SensitivityKey(greek="delta", market_id=MarketId.parse("FX.SPOT.EURUSD"))
    config = VarConfig(confidence=0.99, horizon_days=1, method="historical")
    with pytest.raises(ValueError, match="method must be 'parametric'"):
        parametric_var(report, {key: 0.01}, config)


def test_parametric_var_empty_report_raises() -> None:
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    with pytest.raises(ValueError, match="must not be empty"):
        parametric_var(SensitivitiesReport(rows=[]), {}, config)


def test_parametric_var_no_overlap_raises() -> None:
    report = _make_report([("delta", "FX.SPOT.EURUSD", 1.0)])
    other_key = SensitivityKey(greek="vega", market_id=MarketId.parse("FX.VOL.EURUSD"))
    config = VarConfig(confidence=0.99, horizon_days=1, method="parametric")
    with pytest.raises(ValueError, match="No overlap"):
        parametric_var(report, {other_key: 0.01}, config)
