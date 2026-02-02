"""Unit tests for Greeks aggregation."""

from __future__ import annotations

from src.marketdata.core.ids import MarketId
from src.risk.sensitivities.aggregation import GreeksSummary, aggregate_sensitivities
from src.risk.sensitivities.result import SensitivityKey, SensitivityRow, SensitivitiesReport


def _row(greek: str, market_id_str: str | None, value: float) -> SensitivityRow:
    mid = MarketId.parse(market_id_str) if market_id_str else None
    return SensitivityRow(
        key=SensitivityKey(greek=greek, market_id=mid),
        value=value,
        method="analytic",
        bump=None,
        units="",
    )


def test_aggregate_sensitivities_totals_by_greek() -> None:
    report = SensitivitiesReport(rows=[
        _row("delta", "FX.SPOT.EURUSD", 100.0),
        _row("vega", "FX.VOL.EURUSD", 50.0),
        _row("delta", "FX.SPOT.GBPUSD", 30.0),
    ])
    summary = aggregate_sensitivities(report)
    assert isinstance(summary, GreeksSummary)
    assert summary.totals_by_greek["delta"] == 130.0
    assert summary.totals_by_greek["vega"] == 50.0
    assert summary.totals_by_factor["spot"] == 130.0
    assert summary.totals_by_factor["vol"] == 50.0


def test_aggregate_sensitivities_per_market_id() -> None:
    report = SensitivitiesReport(rows=[
        _row("delta", "FX.SPOT.EURUSD", 100.0),
        _row("vega", "FX.VOL.EURUSD", 50.0),
    ])
    summary = aggregate_sensitivities(report, include_per_market_id=True)
    assert len(summary.per_market_id) == 2
    mids = [r[0] for r in summary.per_market_id]
    assert any((m.key() if m else "") == "FX.SPOT.EURUSD" for m in mids if m)
    values = {r[1]: r[2] for r in summary.per_market_id}
    assert values.get("delta") == 100.0
    assert values.get("vega") == 50.0


def test_aggregate_sensitivities_empty_report() -> None:
    report = SensitivitiesReport(rows=[])
    summary = aggregate_sensitivities(report)
    assert summary.totals_by_greek == {}
    assert summary.totals_by_factor == {}
    assert summary.per_market_id == []
