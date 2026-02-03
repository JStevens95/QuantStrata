"""Unit tests for combined RiskReport."""

from __future__ import annotations

import pytest

from src.risk.reporting.risk_report import RiskReport
from src.risk.reporting.scenario_report import ScenarioReport, ScenarioRow
from src.risk.reporting.var_summary import VarSummaryReport, build_var_summary_report
from src.risk.sensitivities.aggregation import GreeksSummary
from src.risk.var.config import VarResult


def test_risk_report_empty_to_console() -> None:
    report = RiskReport()
    out = report.to_console()
    assert "empty" in out or "RiskReport" in out


def test_risk_report_partial_var_only() -> None:
    var_result = VarResult(var=1000.0, method="historical", confidence=0.99, horizon_days=1)
    var_summary = build_var_summary_report(var_result)
    report = RiskReport(var_summary=var_summary)
    out = report.to_console()
    assert "VaR" in out or "1000" in out
    d = report.to_dicts()
    assert "var" in d
    assert d["var"]["var"] == pytest.approx(1000.0)


def test_risk_report_with_greeks_and_scenario() -> None:
    greeks = GreeksSummary(totals_by_greek={"delta": 50_000.0}, totals_by_factor={"spot": 50_000.0}, per_market_id=[])
    scenario = ScenarioReport(
        rows=[
            ScenarioRow(scenario="BASE", pv=1_000_000.0, pnl=0.0),
            ScenarioRow(scenario="up", pv=1_005_000.0, pnl=5_000.0),
        ],
        base_scenario="BASE",
    )
    report = RiskReport(greeks_summary=greeks, scenario_report=scenario)
    d = report.to_dicts()
    assert "greeks" in d
    assert d["greeks"]["totals_by_greek"]["delta"] == pytest.approx(50_000.0)
    assert "scenarios" in d
    assert len(d["scenarios"]) == 2


def test_risk_report_to_csv_has_sections() -> None:
    var_result = VarResult(var=500.0, method="historical", confidence=0.95, horizon_days=1)
    report = RiskReport(var_summary=build_var_summary_report(var_result))
    csv_text = report.to_csv()
    assert "section=" in csv_text or "method" in csv_text
