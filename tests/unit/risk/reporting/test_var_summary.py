"""Unit tests for VaR summary report."""

from __future__ import annotations

import pytest

from src.risk.reporting.var_summary import (
    VarBreakdownRow,
    VarSummaryReport,
    build_var_summary_report,
)
from src.risk.var.config import VarResult


def test_var_summary_report_to_dicts() -> None:
    result = VarResult(
        var=1000.0,
        method="historical",
        confidence=0.99,
        horizon_days=1,
        cvar=1200.0,
        metadata={"n_observations": 252},
    )
    summary = VarSummaryReport(result=result, breakdown=[])
    d = summary.to_dicts()
    assert isinstance(d, dict)
    assert d["method"] == "historical"
    assert d["confidence"] == pytest.approx(0.99)
    assert d["var"] == pytest.approx(1000.0)
    assert d["cvar"] == pytest.approx(1200.0)
    assert d["breakdown"] == []


def test_var_summary_report_to_dicts_with_breakdown() -> None:
    result = VarResult(var=500.0, method="parametric", confidence=0.95, horizon_days=1)
    breakdown = [
        VarBreakdownRow("spot", 300.0),
        VarBreakdownRow("vol", 200.0),
    ]
    summary = VarSummaryReport(result=result, breakdown=breakdown)
    d = summary.to_dicts()
    assert len(d["breakdown"]) == 2
    assert d["breakdown"][0]["factor_name"] == "spot"
    assert d["breakdown"][0]["contribution"] == pytest.approx(300.0)


def test_var_summary_report_to_csv_roundtrip() -> None:
    result = VarResult(var=2000.0, method="historical", confidence=0.99, horizon_days=1, cvar=2200.0)
    summary = build_var_summary_report(result)
    csv_text = summary.to_csv()
    lines = csv_text.splitlines()
    assert lines[0] == "method,confidence,horizon_days,var,cvar"
    assert "historical" in lines[1]
    assert "2000" in lines[1]


def test_var_summary_report_to_console() -> None:
    result = VarResult(var=1500.0, method="mc", confidence=0.99, horizon_days=1)
    summary = build_var_summary_report(result)
    out = summary.to_console()
    assert "VaR Summary" in out
    assert "1500" in out
    assert "mc" in out
