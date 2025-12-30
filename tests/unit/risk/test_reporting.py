from __future__ import annotations

import pytest
import numpy as np

from src.risk.reporting import ScenarioReport


class _DummyResult:
    def __init__(self) -> None:
        self.scenario_names = ["BASE", "up", "down"]
        self.pv = np.array([100.0, 110.0, 90.0], dtype=float)
        self.pnl = np.array([0.0, 10.0, -10.0], dtype=float)


def test_scenario_report_from_result_and_dicts() -> None:
    res = _DummyResult()
    report = ScenarioReport.from_result(res)

    assert len(report.rows) == 3
    assert report.rows[0].scenario == "BASE"
    assert report.rows[1].pv == pytest.approx(110.0)

    dicts = report.to_dicts()
    assert dicts[2]["scenario"] == "down"
    assert dicts[2]["pnl"] == pytest.approx(-10.0)


def test_scenario_report_csv_has_header_and_rows() -> None:
    res = _DummyResult()
    report = ScenarioReport.from_result(res)

    csv_text = report.to_csv()
    lines = csv_text.splitlines()

    assert lines[0] == "scenario,pv,pnl"
    assert "BASE" in lines[1]
    assert "down" in lines[3]


def test_scenario_report_to_console_contains_expected_fields() -> None:
    res = _DummyResult()
    report = ScenarioReport.from_result(res)

    out = report.to_console(pv_decimals=2, pnl_decimals=2)

    assert "Scenario" in out
    assert "PV" in out
    assert "PnL" in out

    # Check values are formatted and present
    assert "100.00" in out
    assert "+10.00" in out
    assert "-10.00" in out


def test_scenario_report_validates_lengths() -> None:
    class _Bad:
        scenario_names = ["BASE", "x"]
        pv = [1.0]
        pnl = [0.0, 0.0]

    with pytest.raises(ValueError):
        ScenarioReport.from_result(_Bad())