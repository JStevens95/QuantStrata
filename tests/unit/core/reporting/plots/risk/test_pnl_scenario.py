"""Smoke tests for PnL-by-scenario and attribution bar plotters."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from src.core.reporting.plots.risk.pnl_scenario import (
    plot_attribution_bars,
    plot_pnl_by_scenario,
)
from src.risk.attribution.report import AttributionReport, AttributionRow
from src.risk.reporting.scenario_report import ScenarioReport, ScenarioRow


def test_plot_pnl_by_scenario_smoke() -> None:
    """Call plotter with minimal ScenarioReport; no crash."""
    report = ScenarioReport(
        rows=[
            ScenarioRow(scenario="BASE", pv=100.0, pnl=0.0),
            ScenarioRow(scenario="up", pv=110.0, pnl=10.0),
            ScenarioRow(scenario="down", pv=90.0, pnl=-10.0),
        ],
        base_scenario="BASE",
    )
    fig = plot_pnl_by_scenario(report, title="Stress PnL")
    assert fig is not None
    assert len(fig.axes) == 1


def test_plot_attribution_bars_smoke() -> None:
    """Call plotter with minimal AttributionReport; no crash."""
    report = AttributionReport(
        rows=[
            AttributionRow(
                scenario="spot_up",
                pv_base=100.0,
                pv_scn=105.0,
                pnl=5.0,
                contributions={"delta:FX.SPOT.EURUSD": 4.8},
                predicted_pnl=4.8,
                residual=0.2,
                rel_error=0.04,
            ),
        ],
        base_scenario_name="BASE",
    )
    fig = plot_attribution_bars(report, title="Attribution")
    assert fig is not None
    assert len(fig.axes) == 1
