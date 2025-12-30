from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.risk.attribution.report import AttributionReport, AttributionRow


def test_attribution_report_to_dicts_round_trips_expected_fields() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    rows = [
        AttributionRow(
            scenario="BASE",
            pv_base=100.0,
            pv_scn=100.0,
            pnl=0.0,
            contributions={},
            predicted_pnl=0.0,
            residual=0.0,
            rel_error=0.0,
        ),
        AttributionRow(
            scenario="spot_up_1pct",
            pv_base=100.0,
            pv_scn=110.0,
            pnl=10.0,
            contributions={f"delta:{spot_id}": 10.0},
            predicted_pnl=10.0,
            residual=0.0,
            rel_error=0.0,
        ),
    ]

    report = AttributionReport(rows=rows, base_scenario_name="BASE")
    dicts = report.to_dicts()

    assert isinstance(dicts, list)
    assert dicts[0]["scenario"] == "BASE"
    assert dicts[0]["contributions"] == {}

    assert dicts[1]["scenario"] == "spot_up_1pct"
    assert dicts[1]["pv_base"] == pytest.approx(100.0)
    assert dicts[1]["pv_scn"] == pytest.approx(110.0)
    assert dicts[1]["pnl"] == pytest.approx(10.0)
    assert dicts[1]["predicted_pnl"] == pytest.approx(10.0)
    assert dicts[1]["residual"] == pytest.approx(0.0)

    contribs = dicts[1]["contributions"]
    assert isinstance(contribs, dict)
    assert f"delta:{spot_id}" in contribs
    assert contribs[f"delta:{spot_id}"] == pytest.approx(10.0)


def test_attribution_report_to_console_contains_headers_and_keys() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    report = AttributionReport(
        rows=[
            AttributionRow(
                scenario="BASE",
                pv_base=100.0,
                pv_scn=100.0,
                pnl=0.0,
                contributions={},
                predicted_pnl=0.0,
                residual=0.0,
                rel_error=0.0,
            ),
            AttributionRow(
                scenario="spot_up_1pct",
                pv_base=100.0,
                pv_scn=110.0,
                pnl=10.0,
                contributions={f"delta:{spot_id}": 10.0},
                predicted_pnl=10.0,
                residual=0.0,
                rel_error=0.0,
            ),
        ],
        base_scenario_name="BASE",
    )

    out = report.to_console(decimals=2)

    # Core headers
    assert "Scenario" in out
    assert "PnL" in out
    assert "Pred" in out
    assert "Residual" in out
    assert "RelErr" in out

    # Contribution column key appears
    assert f"delta:{spot_id}" in out

    # Some formatted values appear
    assert "+10.00" in out


def test_attribution_report_max_rel_error_ignores_base_row() -> None:
    report = AttributionReport(
        rows=[
            AttributionRow(
                scenario="BASE",
                pv_base=100.0,
                pv_scn=100.0,
                pnl=0.0,
                contributions={},
                predicted_pnl=0.0,
                residual=0.0,
                rel_error=999.0,  # should be ignored
            ),
            AttributionRow(
                scenario="x",
                pv_base=100.0,
                pv_scn=101.0,
                pnl=1.0,
                contributions={"k": 0.5},
                predicted_pnl=0.5,
                residual=0.5,
                rel_error=0.25,
            ),
            AttributionRow(
                scenario="y",
                pv_base=100.0,
                pv_scn=98.0,
                pnl=-2.0,
                contributions={"k": -1.5},
                predicted_pnl=-1.5,
                residual=-0.5,
                rel_error=0.10,
            ),
        ],
        base_scenario_name="BASE",
    )

    assert report.max_rel_error() == pytest.approx(0.25)