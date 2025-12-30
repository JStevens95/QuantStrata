from __future__ import annotations

import pytest

from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.linear.spot import FxSpot
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import ParallelRateShock, SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry
from src.risk.attribution.runner import AttributionConfig, attribute_portfolio_scenarios


@pytest.fixture(scope="module")
def fx_env():
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(asof="2025-12-29", universe=Universe([spot_id, rd_id, rf_id]))
    )

    pricer = PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())

    return market, pricer, spot_id, rd_id, rf_id


def test_attribute_portfolio_scenarios_includes_base_row_and_empty_contribs(fx_env) -> None:
    market, pricer, spot_id, rd_id, rf_id = fx_env

    portfolio = Portfolio(
        positions=[Position(position_id="SPOT", instrument=FxSpot(spot_id=spot_id), quantity=1.0)]
    )

    pack = ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock("spot_up_1pct", spot_id, 0.01, "relative"),
        }
    )

    report = attribute_portfolio_scenarios(
        portfolio=portfolio,
        base_market=market,
        portfolio_pricer=pricer,
        scenarios=pack,
        config=AttributionConfig(
            base_name="BASE",
            include_gamma_for_spot=True,
            rho_key_by_curve_id={rd_id: "rho_domestic", rf_id: "rho_foreign"},
        ),
    )

    assert len(report.rows) == 2
    assert report.rows[0].scenario == "BASE"
    assert report.rows[0].pnl == pytest.approx(0.0)
    assert report.rows[0].contributions == {}
    assert report.rows[0].predicted_pnl == pytest.approx(0.0)
    assert report.rows[0].residual == pytest.approx(0.0)


def test_attribute_portfolio_scenarios_spot_shock_matches_delta_contribution(fx_env) -> None:
    market, pricer, spot_id, rd_id, rf_id = fx_env

    portfolio = Portfolio(
        positions=[Position(position_id="SPOT", instrument=FxSpot(spot_id=spot_id), quantity=1.0)]
    )

    pack = ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock("spot_up_1pct", spot_id, 0.01, "relative"),
        }
    )

    report = attribute_portfolio_scenarios(
        portfolio=portfolio,
        base_market=market,
        portfolio_pricer=pricer,
        scenarios=pack,
        config=AttributionConfig(
            base_name="BASE",
            include_gamma_for_spot=True,
            rho_key_by_curve_id={rd_id: "rho_domestic", rf_id: "rho_foreign"},
        ),
    )

    row = report.rows[1]
    assert row.scenario == "spot_up_1pct"

    # For pure spot, attribution should be exact (delta explains all).
    assert row.predicted_pnl == pytest.approx(row.pnl, rel=1e-12, abs=1e-10)
    assert row.residual == pytest.approx(0.0, abs=1e-10)

    # Contribution key format: "delta:<MarketId>"
    k = f"delta:{spot_id}"
    assert k in row.contributions
    assert float(row.contributions[k]) == pytest.approx(row.pnl, rel=1e-12, abs=1e-10)


def test_attribute_portfolio_scenarios_rate_shock_uses_configured_rho_key(fx_env) -> None:
    market, pricer, spot_id, rd_id, rf_id = fx_env

    # Build a simple fair forward so base PV ~ 0 (cleaner test).
    S0 = float(market.quote(spot_id))
    T = 1.0
    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))
    K = S0 * df_f / df_d

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="FWD",
                instrument=FxForward(
                    notional=1_000_000.0,
                    strike=K,
                    expiry=T,
                    spot_id=spot_id,
                    domestic_curve_id=rd_id,
                    foreign_curve_id=rf_id,
                ),
                quantity=1.0,
            )
        ]
    )

    pack = ScenarioPack(
        scenarios={
            "rd_up_25bp": ParallelRateShock("rd_up_25bp", rd_id, 0.0025),
        }
    )

    report = attribute_portfolio_scenarios(
        portfolio=portfolio,
        base_market=market,
        portfolio_pricer=pricer,
        scenarios=pack,
        config=AttributionConfig(
            base_name="BASE",
            include_gamma_for_spot=True,
            rho_key_by_curve_id={rd_id: "rho_domestic", rf_id: "rho_foreign"},
        ),
    )

    row = report.rows[1]
    assert row.scenario == "rd_up_25bp"

    # Must attribute via rho_domestic:<curve_id>
    k = f"rho_domestic:{rd_id}"
    assert k in row.contributions

    # Predicted is sum(contributions) by construction, so equality is exact.
    assert row.predicted_pnl == pytest.approx(sum(row.contributions.values()), rel=0.0, abs=0.0)

    # Residual for a rate shift is expected to be small but not necessarily zero
    # because a parallel rate shock changes discount factors nonlinearly.
    # We only assert it is "small" relative to pnl.
    if abs(row.pnl) > 1e-8:
        assert abs(row.residual) / abs(row.pnl) < 5e-3