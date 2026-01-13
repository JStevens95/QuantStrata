from __future__ import annotations

import pytest

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, Universe
from src.marketdata.scenarios.interfaces import ScenarioPack
from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.providers.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry
from src.risk.scenarios.runner import run_portfolio_scenarios


@pytest.fixture(scope="module")
def fx_market_book_and_ids():
    """
    Deterministic synthetic market and a mixed portfolio (FX spot + FX european call).
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(asof="2025-12-29", universe=Universe([spot_id, vol_id, rd_id, rf_id]))
    )

    S0 = float(market.quote(spot_id))

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="SPOT",
                instrument=FxSpot(spot_id=spot_id, contract_multiplier=1.0),
                quantity=1.0,
            ),
            Position(
                position_id="CALL",
                instrument=EuropeanFxVanillaOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=S0,
                    expiry=1.0,
                    spot_id=spot_id,
                    vol_id=vol_id,
                    domestic_curve_id=rd_id,
                    foreign_curve_id=rf_id,
                ),
                quantity=1.0,
            ),
        ]
    )

    pricer = PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())
    return market, portfolio, pricer, spot_id


def test_runner_includes_base_and_zero_pnl(fx_market_book_and_ids) -> None:
    market, portfolio, pricer, spot_id = fx_market_book_and_ids

    pack = ScenarioPack(
        scenarios={
            "up_1pct": SpotShock(name="up_1pct", spot_id=spot_id, bump=0.01, bump_mode="relative"),
        }
    )

    res = run_portfolio_scenarios(portfolio, market, pricer, pack)

    assert res.scenario_names[0] == "BASE"
    assert res.pnl[0] == pytest.approx(0.0)

    # Dicts align with arrays
    assert res.pv_by_scenario["BASE"] == pytest.approx(res.pv[0])
    assert res.pnl_by_scenario["BASE"] == pytest.approx(res.pnl[0])


def test_scenario_pack_ordering_is_preserved(fx_market_book_and_ids) -> None:
    market, portfolio, pricer, spot_id = fx_market_book_and_ids

    # Dict insertion order should be preserved (Python 3.7+).
    pack = ScenarioPack(
        scenarios={
            "up": SpotShock(name="up", spot_id=spot_id, bump=0.01, bump_mode="relative"),
            "down": SpotShock(name="down", spot_id=spot_id, bump=-0.01, bump_mode="relative"),
        }
    )

    res = run_portfolio_scenarios(portfolio, market, pricer, pack)

    assert res.scenario_names == ["BASE", "up", "down"]
    assert len(res.pv) == 3
    assert len(res.pnl) == 3


def test_sequence_of_shocks_path_uses_shock_names(fx_market_book_and_ids) -> None:
    market, portfolio, pricer, spot_id = fx_market_book_and_ids

    shocks = [
        SpotShock(name="shock_up", spot_id=spot_id, bump=0.01, bump_mode="relative"),
        SpotShock(name="shock_down", spot_id=spot_id, bump=-0.01, bump_mode="relative"),
    ]

    res = run_portfolio_scenarios(portfolio, market, pricer, shocks)

    assert res.scenario_names == ["BASE", "shock_up", "shock_down"]
    assert res.pv_by_scenario["shock_up"] == pytest.approx(res.pv[1])
    assert res.pnl_by_scenario["shock_down"] == pytest.approx(res.pnl[2])


def test_directional_pnl_long_spot_and_call(fx_market_book_and_ids) -> None:
    """
    For long spot + long call:
      - spot up => PV up
      - spot down => PV down
    """
    market, portfolio, pricer, spot_id = fx_market_book_and_ids

    pack = ScenarioPack(
        scenarios={
            "up_1pct": SpotShock(name="up_1pct", spot_id=spot_id, bump=0.01, bump_mode="relative"),
            "down_1pct": SpotShock(name="down_1pct", spot_id=spot_id, bump=-0.01, bump_mode="relative"),
        }
    )

    res = run_portfolio_scenarios(portfolio, market, pricer, pack)

    assert res.pnl_by_scenario["up_1pct"] > 0.0
    assert res.pnl_by_scenario["down_1pct"] < 0.0

    # Also check the array positions match the dict values
    idx_up = res.scenario_names.index("up_1pct")
    idx_dn = res.scenario_names.index("down_1pct")
    assert res.pnl[idx_up] == pytest.approx(res.pnl_by_scenario["up_1pct"])
    assert res.pnl[idx_dn] == pytest.approx(res.pnl_by_scenario["down_1pct"])


def test_custom_base_name_is_respected(fx_market_book_and_ids) -> None:
    market, portfolio, pricer, spot_id = fx_market_book_and_ids

    pack = ScenarioPack(
        scenarios={
            "up": SpotShock(name="up", spot_id=spot_id, bump=0.01, bump_mode="relative"),
        }
    )

    res = run_portfolio_scenarios(portfolio, market, pricer, pack, base_name="ASOF")

    assert res.scenario_names[0] == "ASOF"
    assert res.pnl[0] == pytest.approx(0.0)