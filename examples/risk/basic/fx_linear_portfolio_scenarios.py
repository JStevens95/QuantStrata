from __future__ import annotations

from src.instruments.fx.linear.forward import FxForward
from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.european import EuropeanFxOption
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.pricers.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry
from src.risk.scenario_analysis import run_portfolio_scenarios


def main() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([spot_id, vol_id, rd_id, rf_id])))

    S0 = float(market.quote(spot_id))
    T = 1.0

    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))
    F0 = S0 * df_f / df_d

    portfolio = Portfolio(
        positions=[
            Position("SPOT", FxSpot(spot_id=spot_id), quantity=1.0),
            Position(
                "FWD",
                FxForward(
                    notional_foreign=1_000_000.0,
                    strike=F0,
                    expiry=T,
                    spot_id=spot_id,
                    domestic_curve_id=rd_id,
                    foreign_curve_id=rf_id,
                ),
                quantity=1.0,
            ),
            Position(
                "CALL",
                EuropeanFxOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=S0,
                    expiry=T,
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

    pack = ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock("spot_up_1pct", spot_id, 0.01, "relative"),
            "spot_down_1pct": SpotShock("spot_down_1pct", spot_id, -0.01, "relative"),
        }
    )

    res = run_portfolio_scenarios(portfolio, market, pricer, pack)

    print("\nFX Portfolio Scenario Analysis")
    print("------------------------------")
    for name, pv, pnl in zip(res.scenario_names, res.pv, res.pnl):
        print(f"{name:>14s} | PV: {pv:>14.6f} | PnL: {pnl:>+14.6f}")
    print()


if __name__ == "__main__":
    main()