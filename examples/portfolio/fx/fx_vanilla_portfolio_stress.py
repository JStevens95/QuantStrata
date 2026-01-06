# examples/portfolio/fx/02_fx_vanilla_portfolio_stress_pack.py
from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock

from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry


@dataclass(frozen=True, slots=True)
class FxIds:
    spot: MarketId
    vol: MarketId
    rd: MarketId
    rf: MarketId


def _build_fx_ids() -> FxIds:
    return FxIds(
        spot=MarketId("FX", "SPOT", "EURUSD"),
        vol=MarketId("FX", "VOL", "EURUSD.VOL"),
        rd=MarketId("IR", "CURVE", "USD.OIS"),
        rf=MarketId("IR", "CURVE", "EUR.OIS"),
    )


def main() -> None:
    ids = _build_fx_ids()

    provider = SyntheticProvider(seed=123)
    base_market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([ids.spot, ids.vol, ids.rd, ids.rf]),
        )
    )

    s0 = float(base_market.quote(ids.spot))

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="SPOT_1",
                instrument=FxSpot(spot_id=ids.spot, contract_multiplier=1.0),
                quantity=100_000.0,
            ),
            Position(
                position_id="CALL_ATM_1Y",
                instrument=EuropeanFxVanillaOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=s0,
                    expiry=1.0,
                    spot_id=ids.spot,
                    vol_id=ids.vol,
                    domestic_curve_id=ids.rd,
                    foreign_curve_id=ids.rf,
                ),
                quantity=1.0,
            ),
        ]
    )

    pack = ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock(name="spot_up_1pct", spot_id=ids.spot, bump=+0.01, bump_mode="relative"),
            "spot_dn_1pct": SpotShock(name="spot_dn_1pct", spot_id=ids.spot, bump=-0.01, bump_mode="relative"),
            "vol_up_1pt": FlatVolShock(name="vol_up_1pt", vol_id=ids.vol, vol_bump=+0.01),
            "rd_up_10bp": ParallelRateShock(name="rd_up_10bp", curve_id=ids.rd, rate_shift=+0.001),
        }
    )

    registry = DefaultPricerRegistry().build()
    pricer = PortfolioPricer(pricer_registry=registry)

    base_res = pricer.price(portfolio, base_market)
    shocked_markets = pack.apply_all(base_market)

    print("=== FX Portfolio Stress (ScenarioPack) ===")
    print(f"asof : {base_market.asof}")
    print(f"base spot: {s0:.8f}")
    print(f"BASE PV : {base_res.totals.pv:,.6f}")
    print()

    for name, shocked_market in shocked_markets.items():
        s = float(shocked_market.quote(ids.spot))
        res = pricer.price(portfolio, shocked_market)
        dpv = float(res.totals.pv) - float(base_res.totals.pv)

        print(f"--- {name} ---")
        print(f"spot : {s:.8f} (Δ {s - s0:+.8f})")
        print(f"PV   : {res.totals.pv:,.6f}")
        print(f"ΔPV  : {dpv:+,.6f}")
        print()


if __name__ == "__main__":
    main()