from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

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
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([ids.spot, ids.vol, ids.rd, ids.rf]),
        )
    )

    s = float(market.quote(ids.spot))

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="SPOT_1",
                instrument=FxSpot(spot_id=ids.spot, contract_multiplier=1.0),
                quantity=100_000.0,  # linear spot units
            ),
            Position(
                position_id="CALL_ATM_1Y",
                instrument=EuropeanFxVanillaOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=s,
                    expiry=1.0,
                    spot_id=ids.spot,
                    vol_id=ids.vol,
                    domestic_curve_id=ids.rd,
                    foreign_curve_id=ids.rf,
                ),
                quantity=1.0,
            ),
            Position(
                position_id="PUT_OTM_6M",
                instrument=EuropeanFxVanillaOption(
                    option_type="put",
                    notional=500_000.0,
                    strike=0.98 * s,
                    expiry=0.5,
                    spot_id=ids.spot,
                    vol_id=ids.vol,
                    domestic_curve_id=ids.rd,
                    foreign_curve_id=ids.rf,
                ),
                quantity=2.0,
            ),
            Position(
                position_id="CALL_ITM_2Y_SHORT",
                instrument=EuropeanFxVanillaOption(
                    option_type="call",
                    notional=750_000.0,
                    strike=0.95 * s,
                    expiry=2.0,
                    spot_id=ids.spot,
                    vol_id=ids.vol,
                    domestic_curve_id=ids.rd,
                    foreign_curve_id=ids.rf,
                ),
                quantity=-0.5,  # short half
            ),
        ]
    )

    registry = DefaultPricerRegistry().build()
    pricer = PortfolioPricer(pricer_registry=registry)

    result = pricer.price(portfolio, market)

    print("=== FX Portfolio (Base) ===")
    print(f"asof : {market.asof}")
    print(f"spot : {ids.spot} -> {s:.8f}")
    print()
    print("Per-position:")
    for r in result.per_position:
        print(f"  {r.position_id:>18s} | qty={r.quantity: .6f} | pv={r.pv: .6f}")
        if r.greeks:
            g = ", ".join(f"{k}={v: .6e}" for k, v in sorted(r.greeks.items()))
            print(f"    greeks: {g}")

    print()
    print(f"TOTAL PV   : {result.totals.pv:,.6f}")
    if result.totals.greeks:
        print("TOTAL GREEKS:")
        for k, v in sorted(result.totals.greeks.items()):
            print(f"  {k:>8s}: {v: .10f}")


if __name__ == "__main__":
    main()