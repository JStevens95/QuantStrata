# examples/pricers/fx/01_price_single_vanilla_bsm.py
from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.core.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer


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

    trade = EuropeanFxVanillaOption(
        option_type="call",
        notional=1_000_000.0,
        strike=s,      # ATM
        expiry=1.0,    # 1y
        spot_id=ids.spot,
        vol_id=ids.vol,
        domestic_curve_id=ids.rd,
        foreign_curve_id=ids.rf,
    )

    pricer = FxEuropeanVanillaBsmPricer()

    pv = float(pricer.price(trade, market))
    greeks = {}
    if hasattr(pricer, "greeks"):
        greeks = dict(pricer.greeks(trade, market))  # type: ignore[misc]

    print("=== FX European Vanilla (BSM Adapter) ===")
    print(f"asof   : {market.asof}")
    print(f"type   : {trade.option_type}")
    print(f"notional: {trade.notional:,.2f}")
    print(f"S      : {s:.8f}")
    print(f"K      : {trade.strike:.8f}")
    print(f"T      : {trade.expiry:.6f}")
    print(f"PV     : {pv:,.6f}")
    if greeks:
        print("Greeks :")
        for k, v in greeks.items():
            print(f"  {k:>8s}: {float(v): .10f}")


if __name__ == "__main__":
    main()