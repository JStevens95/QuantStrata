from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider


@dataclass(frozen=True, slots=True)
class FxIds:
    spot: MarketId
    vol: MarketId
    rd: MarketId
    rf: MarketId


def _build_fx_ids() -> FxIds:
    # Note: your SyntheticProvider uses "EURUSD.VOL" (based on your test output).
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
    df_d_1y = float(market.curve(ids.rd).df(1.0))
    df_f_1y = float(market.curve(ids.rf).df(1.0))
    sigma_atm_1y = float(market.vol_surface(ids.vol).vol(expiry=1.0, strike=s))

    print("=== Synthetic FX Market Snapshot ===")
    print(f"asof          : {market.asof}")
    print(f"spot          : {ids.spot} -> {s:.8f}")
    print(f"df_domestic_1y: {ids.rd} -> {df_d_1y:.8f}")
    print(f"df_foreign_1y : {ids.rf} -> {df_f_1y:.8f}")
    print(f"vol(1y, ATM)  : {ids.vol} -> {sigma_atm_1y:.6f}")


if __name__ == "__main__":
    main()