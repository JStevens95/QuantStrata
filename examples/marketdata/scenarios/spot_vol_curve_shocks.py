from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider

from src.marketdata.scenarios.base import ScenarioPack
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock


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
    df_d_1y = float(base_market.curve(ids.rd).df(1.0))
    sigma_atm_1y = float(base_market.vol_surface(ids.vol).vol(expiry=1.0, strike=s0))

    pack = ScenarioPack(
        scenarios={
            "spot_up_1pct": SpotShock(name="spot_up_1pct", spot_id=ids.spot, bump=+0.01, bump_mode="relative"),
            "vol_up_1pt": FlatVolShock(name="vol_up_1pt", vol_id=ids.vol, vol_bump=+0.01),
            "rd_up_10bp": ParallelRateShock(name="rd_up_10bp", curve_id=ids.rd, rate_shift=+0.001),
        }
    )

    shocked = pack.apply_all(base_market)

    print("=== Base Market ===")
    print(f"spot          : {s0:.8f}")
    print(f"df_domestic_1y: {df_d_1y:.8f}")
    print(f"vol(1y, ATM)  : {sigma_atm_1y:.6f}")
    print()

    for name, m in shocked.items():
        s = float(m.quote(ids.spot))
        df_d = float(m.curve(ids.rd).df(1.0))
        sigma = float(m.vol_surface(ids.vol).vol(expiry=1.0, strike=s))

        print(f"=== Shock: {name} ===")
        print(f"spot          : {s:.8f}   (Δ {s - s0:+.8f})")
        print(f"df_domestic_1y: {df_d:.8f} (Δ {df_d - df_d_1y:+.8f})")
        print(f"vol(1y, ATM)  : {sigma:.6f}  (Δ {sigma - sigma_atm_1y:+.6f})")
        print()


if __name__ == "__main__":
    main()