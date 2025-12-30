from __future__ import annotations

from src.marketdata.ids import MarketId
from src.instruments.fx.linear.spot import FxSpot
from src.marketdata.scenarios.shocks import SpotShock
from src.pricers.linear.spot import LinearFxSpotPricer
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider


def main() -> None:
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([spot_id])))

    instrument = FxSpot(spot_id=spot_id, contract_multiplier=2.5, description="EURUSD spot exposure")
    pricer = LinearFxSpotPricer()

    base_spot = float(market.quote(spot_id))
    base_pv = pricer.price(instrument, market)

    shock = SpotShock(name="eurusd_up_1pct", spot_id=spot_id, bump=0.01, bump_mode="relative")
    shocked_market = shock.apply(market)

    shocked_spot = float(shocked_market.quote(spot_id))
    shocked_pv = pricer.price(instrument, shocked_market)

    pnl = shocked_pv - base_pv

    print("\nFX Spot Stress Demo")
    print("-------------------")
    print(f"Spot ID:        {spot_id}")
    print(f"Base spot:      {base_spot:.6f}")
    print(f"Shocked spot:   {shocked_spot:.6f}")
    print(f"Multiplier:     {instrument.contract_multiplier:.4f}")
    print(f"Base PV:        {base_pv:.6f}")
    print(f"Shocked PV:     {shocked_pv:.6f}")
    print(f"PnL:            {pnl:.6f}\n")


if __name__ == "__main__":
    main()