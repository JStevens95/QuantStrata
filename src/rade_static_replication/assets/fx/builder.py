"""FX risk-factor builder — assembles an FXSnapshot + FXScenarioSet for one pair."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.fx.instruments import cob_to_date, parse_fx_factor
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot


class FXRiskFactorBuilder:
    """Turn raw client payloads into the typed FX market objects for one factor.

    A factor id like ``FX.SPOT.USD.EUR`` resolves to pair ``EURUSD`` with domestic
    (quote) ccy USD and foreign (base) ccy EUR; we fetch both discount curves, the spot,
    the vol surface, and the scenario shocks, then validate the scenario grids line up.
    """

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        foreign_ccy, domestic_ccy, pair = parse_fx_factor(spec.factor_id)

        # --- COB base market ---
        spot = Spot(pair=pair, value=client.fx_spot(pair, cob_date))
        domestic_payload = client.discount_curve(domestic_ccy, cob_date)
        foreign_payload = client.discount_curve(foreign_ccy, cob_date)
        surface_payload = client.fx_vol_surface(pair, cob_date)

        snapshot = FXSnapshot(
            factor_id=spec.factor_id, asset_class="fx", as_of=cob_to_date(cob_date),
            spot=spot,
            domestic_curve=DiscountCurve(domestic_ccy, domestic_payload.tenors, domestic_payload.zero_rates),
            foreign_curve=DiscountCurve(foreign_ccy, foreign_payload.tenors, foreign_payload.zero_rates),
            vol_surface=VolSurface(
                surface_payload.expiries, surface_payload.strikes, surface_payload.vols,
                surface_payload.strike_convention, surface_payload.vol_type,
            ),
        )

        # --- shocked scenario states ---
        shocks = client.fx_shocks(pair, cob_date)
        scenarios = FXScenarioSet(
            factor_id=spec.factor_id, asset_class="fx", scenario_ids=shocks.scenario_ids,
            spot=shocks.spot, vol=shocks.vol, vol_expiries=shocks.vol_expiries, vol_strikes=shocks.vol_strikes,
            domestic_rate=shocks.domestic_rate, domestic_tenors=shocks.domestic_tenors,
            foreign_rate=shocks.foreign_rate, foreign_tenors=shocks.foreign_tenors,
        )
        scenarios.validate_against(snapshot)  # fail fast on any grid mismatch
        return RiskFactorData(spec=spec, snapshot=snapshot, scenarios=scenarios, dependencies=dependencies)
