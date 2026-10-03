"""Rates risk-factor builder — assembles a RatesSnapshot + RatesScenarioSet."""
from __future__ import annotations

from typing import Dict

from src.rade_static_replication.assets.rates.instruments import cob_to_date, parse_rates_factor
from src.rade_static_replication.clients.base import MarketDataClient
from src.rade_static_replication.domain.contracts import RiskFactorData, RiskFactorSpec
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.rates.instruments import VolCube
from src.rade_static_replication.marketdata.rates.scenarios import RatesScenarioSet
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot


class RatesRiskFactorBuilder:
    """Turn raw client payloads into the typed rates market objects for one factor.

    The currency is the last token of the factor id (e.g. ``IR_CURVE_SWAP.GBP`` -> GBP).
    We fetch the discount curve, the optional swaption cube, and the scenario shocks.
    """

    def build(
        self, spec: RiskFactorSpec, dependencies: Dict[str, RiskFactorData],
        client: MarketDataClient, cob_date: str,
    ) -> RiskFactorData:
        currency = parse_rates_factor(spec.factor_id)

        # --- COB base market ---
        curve_payload = client.discount_curve(currency, cob_date)
        cube_payload = client.rates_vol_cube(spec.factor_id, cob_date)  # may be None
        snapshot = RatesSnapshot(
            factor_id=spec.factor_id, asset_class="rates", as_of=cob_to_date(cob_date),
            discount_curve=DiscountCurve(currency, curve_payload.tenors, curve_payload.zero_rates),
            vol_cube=(
                None if cube_payload is None else
                VolCube(
                    cube_payload.expiries, cube_payload.swap_tenors,
                    cube_payload.strikes, cube_payload.vols, cube_payload.vol_type,
                )
            ),
        )

        # --- shocked scenario states ---
        shocks = client.rates_shocks(spec.factor_id, cob_date)
        scenarios = RatesScenarioSet(
            factor_id=spec.factor_id, asset_class="rates", scenario_ids=shocks.scenario_ids,
            curve=shocks.curve, curve_tenors=shocks.curve_tenors,
            vol=shocks.vol, vol_expiries=shocks.vol_expiries,
            vol_swap_tenors=shocks.vol_swap_tenors, vol_strikes=shocks.vol_strikes,
        )
        scenarios.validate_against(snapshot)  # fail fast on any grid mismatch
        return RiskFactorData(spec=spec, snapshot=snapshot, scenarios=scenarios, dependencies=dependencies)
