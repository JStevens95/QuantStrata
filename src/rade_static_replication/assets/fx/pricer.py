"""
FX pricer — Garman-Kohlhagen base PVs and vectorised scenario PnL.

Vols are read off the surface by moneyness (``strike / forward``) on the surface's own
strike axis. The scenario path interpolates rates and the vol plane *per expiry* (cached,
since many strikes share an expiry) and prices the whole window in compiled kernels.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.errors import PricingError
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.fx.scenarios import FXScenarioSet
from src.rade_static_replication.marketdata.fx.snapshot import FXSnapshot
from src.rade_static_replication.pricing.kernels.fx import (
    price_fx_digital,
    price_fx_digital_vec,
    price_fx_forward,
    price_fx_forward_vec,
    price_fx_vanilla,
    price_fx_vanilla_vec,
)


def _interp_rate_path(rate_matrix: np.ndarray, tenors: np.ndarray, expiry: float) -> np.ndarray:
    """Interpolate a zero rate at ``expiry`` for every scenario row -> ``(n_scenarios,)``."""
    return np.array([np.interp(expiry, tenors, rate_matrix[i]) for i in range(rate_matrix.shape[0])])


def _vol_plane_at_expiry(vol_cube: np.ndarray, expiries: np.ndarray, expiry: float) -> np.ndarray:
    """Linearly interpolate the (scenario × strike) vol plane at a target ``expiry``."""
    if expiry <= expiries[0]:
        return vol_cube[:, 0, :]
    if expiry >= expiries[-1]:
        return vol_cube[:, -1, :]
    upper = int(np.searchsorted(expiries, expiry))
    lower_exp, upper_exp = expiries[upper - 1], expiries[upper]
    weight = (expiry - lower_exp) / (upper_exp - lower_exp)
    return (1.0 - weight) * vol_cube[:, upper - 1, :] + weight * vol_cube[:, upper, :]


class FXPricer:
    """Garman-Kohlhagen pricer for the FX elementary universe."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        """COB present value of every elementary trade for this factor."""
        snapshot: FXSnapshot = rf.snapshot  # type: ignore[assignment]
        spot = snapshot.spot.value
        prices: Dict[str, float] = {}
        for trade in trades:
            expiry, strike = trade.parameters["expiry"], trade.parameters["strike"]
            rate_dom = snapshot.domestic_curve.zero(expiry)
            rate_for = snapshot.foreign_curve.zero(expiry)
            # Surface is keyed by moneyness = strike / forward.
            vol = snapshot.vol_surface.vol(strike / snapshot.forward(expiry), expiry)
            prices[trade.trade_id] = self._price_scalar(
                trade.payoff_type, spot, strike, expiry, rate_dom, rate_for, vol, trade.notional,
            )
        return prices

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        """Scenario PnL matrix ``(n_trades, n_scenarios)`` = PV(scenario) - PV(base)."""
        scenarios: FXScenarioSet = rf.scenarios  # type: ignore[assignment]
        n_scenarios = scenarios.n_scenarios
        spot_path = scenarios.spot
        strike_axis = scenarios.vol_strikes

        # Market quantities depend only on expiry, so cache them across trades.
        market_by_expiry: Dict[float, dict] = {}

        def _market_at(expiry: float) -> dict:
            key = round(expiry, 8)
            if key not in market_by_expiry:
                rate_dom = _interp_rate_path(scenarios.domestic_rate, scenarios.domestic_tenors, expiry)
                rate_for = _interp_rate_path(scenarios.foreign_rate, scenarios.foreign_tenors, expiry)
                market_by_expiry[key] = {
                    "rate_dom": rate_dom,
                    "rate_for": rate_for,
                    "forward": spot_path * np.exp((rate_dom - rate_for) * expiry),
                    "vol_plane": _vol_plane_at_expiry(scenarios.vol, scenarios.vol_expiries, expiry),
                }
            return market_by_expiry[key]

        pnl = np.empty((len(trades), n_scenarios), dtype=np.float64)
        for row, trade in enumerate(trades):
            expiry, strike = trade.parameters["expiry"], trade.parameters["strike"]
            market = _market_at(expiry)
            moneyness = strike / market["forward"]
            # Per-scenario vol: interpolate along the strike axis at each scenario's moneyness.
            vol_path = np.array([
                np.interp(moneyness[i], strike_axis, market["vol_plane"][i]) for i in range(n_scenarios)
            ])
            scenario_pv = self._price_vector(
                trade.payoff_type, spot_path, strike, expiry,
                market["rate_dom"], market["rate_for"], vol_path, trade.notional,
            )
            pnl[row] = scenario_pv - base[trade.trade_id]
        return pnl

    # ---- payoff routing ----

    @staticmethod
    def _price_scalar(payoff, spot, strike, expiry, rate_dom, rate_for, vol, notional) -> float:
        if payoff == "call":
            return price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, True, notional)
        if payoff == "put":
            return price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, False, notional)
        if payoff == "digital_call":
            return price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, True, notional)
        if payoff == "digital_put":
            return price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, False, notional)
        if payoff == "forward":
            return price_fx_forward(spot, strike, expiry, rate_dom, rate_for, notional, 1.0)
        raise PricingError(f"unknown FX payoff {payoff!r}")

    @staticmethod
    def _price_vector(payoff, spot_path, strike, expiry, rate_dom, rate_for, vol_path, notional) -> np.ndarray:
        if payoff == "call":
            return price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, True, notional)
        if payoff == "put":
            return price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, False, notional)
        if payoff == "digital_call":
            return price_fx_digital_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, True, notional)
        if payoff == "digital_put":
            return price_fx_digital_vec(spot_path, strike, expiry, rate_dom, rate_for, vol_path, False, notional)
        if payoff == "forward":
            return price_fx_forward_vec(spot_path, strike, expiry, rate_dom, rate_for, notional, 1.0)
        raise PricingError(f"unknown FX payoff {payoff!r}")
