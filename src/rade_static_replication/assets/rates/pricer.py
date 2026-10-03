"""
Rates pricer — Bachelier swaptions, base PV + vectorised scenario PnL.

Scenario PnL reprices each swaption across the shocked curve matrix; normal vols come
from the scenario cube (trilinear, vectorised over scenarios) when present, otherwise the
base cube vol is held flat across the window.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.rade_static_replication.domain.contracts import RiskFactorData
from src.rade_static_replication.domain.errors import PricingError
from src.rade_static_replication.domain.instruments import ElementaryTrade
from src.rade_static_replication.marketdata.rates.scenarios import RatesScenarioSet
from src.rade_static_replication.marketdata.rates.snapshot import RatesSnapshot
from src.rade_static_replication.pricing.kernels.rates import (
    price_ir_swaption_bachelier,
    price_ir_swaption_bachelier_vec,
)

# Used only when no swaption cube is supplied (≈ 80bp normal vol).
_FALLBACK_NORMAL_VOL = 0.008


def _bracket(axis: np.ndarray, value: float) -> Tuple[int, int, float]:
    """Return ``(lower_idx, upper_idx, weight)`` to linearly interpolate ``value`` on ``axis``."""
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        return axis.size - 1, axis.size - 1, 0.0
    upper = int(np.searchsorted(axis, value))
    lower = upper - 1
    weight = (value - axis[lower]) / (axis[upper] - axis[lower])
    return lower, upper, weight


def _interp_cube_vol_path(
    vol_cube_path: np.ndarray, expiries, swap_tenors, strikes, expiry, swap_tenor, strike,
) -> np.ndarray:
    """Trilinear vol at one (expiry, tenor, strike) for every scenario -> ``(n_scenarios,)``.

    ``vol_cube_path`` is ``(n_scenarios, n_exp, n_ten, n_k)``; we collapse the last three
    axes with the bracketing weights, keeping the scenario axis vectorised.
    """
    e_lo, e_hi, e_w = _bracket(expiries, expiry)
    t_lo, t_hi, t_w = _bracket(swap_tenors, swap_tenor)
    k_lo, k_hi, k_w = _bracket(strikes, strike)
    vol = (1 - e_w) * vol_cube_path[:, e_lo] + e_w * vol_cube_path[:, e_hi]   # -> (n, n_ten, n_k)
    vol = (1 - t_w) * vol[:, t_lo] + t_w * vol[:, t_hi]                       # -> (n, n_k)
    return (1 - k_w) * vol[:, k_lo] + k_w * vol[:, k_hi]                      # -> (n,)


class RatesPricer:
    """Bachelier pricer for the rates (swaption) elementary universe."""

    def base_prices(self, trades: List[ElementaryTrade], rf: RiskFactorData) -> Dict[str, float]:
        """COB present value of every swaption for this factor."""
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        tenors, zero_rates = snapshot.discount_curve.tenors, snapshot.discount_curve.zero_rates
        prices: Dict[str, float] = {}
        for trade in trades:
            params = trade.parameters
            normal_vol = (
                snapshot.vol_cube.vol(params["expiry"], params["swap_tenor"], params["strike"])
                if snapshot.vol_cube is not None else _FALLBACK_NORMAL_VOL
            )
            prices[trade.trade_id] = price_ir_swaption_bachelier(
                tenors, zero_rates, params["expiry"], params["swap_tenor"], params["strike"],
                trade.notional, self._is_payer(trade.payoff_type), normal_vol, params["freq"],
            )
        return prices

    def scenario_pnl(
        self, trades: List[ElementaryTrade], rf: RiskFactorData, base: Dict[str, float],
    ) -> np.ndarray:
        """Scenario PnL matrix ``(n_trades, n_scenarios)`` = PV(scenario) - PV(base)."""
        snapshot: RatesSnapshot = rf.snapshot  # type: ignore[assignment]
        scenarios: RatesScenarioSet = rf.scenarios  # type: ignore[assignment]
        n_scenarios = scenarios.n_scenarios
        curve_tenors = scenarios.curve_tenors
        curve_paths = scenarios.curve

        pnl = np.empty((len(trades), n_scenarios), dtype=np.float64)
        for row, trade in enumerate(trades):
            params = trade.parameters
            if scenarios.vol is not None:
                normal_vol_path = _interp_cube_vol_path(
                    scenarios.vol, scenarios.vol_expiries, scenarios.vol_swap_tenors, scenarios.vol_strikes,
                    params["expiry"], params["swap_tenor"], params["strike"],
                )
            else:
                base_vol = (
                    snapshot.vol_cube.vol(params["expiry"], params["swap_tenor"], params["strike"])
                    if snapshot.vol_cube is not None else _FALLBACK_NORMAL_VOL
                )
                normal_vol_path = np.full(n_scenarios, base_vol)

            scenario_pv = price_ir_swaption_bachelier_vec(
                curve_tenors, curve_paths, params["expiry"], params["swap_tenor"], params["strike"],
                trade.notional, self._is_payer(trade.payoff_type), normal_vol_path, params["freq"],
            )
            pnl[row] = scenario_pv - base[trade.trade_id]
        return pnl

    @staticmethod
    def _is_payer(payoff: str) -> bool:
        if payoff == "payer":
            return True
        if payoff == "receiver":
            return False
        raise PricingError(f"unknown rates payoff {payoff!r}")
