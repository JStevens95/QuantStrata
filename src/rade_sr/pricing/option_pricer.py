"""
Option pricer — vectorised batch pricing across scenarios via @njit kernels.

This is the computational core for PnL generation. The VectorisedPnLEngine
(in replication/pnl_engine.py) delegates to this pricer for each factor group.

The pricer bridges two worlds:
  - OOP side: Asset objects (FXAsset, IRAsset) with market data
  - Numeric side: @njit kernels that take only float/ndarray

For each factor group it:
  1. Converts typed shocks (FXShocks, IRShocks) to scenario dicts
  2. Groups trades by payoff_type
  3. Extracts numeric params and calls the appropriate @njit batch kernel
  4. Returns (n_trades, n_scenarios) PnL matrix
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..assets.types import FXShocks, IRShocks
from ..core.types import ElementaryTrade
from ..core.exceptions import PnLComputationError
from .kernels.fx import (
    price_fx_forward,
    price_fx_vanilla,
    price_fx_digital,
)
from .kernels.rates import (
    price_ir_swap,
    price_ir_swap_vec,
    price_ir_swaption_bachelier,
    price_ir_swaption_black,
    price_ir_capfloor_bachelier,
    price_ir_capfloor_black,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Shocks → Scenarios adapter
# ─────────────────────────────────────────────────────────────────────

def shocks_to_scenarios(shocks: Any) -> Dict[str, np.ndarray]:
    """Convert typed shock objects to the flat dict format kernels expect.

    Parameters
    ----------
    shocks : FXShocks | IRShocks | dict | None
        Typed shock dataclass from an Asset, or an already-flat dict.

    Returns
    -------
    dict[str, np.ndarray]
        Keys depend on asset class:
        - FX: ``spot``, ``vol``, ``domestic_rate``, ``foreign_rate``,
          optionally ``forward_points``
        - IR: ``rate``, optionally ``vol``
    """
    if shocks is None:
        return {}

    if isinstance(shocks, dict):
        return shocks

    if isinstance(shocks, FXShocks):
        result: Dict[str, np.ndarray] = {
            "spot": shocks.spot,
            "vol": shocks.vol_surface,
            "domestic_rate": shocks.domestic_rate,
            "foreign_rate": shocks.foreign_rate,
        }
        if shocks.forward_points is not None:
            result["forward_points"] = shocks.forward_points
        return result

    if isinstance(shocks, IRShocks):
        result = {"rate": shocks.curve}
        if shocks.vol is not None:
            result["vol"] = shocks.vol
        return result

    raise PnLComputationError(
        f"Cannot convert shocks of type {type(shocks).__name__} to scenarios"
    )


class OptionPricer:
    """Batch pricer dispatching to @njit kernels by payoff type.

    Parameters
    ----------
    config : dict or None
        Optional configuration overrides. Reserved for model selection
        (SABR, local vol, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    # ── Main entry point ──────────────────────────────────────────────

    def price_trades(
        self,
        trades: List[ElementaryTrade],
        market_data: Any,
    ) -> np.ndarray:
        """Price all trades for one factor across all scenarios.

        Groups trades by payoff_type, dispatches each group to the
        correct batch kernel, and reassembles the result.

        Parameters
        ----------
        trades : list[ElementaryTrade]
            All trades for this risk factor.
        market_data : Asset
            Fully loaded Asset with market data and shocks.

        Returns
        -------
        np.ndarray
            PnL matrix, shape ``(n_trades, n_scenarios)``.
        """
        scenarios = shocks_to_scenarios(
            market_data.shocks if hasattr(market_data, "shocks") else None
        )
        n_scenarios = self._resolve_n_scenarios(scenarios, market_data)
        n_trades = len(trades)

        if n_trades == 0 or n_scenarios == 0:
            return np.zeros((n_trades, n_scenarios), dtype=np.float64)

        asset_class = getattr(market_data, "asset_class", "unknown")

        # Group trades by payoff_type, preserving original order
        groups: Dict[str, List[int]] = {}
        for i, t in enumerate(trades):
            groups.setdefault(t.payoff_type, []).append(i)

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)

        for payoff_type, indices in groups.items():
            group_trades = [trades[i] for i in indices]
            try:
                group_pnl = self._dispatch(
                    payoff_type, asset_class, group_trades,
                    market_data, scenarios,
                )
                for local_idx, global_idx in enumerate(indices):
                    pnl[global_idx] = group_pnl[local_idx]
            except Exception as exc:
                raise PnLComputationError(
                    f"Pricing failed for payoff={payoff_type!r}, "
                    f"asset_class={asset_class!r}: {exc}"
                ) from exc

        return pnl

    # ── Dispatch by (asset_class, payoff_type) ────────────────────────

    def _dispatch(
        self,
        payoff_type: str,
        asset_class: str,
        trades: List[ElementaryTrade],
        market_data: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Route a homogeneous group to the correct batch kernel."""
        if asset_class == "fx":
            if payoff_type == "vanilla":
                return self._price_fx_vanillas(trades, market_data, scenarios)
            elif payoff_type == "forward":
                return self._price_fx_forwards(trades, market_data, scenarios)
            elif payoff_type == "digital":
                return self._price_fx_digitals(trades, market_data, scenarios)
            else:
                raise PnLComputationError(
                    f"No FX pricer for payoff_type={payoff_type!r}"
                )
        elif asset_class == "rates":
            if payoff_type == "swap":
                return self._price_ir_swaps(trades, market_data, scenarios)
            elif payoff_type == "swaption":
                return self._price_ir_swaptions(trades, market_data, scenarios)
            elif payoff_type == "cap_floor":
                return self._price_ir_capfloors(trades, market_data, scenarios)
            else:
                raise PnLComputationError(
                    f"No IR pricer for payoff_type={payoff_type!r}"
                )
        else:
            raise PnLComputationError(
                f"No batch pricer for asset_class={asset_class!r}"
            )

    # ── FX pricing helpers ───────────────────────────────────────────

    @staticmethod
    def _fx_rates(fx_asset: Any, T: float) -> tuple:
        """Extract GK-convention rates from FXAsset for a given tenor.

        FXAsset naming is opposite to GK:
            domestic_ir = base ccy  (asset)    → r_f in GK
            foreign_ir  = quote ccy (numeraire) → r_d in GK
        """
        r_d = fx_asset.foreign_ir.curve.rate_at(T)
        r_f = fx_asset.domestic_ir.curve.rate_at(T)
        return r_d, r_f

    # ── FX scenario repricing ─────────────────────────────────────────

    def _price_fx_vanillas(
        self,
        trades: List[ElementaryTrade],
        fx_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        S_base = fx_asset.spot
        spot_scenarios = scenarios["spot"]
        n_trades = len(trades)
        n_scenarios = spot_scenarios.shape[0]

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            K = p["strike"]
            T = p["expiry"]
            is_call = p.get("option_type", "call") == "call"
            N = p.get("notional", trade.notional)

            r_d_base, r_f_base = self._fx_rates(fx_asset, T)
            vol_base = (
                fx_asset.vol_surface.vol_at(T, K)
                if fx_asset.vol_surface is not None
                else 0.10
            )

            base_pv = price_fx_vanilla(S_base, K, T, r_d_base, r_f_base, vol_base, is_call, N)

            for j in range(n_scenarios):
                S_j = spot_scenarios[j]
                r_d_j = self._scenario_rate(scenarios, "foreign_rate", j, r_d_base)
                r_f_j = self._scenario_rate(scenarios, "domestic_rate", j, r_f_base)
                vol_j = self._scenario_vol(scenarios, j, vol_base)

                shocked_pv = price_fx_vanilla(S_j, K, T, r_d_j, r_f_j, vol_j, is_call, N)
                pnl[i, j] = shocked_pv - base_pv

        return pnl

    def _price_fx_forwards(
        self,
        trades: List[ElementaryTrade],
        fx_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        S_base = fx_asset.spot
        spot_scenarios = scenarios["spot"]
        n_trades = len(trades)
        n_scenarios = spot_scenarios.shape[0]

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            T = p["expiry"]
            K = p.get("forward_rate", 0.0)
            N = p.get("notional", trade.notional)
            phi = p.get("direction_sign", 1.0)

            r_d_base, r_f_base = self._fx_rates(fx_asset, T)
            base_pv = price_fx_forward(S_base, K, T, r_d_base, r_f_base, N, phi)

            for j in range(n_scenarios):
                S_j = spot_scenarios[j]
                r_d_j = self._scenario_rate(scenarios, "foreign_rate", j, r_d_base)
                r_f_j = self._scenario_rate(scenarios, "domestic_rate", j, r_f_base)

                shocked_pv = price_fx_forward(S_j, K, T, r_d_j, r_f_j, N, phi)
                pnl[i, j] = shocked_pv - base_pv

        return pnl

    def _price_fx_digitals(
        self,
        trades: List[ElementaryTrade],
        fx_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        S_base = fx_asset.spot
        spot_scenarios = scenarios["spot"]
        n_trades = len(trades)
        n_scenarios = spot_scenarios.shape[0]

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            K = p["strike"]
            T = p["expiry"]
            is_call = p.get("option_type", "call") == "call"
            payout = p.get("payout", 1.0)

            r_d_base, r_f_base = self._fx_rates(fx_asset, T)
            vol_base = (
                fx_asset.vol_surface.vol_at(T, K)
                if fx_asset.vol_surface is not None
                else 0.10
            )

            base_pv = price_fx_digital(S_base, K, T, r_d_base, r_f_base, vol_base, is_call, payout)

            for j in range(n_scenarios):
                S_j = spot_scenarios[j]
                r_d_j = self._scenario_rate(scenarios, "foreign_rate", j, r_d_base)
                r_f_j = self._scenario_rate(scenarios, "domestic_rate", j, r_f_base)
                vol_j = self._scenario_vol(scenarios, j, vol_base)

                shocked_pv = price_fx_digital(S_j, K, T, r_d_j, r_f_j, vol_j, is_call, payout)
                pnl[i, j] = shocked_pv - base_pv

        return pnl

    # ── FX scenario extraction helpers ────────────────────────────────

    @staticmethod
    def _scenario_rate(
        scenarios: Dict[str, np.ndarray], key: str, j: int, fallback: float,
    ) -> float:
        """Extract a scalar rate for scenario j, falling back to base value."""
        arr = scenarios.get(key)
        if arr is None:
            return fallback
        if arr.ndim == 1:
            return float(arr[j])
        return fallback

    @staticmethod
    def _scenario_vol(
        scenarios: Dict[str, np.ndarray], j: int, fallback: float,
    ) -> float:
        """Extract a scalar vol for scenario j, falling back to base value."""
        arr = scenarios.get("vol")
        if arr is None:
            return fallback
        if arr.ndim == 1:
            return float(arr[j])
        return fallback

    # ── IR batch kernels ──────────────────────────────────────────────

    def _price_ir_swaps(
        self,
        trades: List[ElementaryTrade],
        ir_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        curve = ir_asset.curve
        curve_shocks = scenarios.get("rate", np.zeros((1, curve.n_pillars)))
        n_trades = len(trades)

        pnl = np.zeros((n_trades, curve_shocks.shape[0]), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            pnl[i] = price_ir_swap_vec(
                curve.tenors, curve.values, curve_shocks,
                p["maturity"], p["fixed_rate"],
                p.get("notional", trade.notional),
                p.get("pay_receive_sign", 1.0),
                p.get("frequency", 0.5),
            )
        return pnl

    def _price_ir_swaptions(
        self,
        trades: List[ElementaryTrade],
        ir_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        curve = ir_asset.curve
        curve_shocks = scenarios.get("rate", np.zeros((1, curve.n_pillars)))
        n_scenarios = curve_shocks.shape[0]
        n_trades = len(trades)

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            opt_exp = p["option_expiry"]
            sw_ten = p["swap_tenor"]
            strike = p["strike"]
            notl = p.get("notional", trade.notional)
            is_payer = p.get("option_type", "payer") == "payer"
            sigma = self._resolve_ir_vol(ir_asset, opt_exp, sw_ten)
            freq = p.get("frequency", 0.5)

            price_fn = price_ir_swaption_bachelier
            if p.get("vol_type") == "lognormal":
                price_fn = price_ir_swaption_black

            base_pv = price_fn(
                curve.tenors, curve.values, opt_exp, sw_ten,
                strike, notl, is_payer, sigma, freq,
            )
            shocked_values = np.empty_like(curve.values)
            for j in range(n_scenarios):
                for k in range(len(curve.values)):
                    shocked_values[k] = curve.values[k] + curve_shocks[j, k]
                shocked_pv = price_fn(
                    curve.tenors, shocked_values, opt_exp, sw_ten,
                    strike, notl, is_payer, sigma, freq,
                )
                pnl[i, j] = shocked_pv - base_pv

        return pnl

    def _price_ir_capfloors(
        self,
        trades: List[ElementaryTrade],
        ir_asset: Any,
        scenarios: Dict[str, np.ndarray],
    ) -> np.ndarray:
        curve = ir_asset.curve
        curve_shocks = scenarios.get("rate", np.zeros((1, curve.n_pillars)))
        n_scenarios = curve_shocks.shape[0]
        n_trades = len(trades)

        pnl = np.zeros((n_trades, n_scenarios), dtype=np.float64)
        for i, trade in enumerate(trades):
            p = trade.parameters
            mat = p["maturity"]
            strike = p["strike"]
            notl = p.get("notional", trade.notional)
            is_cap = p.get("option_type", "cap") == "cap"
            sigma = self._resolve_caplet_vol(ir_asset, mat)
            freq = p.get("frequency", 0.25)

            price_fn = price_ir_capfloor_bachelier
            if p.get("vol_type") == "lognormal":
                price_fn = price_ir_capfloor_black

            base_pv = price_fn(
                curve.tenors, curve.values, mat, strike,
                notl, is_cap, sigma, freq,
            )
            shocked_values = np.empty_like(curve.values)
            for j in range(n_scenarios):
                for k in range(len(curve.values)):
                    shocked_values[k] = curve.values[k] + curve_shocks[j, k]
                shocked_pv = price_fn(
                    curve.tenors, shocked_values, mat, strike,
                    notl, is_cap, sigma, freq,
                )
                pnl[i, j] = shocked_pv - base_pv

        return pnl

    # ── Vol resolution helpers ────────────────────────────────────────

    @staticmethod
    def _resolve_ir_vol(ir_asset: Any, option_expiry: float, swap_tenor: float) -> float:
        if ir_asset.vol_cube is not None:
            cube = ir_asset.vol_cube
            atm_idx = int(np.argmin(np.abs(cube.strikes)))
            exp_idx = int(np.argmin(np.abs(cube.expiries - option_expiry)))
            ten_idx = int(np.argmin(np.abs(cube.swap_tenors - swap_tenor)))
            return float(cube.values[exp_idx, ten_idx, atm_idx])
        return 0.005

    @staticmethod
    def _resolve_caplet_vol(ir_asset: Any, maturity: float) -> float:
        if ir_asset.vol_cube is not None:
            cube = ir_asset.vol_cube
            atm_idx = int(np.argmin(np.abs(cube.strikes)))
            exp_idx = int(np.argmin(np.abs(cube.expiries - maturity * 0.5)))
            return float(cube.values[exp_idx, 0, atm_idx])
        return 0.005

    @staticmethod
    def _resolve_n_scenarios(scenarios: Dict[str, np.ndarray], market_data: Any) -> int:
        for v in scenarios.values():
            return v.shape[0]
        if hasattr(market_data, "n_scenarios"):
            return market_data.n_scenarios
        return 0
