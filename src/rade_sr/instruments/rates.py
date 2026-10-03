"""
IR instrument definitions — elementary trades for interest rate risk factors.

Each dataclass represents a single elementary trade. Every instrument
knows how to:
  - to_pricer_params()  → flat dict for vectorised batch pricing
  - to_trade_id()       → stable globally-unique ID
  - price()             → single-trade PV given an IRAsset
  - sensitivities()     → first-order greeks (delta/DV01, gamma, vega, theta)

Pricing & greeks delegate to @njit compiled kernels in
pricing/kernels/rates.py for maximum throughput.

Instrument types:
  - IRSwap       : fixed-for-floating interest rate swap
  - IRSwaption   : European payer/receiver swaption
  - IRCapFloor   : interest rate cap or floor

The generate_ir_elementary_trades() helper creates the full trade
universe for a given IR risk factor from tenor grids and strike grids.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from .base import InstrumentSpec, StrikeGrid, TenorGrid
from ..pricing.kernels.rates import (
    price_ir_swap,
    greeks_ir_swap,
    price_ir_swaption_bachelier,
    price_ir_swaption_black,
    greeks_ir_swaption_bachelier,
    price_ir_capfloor_bachelier,
    price_ir_capfloor_black,
)


# ─────────────────────────────────────────────────────────────────────
# IR Swap
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRSwap(InstrumentSpec):
    """Plain-vanilla fixed-for-floating interest rate swap.

    Parameters
    ----------
    maturity : float
        Swap maturity in year fractions.
    fixed_rate : float
        Fixed leg coupon rate.
    notional : float
        Notional principal.
    pay_receive : str
        ``"pay"`` or ``"receive"`` fixed.
    frequency : float
        Payment frequency in year fractions (e.g. 0.5 for semi-annual).
    """
    maturity: float = 5.0
    fixed_rate: float = 0.0
    notional: float = 1_000_000.0
    pay_receive: str = "pay"
    frequency: float = 0.5

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "maturity": self.maturity,
            "fixed_rate": self.fixed_rate,
            "notional": self.notional,
            "pay_receive_sign": 1.0 if self.pay_receive == "pay" else -1.0,
            "frequency": self.frequency,
        }

    def to_trade_id(self) -> str:
        pr = "P" if self.pay_receive == "pay" else "R"
        return f"{self.factor_id}|SWAP|{self.maturity:.2f}|{self.fixed_rate:.6f}|{pr}"

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: price_ir_swap."""
        tenors, values = _extract_curve(market_data)
        sign = 1.0 if self.pay_receive == "pay" else -1.0
        return price_ir_swap(
            tenors, values, self.maturity, self.fixed_rate,
            self.notional, sign, self.frequency,
        )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Greeks via @njit kernel: greeks_ir_swap."""
        tenors, values = _extract_curve(market_data)
        sign = 1.0 if self.pay_receive == "pay" else -1.0
        dv01, gamma, vega, theta = greeks_ir_swap(
            tenors, values, self.maturity, self.fixed_rate,
            self.notional, sign, self.frequency,
        )
        return {"dv01": dv01, "gamma": gamma, "vega": vega, "theta": theta}


# ─────────────────────────────────────────────────────────────────────
# IR Swaption
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRSwaption(InstrumentSpec):
    """European payer or receiver swaption.

    Parameters
    ----------
    option_expiry : float
        Time to option expiry in year fractions.
    swap_tenor : float
        Underlying swap tenor in year fractions.
    strike : float
        Strike rate (fixed coupon on the underlying swap).
    notional : float
        Notional principal of the underlying swap.
    option_type : str
        ``"payer"`` (right to pay fixed) or ``"receiver"`` (right to receive fixed).
    vol_type : str
        ``"normal"`` (Bachelier) or ``"lognormal"`` (Black).
    """
    option_expiry: float = 1.0
    swap_tenor: float = 5.0
    strike: float = 0.0
    notional: float = 1_000_000.0
    option_type: str = "payer"
    vol_type: str = "normal"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "option_expiry": self.option_expiry,
            "swap_tenor": self.swap_tenor,
            "strike": self.strike,
            "notional": self.notional,
            "option_type": self.option_type,
            "vol_type": self.vol_type,
        }

    def to_trade_id(self) -> str:
        ot = "PAY" if self.option_type == "payer" else "RCV"
        return (
            f"{self.factor_id}|SWPTN|"
            f"{self.option_expiry:.2f}x{self.swap_tenor:.2f}|"
            f"{self.strike:.6f}|{ot}"
        )

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: Bachelier or Black swaption."""
        tenors, values = _extract_curve(market_data)
        sigma = _extract_swaption_vol(market_data, self.option_expiry, self.swap_tenor, self.vol_type)
        is_payer = self.option_type == "payer"

        if self.vol_type == "normal":
            return price_ir_swaption_bachelier(
                tenors, values, self.option_expiry, self.swap_tenor,
                self.strike, self.notional, is_payer, sigma, 0.5,
            )
        else:
            return price_ir_swaption_black(
                tenors, values, self.option_expiry, self.swap_tenor,
                self.strike, self.notional, is_payer, sigma, 0.5,
            )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Greeks via @njit kernel: greeks_ir_swaption_bachelier.

        Note: Black greeks use the same Bachelier kernel structure;
        extend with a greeks_ir_swaption_black kernel if needed.
        """
        tenors, values = _extract_curve(market_data)
        sigma = _extract_swaption_vol(market_data, self.option_expiry, self.swap_tenor, self.vol_type)
        is_payer = self.option_type == "payer"

        delta, gamma, vega, theta = greeks_ir_swaption_bachelier(
            tenors, values, self.option_expiry, self.swap_tenor,
            self.strike, self.notional, is_payer, sigma, 0.5,
        )
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ─────────────────────────────────────────────────────────────────────
# IR Cap / Floor
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IRCapFloor(InstrumentSpec):
    """Interest rate cap or floor (strip of caplets/floorlets).

    Parameters
    ----------
    maturity : float
        Cap/floor maturity in year fractions.
    strike : float
        Strike rate.
    notional : float
        Notional principal.
    option_type : str
        ``"cap"`` or ``"floor"``.
    frequency : float
        Caplet reset frequency in year fractions.
    vol_type : str
        ``"normal"`` or ``"lognormal"``.
    """
    maturity: float = 5.0
    strike: float = 0.0
    notional: float = 1_000_000.0
    option_type: str = "cap"
    frequency: float = 0.25
    vol_type: str = "normal"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "maturity": self.maturity,
            "strike": self.strike,
            "notional": self.notional,
            "option_type": self.option_type,
            "frequency": self.frequency,
            "vol_type": self.vol_type,
        }

    def to_trade_id(self) -> str:
        cf = "CAP" if self.option_type == "cap" else "FLR"
        return f"{self.factor_id}|{cf}|{self.maturity:.2f}|{self.strike:.6f}"

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: capfloor Bachelier or Black."""
        tenors, values = _extract_curve(market_data)
        sigma = _extract_caplet_vol(market_data, self.maturity, self.vol_type)
        is_cap = self.option_type == "cap"

        if self.vol_type == "normal":
            return price_ir_capfloor_bachelier(
                tenors, values, self.maturity, self.strike,
                self.notional, is_cap, sigma, self.frequency,
            )
        else:
            return price_ir_capfloor_black(
                tenors, values, self.maturity, self.strike,
                self.notional, is_cap, sigma, self.frequency,
            )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Aggregate greeks via bump-and-reprice through @njit kernels.

        TODO(wire): For production, add per-caplet analytic greeks kernel.
        """
        pv_base = self.price(market_data)
        return {
            "dv01": 0.0,   # TODO(wire): bump-and-reprice via kernel
            "gamma": 0.0,
            "vega": 0.0,   # TODO(wire): parallel vol bump via kernel
            "theta": 0.0,
            "pv": pv_base,
        }


# ─────────────────────────────────────────────────────────────────────
# Market data extraction helpers (OOP → numeric for kernels)
# ─────────────────────────────────────────────────────────────────────

def _extract_curve(market_data: Any) -> tuple:
    """Extract (tenors, values) arrays from IRAsset's discount curve."""
    curve = market_data.curve
    return curve.tenors, curve.values


def _extract_swaption_vol(
    market_data: Any,
    option_expiry: float,
    swap_tenor: float,
    vol_type: str,
) -> float:
    """Extract ATM swaption vol from IRAsset's vol cube."""
    if market_data.vol_cube is not None:
        cube = market_data.vol_cube
        atm_idx = int(np.argmin(np.abs(cube.strikes)))
        exp_idx = int(np.argmin(np.abs(cube.expiries - option_expiry)))
        ten_idx = int(np.argmin(np.abs(cube.swap_tenors - swap_tenor)))
        return float(cube.values[exp_idx, ten_idx, atm_idx])
    return 0.005 if vol_type == "normal" else 0.20


def _extract_caplet_vol(market_data: Any, maturity: float, vol_type: str) -> float:
    """Extract representative caplet vol (mid-maturity ATM) from IRAsset."""
    if market_data.vol_cube is not None:
        cube = market_data.vol_cube
        atm_idx = int(np.argmin(np.abs(cube.strikes)))
        exp_idx = int(np.argmin(np.abs(cube.expiries - maturity * 0.5)))
        ten_idx = 0
        return float(cube.values[exp_idx, ten_idx, atm_idx])
    return 0.005 if vol_type == "normal" else 0.20


def generate_ir_elementary_trades(
    factor_id: str,
    tenor_grid: TenorGrid,
    strike_grid: StrikeGrid,
    include_swaps: bool = True,
    include_swaptions: bool = True,
    include_caps_floors: bool = True,
    vol_type: str = "normal",
) -> List[InstrumentSpec]:
    """Generate the elementary trade universe for one IR risk factor.

    Creates a cross-product of tenors × strikes × types, producing
    the full set of instruments for the PnL engine.

    Parameters
    ----------
    factor_id : str
        IR curve identifier (e.g. ``"EUR_6M"``).
    tenor_grid : TenorGrid
        Swap maturities / swaption expiries.
    strike_grid : StrikeGrid
        Strike rates (absolute bp or rate values).
    include_swaps : bool
        Generate pay/receive swaps at each maturity.
    include_swaptions : bool
        Generate payer/receiver swaptions at each expiry × strike.
    include_caps_floors : bool
        Generate caps/floors at each maturity × strike.
    vol_type : str
        ``"normal"`` or ``"lognormal"`` for swaptions/caps.

    Returns
    -------
    list of InstrumentSpec
    """
    trades: List[InstrumentSpec] = []

    for tenor in tenor_grid.values:
        if include_swaps:
            for pr in ("pay", "receive"):
                trades.append(IRSwap(
                    asset_class="rates",
                    payoff_type="swap",
                    factor_id=factor_id,
                    maturity=tenor,
                    pay_receive=pr,
                ))

        for strike in strike_grid.values:
            if include_swaptions:
                for opt_type in ("payer", "receiver"):
                    trades.append(IRSwaption(
                        asset_class="rates",
                        payoff_type="swaption",
                        factor_id=factor_id,
                        option_expiry=tenor,
                        swap_tenor=min(tenor * 2, 30.0),
                        strike=strike,
                        option_type=opt_type,
                        vol_type=vol_type,
                    ))

            if include_caps_floors:
                for opt_type in ("cap", "floor"):
                    trades.append(IRCapFloor(
                        asset_class="rates",
                        payoff_type="cap_floor",
                        factor_id=factor_id,
                        maturity=tenor,
                        strike=strike,
                        option_type=opt_type,
                        vol_type=vol_type,
                    ))

    return trades
