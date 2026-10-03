"""
FX instrument definitions — elementary trades for FX risk factors.

Each dataclass represents a single elementary trade that the trade
generator creates and the pricer/PnL engine consumes. Every instrument
knows how to:
  - to_pricer_params()  → flat dict for vectorised batch pricing
  - to_trade_id()       → stable globally-unique ID
  - price()             → single-trade PV given an FXAsset
  - sensitivities()     → first-order greeks (delta, gamma, vega, theta, rho)

Pricing & greeks delegate to @njit compiled kernels in
pricing/kernels/fx.py for maximum throughput. The class methods
extract numeric inputs from the Asset object, call the kernel,
and return the result.

Instrument types:
  - FXForward        : outright / NDF
  - FXVanillaOption  : European/American call/put
  - FXDigital        : cash-or-nothing digital
  - FXBarrierOption  : single-barrier knock-in / knock-out (future)

The generate_fx_elementary_trades() helper creates the full trade
universe for a given FX risk factor from strike and tenor grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import InstrumentSpec, StrikeGrid, TenorGrid
from ..pricing.kernels.fx import (
    price_fx_forward,
    greeks_fx_forward,
    price_fx_vanilla,
    greeks_fx_vanilla,
    price_fx_digital,
    greeks_fx_digital,
)


# ─────────────────────────────────────────────────────────────────────
# FX Forward
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FXForward(InstrumentSpec):
    """FX outright forward / NDF.

    Parameters
    ----------
    expiry : float
        Time to maturity in year fractions.
    forward_rate : float
        Agreed forward rate.
    notional : float
        Notional in base currency.
    direction : str
        ``"buy"`` or ``"sell"`` base currency.
    """
    expiry: float = 1.0
    forward_rate: float = 0.0
    notional: float = 1.0
    direction: str = "buy"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "expiry": self.expiry,
            "forward_rate": self.forward_rate,
            "notional": self.notional,
            "direction_sign": 1.0 if self.direction == "buy" else -1.0,
        }

    def to_trade_id(self) -> str:
        d = "B" if self.direction == "buy" else "S"
        return f"{self.factor_id}|FWD|{self.expiry:.4f}|{d}"

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: price_fx_forward."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        phi = 1.0 if self.direction == "buy" else -1.0
        return price_fx_forward(
            S, self.forward_rate, self.expiry, r_d, r_f, self.notional, phi,
        )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Greeks via @njit kernel: greeks_fx_forward."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        phi = 1.0 if self.direction == "buy" else -1.0
        delta, gamma, vega, theta, rho_d, rho_f = greeks_fx_forward(
            S, self.forward_rate, self.expiry, r_d, r_f, self.notional, phi,
        )
        return {
            "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho_d": rho_d, "rho_f": rho_f,
        }


# ─────────────────────────────────────────────────────────────────────
# FX Vanilla Option
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FXVanillaOption(InstrumentSpec):
    """European/American FX vanilla call or put.

    Parameters
    ----------
    expiry : float
        Time to expiry in year fractions.
    strike : float
        Option strike (absolute or delta-derived).
    notional : float
        Notional in base currency.
    option_type : str
        ``"call"`` or ``"put"``.
    exercise : str
        ``"european"`` or ``"american"``.
    """
    expiry: float = 1.0
    strike: float = 0.0
    notional: float = 1.0
    option_type: str = "call"
    exercise: str = "european"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "expiry": self.expiry,
            "strike": self.strike,
            "notional": self.notional,
            "option_type": self.option_type,
            "exercise": self.exercise,
        }

    def to_trade_id(self) -> str:
        cp = "C" if self.option_type == "call" else "P"
        return f"{self.factor_id}|VAN|{self.expiry:.4f}|{self.strike:.6f}|{cp}"

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: price_fx_vanilla (Garman-Kohlhagen)."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        vol = _extract_fx_vol(market_data, self.expiry, self.strike)
        return price_fx_vanilla(
            S, self.strike, self.expiry, r_d, r_f,
            vol, self.option_type == "call", self.notional,
        )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Greeks via @njit kernel: greeks_fx_vanilla."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        vol = _extract_fx_vol(market_data, self.expiry, self.strike)
        delta, gamma, vega, theta, rho_d, rho_f = greeks_fx_vanilla(
            S, self.strike, self.expiry, r_d, r_f,
            vol, self.option_type == "call", self.notional,
        )
        return {
            "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho_d": rho_d, "rho_f": rho_f,
        }


# ─────────────────────────────────────────────────────────────────────
# FX Digital
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FXDigital(InstrumentSpec):
    """Cash-or-nothing FX digital option.

    Parameters
    ----------
    expiry : float
        Time to expiry in year fractions.
    strike : float
        Barrier/strike level.
    payout : float
        Fixed payout amount if in-the-money at expiry.
    option_type : str
        ``"call"`` (pays if S > K) or ``"put"`` (pays if S < K).
    """
    expiry: float = 1.0
    strike: float = 0.0
    payout: float = 1.0
    option_type: str = "call"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "expiry": self.expiry,
            "strike": self.strike,
            "payout": self.payout,
            "option_type": self.option_type,
        }

    def to_trade_id(self) -> str:
        cp = "C" if self.option_type == "call" else "P"
        return f"{self.factor_id}|DIG|{self.expiry:.4f}|{self.strike:.6f}|{cp}"

    def price(self, market_data: Any) -> float:
        """PV via @njit kernel: price_fx_digital."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        vol = _extract_fx_vol(market_data, self.expiry, self.strike)
        return price_fx_digital(
            S, self.strike, self.expiry, r_d, r_f,
            vol, self.option_type == "call", self.payout,
        )

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """Greeks via @njit kernel: greeks_fx_digital."""
        S, r_d, r_f = _extract_fx_rates(market_data, self.expiry)
        vol = _extract_fx_vol(market_data, self.expiry, self.strike)
        delta, gamma, vega, theta = greeks_fx_digital(
            S, self.strike, self.expiry, r_d, r_f,
            vol, self.option_type == "call", self.payout,
        )
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ─────────────────────────────────────────────────────────────────────
# FX Barrier Option (placeholder for extension)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class FXBarrierOption(InstrumentSpec):
    """Single-barrier knock-in / knock-out FX option.

    TODO(wire): Full implementation requires barrier monitoring
    convention (continuous/discrete), rebate, and the appropriate
    closed-form or PDE pricer.
    """
    expiry: float = 1.0
    strike: float = 0.0
    barrier: float = 0.0
    notional: float = 1.0
    option_type: str = "call"
    barrier_type: str = "knock_out"
    barrier_direction: str = "up"

    def to_pricer_params(self) -> Dict[str, Any]:
        return {
            "payoff_type": self.payoff_type,
            "expiry": self.expiry,
            "strike": self.strike,
            "barrier": self.barrier,
            "notional": self.notional,
            "option_type": self.option_type,
            "barrier_type": self.barrier_type,
            "barrier_direction": self.barrier_direction,
        }

    def to_trade_id(self) -> str:
        cp = "C" if self.option_type == "call" else "P"
        bt = "KO" if self.barrier_type == "knock_out" else "KI"
        bd = "U" if self.barrier_direction == "up" else "D"
        return f"{self.factor_id}|BAR|{self.expiry:.4f}|{self.strike:.6f}|{self.barrier:.6f}|{cp}{bt}{bd}"

    def price(self, market_data: Any) -> float:
        """TODO(wire): Implement barrier option pricing (Merton/Reiner-Rubinstein)."""
        raise NotImplementedError("Wire barrier option pricer")

    def sensitivities(self, market_data: Any) -> Dict[str, float]:
        """TODO(wire): Implement barrier option greeks."""
        raise NotImplementedError("Wire barrier option greeks")


# ─────────────────────────────────────────────────────────────────────
# Market data extraction helpers (OOP → numeric for kernels)
# ─────────────────────────────────────────────────────────────────────

def _extract_fx_rates(market_data: Any, T: float) -> tuple:
    """Extract (spot, r_d, r_f) from an FXAsset for GK kernel input.

    FXAsset naming vs GK convention:
        domestic_ir = base ccy  (asset)     → r_f  (GK foreign)
        foreign_ir  = quote ccy (numeraire) → r_d  (GK domestic)
    """
    S = market_data.spot
    r_d = market_data.foreign_ir.curve.rate_at(T)
    r_f = market_data.domestic_ir.curve.rate_at(T)
    return S, r_d, r_f


def _extract_fx_vol(market_data: Any, T: float, K: float) -> float:
    """Extract implied vol from FXAsset's vol surface, with fallback."""
    if market_data.vol_surface is not None:
        return market_data.vol_surface.vol_at(T, K)
    return 0.10


def generate_fx_elementary_trades(
    factor_id: str,
    tenor_grid: TenorGrid,
    strike_grid: StrikeGrid,
    include_forwards: bool = True,
    include_vanillas: bool = True,
    include_digitals: bool = True,
) -> List[InstrumentSpec]:
    """Generate the elementary trade universe for one FX risk factor.

    Creates a cross-product of tenors × strikes × option types,
    producing the full set of instruments that a pricer can batch-price.

    Parameters
    ----------
    factor_id : str
        FX pair identifier (e.g. ``"EURUSD"``).
    tenor_grid : TenorGrid
        Expiry tenors.
    strike_grid : StrikeGrid
        Option strikes.
    include_forwards : bool
        Generate FX forwards at each tenor.
    include_vanillas : bool
        Generate call + put at each tenor × strike.
    include_digitals : bool
        Generate digital call + put at each tenor × strike.

    Returns
    -------
    list of InstrumentSpec
        Elementary trades ready for pricer.
    """
    trades: List[InstrumentSpec] = []

    for tenor in tenor_grid.values:
        if include_forwards:
            for direction in ("buy", "sell"):
                trades.append(FXForward(
                    asset_class="fx",
                    payoff_type="forward",
                    factor_id=factor_id,
                    expiry=tenor,
                    direction=direction,
                ))

        for strike in strike_grid.values:
            if include_vanillas:
                for opt_type in ("call", "put"):
                    trades.append(FXVanillaOption(
                        asset_class="fx",
                        payoff_type="vanilla",
                        factor_id=factor_id,
                        expiry=tenor,
                        strike=strike,
                        option_type=opt_type,
                    ))

            if include_digitals:
                for opt_type in ("call", "put"):
                    trades.append(FXDigital(
                        asset_class="fx",
                        payoff_type="digital",
                        factor_id=factor_id,
                        expiry=tenor,
                        strike=strike,
                        option_type=opt_type,
                    ))

    return trades
