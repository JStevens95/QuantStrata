from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Union

import numpy as np

from src.marketdata.market import Market
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

from src.models.analytic.black_scholes_merton.vanilla import BlackScholesMertonVanilla
from src.models.analytic.black_scholes_merton.digital import (
    BlackScholesMertonDigitalCash,
    BlackScholesMertonDigitalAsset,
)

# Payoff library: single source of truth for terminal condition + payout semantics
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff
from src.models.payoffs.base import BasePayoff1D
from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff

GreekName = Literal[
    "delta",
    "gamma",
    "vega",
    "rho_domestic",
    "rho_foreign",
]


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    df = exp(-rT)  =>  r = -ln(df)/T
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


def _terminal_value(payoff: BasePayoff1D, spot: float) -> float:
    """
    Evaluate payoff at a single spot using the payoff library (vectorized API).
    """
    return float(payoff.terminal(np.asarray([float(spot)], dtype=np.float64))[0])


@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaBsmPricer:
    """
    Adapter pricer: EuropeanFxVanillaOption -> BlackScholesMertonVanilla (generic carry).

    Alignment policy with MC/FD payoff library
    -----------------------------------------
    - Always build the terminal payoff via payoff library.
    - Handle degenerate cases consistently:
        * T == 0: PV = payoff(S0) (no discounting), then scale by notional.
        * sigma == 0: PV = exp(-r_d T) * payoff(F0) where F0 = S0 * exp((r_d - r_f) T),
                      then scale by notional.

    FX mapping (analytic engine)
    ----------------------------
    - r = r_d (domestic rate)
    - b = r_d - r_f  (cost-of-carry)
    - engine PV is "per 1 unit of foreign notional"
    - we multiply by notional to get domestic PV

    Greeks mapping (important!)
    ---------------------------
    The generic engine returns:
      - rho_discount: dPV/dr holding b fixed
      - rho_carry:    dPV/db holding r fixed

    For FX:
      b = r_d - r_f,  r = r_d

    Chain rule:
      dPV/dr_d = dPV/dr * 1 + dPV/db * 1 = rho_discount + rho_carry
      dPV/dr_f = dPV/db * (-1)           = -rho_carry
    """

    engine: BlackScholesMertonVanilla = BlackScholesMertonVanilla()

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        # ---- Read market inputs ----
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        notional = float(trade.notional)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # Terminal payoff (single source of truth).
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Consistent degenerate shortcut: expiry now => immediate payoff (no discounting).
        if T == 0.0:
            return notional * _terminal_value(payoff, S0)

        # Discount factors from curves.
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs -> continuous rates.
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)

        # Generic carry for the analytic engine.
        b = float(r_d - r_f)

        # Implied vol from the surface.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Consistent degenerate shortcut: zero vol => deterministic forward terminal spot.
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            return float(notional * disc * _terminal_value(payoff, F0))

        # ---- Analytic engine PV is per 1 unit foreign; scale by notional ----
        pv_per_unit = self.engine.price(
            option_type=trade.option_type,
            spot=S0,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )
        return float(notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        # ---- Read market inputs ----
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        notional = float(trade.notional)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # Policy match with FD: at expiry, return stable zeros (kinked payoff greeks unstable).
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho_domestic": 0.0,
                "rho_foreign": 0.0,
            }

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Note: we do NOT special-case sigma==0 in greeks; analytic greeks can be ill-conditioned.
        # Keeping this consistent with your FD policy (where greeks are computed via PDE/bump logic)
        # is fine; if you want a deterministic-limit greek policy later, do it explicitly.

        g = self.engine.greeks(
            option_type=trade.option_type,
            spot=S0,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )

        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])

        rho_r = notional * float(g["rho_discount"])
        rho_carry = notional * float(g["rho_carry"])

        rho_domestic = float(rho_r + rho_carry)
        rho_foreign = float(-rho_carry)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }


@dataclass(frozen=True, slots=True)
class FxEuropeanDigitalBsmPricer:
    """
    FX adapter for EuropeanFxDigitalOption using BSM-with-carry analytic engines.

    Alignment policy with MC/FD payoff library
    -----------------------------------------
    - Build payoff via payoff library (single source of truth).
    - Handle degenerate cases consistently:
        * T == 0: PV = payoff(S0) (no discounting)
        * sigma == 0: PV = exp(-r_d T) * payoff(F0) with F0 = S0 * exp((r_d - r_f) T)

    Engine routing
    --------------
    We route cash-vs-asset engine selection from the payoff object type
    (DigitalCashPayoff vs DigitalAssetPayoff) to avoid duplicating trade.payoff logic.
    """

    cash_engine: BlackScholesMertonDigitalCash = BlackScholesMertonDigitalCash()
    asset_engine: BlackScholesMertonDigitalAsset = BlackScholesMertonDigitalAsset()

    def price(self, trade: EuropeanFxDigitalOption, market: Market) -> float:
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Expiry-now shortcut (no discounting).
        if T == 0.0:
            return _terminal_value(payoff, S0)

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero-vol shortcut: deterministic forward.
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            return float(disc * _terminal_value(payoff, F0))

        # Route engine choice from payoff type (single source of truth via build_payoff_1d).
        if isinstance(payoff, DigitalCashPayoff):
            pv = self.cash_engine.price(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                cash=float(payoff.cash),
            )
            return float(pv)

        if isinstance(payoff, DigitalAssetPayoff):
            pv = self.asset_engine.price(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                asset_units=float(payoff.asset_units),
            )
            return float(pv)

        # Defensive: if a new payoff type is added later, fail loudly.
        raise TypeError(f"Unsupported payoff type for FxEuropeanDigitalBsmPricer: {type(payoff)!r}")

    def greeks(self, trade: EuropeanFxDigitalOption, market: Market) -> Dict[GreekName, float]:
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # Policy match with your FD digital greeks: stable zeros at/after expiry.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho_domestic": 0.0,
                "rho_foreign": 0.0,
            }

        payoff = require_terminal_payoff(build_payoff_1d(trade))

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        if isinstance(payoff, DigitalCashPayoff):
            g = self.cash_engine.greeks(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                cash=float(payoff.cash),
            )
        elif isinstance(payoff, DigitalAssetPayoff):
            g = self.asset_engine.greeks(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                asset_units=float(payoff.asset_units),
            )
        else:
            raise TypeError(f"Unsupported payoff type for FxEuropeanDigitalBsmPricer: {type(payoff)!r}")

        rho_domestic = float(g["rho_discount"] + g["rho_carry"])
        rho_foreign = float(-g["rho_carry"])

        return {
            "delta": float(g["delta"]),
            "gamma": float(g["gamma"]),
            "vega": float(g["vega"]),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }