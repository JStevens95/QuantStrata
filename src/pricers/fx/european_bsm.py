# src/pricers/fx/european_bsm.py
"""
FX European BSM Pricers.

Adapter pricers that map FX instruments to the generic BSM model.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}

from src.marketdata.core.market import Market
from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption

# Import pure functions from the generic BSM model.
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
    vanilla_greeks,
    digital_cash_price,
    digital_cash_greeks,
    digital_asset_price,
    digital_asset_greeks,
)

# Payoff library: single source of truth for terminal condition + payout semantics.
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff
from src.models.payoffs.base import BasePayoff1D
from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff

GreekName = Literal[
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho_domestic",
    "rho_foreign",
]


# Import common rate utility.
from src.core.math.rates import rate_from_df as _rate_from_df


def _terminal_value(payoff: BasePayoff1D, spot: float) -> float:
    """
    Evaluate payoff at a single spot using the payoff library (vectorized API).
    """
    return float(payoff.terminal(np.asarray([float(spot)], dtype=np.float64))[0])


@dataclass(**_DATACLASS_KW)
class FxVanillaEuropeanOptionBsmPricer:
    """
    Adapter pricer: EuropeanFxVanillaOption -> BlackScholesMertonVanilla formulas (generic carry).

    FX Mapping
    ----------
    - discount_rate = r_d (domestic rate)
    - carry = r_d - r_f (cost-of-carry)
    - PV per unit foreign notional, scaled by notional

    Greeks Mapping
    --------------
    The generic model returns rho_discount and rho_carry.
    For FX we compute:
    - rho_domestic = rho_discount + rho_carry (since both r and b depend on r_d)
    - rho_foreign = -rho_carry (since only b depends on r_f)
    """

    def price(self, trade: FxVanillaEuropeanOption, market: Market) -> float:
        """Price FX vanilla option using BSM."""
        # Read market inputs.
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        notional = float(trade.notional)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # Build terminal payoff (single source of truth).
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Handle T=0: immediate payoff, no discounting.
        if T == 0.0:
            return notional * _terminal_value(payoff, S0)

        # Get discount factors from curves.
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs to continuous rates.
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)

        # FX cost-of-carry.
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

        # Call generic BSM formula.
        pv_per_unit = vanilla_price(
            option_type=trade.option_type,
            spot=S0,
            strike=K,
            expiry=T,
            discount_rate=r_d,
            carry=b,
            vol=sigma,
        )
        return float(notional) * float(pv_per_unit)

    def greeks(self, trade: FxVanillaEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for FX vanilla option."""
        # Read market inputs.
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        notional = float(trade.notional)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional < 0.0:
            raise ValueError("notional must be >= 0.")

        # Handle T=0: return stable zeros.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho_domestic": 0.0,
                "rho_foreign": 0.0,
            }

        # Get rates.
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        # Get vol.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Call generic BSM Greeks.
        g = vanilla_greeks(
            option_type=trade.option_type,
            spot=S0,
            strike=K,
            expiry=T,
            discount_rate=r_d,
            carry=b,
            vol=sigma,
        )

        # Scale by notional.
        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])
        theta = notional * float(g["theta"])

        # Map generic rhos to FX-specific rhos.
        rho_r = notional * float(g["rho_discount"])
        rho_carry = notional * float(g["rho_carry"])
        rho_domestic = float(rho_r + rho_carry)
        rho_foreign = float(-rho_carry)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }


# Alias for backward compatibility with examples and docs
FxEuropeanVanillaBsmPricer = FxVanillaEuropeanOptionBsmPricer


@dataclass(**_DATACLASS_KW)
class FxDigitalEuropeanOptionBsmPricer:
    """
    Adapter pricer: EuropeanFxDigitalOption -> BSM digital formulas.

    Routes to cash-or-nothing or asset-or-nothing based on payoff type.
    """

    def price(self, trade: FxDigitalEuropeanOption, market: Market) -> float:
        """Price FX digital option using BSM."""
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # Build payoff.
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Handle T=0: immediate payoff.
        if T == 0.0:
            return _terminal_value(payoff, S0)

        # Get rates.
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        # Get vol.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Handle σ=0: deterministic forward.
        if sigma == 0.0:
            F0 = S0 * math.exp((r_d - r_f) * T)
            disc = math.exp(-r_d * T)
            return float(disc * _terminal_value(payoff, F0))

        # Route to appropriate formula based on payoff type.
        if isinstance(payoff, DigitalCashPayoff):
            return float(digital_cash_price(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                expiry=T,
                discount_rate=r_d,
                carry=b,
                vol=sigma,
                cash=float(payoff.cash),
            ))

        if isinstance(payoff, DigitalAssetPayoff):
            return float(digital_asset_price(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                expiry=T,
                discount_rate=r_d,
                carry=b,
                vol=sigma,
            ) * float(payoff.asset_units))

        raise TypeError(f"Unsupported payoff type: {type(payoff)!r}")

    def greeks(self, trade: FxDigitalEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for FX digital option."""
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S0 <= 0.0:
            raise ValueError("spot must be > 0.")

        # Handle T=0: return stable zeros.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho_domestic": 0.0,
                "rho_foreign": 0.0,
            }

        # Build payoff.
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Get rates.
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        # Get vol.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Route to appropriate Greeks function.
        if isinstance(payoff, DigitalCashPayoff):
            g = digital_cash_greeks(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                expiry=T,
                discount_rate=r_d,
                carry=b,
                vol=sigma,
                cash=float(payoff.cash),
            )
        elif isinstance(payoff, DigitalAssetPayoff):
            g = digital_asset_greeks(
                option_type=trade.option_type,
                spot=S0,
                strike=K,
                expiry=T,
                discount_rate=r_d,
                carry=b,
                vol=sigma,
            )
            # Scale by asset_units.
            g = {k: v * float(payoff.asset_units) for k, v in g.items()}
        else:
            raise TypeError(f"Unsupported payoff type: {type(payoff)!r}")

        # Map generic rhos to FX-specific rhos.
        rho_domestic = float(g["rho_discount"] + g["rho_carry"])
        rho_foreign = float(-g["rho_carry"])

        return {
            "delta": float(g["delta"]),
            "gamma": float(g["gamma"]),
            "vega": float(g["vega"]),
            "theta": float(g["theta"]),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }
