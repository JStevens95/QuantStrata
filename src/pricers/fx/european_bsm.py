from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.market import Market
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

from src.models.analytic.black_scholes_merton.vanilla import BlackScholesMertonVanilla
from src.models.analytic.black_scholes_merton.digital import (
    BlackScholesMertonDigitalCash, BlackScholesMertonDigitalAsset
)


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


@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaBsmPricer:
    """
    Adapter pricer: EuropeanFxVanillaOption -> BlackScholesMertonEuropean (generic carry).

    FX mapping
    ----------
    - r = r_d (domestic rate)
    - b = r_d - r_f  (cost-of-carry)
    - engine PV is "per 1 unit of foreign notional"
    - we multiply by notional to get domestic PV

    Greeks mapping (important!)
    ---------------------------
    The generic engine returns:
      - rho: dPV/dr holding b fixed
      - rho_carry: dPV/db holding r fixed  (name in engine may differ)

    For FX:
      b = r_d - r_f,  r = r_d

    Chain rule:
      dPV/dr_d = dPV/dr * (dr/dr_d) + dPV/db * (db/dr_d) = rho + rho_carry
      dPV/dr_f = dPV/db * (db/dr_f) = -rho_carry
    """

    # define model engine to use.
    engine: BlackScholesMertonVanilla = BlackScholesMertonVanilla()

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        # ---- Read market inputs ----
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Discount factors from curves (domestic and foreign).
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs -> continuous rates for the engine.
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)

        # Generic carry parameter for the engine.
        b = float(r_d - r_f)

        # Implied vol from the surface.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        # ---- Engine PV is per 1 unit foreign; scale by notional_foreign ----
        pv_per_unit = self.engine.price(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        # ---- Read market inputs ----
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        # ---- Engine greeks are per 1 unit foreign; scale by notional_foreign ----
        g = self.engine.greeks(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )

        notional = float(trade.notional)

        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])

        # Engine provides:
        # - rho (wrt discount_rate r, holding carry fixed)
        # - rho_carry (wrt carry_rate b, holding r fixed)
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

    FX mapping
    ----------
    discount_rate = r_d
    carry         = r_d - r_f
    """

    cash_engine: BlackScholesMertonDigitalCash = BlackScholesMertonDigitalCash()  # Cash digital engine.
    asset_engine: BlackScholesMertonDigitalAsset = BlackScholesMertonDigitalAsset()  # Asset digital engine.

    def price(self, trade: EuropeanFxDigitalOption, market: Market) -> float:
        S = float(market.quote(trade.spot_id))  # Read spot.
        K = float(trade.strike)  # Read strike.
        T = float(trade.expiry)  # Read expiry.

        df_d = float(market.curve(trade.domestic_curve_id).df(T))  # Domestic DF.
        df_f = float(market.curve(trade.foreign_curve_id).df(T))  # Foreign DF.

        r_d = _rate_from_df(df=df_d, t=T)  # Domestic rate.
        r_f = _rate_from_df(df=df_f, t=T)  # Foreign rate.

        b = float(r_d - r_f)  # Carry for FX.

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))  # Implied vol.

        payout = float(trade.payout_amount)  # Payout amount (domestic if cash; foreign units if asset).

        if trade.payoff == "cash":  # Cash-or-nothing.
            pv = self.cash_engine.price(  # Price via cash digital engine.
                option_type=trade.option_type,
                spot=S,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                cash=payout,
            )
            return float(pv)  # PV already domestic.

        # Asset-or-nothing: payout is foreign units; engine returns domestic PV via S*exp(-r_f T).
        pv = self.asset_engine.price(  # Price via asset digital engine.
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
            asset_units=payout,
        )
        return float(pv)  # PV already domestic.

    def greeks(self, trade: EuropeanFxDigitalOption, market: Market) -> Dict[GreekName, float]:
        S = float(market.quote(trade.spot_id))  # Read spot.
        K = float(trade.strike)  # Read strike.
        T = float(trade.expiry)  # Read expiry.

        df_d = float(market.curve(trade.domestic_curve_id).df(T))  # Domestic DF.
        df_f = float(market.curve(trade.foreign_curve_id).df(T))  # Foreign DF.

        r_d = _rate_from_df(df=df_d, t=T)  # Domestic rate.
        r_f = _rate_from_df(df=df_f, t=T)  # Foreign rate.

        b = float(r_d - r_f)  # Carry.

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))  # Implied vol.

        payout = float(trade.payout_amount)  # Payout amount.

        if trade.payoff == "cash":  # Cash digital greeks.
            g = self.cash_engine.greeks(
                option_type=trade.option_type,
                spot=S,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                cash=payout,
            )
        else:  # Asset digital greeks.
            g = self.asset_engine.greeks(
                option_type=trade.option_type,
                spot=S,
                strike=K,
                time_to_expiry=T,
                discount_rate=r_d,
                carry=b,
                sigma=sigma,
                asset_units=payout,
            )

        # Engine greeks:
        #   rho_discount = ∂PV/∂r  holding b fixed
        #   rho_carry    = ∂PV/∂b  holding r fixed
        #
        # FX mapping:
        #   r = r_d
        #   b = r_d - r_f
        #
        # Chain rule:
        #   ∂PV/∂r_d = rho_discount*(∂r/∂r_d) + rho_carry*(∂b/∂r_d) = rho_discount + rho_carry
        #   ∂PV/∂r_f = rho_carry*(∂b/∂r_f) = -rho_carry
        rho_domestic = float(g["rho_discount"] + g["rho_carry"])  # Domestic rho.
        rho_foreign = float(-g["rho_carry"])  # Foreign rho.

        return {
            "delta": float(g["delta"]),
            "gamma": float(g["gamma"]),
            "vega": float(g["vega"]),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }