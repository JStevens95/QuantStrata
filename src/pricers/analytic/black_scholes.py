from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.market import Market
from src.instruments.fx.options.european import EuropeanFxOption

# Keep the Greek key names explicit and type-checked.
GreekName = Literal["delta", "gamma", "vega"]


@dataclass(frozen=True, slots=True)
class BlackScholesPricer:
    """
    Analytic European FX option pricer (Garman–Kohlhagen / Black–Scholes).

    Design choices
    --------------
    - `price(...)` returns PV only (not mixed with Greeks).
    - `greeks(...)` returns *only* Greeks (delta/gamma/vega).
    - Uses continuous compounding for rates (consistent with your FlatDiscountCurve).
    - Uses a flat implied vol surface (for V1), but the interface supports full surfaces.

    Inputs from Market
    ------------------
    - Spot quote S from market.quote(trade.spot_id)
    - Domestic DF df_d(T) from market.curve(trade.domestic_curve_id).df(T)
    - Foreign DF df_f(T) from market.curve(trade.foreign_curve_id).df(T)
    - Vol sigma from market.vol_surface(trade.vol_id).vol(expiry=T, strike=K)

    Conventions
    -----------
    - PV is returned in *domestic currency*.
    - Vega is returned per 1.00 absolute vol (e.g., sigma from 0.12 -> 0.13 is +0.01).
      If you want "per 1 vol point" = per 1%, use vega_per_1pct = vega * 0.01.
    """

    def price(self, trade: EuropeanFxOption, market: Market) -> float:
        """
        Compute the option PV (domestic currency).

        We keep PV separate from Greeks to avoid mixing concepts and
        to keep the public API clean.
        """
        # Extract model inputs from the market snapshot.
        S = float(market.quote(trade.spot_id))  # spot FX
        K = float(trade.strike)                 # strike
        T = float(trade.expiry)                 # time to expiry in years

        if S <= 0.0:
            raise ValueError(f"Spot must be > 0; got {S}.")
        if K <= 0.0:
            raise ValueError(f"Strike must be > 0; got {K}.")
        if T < 0.0:
            raise ValueError(f"Expiry must be >= 0; got {T}.")

        # Handle expiry == 0 with intrinsic value to avoid numerical issues.
        if T == 0.0:
            intrinsic = max(S - K, 0.0) if trade.option_type == "call" else max(K - S, 0.0)
            return float(trade.notional * intrinsic)

        # Discount factors from curves (continuous compounding under the hood).
        df_d = float(market.curve(trade.domestic_curve_id).df(T))  # domestic DF
        df_f = float(market.curve(trade.foreign_curve_id).df(T))   # foreign DF

        # Implied vol from the volatility surface (flat surface in V1).
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError(f"Implied vol must be > 0; got {sigma}.")

        # Convert discount factors to continuously-compounded rates for d1/d2.
        # df = exp(-rT) => r = -log(df)/T
        r_d = -math.log(df_d) / T
        r_f = -math.log(df_f) / T

        # Compute d1/d2 for Garman–Kohlhagen.
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        # Standard normal CDF values used in analytic pricing.
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)

        # Price formulas (domestic PV):
        # Call:  N * ( S*df_f*N(d1) - K*df_d*N(d2) )
        # Put:   N * ( K*df_d*N(-d2) - S*df_f*N(-d1) )
        if trade.option_type == "call":
            pv = trade.notional * (S * df_f * Nd1 - K * df_d * Nd2)
        else:
            pv = trade.notional * (K * df_d * _norm_cdf(-d2) - S * df_f * _norm_cdf(-d1))

        return float(pv)

    def greeks(self, trade: EuropeanFxOption, market: Market) -> Dict[GreekName, float]:
        """
        Compute key Greeks for the option.

        Returns a dict containing:
        - delta : dPV/dS
        - gamma : d²PV/dS²
        - vega  : dPV/dsigma (per 1.00 absolute vol)

        Notes
        -----
        At expiry (T==0), Greeks are not well-defined; we return zeros for stability.
        """
        # Read inputs from the market snapshot (same as price()).
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        if S <= 0.0:
            raise ValueError(f"Spot must be > 0; got {S}.")
        if K <= 0.0:
            raise ValueError(f"Strike must be > 0; got {K}.")
        if T < 0.0:
            raise ValueError(f"Expiry must be >= 0; got {T}.")

        # At expiry we return zeros to avoid noisy/discontinuous Greeks.
        if T == 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError(f"Implied vol must be > 0; got {sigma}.")

        # Convert DFs to continuous rates for d1/d2.
        r_d = -math.log(df_d) / T
        r_f = -math.log(df_f) / T

        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)

        # Standard normal PDF appears in vega/gamma.
        n_d1 = _norm_pdf(d1)

        # N(d1) is used in delta.
        Nd1 = _norm_cdf(d1)

        # Delta (domestic PV sensitivity to spot).
        # Call:  N * df_f * N(d1)
        # Put:   N * df_f * (N(d1) - 1)
        if trade.option_type == "call":
            delta = trade.notional * (df_f * Nd1)
        else:
            delta = trade.notional * (df_f * (Nd1 - 1.0))

        # Vega (per 1.00 vol): N * S * df_f * n(d1) * sqrt(T)
        vega = trade.notional * (S * df_f * n_d1 * sqrtT)

        # Gamma: N * df_f * n(d1) / (S * sigma * sqrt(T))
        gamma = trade.notional * (df_f * n_d1 / (S * sigma * sqrtT))

        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return float(math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf (fast + stable)."""
    return float(0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))