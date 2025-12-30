from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.market import Market
from src.instruments.fx.linear.forward import FxForward

# Keep greek key names explicit and type-checked.
GreekName = Literal["delta", "rho_domestic", "rho_foreign"]


@dataclass(frozen=True, slots=True)
class FxForwardPricer:
    """
    Linear FX Forward pricer.

    Product definition (domestic PV)
    -------------------------------
    For an FX forward paying/receiving foreign notional Nf at strike K:

        PV = Nf * ( S0 * df_f(T) - K * df_d(T) )

    where
      - S0 is the spot quote (domestic per 1 foreign; e.g. EURUSD is USD per 1 EUR)
      - df_d(T) is the domestic discount factor to expiry
      - df_f(T) is the foreign discount factor to expiry
      - PV is in domestic currency units

    Greeks (first-order)
    --------------------
      - delta        = dPV/dS0      = Nf * df_f(T)
      - rho_domestic = dPV/dr_d     = Nf * K * T * df_d(T)
      - rho_foreign  = dPV/dr_f     = -Nf * S0 * T * df_f(T)

    Conventions
    -----------
    - Rates are continuously compounded implicitly via df(T)=exp(-rT) in your curves.
    - Rhos are per 1.00 absolute rate (so 1bp = 0.0001).
    """

    def price(self, trade: FxForward, market: Market) -> float:
        """
        Price the FX forward.

        Returns
        -------
        float
            PV in domestic currency.
        """
        # --- Extract core trade terms ---
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Notional naming can differ across iterations; support both safely.
        # Prefer notional_foreign (explicit), otherwise fallback to notional.
        Nf = getattr(trade, "notional_foreign", None)
        if Nf is None:
            Nf = getattr(trade, "notional", None)
        if Nf is None:
            raise AttributeError("FxForward must define 'notional_foreign' or 'notional'.")
        Nf = float(Nf)

        # --- Defensive checks ---
        if not math.isfinite(S0):
            raise ValueError(f"Spot quote must be finite; got {S0}.")
        if not math.isfinite(K) or K <= 0.0:
            raise ValueError(f"Strike must be finite and > 0; got {K}.")
        if not math.isfinite(T) or T < 0.0:
            raise ValueError(f"Expiry must be finite and >= 0; got {T}.")
        if not math.isfinite(Nf):
            raise ValueError(f"Notional must be finite; got {Nf}.")

        # --- Handle expiry = 0 cleanly ---
        # At expiry, discount factors are 1, so PV = Nf*(S-K).
        if T == 0.0:
            return float(Nf * (S0 - K))

        # --- Market discount factors ---
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        if not math.isfinite(df_d) or df_d <= 0.0:
            raise ValueError(f"Domestic DF must be finite and > 0; got {df_d}.")
        if not math.isfinite(df_f) or df_f <= 0.0:
            raise ValueError(f"Foreign DF must be finite and > 0; got {df_f}.")

        # --- Pricing formula ---
        pv = Nf * (S0 * df_f - K * df_d)
        return float(pv)

    def greeks(self, trade: FxForward, market: Market) -> Dict[GreekName, float]:
        """
        Compute first-order greeks for the FX forward.

        Returns
        -------
        Dict[str, float]
            delta, rho_domestic, rho_foreign
        """
        # --- Extract core trade terms ---
        S0 = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        Nf = getattr(trade, "notional_foreign", None)
        if Nf is None:
            Nf = getattr(trade, "notional", None)
        if Nf is None:
            raise AttributeError("FxForward must define 'notional_foreign' or 'notional'.")
        Nf = float(Nf)

        # --- Defensive checks ---
        if not math.isfinite(S0):
            raise ValueError(f"Spot quote must be finite; got {S0}.")
        if not math.isfinite(K) or K <= 0.0:
            raise ValueError(f"Strike must be finite and > 0; got {K}.")
        if not math.isfinite(T) or T < 0.0:
            raise ValueError(f"Expiry must be finite and >= 0; got {T}.")
        if not math.isfinite(Nf):
            raise ValueError(f"Notional must be finite; got {Nf}.")

        # At expiry, greeks are not stable/discontinuous across payoff conventions.
        # For consistency with your other pricers, return zeros at T=0.
        if T == 0.0:
            return {"delta": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}

        # --- Market discount factors ---
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        if not math.isfinite(df_d) or df_d <= 0.0:
            raise ValueError(f"Domestic DF must be finite and > 0; got {df_d}.")
        if not math.isfinite(df_f) or df_f <= 0.0:
            raise ValueError(f"Foreign DF must be finite and > 0; got {df_f}.")

        # --- Greeks ---
        # dPV/dS0 = Nf * df_f
        delta = Nf * df_f

        # Using df(T)=exp(-rT) -> d(df)/dr = -T*df
        # PV contains -Nf*K*df_d so rho_d = Nf*K*T*df_d
        rho_domestic = Nf * K * T * df_d

        # PV contains +Nf*S0*df_f so rho_f = Nf*S0*(-T*df_f)
        rho_foreign = -Nf * S0 * T * df_f

        return {
            "delta": float(delta),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }