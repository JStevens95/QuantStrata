"""
Equity Forward Pricer

Pricer for equity forward contracts with continuous dividend yield.

Author: QuantStrata Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.instruments.equity.linear.forward import EquityForward
from src.marketdata.core.market import Market

GreekName = Literal["delta", "gamma", "vega", "rho", "theta"]


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    df = exp(-r × T) => r = -ln(df) / T

    Parameters
    ----------
    df : float
        Discount factor
    t : float
        Time to maturity

    Returns
    -------
    float
        Continuously compounded rate
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class EquityForwardPricer:
    """
    Pricer for equity forward contracts.

    Pricing Formula
    ---------------
    Forward price: F = S × exp((r - q) × T)

    PV = notional × (F - K) × exp(-r × T)
       = notional × (S × exp(-q × T) - K × exp(-r × T))

    Where:
    - S = spot price
    - K = strike (delivery price)
    - r = risk-free rate
    - q = dividend yield
    - T = time to expiry

    Greeks
    ------
    - Delta = notional × exp(-q × T)
    - Gamma = 0 (linear in S after discounting)
    - Vega = 0 (no optionality)
    - Rho = notional × T × K × exp(-r × T) (sensitivity to rate)
    - Theta ≈ dPV/dt (time decay from discounting)
    """

    def price(self, trade: EquityForward, market: Market) -> float:
        """
        Calculate present value of equity forward.

        Parameters
        ----------
        trade : EquityForward
            Forward contract to price
        market : Market
            Market snapshot with spot and curve

        Returns
        -------
        float
            Present value in currency units
        """
        # Read market data
        S = float(market.quote(trade.spot_id))
        T = float(trade.expiry)
        K = float(trade.strike)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Validate
        if S <= 0.0:
            raise ValueError(f"Spot must be > 0, got {S}")
        if T < 0.0:
            raise ValueError("Expiry must be >= 0")

        # At expiry, PV = notional × (S - K)
        if T == 0.0:
            return notional * (S - K)

        # Get discount factor and rate
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        # PV = notional × (S × exp(-q×T) - K × exp(-r×T))
        pv_per_unit = S * math.exp(-q * T) - K * math.exp(-r * T)

        return notional * pv_per_unit

    def forward_price(self, trade: EquityForward, market: Market) -> float:
        """
        Calculate the fair forward price.

        F = S × exp((r - q) × T)

        Parameters
        ----------
        trade : EquityForward
            Forward contract
        market : Market
            Market snapshot

        Returns
        -------
        float
            Fair forward price
        """
        S = float(market.quote(trade.spot_id))
        T = float(trade.expiry)
        q = float(trade.dividend_yield)

        if T <= 0.0:
            return S

        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        return S * math.exp((r - q) * T)

    def greeks(self, trade: EquityForward, market: Market) -> Dict[GreekName, float]:
        """
        Calculate Greeks for forward contract.

        Parameters
        ----------
        trade : EquityForward
            Forward contract
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary
        """
        S = float(market.quote(trade.spot_id))
        T = float(trade.expiry)
        K = float(trade.strike)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # At expiry, use limit values
        if T <= 0.0:
            return {
                "delta": notional,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        # Delta = notional × exp(-q × T)
        # (Change in PV per unit change in spot)
        delta = notional * math.exp(-q * T)

        # Rho = notional × T × K × exp(-r × T)
        # (Change in PV per unit change in rate)
        rho = notional * T * K * math.exp(-r * T)

        # Theta: rate of change of PV as time passes (T decreases)
        # PV = notional × (S×exp(-qT) - K×exp(-rT))
        # ∂PV/∂T = notional × (-q×S×exp(-qT) + r×K×exp(-rT))
        # Theta = -∂PV/∂T (since theta measures value change as time passes, not as T increases)
        # Theta = notional × (q×S×exp(-qT) - r×K×exp(-rT))
        # 
        # For a long forward (notional > 0):
        # - As time passes, discounting effect makes K worth more in PV (negative contribution)
        # - As time passes, dividend yield effect reduces forward price (positive contribution for long)
        theta = notional * (q * S * math.exp(-q * T) - r * K * math.exp(-r * T))

        return {
            "delta": float(delta),
            "gamma": 0.0,
            "vega": 0.0,
            "rho": float(rho),
            "theta": float(theta),
        }
