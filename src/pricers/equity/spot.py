"""
Equity Spot Pricer

Simple pricer for spot equity positions.

Author: QuantStrata Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from src.instruments.equity.linear.spot import EquitySpot
from src.marketdata.core.market import Market

GreekName = Literal["delta", "gamma", "vega", "rho", "theta"]


@dataclass(frozen=True, slots=True)
class EquitySpotPricer:
    """
    Pricer for equity spot positions.

    Pricing
    -------
    PV = quantity × spot_price

    Greeks
    ------
    - Delta = quantity (linear exposure)
    - Gamma = 0 (linear instrument)
    - Vega = 0 (no optionality)
    - Rho = 0 (spot position, no discounting)
    - Theta = 0 (no time decay)
    """

    def price(self, trade: EquitySpot, market: Market) -> float:
        """
        Calculate present value of spot equity position.

        Parameters
        ----------
        trade : EquitySpot
            The spot position to price
        market : Market
            Market snapshot with spot price

        Returns
        -------
        float
            Present value in currency units
        """
        # Read spot price from market
        spot = float(market.quote(trade.spot_id))

        # Validate spot price
        if spot <= 0.0:
            raise ValueError(f"Spot price must be > 0, got {spot}")

        # PV = quantity × spot
        return float(trade.quantity) * spot

    def greeks(self, trade: EquitySpot, market: Market) -> Dict[GreekName, float]:
        """
        Calculate Greeks for spot position.

        For a linear spot position:
        - Delta = quantity (per dollar move in spot)
        - All other Greeks are zero

        Parameters
        ----------
        trade : EquitySpot
            The spot position
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary
        """
        return {
            "delta": float(trade.quantity),
            "gamma": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "theta": 0.0,
        }
