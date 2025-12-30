from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.market import Market
from src.instruments.fx.linear.spot import FxSpot

# Keep greek key names explicit and type-checked.
GreekName = Literal["delta"]


@dataclass(frozen=True, slots=True)
class FxSpotPricer:
    """
    Linear FX Spot pricer.

    Instrument
    ----------
    FxSpot represents a position in the FX spot quote itself:
        PV = quantity * contract_multiplier * S

    Conventions
    -----------
    - PV is returned in the same units as the spot quote in Market.
      (For EURUSD this is USD per 1 EUR; so a 1-unit spot position has PV ~ 1.10 USD.)
    - Delta is dPV/dS and is constant for this linear product.

    Notes
    -----
    - We keep this pricer intentionally minimal and deterministic.
    - Higher-order greeks (gamma/vega/rho) are zero for spot.
    """

    def price(self, trade: FxSpot, market: Market) -> float:
        """
        Price the FX spot instrument.

        Parameters
        ----------
        trade:
            FxSpot instrument (contains spot_id and optional contract multiplier).
        market:
            Market snapshot providing quote(spot_id).

        Returns
        -------
        float
            PV in the same units as the spot quote.
        """
        # Read the spot quote from the market snapshot.
        spot = float(market.quote(trade.spot_id))

        # Defensive checks: spot must be finite.
        if not (spot == spot) or spot in (float("inf"), float("-inf")):
            raise ValueError(f"Spot quote must be finite; got {spot}.")

        # Contract multiplier allows FxSpot to represent "1 unit" or "N units" cleanly.
        # If your FxSpot doesn't define contract_multiplier, default to 1.0.
        contract_multiplier = float(getattr(trade, "contract_multiplier", 1.0))

        # Linear PV.
        pv = contract_multiplier * spot
        return float(pv)

    def greeks(self, trade: FxSpot, market: Market) -> Dict[GreekName, float]:
        """
        Spot Greeks.

        Returns
        -------
        Dict[str, float]
            delta: dPV/dS = contract_multiplier
        """
        # Delta is constant for a linear spot instrument.
        contract_multiplier = float(getattr(trade, "contract_multiplier", 1.0))
        return {"delta": float(contract_multiplier)}