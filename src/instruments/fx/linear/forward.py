from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId


@dataclass(frozen=True, slots=True)
class FxForward:
    """
    FX forward instrument.

    Parameters
    ----------
    notional:
        Foreign (base currency) notional. Example: EUR notional for EURUSD.
    strike:
        Forward strike in domestic per foreign (e.g., USD per EUR).
    expiry:
        Time to expiry in years.
    spot_id:
        MarketId of FX spot quote.
    domestic_curve_id:
        Domestic discount curve MarketId (e.g., USD curve for EURUSD).
    foreign_curve_id:
        Foreign discount curve MarketId (e.g., EUR curve for EURUSD).

    Notes
    -----
    This instrument definition stays generic and stable even as pricing models evolve.
    """
    notional: float
    strike: float
    expiry: float
    spot_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")