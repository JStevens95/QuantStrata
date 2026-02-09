"""
Equity Spot Instrument

Represents a spot position in an equity (stock).

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class EquitySpot:
    """
    Spot equity position.

    Represents holding `quantity` shares of a stock at the current spot price.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    quantity : float
        Number of shares (positive for long, negative for short)
    spot_id : MarketId
        Market identifier for spot price lookup

    Notes
    -----
    - PV = quantity × spot_price
    - Delta = quantity (per share)
    - No optionality, no expiry

    Examples
    --------
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> position = EquitySpot(ticker="AAPL", quantity=100, spot_id=spot_id)
    """

    ticker: str
    quantity: float
    spot_id: MarketId

    def __post_init__(self) -> None:
        # Validate ticker
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Validate quantity (can be zero for flat position)
        if not isinstance(self.quantity, (int, float)):
            raise ValueError("quantity must be a number.")

        # Validate spot_id
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
