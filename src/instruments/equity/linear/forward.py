"""
Equity Forward Instrument

Represents a forward contract on an equity with dividend handling.

Author: QuantStrata Team
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId


@dataclass(frozen=True, slots=True)
class EquityForward:
    """
    Equity forward contract.

    Agreement to buy/sell `notional` shares at price `strike` at time `expiry`.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    strike : float
        Forward price (delivery price)
    expiry : float
        Time to maturity in years
    notional : float
        Number of shares (positive = long forward, negative = short)
    dividend_yield : float
        Continuous dividend yield (annualized)
    spot_id : MarketId
        Market identifier for spot price
    curve_id : MarketId
        Market identifier for discount curve

    Pricing
    -------
    Forward price: F = S × exp((r - q) × T)

    PV = notional × (F - K) × exp(-r × T)
       = notional × (S × exp(-q × T) - K × exp(-r × T))

    Where:
    - S = spot price
    - K = strike (delivery price)
    - r = risk-free rate
    - q = dividend yield
    - T = time to expiry

    Examples
    --------
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> fwd = EquityForward(
    ...     ticker="AAPL",
    ...     strike=150.0,
    ...     expiry=1.0,
    ...     notional=100,
    ...     dividend_yield=0.005,
    ...     spot_id=spot_id,
    ...     curve_id=curve_id,
    ... )
    """

    ticker: str
    strike: float
    expiry: float
    notional: float
    dividend_yield: float
    spot_id: MarketId
    curve_id: MarketId

    def __post_init__(self) -> None:
        # Validate ticker
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Validate strike
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate expiry
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate notional (can be zero)
        if not isinstance(self.notional, (int, float)):
            raise ValueError("notional must be a number.")

        # Validate dividend_yield
        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")
