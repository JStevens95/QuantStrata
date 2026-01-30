# src/instruments/equity/options/futures.py
"""
Equity Index Futures Options.

Options on equity index futures, priced using the Black76 model.

Mathematical Framework
----------------------
An equity index futures option is an option on a futures contract.
The futures price F at time t for delivery at T is typically:

    F(t, T) = S(t) × exp((r - q) × (T - t))

Where:
    S = spot index level
    r = risk-free rate
    q = dividend yield

However, for futures options, we often use the futures price directly
since it's observable in the market.

The option payoff at expiry:
    Call: max(F_T - K, 0)
    Put:  max(K - F_T, 0)

Use Cases
---------
- S&P 500 futures options (ES options)
- E-mini options
- Index futures volatility trading
- Delta-one hedging instruments

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class EuropeanEquityFuturesOption:
    """
    European option on an equity index futures contract.

    This instrument represents an option on a futures contract.
    Priced using Black76.

    Parameters
    ----------
    ticker : str
        Index/futures ticker (e.g., "SPX", "ES", "NQ").
    option_type : OptionType
        "call" or "put".
    strike : float
        Strike price (in index points).
    expiry : float
        Time to option expiry in years.
    futures_expiry : float
        Time to futures delivery in years (>= option expiry).
    notional : float
        Contract multiplier (e.g., $50 for E-mini S&P 500).
    spot_id : MarketId
        Market identifier for spot index level.
    vol_id : MarketId
        Market identifier for volatility surface.
    curve_id : MarketId
        Market identifier for discount curve.
    dividend_yield : float
        Continuous dividend yield for forward calculation.
        Set to 0 if using futures price directly.

    Conventions
    -----------
    - Strike is in index points
    - PV is returned in currency units (notional × option value)

    Forward Calculation
    -------------------
    The futures price used for pricing is computed as:
        F = S × exp((r - q) × T_fut)

    If futures price is directly observable, can use
    EuropeanEquityFuturesOptionSimple instead.

    Examples
    --------
    >>> # Option on S&P 500 futures
    >>> opt = EuropeanEquityFuturesOption(
    ...     ticker="SPX",
    ...     option_type="call",
    ...     strike=5000.0,
    ...     expiry=0.25,        # 3 month option
    ...     futures_expiry=0.25, # Same expiry
    ...     notional=50.0,       # E-mini multiplier
    ...     spot_id=spx_spot_id,
    ...     vol_id=spx_vol_id,
    ...     curve_id=usd_curve_id,
    ...     dividend_yield=0.015,
    ... )
    """

    ticker: str
    option_type: OptionType
    strike: float
    expiry: float
    futures_expiry: float
    notional: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId
    dividend_yield: float

    def __post_init__(self) -> None:
        """Validate inputs."""
        # Ticker validation.
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Option type validation.
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Strike validation.
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Expiry validation.
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Futures expiry validation.
        if float(self.futures_expiry) < 0.0:
            raise ValueError("futures_expiry must be >= 0.")

        if float(self.futures_expiry) < float(self.expiry):
            raise ValueError("futures_expiry must be >= expiry.")

        # Notional validation.
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Dividend yield validation.
        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # MarketId type validation.
        if not isinstance(self.spot_id, MarketId):
            raise TypeError("spot_id must be a MarketId.")
        if not isinstance(self.vol_id, MarketId):
            raise TypeError("vol_id must be a MarketId.")
        if not isinstance(self.curve_id, MarketId):
            raise TypeError("curve_id must be a MarketId.")


@dataclass(frozen=True, slots=True)
class EuropeanEquityFuturesOptionSimple:
    """
    Simplified equity futures option for direct Black76 pricing.

    Use this when you have the futures price and vol directly,
    without needing to compute them from spot and curves.

    Parameters
    ----------
    ticker : str
        Index/futures ticker.
    option_type : OptionType
        "call" or "put".
    strike : float
        Strike price.
    expiry : float
        Time to expiry in years.
    futures_price : float
        Current futures price F.
    vol : float
        Futures volatility σ (log-normal).
    discount_factor : float
        Discount factor DF = exp(-r × T).
    notional : float
        Contract multiplier.

    Examples
    --------
    >>> # Direct Black76 pricing with known futures
    >>> opt = EuropeanEquityFuturesOptionSimple(
    ...     ticker="ES",
    ...     option_type="call",
    ...     strike=5000.0,
    ...     expiry=0.25,
    ...     futures_price=5050.0,
    ...     vol=0.18,
    ...     discount_factor=0.9875,
    ...     notional=50.0,
    ... )
    """

    ticker: str
    option_type: OptionType
    strike: float
    expiry: float
    futures_price: float
    vol: float
    discount_factor: float
    notional: float

    def __post_init__(self) -> None:
        """Validate inputs."""
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.futures_price) <= 0.0:
            raise ValueError("futures_price must be > 0.")

        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")

        if float(self.discount_factor) <= 0.0 or float(self.discount_factor) > 1.0:
            raise ValueError("discount_factor must be in (0, 1].")

        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")


__all__ = [
    "EuropeanEquityFuturesOption",
    "EuropeanEquityFuturesOptionSimple",
]
