"""
Equity Vanilla Options

European and American vanilla options on equities.

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class EquityVanillaEuropeanOption:
    """
    European vanilla equity option (call/put).

    Payoff at expiry: max(±(S_T - K), 0) × notional

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "MSFT")
    option_type : OptionType
        "call" or "put"
    strike : float
        Strike price
    expiry : float
        Time to expiry in years
    notional : float
        Number of shares (contract multiplier)
    dividend_yield : float
        Continuous dividend yield (annualized)
    spot_id : MarketId
        Market identifier for spot price
    vol_id : MarketId
        Market identifier for volatility surface
    curve_id : MarketId
        Market identifier for discount curve (risk-free rate)

    Pricing
    -------
    Uses Black-Scholes-Merton with cost-of-carry:
    - b = r - q (where r = risk-free rate, q = dividend yield)
    - PV = notional × BSM(S, K, T, r, b, σ)

    The dividend yield affects:
    - Forward price: F = S × exp((r - q) × T)
    - Option delta: exp(-qT) × N(d1) for calls

    Examples
    --------
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> option = EquityVanillaEuropeanOption(
    ...     ticker="AAPL",
    ...     option_type="call",
    ...     strike=150.0,
    ...     expiry=1.0,
    ...     notional=100,
    ...     dividend_yield=0.005,
    ...     spot_id=spot_id,
    ...     vol_id=vol_id,
    ...     curve_id=curve_id,
    ... )
    """

    ticker: str
    option_type: OptionType
    strike: float
    expiry: float
    notional: float
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId

    def __post_init__(self) -> None:
        # Validate ticker
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Validate option_type
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate strike
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate expiry
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate notional
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Validate dividend_yield
        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.vol_id, MarketId):
            raise ValueError("vol_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")


@dataclass(**_DATACLASS_KW)
class EquityVanillaAmericanOption:
    """
    American vanilla equity option (call/put) with early exercise.

    Can be exercised at any time up to and including expiry.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    option_type : OptionType
        "call" or "put"
    strike : float
        Strike price
    expiry : float
        Time to expiry in years
    notional : float
        Number of shares
    dividend_yield : float
        Continuous dividend yield
    spot_id : MarketId
        Market identifier for spot price
    vol_id : MarketId
        Market identifier for volatility surface
    curve_id : MarketId
        Market identifier for discount curve

    Pricing
    -------
    Requires numerical methods (cannot use closed-form BSM):
    - Finite Difference with PSOR (Projected Successive Over-Relaxation)
    - Binomial/Trinomial trees
    - Longstaff-Schwartz Monte Carlo

    Early Exercise
    --------------
    - American PUT: May exercise early when ITM to receive strike immediately
    - American CALL: May exercise early before ex-dividend date to capture dividend

    Value Relationship:
    - American option ≥ European option (early exercise premium)
    - American call on non-dividend stock = European call (never early exercise)
    - American put always has early exercise premium

    Examples
    --------
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> option = EquityVanillaAmericanOption(
    ...     ticker="AAPL",
    ...     option_type="put",
    ...     strike=150.0,
    ...     expiry=1.0,
    ...     notional=100,
    ...     dividend_yield=0.005,
    ...     spot_id=spot_id,
    ...     vol_id=vol_id,
    ...     curve_id=curve_id,
    ... )
    """

    ticker: str
    option_type: OptionType
    strike: float
    expiry: float
    notional: float
    dividend_yield: float
    spot_id: MarketId
    vol_id: MarketId
    curve_id: MarketId

    def __post_init__(self) -> None:
        # Validate ticker
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Validate option_type
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate strike
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate expiry
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate notional
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Validate dividend_yield
        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.vol_id, MarketId):
            raise ValueError("vol_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")
