"""
Equity Lookback Option Instrument

Fixed and floating strike lookback options on equities.

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType
from src.instruments.core.types import LookbackType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class EquityLookbackEuropeanOption:
    """
    European equity lookback option.

    Lookback Types
    --------------
    **Fixed Strike:**
    - Call: max(S_max - K, 0)  (buy at strike, sell at maximum)
    - Put: max(K - S_min, 0)   (sell at strike, buy at minimum)

    **Floating Strike:**
    - Call: S_T - S_min  (buy at minimum, sell at terminal)
    - Put: S_max - S_T   (buy at terminal, sell at maximum)

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    option_type : OptionType
        "call" or "put"
    lookback_type : LookbackType
        "fixed_strike" or "floating_strike"
    strike : float
        Strike price (used for fixed_strike only)
    expiry : float
        Time to maturity in years
    notional : float
        Number of contracts
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
    - **Floating Strike:** Closed-form solutions exist (Goldman-Sosin-Gatto)
    - **Fixed Strike:** Closed-form solutions exist (Conze-Viswanathan)
    - **Monte Carlo:** Always applicable for discrete monitoring

    Mathematical Notes
    ------------------
    For continuous monitoring under GBM:
        S_max/S_0 and S_min/S_0 have known distributions
        involving normal and log-normal terms.

    For discrete monitoring:
        Use Monte Carlo with path simulation.

    Examples
    --------
    >>> from src.marketdata.core.ids import MarketId
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> lookback_call = EquityLookbackEuropeanOption(
    ...     ticker="AAPL",
    ...     option_type="call",
    ...     lookback_type="floating_strike",
    ...     strike=150.0,  # Ignored for floating_strike
    ...     expiry=1.0,
    ...     notional=100,
    ...     dividend_yield=0.01,
    ...     spot_id=spot_id,
    ...     vol_id=vol_id,
    ...     curve_id=curve_id,
    ... )
    """

    # Instrument identifiers
    ticker: str

    # Option specification
    option_type: OptionType          # "call" or "put"
    lookback_type: LookbackType      # "fixed_strike" or "floating_strike"

    # Contract terms
    strike: float                    # Used for fixed_strike
    expiry: float                    # Year fraction
    notional: float                  # Number of contracts

    # Equity-specific: dividend yield
    dividend_yield: float

    # Market identifiers
    spot_id: MarketId = MarketId("EQ", "SPOT", "UNKNOWN")
    vol_id: MarketId = MarketId("EQ", "VOL", "UNKNOWN")
    curve_id: MarketId = MarketId("IR", "CURVE", "UNKNOWN")

    def __post_init__(self) -> None:
        """Validate instrument parameters."""
        # Validate ticker
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string.")

        # Validate option type
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate lookback type
        if self.lookback_type not in ("fixed_strike", "floating_strike"):
            raise ValueError("lookback_type must be 'fixed_strike' or 'floating_strike'.")

        # Validate numeric parameters
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")

        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.vol_id, MarketId):
            raise ValueError("vol_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")

    @property
    def uses_strike(self) -> bool:
        """
        Check if this lookback type uses the strike parameter.

        Returns
        -------
        bool
            True for fixed_strike, False for floating_strike
        """
        return self.lookback_type == "fixed_strike"
