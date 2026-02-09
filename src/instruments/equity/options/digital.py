"""
Equity Digital Option Instrument

Cash-or-nothing and asset-or-nothing digital options on equities.

Author: QuantStrata Team
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType
from src.instruments.core.types import DigitalType

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class EquityDigitalEuropeanOption:
    """
    European equity digital (binary) option.

    Payoff Types
    ------------
    **Cash-or-Nothing:**
    - Call: Pays `payout` if S_T > K, else 0
    - Put: Pays `payout` if S_T < K, else 0

    **Asset-or-Nothing:**
    - Call: Pays S_T if S_T > K, else 0
    - Put: Pays S_T if S_T < K, else 0

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    option_type : OptionType
        "call" or "put"
    digital_type : DigitalType
        "cash" (cash-or-nothing) or "asset" (asset-or-nothing)
    strike : float
        Strike price
    expiry : float
        Time to maturity in years
    notional : float
        Number of contracts
    payout : float
        Cash payout amount (for cash digital, ignored for asset digital)
    dividend_yield : float
        Continuous dividend yield
    spot_id : MarketId
        Market identifier for spot price
    vol_id : MarketId
        Market identifier for volatility surface
    curve_id : MarketId
        Market identifier for discount curve

    Pricing (BSM Closed-Form)
    -------------------------
    Cash-or-Nothing Call:
        V = payout × exp(-rT) × N(d2)

    Cash-or-Nothing Put:
        V = payout × exp(-rT) × N(-d2)

    Asset-or-Nothing Call:
        V = S × exp(-qT) × N(d1)

    Asset-or-Nothing Put:
        V = S × exp(-qT) × N(-d1)

    Where:
        d1 = (ln(S/K) + (r-q+σ²/2)T) / (σ√T)
        d2 = d1 - σ√T

    Examples
    --------
    >>> from src.marketdata.core.ids import MarketId
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> cash_digital = EquityDigitalEuropeanOption(
    ...     ticker="AAPL",
    ...     option_type="call",
    ...     digital_type="cash",
    ...     strike=150.0,
    ...     expiry=0.5,
    ...     notional=1000,
    ...     payout=10.0,  # Pays $10 per contract if ITM
    ...     dividend_yield=0.01,
    ...     spot_id=spot_id,
    ...     vol_id=vol_id,
    ...     curve_id=curve_id,
    ... )
    """

    # Instrument identifiers
    ticker: str

    # Option specification
    option_type: OptionType      # "call" or "put"
    digital_type: DigitalType    # "cash" or "asset"

    # Contract terms
    strike: float
    expiry: float                # Year fraction
    notional: float              # Number of contracts

    # Digital payout (for cash-or-nothing)
    payout: float

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

        # Validate digital type
        if self.digital_type not in ("cash", "asset"):
            raise ValueError("digital_type must be 'cash' or 'asset'.")

        # Validate numeric parameters
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")

        if float(self.payout) < 0.0:
            raise ValueError("payout must be >= 0.")

        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.vol_id, MarketId):
            raise ValueError("vol_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")
