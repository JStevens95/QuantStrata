"""
Equity Asian Option Instrument

Average price/strike options on equities with continuous dividend yield.

Author: QuantStrata Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType
from src.instruments.core.types import AsianAveragingType


@dataclass(frozen=True, slots=True)
class EuropeanEquityAsianOption:
    """
    European equity Asian option (average price).

    Averaging Types
    ---------------
    **Arithmetic Average:**
        A = (1/n) × Σ S_i
        Payoff (call): max(A - K, 0)
        Payoff (put): max(K - A, 0)

    **Geometric Average:**
        A = (∏ S_i)^(1/n) = exp((1/n) × Σ ln(S_i))
        Payoff (call): max(A - K, 0)
        Payoff (put): max(K - A, 0)

    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    option_type : OptionType
        "call" or "put"
    averaging_type : AveragingType
        "arithmetic" or "geometric"
    strike : float
        Strike price
    expiry : float
        Time to maturity in years
    notional : float
        Number of contracts
    n_averaging : int
        Number of averaging observations
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
    - **Geometric:** Closed-form adjustment (log-average is normal)
    - **Arithmetic:** Monte Carlo simulation (no closed form)

    Mathematical Notes
    ------------------
    For geometric averaging, under GBM:
        ln(A_geo) ~ N(μ_A, σ_A²)
    where adjusted parameters account for averaging effect.

    For arithmetic averaging:
        A_arith > A_geo (Jensen's inequality)
        No closed form; use MC or moment matching.

    Examples
    --------
    >>> from src.marketdata.core.ids import MarketId
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> asian_call = EuropeanEquityAsianOption(
    ...     ticker="AAPL",
    ...     option_type="call",
    ...     averaging_type="arithmetic",
    ...     strike=150.0,
    ...     expiry=1.0,
    ...     notional=100,
    ...     n_averaging=12,  # Monthly averaging
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
    averaging_type: AsianAveragingType    # "arithmetic" or "geometric"

    # Contract terms
    strike: float
    expiry: float                    # Year fraction
    notional: float                  # Number of contracts
    n_averaging: int                 # Number of averaging points

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

        # Validate averaging type
        if self.averaging_type not in ("arithmetic", "geometric"):
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'.")

        # Validate numeric parameters
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")

        if int(self.n_averaging) < 1:
            raise ValueError("n_averaging must be >= 1.")

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
    def averaging_period(self) -> float:
        """
        Get the time between averaging observations.

        Returns
        -------
        float
            Time in years between observations
        """
        if self.n_averaging <= 1:
            return self.expiry
        return self.expiry / self.n_averaging
