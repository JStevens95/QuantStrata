"""
Equity Barrier Option Instrument

Single-barrier options on equities with continuous dividend yield.

Author: QuantStrata Team
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType, BarrierStyle, BarrierDirection


@dataclass(frozen=True, slots=True)
class EquityBarrierEuropeanOption:
    """
    European equity single-barrier option.

    Barrier Types
    -------------
    - Up-and-Out: Knocked out if spot rises above barrier
    - Up-and-In: Knocked in if spot rises above barrier
    - Down-and-Out: Knocked out if spot falls below barrier
    - Down-and-In: Knocked in if spot falls below barrier

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL")
    option_type : OptionType
        "call" or "put"
    barrier_direction : BarrierDirection
        "up" or "down"
    barrier_style : BarrierStyle
        "knock_out" or "knock_in"
    strike : float
        Strike price
    barrier_level : float
        Barrier level
    expiry : float
        Time to maturity in years
    notional : float
        Number of shares
    dividend_yield : float
        Continuous dividend yield
    rebate_amount : float
        Rebate paid at expiry if knocked out (default 0)
    spot_id : MarketId
        Market identifier for spot price
    vol_id : MarketId
        Market identifier for volatility surface
    curve_id : MarketId
        Market identifier for discount curve

    Pricing
    -------
    Barrier options are typically priced via:
    - Monte Carlo simulation (path-dependent)
    - Analytic formulas (for continuous monitoring)
    - Finite difference PDE methods

    Notes
    -----
    - Monitoring: Discrete monitoring on simulated path points (MC)
    - Rebate: Paid at expiry (not at hit time) for simplicity
    - Cost-of-carry: b = r - q (single curve + dividend yield)

    Examples
    --------
    >>> from src.marketdata.core.ids import MarketId
    >>> spot_id = MarketId(asset_class="EQ", mkt_type="SPOT", name="AAPL")
    >>> vol_id = MarketId(asset_class="EQ", mkt_type="VOL", name="AAPL")
    >>> curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD_OIS")
    >>> barrier_call = EquityBarrierEuropeanOption(
    ...     ticker="AAPL",
    ...     option_type="call",
    ...     barrier_direction="up",
    ...     barrier_style="knock_out",
    ...     strike=150.0,
    ...     barrier_level=180.0,
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
    option_type: OptionType           # "call" or "put"
    barrier_direction: BarrierDirection  # "up" or "down"
    barrier_style: BarrierStyle       # "knock_out" or "knock_in"

    # Strike and barrier
    strike: float
    barrier_level: float

    # Contract terms
    expiry: float                     # Year fraction
    notional: float                   # Number of shares

    # Equity-specific: dividend yield
    dividend_yield: float

    # Optional rebate (paid at expiry if knocked out)
    rebate_amount: float = 0.0

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

        # Validate barrier direction
        if self.barrier_direction not in ("up", "down"):
            raise ValueError("barrier_direction must be 'up' or 'down'.")

        # Validate barrier style
        if self.barrier_style not in ("knock_out", "knock_in"):
            raise ValueError("barrier_style must be 'knock_out' or 'knock_in'.")

        # Validate numeric parameters
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.barrier_level) <= 0.0:
            raise ValueError("barrier_level must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")

        if float(self.dividend_yield) < 0.0:
            raise ValueError("dividend_yield must be >= 0.")

        if float(self.rebate_amount) < 0.0:
            raise ValueError("rebate_amount must be >= 0.")

        # Validate market IDs
        if not isinstance(self.spot_id, MarketId):
            raise ValueError("spot_id must be a MarketId instance.")
        if not isinstance(self.vol_id, MarketId):
            raise ValueError("vol_id must be a MarketId instance.")
        if not isinstance(self.curve_id, MarketId):
            raise ValueError("curve_id must be a MarketId instance.")

    @property
    def barrier_type(self) -> str:
        """
        Get the combined barrier type string.

        Returns
        -------
        str
            One of: "up_and_out", "up_and_in", "down_and_out", "down_and_in"
        """
        return f"{self.barrier_direction}_and_{self.barrier_style.replace('knock_', '')}"
