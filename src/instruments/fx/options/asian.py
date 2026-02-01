from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType
from src.instruments.core.types import AsianAveragingType


@dataclass(frozen=True, slots=True)
class FxAsianEuropeanOption:
    """
    European FX Asian option (average price option).

    An Asian option pays based on the average price of the underlying over a specified
    period, rather than the terminal price. This reduces volatility and makes the option
    cheaper than a standard European option.

    Conventions
    -----------
    - Underlying spot S is "domestic per 1 foreign" (e.g. USD per EUR for EURUSD).
    - `notional` is in *foreign* currency units.
    - PV is returned in *domestic* currency units.
    - Average is computed over discrete monitoring points (daily, weekly, monthly).

    Averaging Types
    ---------------
    - arithmetic: Average = (S_1 + S_2 + ... + S_n) / n
    - geometric: Average = (S_1 * S_2 * ... * S_n)^(1/n)

    Notes
    -----
    - This is a pure product definition: no model choice here (MC/FD live in pricers/models).
    - Geometric averaging has closed-form solutions, but arithmetic is more common in practice.
    - For arithmetic averaging, MC pricing is standard (no closed-form exists).
    - Monitoring points are determined by the pricer's time step configuration.
    """

    option_type: OptionType  # "call" or "put"
    notional: float  # Foreign currency units
    strike: float  # Strike price K
    expiry: float  # Year fraction to expiry T

    # Market identifiers
    spot_id: MarketId  # FX spot rate identifier
    vol_id: MarketId  # Volatility surface identifier
    domestic_curve_id: MarketId  # Domestic interest rate curve
    foreign_curve_id: MarketId  # Foreign interest rate curve

    averaging_type: AsianAveragingType = "arithmetic"  # "arithmetic" or "geometric"

    def __post_init__(self) -> None:
        """
        Validate instrument parameters.

        This validation ensures that pricers receive well-formed instruments and can
        assume reasonable input values (e.g., positive strikes, non-negative expiry).
        """
        # Validate option type: must be "call" or "put"
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate notional: must be non-zero (zero notional would give zero PV)
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Validate strike: must be positive (negative strikes are not economically meaningful)
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Validate expiry: must be non-negative (negative expiry would be in the past)
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate averaging type: must be "arithmetic" or "geometric"
        if self.averaging_type not in ("arithmetic", "geometric"):
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'.")
