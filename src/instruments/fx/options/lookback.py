"""
European FX Lookback Option instrument definition.

A lookback option's payoff depends on the maximum or minimum price of the underlying
asset over the option's life. This provides "perfect hindsight" - you always get
the optimal entry or exit point.

Two main variants exist:
- Floating strike: Strike is set to min (call) or max (put) of path
- Fixed strike: Payoff depends on max (call) or min (put) of path vs fixed strike
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
class FxLookbackEuropeanOption:
    """
    European FX Lookback option (path-dependent exotic).

    A lookback option pays based on the maximum or minimum price of the underlying
    over the option's life, providing "perfect hindsight" - the holder always gets
    the optimal entry or exit point.

    Conventions
    -----------
    - Underlying spot S is "domestic per 1 foreign" (e.g. USD per EUR for EURUSD).
    - `notional` is in *foreign* currency units.
    - PV is returned in *domestic* currency units.
    - Extrema are computed over discrete monitoring points.

    Lookback Types
    --------------
    **Floating Strike** (strike is determined by path extremum):
    - Call: Payoff = S_T - min(S_t)  (strike is the minimum spot)
    - Put:  Payoff = max(S_t) - S_T  (strike is the maximum spot)
    - Note: Floating strike lookbacks are ALWAYS in-the-money (payoff >= 0)

    **Fixed Strike** (payoff depends on path extremum vs fixed strike K):
    - Call: Payoff = max(max(S_t) - K, 0)  (payoff on maximum spot)
    - Put:  Payoff = max(K - min(S_t), 0)  (payoff on minimum spot)

    Key Properties
    --------------
    - "No regret" options: holder captures optimal entry/exit
    - Always more expensive than standard European options
    - Floating strike lookbacks are always ITM (payoff >= 0)
    - Closed-form solutions exist for continuous monitoring (Goldman-Sosin-Gatto)
    - Discrete monitoring requires numerical methods (MC)

    Interview Points
    ----------------
    - Lookback options are always >= vanilla value (captures optimal timing)
    - Floating strike lookback call delta at inception = 2 (sensitive to extremum)
    - Continuous monitoring formula: more complex than BSM
    - Discrete vs continuous monitoring creates "continuation value" premium

    Notes
    -----
    - This is a pure product definition: no model choice here (MC lives in pricers).
    - Monitoring points are determined by the pricer's time step configuration.
    - For discrete monitoring, more steps = better approximation to continuous.
    """

    option_type: OptionType  # "call" or "put"
    notional: float  # Foreign currency units
    expiry: float  # Year fraction to expiry T

    # Market identifiers
    spot_id: MarketId  # FX spot rate identifier
    vol_id: MarketId  # Volatility surface identifier
    domestic_curve_id: MarketId  # Domestic interest rate curve
    foreign_curve_id: MarketId  # Foreign interest rate curve

    # "floating_strike" or "fixed_strike"
    lookback_type: LookbackType = "floating_strike"

    # Strike is only used for fixed_strike lookbacks
    # For floating_strike, strike is determined by path extremum
    strike: float = 0.0  # Strike price K (only relevant for fixed_strike)

    def __post_init__(self) -> None:
        """
        Validate instrument parameters.

        This validation ensures that pricers receive well-formed instruments and can
        assume reasonable input values (e.g., positive strikes for fixed type).
        """
        # Validate option type: must be "call" or "put"
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate notional: must be non-zero (zero notional would give zero PV)
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Validate expiry: must be non-negative (negative expiry would be in the past)
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate lookback type: must be "floating_strike" or "fixed_strike"
        if self.lookback_type not in ("floating_strike", "fixed_strike"):
            raise ValueError("lookback_type must be 'floating_strike' or 'fixed_strike'.")

        # Validate strike for fixed_strike lookbacks
        if self.lookback_type == "fixed_strike":
            if float(self.strike) <= 0.0:
                raise ValueError("strike must be > 0 for fixed_strike lookback.")
