"""
European FX Double Barrier Option Instrument Definition.

This module defines the `EuropeanFxDoubleBarrierOption` dataclass representing
a double barrier option with both upper and lower barriers. The option is knocked
out (or knocked in) if either barrier is hit during the option's life.

Mathematical Framework
----------------------
A double barrier option has payoff:
    - Knock-Out: max(S_T - K, 0) if L < S_t < U for all t, else rebate
    - Knock-In:  max(S_T - K, 0) if S_t hits L or U at any time, else rebate

where L is the lower barrier and U is the upper barrier.

Key Properties
--------------
- Both barriers must satisfy: L < S_0 < U (spot must start inside the corridor)
- Can be knock-out (survives if stays in corridor) or knock-in (activates if exits)
- Cheaper than single barriers (more restrictive, less likely to pay)
- Common in structured products (e.g., range accruals, turbo warrants)

Use Cases
---------
- Structured products with defined ranges
- Cheaper hedging than single barriers
- Betting on range-bound markets
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType, BarrierStyle

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**_DATACLASS_KW)
class FxDoubleBarrierEuropeanOption:
    """
    European FX double-barrier option with upper and lower barriers.

    Conventions
    -----------
    - Underlying spot S is domestic-per-foreign (e.g. EURUSD).
    - `notional` is in foreign units.
    - Any payoff returned by pricers should be in domestic currency.

    Barrier Semantics
    -----------------
    - Both barriers are monitored simultaneously.
    - Knock-out: pays vanilla payoff if NEITHER barrier hit, else rebate.
    - Knock-in:  pays vanilla payoff if EITHER barrier hit, else rebate.
    - Monitoring is discrete (at simulated path points in MC).

    Constraints
    -----------
    - Must satisfy: lower_barrier < current_spot < upper_barrier
    - Both barriers must be positive.

    Rebate
    ------
    - `rebate_amount` is a domestic-currency amount paid at expiry.
    - Interpreted as "per unit foreign notional" so pricers can scale.

    Parameters
    ----------
    option_type : OptionType
        "call" or "put" determining vanilla payoff at expiry.
    notional : float
        Notional amount in foreign currency units.
    strike : float
        Strike price for the vanilla payoff.
    expiry : float
        Time to expiry in year fractions.
    barrier_style : BarrierStyle
        "knock_out" (survives in corridor) or "knock_in" (activates on breach).
    lower_barrier : float
        Lower barrier level (must be < current spot).
    upper_barrier : float
        Upper barrier level (must be > current spot).
    rebate_amount : float, optional
        Rebate paid at expiry if barrier condition not satisfied. Default 0.0.
    spot_id : MarketId
        Market identifier for spot price.
    vol_id : MarketId
        Market identifier for volatility.
    domestic_curve_id : MarketId
        Market identifier for domestic discount curve.
    foreign_curve_id : MarketId
        Market identifier for foreign discount curve.

    Examples
    --------
    >>> from src.instruments.fx.options.double_barrier import FxDoubleBarrierEuropeanOption
    >>> from src.marketdata.core.ids import MarketId
    >>> option = FxDoubleBarrierEuropeanOption(
    ...     option_type="call",
    ...     notional=1_000_000.0,
    ...     strike=1.10,
    ...     expiry=1.0,
    ...     barrier_style="knock_out",
    ...     lower_barrier=1.05,
    ...     upper_barrier=1.15,
    ...     spot_id=MarketId("FX", "SPOT", "EURUSD"),
    ...     vol_id=MarketId("FX", "VOL", "EURUSD.VOL"),
    ...     domestic_curve_id=MarketId("IR", "CURVE", "USD.OIS"),
    ...     foreign_curve_id=MarketId("IR", "CURVE", "EUR.OIS"),
    ... )
    """

    # Vanilla leg specification.
    option_type: OptionType  # "call" | "put"
    notional: float          # Foreign units.
    strike: float            # Strike price.
    expiry: float            # Year fraction to expiry.

    # Double barrier specification.
    barrier_style: BarrierStyle  # "knock_out" | "knock_in"
    lower_barrier: float         # Lower barrier level (L < S_0).
    upper_barrier: float         # Upper barrier level (U > S_0).

    # Optional expiry-paid rebate (domestic per 1 foreign notional).
    rebate_amount: float = 0.0

    # Market identifiers for pricing inputs.
    spot_id: MarketId = MarketId("FX", "SPOT", "UNKNOWN")
    vol_id: MarketId = MarketId("FX", "VOL", "UNKNOWN")
    domestic_curve_id: MarketId = MarketId("IR", "CURVE", "UNKNOWN")
    foreign_curve_id: MarketId = MarketId("IR", "CURVE", "UNKNOWN")

    def __post_init__(self) -> None:
        """
        Validate instrument parameters on construction.

        Raises
        ------
        ValueError
            If any parameter fails validation checks.
        """
        # Validate option type.
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Validate numeric parameters (must be sensible).
        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Validate barrier parameters.
        if self.barrier_style not in ("knock_out", "knock_in"):
            raise ValueError("barrier_style must be 'knock_out' or 'knock_in'.")
        if float(self.lower_barrier) <= 0.0:
            raise ValueError("lower_barrier must be > 0.")
        if float(self.upper_barrier) <= 0.0:
            raise ValueError("upper_barrier must be > 0.")

        # Validate barrier ordering (lower < upper).
        if float(self.lower_barrier) >= float(self.upper_barrier):
            raise ValueError(
                f"lower_barrier ({self.lower_barrier}) must be < upper_barrier ({self.upper_barrier})."
            )

        # Validate rebate.
        if float(self.rebate_amount) < 0.0:
            raise ValueError("rebate_amount must be >= 0.")
