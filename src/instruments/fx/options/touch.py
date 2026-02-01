"""
European FX Touch Option Instrument Definition.

This module defines the `EuropeanFxTouchOption` dataclass representing
touch (binary barrier) options including one-touch and no-touch variants.

Mathematical Framework
----------------------
Touch options are binary options with path-dependent activation:

**One-Touch (pays if barrier hit):**
    Payoff = Q * 1_{max(S_t) >= H}  (up-touch)
    Payoff = Q * 1_{min(S_t) <= H}  (down-touch)

**No-Touch (pays if barrier NOT hit):**
    Payoff = Q * 1_{max(S_t) < H}   (up-no-touch)
    Payoff = Q * 1_{min(S_t) > H}   (down-no-touch)

where Q is the fixed payout amount and H is the barrier level.

Key Properties
--------------
- Binary payout: all-or-nothing based on barrier touch
- Path-dependent: requires monitoring of spot path
- One-touch price increases with volatility and time
- No-touch price decreases with volatility and time
- No strike: payoff depends only on barrier touch, not terminal spot

Use Cases
---------
- Simple barrier bet on whether level will be reached
- Component of structured products
- Hedging extreme moves
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import BarrierDirection
from src.instruments.core.types import TouchStyle


@dataclass(frozen=True, slots=True)
class FxTouchEuropeanOption:
    """
    European FX Touch (binary barrier) option.

    A touch option pays a fixed amount (payout_amount) at expiry based on
    whether the spot price touches (one-touch) or avoids (no-touch) a
    specified barrier level during the option's life.

    Conventions
    -----------
    - Underlying spot S is domestic-per-foreign (e.g. EURUSD).
    - `notional` scales the payout (payout_amount is per unit notional).
    - Payout is in domestic currency at expiry.

    Touch Styles
    ------------
    - one_touch: Pays payout_amount if barrier IS touched.
    - no_touch:  Pays payout_amount if barrier is NOT touched.

    Barrier Directions
    ------------------
    - up:   Barrier is above current spot; touch = max(S_t) >= H.
    - down: Barrier is below current spot; touch = min(S_t) <= H.

    Monitoring
    ----------
    - Discrete monitoring at simulated path points (MC).
    - Continuous monitoring approximation possible with corrections.

    Parameters
    ----------
    touch_style : TouchStyle
        "one_touch" or "no_touch".
    barrier_direction : BarrierDirection
        "up" or "down".
    barrier_level : float
        The touch barrier level H.
    payout_amount : float
        Fixed payout Q (domestic currency per unit notional).
    notional : float
        Scaling factor for total payout.
    expiry : float
        Time to expiry in year fractions.
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
    >>> from src.instruments.fx.options.touch import FxTouchEuropeanOption
    >>> from src.marketdata.core.ids import MarketId
    >>> # One-touch up option: pays $10k if EURUSD touches 1.15
    >>> one_touch = FxTouchEuropeanOption(
    ...     touch_style="one_touch",
    ...     barrier_direction="up",
    ...     barrier_level=1.15,
    ...     payout_amount=10_000.0,
    ...     notional=1.0,
    ...     expiry=1.0,
    ...     spot_id=MarketId("FX", "SPOT", "EURUSD"),
    ...     vol_id=MarketId("FX", "VOL", "EURUSD.VOL"),
    ...     domestic_curve_id=MarketId("IR", "CURVE", "USD.OIS"),
    ...     foreign_curve_id=MarketId("IR", "CURVE", "EUR.OIS"),
    ... )
    """

    # Touch specification.
    touch_style: TouchStyle          # "one_touch" | "no_touch"
    barrier_direction: BarrierDirection  # "up" | "down"
    barrier_level: float             # Barrier H.

    # Payout specification.
    payout_amount: float             # Fixed payout Q (domestic per unit notional).
    notional: float                  # Scaling factor.
    expiry: float                    # Year fraction to expiry.

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
        # Validate touch style.
        if self.touch_style not in ("one_touch", "no_touch"):
            raise ValueError("touch_style must be 'one_touch' or 'no_touch'.")

        # Validate barrier direction.
        if self.barrier_direction not in ("up", "down"):
            raise ValueError("barrier_direction must be 'up' or 'down'.")

        # Validate numeric parameters.
        if float(self.barrier_level) <= 0.0:
            raise ValueError("barrier_level must be > 0.")
        if float(self.payout_amount) < 0.0:
            raise ValueError("payout_amount must be >= 0.")
        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
