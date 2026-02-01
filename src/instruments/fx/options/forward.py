# src/instruments/fx/options/forward.py
"""
FX Forward Options.

Options on FX forwards, priced using the Black76 model.

Mathematical Framework
----------------------
An FX forward option is an option to buy/sell FX at a forward rate.
The forward rate F at time t for delivery at T is:

    F(t, T) = S(t) × exp((r_d - r_f) × (T - t))

The option payoff at expiry:
    Call: max(F_T - K, 0)
    Put:  max(K - F_T, 0)

Where F_T is the forward rate at expiry for immediate delivery.

When to Use
-----------
- OTC FX forward options where the underlying is a forward contract
- Options with deferred settlement
- When forward volatility is quoted directly (common in FX markets)

Relationship to Spot Options
----------------------------
A spot option and forward option with the same strike have the same value
if the forward vol equals the spot vol. However:
- Spot option delta is w.r.t. spot
- Forward option delta is w.r.t. forward

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class FxForwardEuropeanOption:
    """
    European option on an FX forward.

    This instrument represents an option to enter into an FX forward at
    a predetermined rate (strike). Priced using Black76.

    Parameters
    ----------
    option_type : OptionType
        "call" (right to buy foreign) or "put" (right to sell foreign).
    notional : float
        Notional in foreign currency units.
    strike : float
        Strike forward rate (domestic per foreign).
    expiry : float
        Time to option expiry in years.
    forward_expiry : float
        Time to forward delivery in years (>= expiry).
        If equal to expiry, this is a standard European option.
    spot_id : MarketId
        Market identifier for spot rate.
    vol_id : MarketId
        Market identifier for forward volatility surface.
    domestic_curve_id : MarketId
        Market identifier for domestic discount curve.
    foreign_curve_id : MarketId
        Market identifier for foreign discount curve.

    Conventions
    -----------
    - Underlying F is "domestic per 1 foreign" (e.g., USD per EUR).
    - Notional is in foreign currency units.
    - PV is returned in domestic currency units.

    Forward Calculation
    -------------------
    The forward rate used for pricing is computed as:
        F = S × exp((r_d - r_f) × T_fwd)

    Where T_fwd is the forward delivery date.

    Examples
    --------
    >>> # Option to buy EUR vs USD at forward rate 1.12
    >>> opt = FxForwardEuropeanOption(
    ...     option_type="call",
    ...     notional=1_000_000,  # EUR 1M
    ...     strike=1.12,
    ...     expiry=0.5,         # 6 month option
    ...     forward_expiry=0.5, # Standard (option and forward same expiry)
    ...     spot_id=eurusd_spot_id,
    ...     vol_id=eurusd_fwd_vol_id,
    ...     domestic_curve_id=usd_curve_id,
    ...     foreign_curve_id=eur_curve_id,
    ... )
    """

    option_type: OptionType
    notional: float
    strike: float
    expiry: float
    forward_expiry: float
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        """Validate inputs."""
        # Option type validation.
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # Notional validation.
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        # Strike validation.
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        # Expiry validation.
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        # Forward expiry validation.
        if float(self.forward_expiry) < 0.0:
            raise ValueError("forward_expiry must be >= 0.")

        if float(self.forward_expiry) < float(self.expiry):
            raise ValueError("forward_expiry must be >= expiry.")

        # MarketId type validation.
        if not isinstance(self.spot_id, MarketId):
            raise TypeError("spot_id must be a MarketId.")
        if not isinstance(self.vol_id, MarketId):
            raise TypeError("vol_id must be a MarketId.")
        if not isinstance(self.domestic_curve_id, MarketId):
            raise TypeError("domestic_curve_id must be a MarketId.")
        if not isinstance(self.foreign_curve_id, MarketId):
            raise TypeError("foreign_curve_id must be a MarketId.")


@dataclass(frozen=True, slots=True)
class FxForwardEuropeanOptionSimple:
    """
    Simplified FX forward option for direct Black76 pricing.

    Use this when you have the forward rate and vol directly,
    without needing to compute them from spot and curves.

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    notional : float
        Notional in foreign currency units.
    strike : float
        Strike forward rate.
    expiry : float
        Time to expiry in years.
    forward_rate : float
        Current forward rate F.
    vol : float
        Forward volatility σ (log-normal).
    discount_factor : float
        Discount factor DF = exp(-r_d × T).

    Examples
    --------
    >>> # Direct Black76 pricing with known forward
    >>> opt = FxForwardEuropeanOptionSimple(
    ...     option_type="call",
    ...     notional=1_000_000,
    ...     strike=1.12,
    ...     expiry=0.5,
    ...     forward_rate=1.1050,
    ...     vol=0.08,
    ...     discount_factor=0.975,
    ... )
    """

    option_type: OptionType
    notional: float
    strike: float
    expiry: float
    forward_rate: float
    vol: float
    discount_factor: float

    def __post_init__(self) -> None:
        """Validate inputs."""
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")

        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")

        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")

        if float(self.forward_rate) <= 0.0:
            raise ValueError("forward_rate must be > 0.")

        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")

        if float(self.discount_factor) <= 0.0 or float(self.discount_factor) > 1.0:
            raise ValueError("discount_factor must be in (0, 1].")


__all__ = [
    "FxForwardEuropeanOption",
    "FxForwardEuropeanOptionSimple",
]
