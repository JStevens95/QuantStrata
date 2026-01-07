from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId

# define parameters values.
OptionType = Literal["call", "put"]


@dataclass(frozen=True, slots=True)
class EuropeanFxVanillaOption:
    """
    European vanilla FX option (call/put): payoff = max(±(S_T - K), 0).

    Conventions
    -----------
    - Underlying S is "domestic per 1 foreign" (e.g. USD per EUR for EURUSD).
    - notional_foreign is in *foreign* currency units.
    - PV is returned in *domestic* currency units.

    This is a pure product definition:
    - No model choice here (BSM/MC/PDE live in pricers/models).
    """

    option_type: OptionType
    notional: float
    strike: float
    expiry: float  # year fraction
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")


@dataclass(frozen=True, slots=True)
class AmericanFxVanillaOption:
    """
    Placeholder for an American vanilla FX option.

    Notes
    -----
    - This is intentionally a *separate instrument type* to keep the registry clean:
        AmericanFxVanillaOption -> American pricer (tree/FD/LSMC)
    - Implement exercise logic later without refactoring the type-driven plumbing.
    """

    option_type: OptionType
    notional: float
    strike: float
    expiry: float
    spot_id: MarketId
    vol_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")