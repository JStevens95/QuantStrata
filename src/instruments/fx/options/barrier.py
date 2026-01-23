from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType, BarrierStyle, BarrierDirection



@dataclass(frozen=True, slots=True)
class EuropeanFxBarrierOption:
    """
    European FX single-barrier option (V1).

    Conventions
    -----------
    - Underlying spot S is domestic-per-foreign (e.g. EURUSD).
    - `notional` is in foreign units.
    - Any payoff returned by pricers should be in domestic currency.

    Barrier semantics (V1)
    ----------------------
    - Monitoring: as implemented in MC, "discrete" monitoring on simulated path points.
      (FD/BSM are typically continuous-monitoring; we’ll handle that at pricer-level.)
    - Knock-out: pays vanilla payoff if barrier NOT hit, else rebate (default 0).
    - Knock-in : pays vanilla payoff if barrier hit, else rebate (default 0).

    Rebate (V1)
    -----------
    - `rebate_amount` is a *domestic-currency* amount paid at expiry (not at hit time).
    - It is interpreted as "per unit foreign notional" so pricers can scale by notional.
    """

    option_type: OptionType  # "call" | "put"

    # Vanilla legs
    notional: float          # foreign units
    strike: float
    expiry: float            # year fraction

    # Barrier definition
    barrier_direction: BarrierDirection  # "up" | "down"
    barrier_style: BarrierStyle          # "knock_out" | "knock_in"
    barrier_level: float

    # Optional expiry-paid rebate (domestic per 1 foreign notional)
    rebate_amount: float = 0.0

    # Market identifiers
    spot_id: MarketId = MarketId("FX", "SPOT", "UNKNOWN")
    vol_id: MarketId = MarketId("FX", "VOL", "UNKNOWN")
    domestic_curve_id: MarketId = MarketId("IR", "CURVE", "UNKNOWN")
    foreign_curve_id: MarketId = MarketId("IR", "CURVE", "UNKNOWN")

    def __post_init__(self) -> None:
        # Basic numeric checks; keep these strict so pricers can assume sane inputs.
        if float(self.notional) < 0.0:
            raise ValueError("notional must be >= 0.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        if float(self.barrier_level) <= 0.0:
            raise ValueError("barrier_level must be > 0.")
        if float(self.rebate_amount) < 0.0:
            raise ValueError("rebate_amount must be >= 0.")
        if self.barrier_direction not in ("up", "down"):
            raise ValueError("barrier_direction must be 'up' or 'down'.")
        if self.barrier_style not in ("knock_out", "knock_in"):
            raise ValueError("barrier_style must be 'knock_out' or 'knock_in'.")