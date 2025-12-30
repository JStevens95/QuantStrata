from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.marketdata.ids import MarketId


@dataclass(frozen=True, slots=True)
class FxForward:
    """
    Deliverable FX forward contract (domestic PV).

    Convention
    ----------
    - Spot quote S is "domestic per 1 foreign" (e.g. EURUSD = USD per 1 EUR).
    - Notional is in foreign currency units (common for FX deliverables).
    - PV is in domestic currency units.

    Payoff at maturity T (domestic currency):
        PV_T = notional_foreign * (S_T - K)

    Risk-neutral PV at time 0 (domestic currency):
        PV_0 = notional_foreign * ( S_0 * df_f(T) - K * df_d(T) )

    where
        df_d(T) = domestic discount factor
        df_f(T) = foreign discount factor (foreign funding curve)

    Parameters
    ----------
    notional_foreign:
        Notional in foreign currency units (must be > 0).
    strike:
        Forward strike K in domestic per 1 foreign (must be > 0).
    expiry:
        Time to maturity in year fractions (>= 0).
    spot_id:
        MarketId for FX spot quote (kind="SPOT").
    domestic_curve_id:
        MarketId for domestic discount curve (kind="CURVE").
    foreign_curve_id:
        MarketId for foreign discount curve (kind="CURVE").
    description:
        Optional label for reporting/debugging.
    """
    notional_foreign: float
    strike: float
    expiry: float

    spot_id: MarketId
    domestic_curve_id: MarketId
    foreign_curve_id: MarketId

    description: Optional[str] = None

    def __post_init__(self) -> None:
        n = float(self.notional_foreign)
        k = float(self.strike)
        t = float(self.expiry)

        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("notional_foreign must be finite and > 0.")
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("strike must be finite and > 0.")
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("expiry must be finite and >= 0.")

        if self.spot_id is None or self.domestic_curve_id is None or self.foreign_curve_id is None:
            raise ValueError("spot_id, domestic_curve_id, foreign_curve_id must not be None.")