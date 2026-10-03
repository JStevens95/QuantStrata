"""
Discount / zero-rate curve — the one genuinely cross-asset market instrument.

A ``DiscountCurve`` is the same object whether an FX factor (which needs two of them) or a
rates factor consumes it, so it lives in ``common`` rather than any asset subpackage.
Stores continuously-compounded zero rates on year-fraction pillars and exposes the
quantities pricers need — ``df``, ``zero``, ``forward_rate`` — under a chosen
interpolation policy. Immutable; mutate via the ``with_*`` helpers (used by the shock
layer to produce scenario curves).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.rade_static_replication.domain.enums import Interpolation
from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.base import (
    as_1d,
    require_increasing,
    require_same_shape,
)


@dataclass(frozen=True)
class DiscountCurve:
    """A continuously-compounded zero curve for one currency.

    Parameters
    ----------
    currency : str
    tenors : np.ndarray
        Pillar tenors in year fractions, strictly increasing.
    zero_rates : np.ndarray
        Continuously-compounded zero rates at each pillar.
    interpolation : Interpolation
        ``LINEAR_ZERO`` (default) or ``LOG_LINEAR_DF`` (flat-forward).
    """
    currency: str
    tenors: np.ndarray
    zero_rates: np.ndarray
    interpolation: Interpolation = Interpolation.LINEAR_ZERO
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = as_1d(f"{self.currency} curve tenors", self.tenors)
        z = as_1d(f"{self.currency} curve zero_rates", self.zero_rates)
        require_same_shape("tenors", t, "zero_rates", z)
        require_increasing(f"{self.currency} curve tenors", t)
        object.__setattr__(self, "tenors", t)
        object.__setattr__(self, "zero_rates", z)
        object.__setattr__(self, "label", self.label or f"DF.{self.currency}")

    # ---- core quantities ----

    def df(self, t: float) -> float:
        """Discount factor to ``t`` (year fraction)."""
        if t <= 0.0:
            return 1.0
        if self.interpolation == Interpolation.LOG_LINEAR_DF:
            log_df_pillars = -self.zero_rates * self.tenors
            return float(np.exp(np.interp(t, self.tenors, log_df_pillars)))
        return float(np.exp(-self.zero(t) * t))

    def zero(self, t: float) -> float:
        """Continuously-compounded zero rate at ``t``."""
        if t <= 0.0:
            return float(self.zero_rates[0])
        if self.interpolation == Interpolation.LOG_LINEAR_DF:
            return float(-np.log(self.df(t)) / t)
        return float(np.interp(t, self.tenors, self.zero_rates))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate between ``t1`` and ``t2``."""
        if t2 <= t1:
            raise MarketDataError(f"forward_rate needs t2 > t1, got t1={t1}, t2={t2}")
        return float((np.log(self.df(t1)) - np.log(self.df(t2))) / (t2 - t1))

    # ---- mutation helpers (return new curves) ----

    def with_zero_rates(self, zero_rates: np.ndarray) -> "DiscountCurve":
        """A copy with replaced zero rates on the same pillars."""
        return DiscountCurve(
            self.currency, self.tenors.copy(), np.asarray(zero_rates, dtype=np.float64),
            self.interpolation, self.label, dict(self.metadata),
        )

    def validate(self) -> None:  # already validated in __post_init__
        return None

    def summary(self) -> dict:
        return {
            "type": "DiscountCurve", "currency": self.currency,
            "n_pillars": int(self.tenors.size),
            "tenor_range": [float(self.tenors[0]), float(self.tenors[-1])],
            "interpolation": self.interpolation.value,
        }
