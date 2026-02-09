from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import math

# slots=True requires Python 3.10+
_DATACLASS_KW = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@runtime_checkable
class Curve(Protocol):
    """Term-structure interface required by pricing and scenarios."""

    def df(self, t: float) -> float:
        """Discount factor DF(t) for year-fraction t from asof."""
        ...

    def zero_rate(self, t: float) -> float:
        """Continuously-compounded zero rate r(t) implied by DF(t)."""
        ...

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate over (t1, t2) consistent with the curve."""
        ...


@runtime_checkable
class VolSurface(Protocol):
    """Implied-vol surface interface used by pricers."""

    def implied_vol(self, expiry: float, strike: float) -> float:
        """Return implied vol σ(T,K)."""
        ...

    def vol(self, expiry: float, strike: float) -> float:
        """Compatibility alias for implied_vol()."""
        ...


@dataclass(**_DATACLASS_KW)
class Quote:
    """
    Scalar market quote wrapper.

    Keeping this as a dataclass (rather than plain float) makes it easier to:
    - attach metadata later (source, timestamp, units)
    - distinguish quotes from parameters
    """
    value: float

    def __post_init__(self) -> None:
        # Validate we always store a finite float.
        v = float(self.value)
        if not math.isfinite(v):
            raise ValueError("Quote.value must be finite.")
        object.__setattr__(self, "value", v)