from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Tuple


@runtime_checkable
class Curve(Protocol):
    """
    Term-structure interface.

    Curves are used by pricers, scenario engines, and label-generation pipelines.
    The minimal pricing contract typically requires discount factors and rates.
    """
    def df(self, t: float) -> float:
        """Discount factor to time t (year fraction from asof date)."""
        ...

    def zero_rate(self, t: float) -> float:
        """Continuously-compounded zero rate to time t (year fraction from asof date). """
        ...

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate over (t1, t2)."""
        ...


@runtime_checkable
class VolSurface(Protocol):
    """
    Implied-vol surface interface.

    Minimal contract:
      - return implied volatility as a function of (expiry, strike)
    """
    def implied_vol(self, expiry: float, strike: float) -> float:
        """Implied volatility for a given expiry and strike."""
        ...

    def vol(self, expiry: float, strike: float) -> float:
        """Compatibility alias for implied_vol()."""
        ...


@dataclass(frozen=True, slots=True)
class Quote:
    """
    Scalar market quote.

    Examples
    --------
    - spot levels
    - fixings
    - dividend yields (if represented as a scalar)
    - convenience spreads
    """
    # initiate required variables.
    value: float


@dataclass(frozen=True, slots=True)
class Panel:
    """
    Numeric panel backing MarketDataset data.

    `Panel` is deliberately lightweight: it’s just (data, axis_names).
    This keeps it suitable for ML/RL pipelines (NumPy-friendly).

    Common shapes
    -------------
    - [T]                 : time series
    - [T, S]              : time x scenarios
    - [T, S, K]           : time x scenarios x tenor/strike-grid parameter blocks

    Notes
    -----
    We store `axis_names` to make debugging and export safer.
    """
    # initiate required variables.
    data: np.ndarray
    axis_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Post initialization of `Panel`."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Panel.data must be a numpy.ndarray.")
        if self.data.ndim != len(self.axis_names):
            raise ValueError(f"Panel axis mismatch: data.ndim={self.data.ndim}, axis_names={self.axis_names}")

    def scalar_at(self, time_idx: int, scenario_idx: int = 0) -> float:
        """
        Extract a scalar at (time_idx, scenario_idx) for panels shaped [T] or [T,S].

        This helper is intentionally strict because it is used to construct Market snapshots.

        :param time_idx: time index
        :param scenario_idx: scenario index
        """
        if self.data.ndim == 1:
            return float(self.data[time_idx])
        if self.data.ndim == 2:
            return float(self.data[time_idx, scenario_idx])
        raise ValueError(f"scalar_at only supports panels with ndim 1 or 2; got ndim={self.data.ndim}.")
