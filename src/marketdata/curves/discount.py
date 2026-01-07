from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.marketdata.core.types import ExtrapolationMode


def _validate_forward_inputs(*, t1: float, t2: float) -> None:
    """Validate forward-rate interval inputs defensively."""
    if not np.isfinite(t1) or not np.isfinite(t2):
        raise ValueError("t1 and t2 must be finite.")
    if t1 < 0.0:
        raise ValueError(f"t1 must be >= 0; got t1={t1}.")
    if t2 <= t1:
        raise ValueError(f"forward_rate requires t2 > t1; got t1={t1}, t2={t2}.")


def _interp_flat(*, x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    1D linear interpolation with flat extrapolation.

    - inside [x0, xN]: linear interpolation
    - below x0: y0
    - above xN: yN
    """
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[-1]):
        return float(y[-1])
    return float(np.interp(float(xq), x, y))


def _interp_linear_extrap(*, x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    1D linear interpolation with linear extrapolation at the ends.

    This is less stable than flat extrapolation but sometimes desired.
    """
    xq = float(xq)

    if xq <= float(x[0]):
        if x.size == 1:
            return float(y[0])
        slope = float((y[1] - y[0]) / (x[1] - x[0]))
        return float(y[0] + slope * (xq - float(x[0])))

    if xq >= float(x[-1]):
        if x.size == 1:
            return float(y[-1])
        slope = float((y[-1] - y[-2]) / (x[-1] - x[-2]))
        return float(y[-1] + slope * (xq - float(x[-1])))

    return float(np.interp(xq, x, y))


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    """
    Flat continuously-compounded discount curve.

    df(t) = exp(-r * t)
    """
    continuously_compounded_rate: float

    def df(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        return float(np.exp(-float(self.continuously_compounded_rate) * float(t)))

    def zero_rate(self, t: float) -> float:
        return float(self.continuously_compounded_rate)

    def forward_rate(self, t1: float, t2: float) -> float:
        _validate_forward_inputs(t1=t1, t2=t2)
        return float(self.continuously_compounded_rate)


@dataclass(frozen=True, slots=True)
class ZeroRateDiscountCurve:
    """
    Continuously-compounded zero rate discount curve defined on a tenor grid, with interpolation.

    Stores:
      - tenors: increasing year fractions
      - zero_rates: continuously-compounded r(T) at each tenor

    Defines:
      df(t) = exp(-r(t) * t)
    """
    tenors: np.ndarray
    zero_rates: np.ndarray
    extrapolation: ExtrapolationMode = "flat"

    def __post_init__(self) -> None:
        tenor_grid = np.asarray(self.tenors, dtype=float).reshape(-1)
        rate_grid = np.asarray(self.zero_rates, dtype=float).reshape(-1)

        if tenor_grid.size == 0:
            raise ValueError("tenors must not be empty.")
        if tenor_grid.size != rate_grid.size:
            raise ValueError("tenors and zero_rates must have the same length.")
        if np.any(~np.isfinite(tenor_grid)) or np.any(~np.isfinite(rate_grid)):
            raise ValueError("tenors and zero_rates must be finite.")
        if np.any(tenor_grid < 0.0):
            raise ValueError("tenors must be >= 0.")
        if np.any(np.diff(tenor_grid) <= 0.0):
            raise ValueError("tenors must be strictly increasing.")
        if self.extrapolation not in ("flat", "linear"):
            raise ValueError("extrapolation must be one of {'flat','linear'}.")

        object.__setattr__(self, "tenors", tenor_grid)
        object.__setattr__(self, "zero_rates", rate_grid)

    def df(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        r_t = self.zero_rate(t)
        return float(np.exp(-float(r_t) * float(t)))

    def zero_rate(self, t: float) -> float:
        if t <= 0.0:
            return float(self.zero_rates[0])

        if self.extrapolation == "flat":
            return float(_interp_flat(x=self.tenors, y=self.zero_rates, xq=float(t)))

        return float(_interp_linear_extrap(x=self.tenors, y=self.zero_rates, xq=float(t)))

    def forward_rate(self, t1: float, t2: float) -> float:
        _validate_forward_inputs(t1=t1, t2=t2)
        df1 = self.df(t1)
        df2 = self.df(t2)
        return float(-np.log(df2 / df1) / (float(t2) - float(t1)))