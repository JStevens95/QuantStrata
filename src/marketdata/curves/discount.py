from __future__ import annotations

import numpy as np
from typing import Literal
from dataclasses import dataclass


# define extrapolation modes
ExtrapolationMode = Literal["flat", "linear"]


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    """
    Flat continuously-compounded discount curve.

    This is the minimal curve implementation you want for V1 because it:
      - is extremely stable numerically,
      - is trivial to calibrate/generate synthetically,
      - supports immediate pricing for BS-style models.

    Parameters
    ----------
    continuously_compounded_rate:
        Constant continuously-compounded zero rate (e.g., 0.05 for 5% p.a.).
        This implies:
            df(t) = exp(-r * t)

    Notes
    -----
    - Time `t` is expressed in year-fractions from the curve's as-of date.
    - We clamp t <= 0 to df=1.0 and rates to r (for convenience).
    """
    continuously_compounded_rate: float

    def df(self, t: float) -> float:
        """
        Discount factor to time `t` (year fraction).

        Returns
        -------
        float
            exp(-r * t) for t > 0, else 1.0.
        """
        if t <= 0.0:
            return 1.0
        return float(np.exp(-self.continuously_compounded_rate * t))

    def zero_rate(self, t: float) -> float:
        """
        Continuously-compounded zero rate to time `t`.

        For a flat curve, this is constant.

        Returns
        -------
        float
        """
        return float(self.continuously_compounded_rate)

    def fwd_rate(self, t1: float, t2: float) -> float:
        """
        Continuously-compounded forward rate over (t1, t2).

        For a flat curve, this is constant (equal to the zero rate).

        Notes
        -----
        Many libraries define forward rates with different compounding conventions.
        Here we remain consistent with continuous compounding.

        Returns
        -------
        float
        """
        _validate_forward_inputs(t1=t1, t2=t2)
        return float(self.continuously_compounded_rate)


@dataclass(frozen=True, slots=True)
class ZeroRateDiscountCurve:
    """
    Continuously-compounded zero rate discount curve defined on a tenor grid, with interpolation.

    This is the natural "complexity upgrade" from FlatDiscountCurve:
    - still exposes the same stable curve interface (df/zero_rate/fwd_rate),
    - but now supports a term-structure rather than a single flat rate.

    The curve stores:
      - tenors: increasing times in year fractions (e.g., [0.25, 0.5, 1.0, 2.0, 5.0])
      - zero_rates: continuously-compounded zero rates at those tenors

    Discount factors are computed via:
        r(t) = interp(tenors, zero_rates, t)
        df(t) = exp(-r(t) * t)

    Parameters
    ----------
    tenors:
        1D array of strictly increasing tenor points in year fractions (t >= 0).
    zero_rates:
        1D array of continuously-compounded zero rates at each tenor point.
    extrapolation:
        Extrapolation policy beyond the grid. Supported:
        - "flat": use the nearest endpoint rate (common and stable)
        - "linear": linear extrapolation based on end segments (less stable)

    Notes
    -----
    - For pricing stability, "flat" extrapolation is recommended.
    - This is intentionally minimal; bootstrapping belongs in a separate module.
    """
    tenors: np.ndarray
    zero_rates: np.ndarray
    extrapolation: ExtrapolationMode = "flat"

    def __post_init__(self) -> None:
        # Canonicalize inputs to 1D float arrays.
        tenor_grid = np.asarray(self.tenors, dtype=float).reshape(-1)
        rate_grid = np.asarray(self.zero_rates, dtype=float).reshape(-1)

        # Defensive validation (fast + explicit).
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

        # Store the validated arrays on the frozen dataclass.
        object.__setattr__(self, "tenors", tenor_grid)
        object.__setattr__(self, "zero_rates", rate_grid)

    def df(self, t: float) -> float:
        """
        Discount factor to time `t`.

        Returns
        -------
        float
            exp(-r(t) * t) for t > 0, else 1.0.
        """
        if t <= 0.0:
            return 1.0
        r_t = self.zero_rate(t)
        return float(np.exp(-r_t * t))

    def zero_rate(self, t: float) -> float:
        """
        Interpolated continuously-compounded zero rate r(t).

        Returns
        -------
        float
        """
        if t <= 0.0:
            # Define r(0) as the shortest tenor rate for convenience.
            return float(self.zero_rates[0])

        if self.extrapolation == "flat":
            return float(_interp_flat(x=self.tenors, y=self.zero_rates, xq=t))

        # linear extrapolation
        return float(_interp_linear_extrap(x=self.tenors, y=self.zero_rates, xq=t))

    def fwd_rate(self, t1: float, t2: float) -> float:
        """
        Continuously-compounded forward rate over (t1, t2).

        We compute it via discount factors:
            f = -log(df(t2)/df(t1)) / (t2 - t1)

        This definition is consistent with continuous compounding.

        Returns
        -------
        float
        """
        _validate_forward_inputs(t1=t1, t2=t2)
        df1 = self.df(t1)
        df2 = self.df(t2)
        return float(-np.log(df2 / df1) / (t2 - t1))


def _validate_forward_inputs(t1: float, t2: float) -> None:
    """Validate forward-rate interval inputs defensively."""
    if not np.isfinite(t1) or not np.isfinite(t2):
        raise ValueError("t1 and t2 must be finite.")
    if t2 <= t1:
        raise ValueError(f"Forward rate requires t2 > t1; got t1={t1}, t2={t2}.")
    if t1 < 0.0:
        raise ValueError(f"t1 must be >= 0; got t1={t1}.")


def _interp_flat(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    1D linear interpolation with flat extrapolation.

    - inside [x0, xN]: linear interpolation
    - below x0: y0
    - above xN: yN
    """
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[-1]:
        return float(y[-1])
    return float(np.interp(xq, x, y))


def _interp_linear_extrap(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """
    1D linear interpolation with linear extrapolation at the ends.

    This is less stable than flat extrapolation but sometimes desired.
    """
    if xq <= x[0]:
        # Extrapolate using first segment.
        slope = (y[1] - y[0]) / (x[1] - x[0]) if x.size > 1 else 0.0
        return float(y[0] + slope * (xq - x[0]))
    if xq >= x[-1]:
        # Extrapolate using last segment.
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2]) if x.size > 1 else 0.0
        return float(y[-1] + slope * (xq - x[-1]))
    return float(np.interp(xq, x, y))