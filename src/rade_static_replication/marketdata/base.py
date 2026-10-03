"""
Foundations for the typed market-data layer.

This is the "stronger base" the market objects build on:

* :class:`Interpolator` — a pluggable interpolation strategy (linear today, with a
  clear seam for log-linear-DF / variance / SABR) so a desk can change interpolation
  policy in one place without touching pricers.
* validation helpers (:func:`require_increasing`, :func:`require_same_shape`, …) that
  give every object fail-fast, self-describing ``__post_init__`` checks.
* :class:`MarketObject` — the structural protocol every market object satisfies
  (``label`` + ``validate`` + ``summary``), used for typing and audit logging.
* :func:`year_fraction` — the single day-count implementation.

Keeping all of this here means the curves/surfaces/cubes stay small and consistent.
"""
from __future__ import annotations

import datetime as _dt
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.rade_static_replication.domain.enums import DayCount
from src.rade_static_replication.domain.errors import MarketDataError


# =====================================================================
#  Interpolation strategies
# =====================================================================

@runtime_checkable
class Interpolator(Protocol):
    """Maps a query point onto sampled ``(xp, fp)`` data."""

    def __call__(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        ...


class LinearInterpolator:
    """Piecewise-linear with flat extrapolation (the safe default)."""

    def __call__(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        return float(np.interp(x, xp, fp))


DEFAULT_INTERPOLATOR: Interpolator = LinearInterpolator()


def total_variance_interp(
    x: float, xp: np.ndarray, vols_at_x: np.ndarray, axis_vals: np.ndarray,
) -> float:
    """Interpolate a vol at expiry ``x`` in *total variance* (the arb-aware default).

    ``vols_at_x`` are the per-expiry vols (already strike-interpolated) on the
    ``axis_vals`` expiry pillars.
    """
    if x <= axis_vals[0]:
        return float(vols_at_x[0])
    if x >= axis_vals[-1]:
        return float(vols_at_x[-1])
    j = int(np.searchsorted(axis_vals, x))
    t0, t1 = axis_vals[j - 1], axis_vals[j]
    w = (x - t0) / (t1 - t0)
    var = (1.0 - w) * vols_at_x[j - 1] ** 2 * t0 + w * vols_at_x[j] ** 2 * t1
    return float(np.sqrt(max(var, 0.0) / x))


# =====================================================================
#  Validation helpers
# =====================================================================

def as_1d(name: str, arr: Sequence) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise MarketDataError(f"{name}: expected non-empty 1-D array, got shape {a.shape}")
    return a


def require_increasing(name: str, arr: np.ndarray) -> None:
    if arr.size > 1 and np.any(np.diff(arr) <= 0):
        raise MarketDataError(f"{name}: values must be strictly increasing")


def require_same_shape(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise MarketDataError(f"{name_a} {a.shape} != {name_b} {b.shape}")


def require_shape(name: str, a: np.ndarray, expected: tuple) -> None:
    if a.shape != expected:
        raise MarketDataError(f"{name} shape {a.shape} != {expected}")


def require_positive(name: str, value: float) -> None:
    if not (value > 0.0):
        raise MarketDataError(f"{name} must be positive, got {value}")


# =====================================================================
#  Day count
# =====================================================================

def year_fraction(start: _dt.date, end: _dt.date, convention: DayCount = DayCount.ACT_365) -> float:
    """Year fraction between two dates (non-negative)."""
    days = max((end - start).days, 0)
    basis = 360.0 if convention == DayCount.ACT_360 else 365.0
    return days / basis


# =====================================================================
#  Structural protocol
# =====================================================================

@runtime_checkable
class MarketObject(Protocol):
    """Everything a market object exposes for typing + audit."""
    label: str

    def validate(self) -> None: ...

    def summary(self) -> dict: ...
