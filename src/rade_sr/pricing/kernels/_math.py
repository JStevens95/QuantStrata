"""
Shared numeric utilities for pricing kernels.

All functions are @njit compiled. No scipy dependency — we implement
normal CDF/PDF directly so the entire pricing path stays in compiled code.
"""
from __future__ import annotations

import math

import numpy as np
from ._numba import njit, float64


# ─────────────────────────────────────────────────────────────────────
# Normal distribution (no scipy needed inside @njit)
# ─────────────────────────────────────────────────────────────────────

@njit(float64(float64), cache=True)
def norm_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function.

    Accuracy: ~15 significant digits (same as scipy.stats.norm.cdf).
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(float64(float64), cache=True)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ─────────────────────────────────────────────────────────────────────
# Discount factor helpers
# ─────────────────────────────────────────────────────────────────────

@njit(float64(float64, float64), cache=True)
def df_from_rate(rate: float, t: float) -> float:
    """Continuous discount factor: exp(-r * t)."""
    return math.exp(-rate * t)


@njit(float64(float64, float64, float64), cache=True)
def forward_rate(r1: float, t1: float, dt: float) -> float:
    """Simple forward rate between t1 and t1+dt from a flat rate.

    f = (DF(t1) / DF(t1+dt) - 1) / dt
    """
    df1 = math.exp(-r1 * t1)
    df2 = math.exp(-r1 * (t1 + dt))
    if df2 == 0.0:
        return 0.0
    return (df1 / df2 - 1.0) / dt


# ─────────────────────────────────────────────────────────────────────
# Annuity
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_annuity(
    rate_func_tenors: np.ndarray,
    rate_func_values: np.ndarray,
    t_start: float,
    tenor: float,
    freq: float,
) -> float:
    """Compute swap annuity from curve pillar data.

    Uses linear interpolation on the rate curve for each payment date.

    Parameters
    ----------
    rate_func_tenors : ndarray
        Curve pillar tenors.
    rate_func_values : ndarray
        Curve pillar rates.
    t_start : float
        Start of the swap (option expiry for swaptions).
    tenor : float
        Swap tenor in years.
    freq : float
        Payment frequency in year fractions.
    """
    n_periods = int(tenor / freq)
    annuity = 0.0
    for i in range(1, n_periods + 1):
        t_i = t_start + i * freq
        r_i = np.interp(t_i, rate_func_tenors, rate_func_values)
        annuity += math.exp(-r_i * t_i) * freq
    return annuity


@njit(cache=True)
def compute_forward_swap_rate(
    rate_func_tenors: np.ndarray,
    rate_func_values: np.ndarray,
    t_start: float,
    tenor: float,
    freq: float,
) -> float:
    """Compute forward par swap rate.

    F = (DF(t_start) - DF(t_end)) / annuity
    """
    r_start = np.interp(t_start, rate_func_tenors, rate_func_values)
    r_end = np.interp(t_start + tenor, rate_func_tenors, rate_func_values)
    df_start = math.exp(-r_start * t_start)
    df_end = math.exp(-r_end * (t_start + tenor))
    annuity = compute_annuity(rate_func_tenors, rate_func_values, t_start, tenor, freq)
    if annuity == 0.0:
        return 0.0
    return (df_start - df_end) / annuity
