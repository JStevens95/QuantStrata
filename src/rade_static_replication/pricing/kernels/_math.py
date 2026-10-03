"""
Shared numeric utilities for pricing kernels — all ``@njit`` compiled.

Normal CDF/PDF via ``erfc`` so the entire pricing path stays in compiled code with
no scipy dependency inside the JIT boundary.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit, float64


@njit(float64(float64), cache=True)
def norm_cdf(x: float) -> float:
    """Standard normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@njit(float64(float64), cache=True)
def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit(float64(float64, float64), cache=True)
def df_from_rate(rate: float, t: float) -> float:
    """Continuous discount factor ``exp(-r t)``."""
    return math.exp(-rate * t)


@njit(cache=True)
def compute_annuity(tenors: np.ndarray, values: np.ndarray, t_start: float, tenor: float, freq: float) -> float:
    """Swap annuity from curve pillars (linear interp on the zero rate per pay date)."""
    n_periods = int(tenor / freq)
    annuity = 0.0
    for i in range(1, n_periods + 1):
        t_i = t_start + i * freq
        r_i = np.interp(t_i, tenors, values)
        annuity += math.exp(-r_i * t_i) * freq
    return annuity


@njit(cache=True)
def compute_forward_swap_rate(tenors: np.ndarray, values: np.ndarray, t_start: float, tenor: float, freq: float) -> float:
    """Forward par swap rate ``(DF(t_start) - DF(t_end)) / annuity``."""
    r_start = np.interp(t_start, tenors, values)
    r_end = np.interp(t_start + tenor, tenors, values)
    df_start = math.exp(-r_start * t_start)
    df_end = math.exp(-r_end * (t_start + tenor))
    annuity = compute_annuity(tenors, values, t_start, tenor, freq)
    if annuity == 0.0:
        return 0.0
    return (df_start - df_end) / annuity
