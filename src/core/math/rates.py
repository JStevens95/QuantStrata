"""
Rate and Discount Factor Utilities.

Common utility functions for converting between discount factors and rates.
"""

from __future__ import annotations

import math


def rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    Given DF(t) = exp(-r × t), we have r = -ln(DF) / t.

    Parameters
    ----------
    df : float
        Discount factor DF(t).
    t : float
        Time to maturity in years.

    Returns
    -------
    float
        Continuously-compounded zero rate r.

    Raises
    ------
    ValueError
        If df <= 0.

    Examples
    --------
    >>> rate_from_df(df=0.97, t=1.0)  # ~3%
    0.030459...
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


def df_from_rate(*, r: float, t: float) -> float:
    """
    Convert continuously-compounded rate to discount factor.

    DF(t) = exp(-r × t).

    Parameters
    ----------
    r : float
        Continuously-compounded zero rate.
    t : float
        Time to maturity in years.

    Returns
    -------
    float
        Discount factor DF(t).

    Examples
    --------
    >>> df_from_rate(r=0.03, t=1.0)  # ~97%
    0.970446...
    """
    if t <= 0.0:
        return 1.0
    return float(math.exp(-r * t))


def forward_rate(*, df1: float, df2: float, t1: float, t2: float) -> float:
    """
    Compute continuously-compounded forward rate between two dates.

    F(t1, t2) = -ln(DF2/DF1) / (t2 - t1).

    Parameters
    ----------
    df1 : float
        Discount factor to t1.
    df2 : float
        Discount factor to t2.
    t1 : float
        Start time in years.
    t2 : float
        End time in years.

    Returns
    -------
    float
        Forward rate F(t1, t2).

    Raises
    ------
    ValueError
        If t2 <= t1 or discount factors <= 0.
    """
    if t2 <= t1:
        raise ValueError(f"t2 must be > t1; got t1={t1}, t2={t2}.")
    if df1 <= 0.0 or df2 <= 0.0:
        raise ValueError(f"Discount factors must be > 0; got df1={df1}, df2={df2}.")
    return float(-math.log(df2 / df1) / (t2 - t1))


__all__ = [
    "rate_from_df",
    "df_from_rate",
    "forward_rate",
]
