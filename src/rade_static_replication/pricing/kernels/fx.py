"""
FX pricing kernels — Garman-Kohlhagen, ``@njit`` compiled.

Two flavours per instrument:

* **scalar** kernels (``price_fx_*``) price one trade in one market state;
* **vectorised** kernels (``price_fx_*_vec``) price one trade across a whole scenario
  window, looping in compiled code.

Conventions: ``spot`` is domestic-per-foreign, ``rate_dom`` the domestic (quote-ccy)
continuously-compounded rate, ``rate_for`` the foreign (base-ccy) rate. The foreign
currency behaves like a continuous dividend yield, hence the dual discounting. Inputs
are primitives / NumPy arrays only so the whole call stays inside the JIT boundary.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit
from src.rade_static_replication.pricing.kernels._math import norm_cdf, df_from_rate


@njit(cache=True)
def price_fx_forward(spot, strike, expiry, rate_dom, rate_for, notional, direction):
    """PV of an FX forward.

    ``direction`` is +1 for a long (buy-foreign) forward, -1 for short. The value is the
    discounted difference between the delivered spot and the agreed strike.
    """
    foreign_leg = spot * df_from_rate(rate_for, expiry)   # PV of receiving 1 unit foreign
    strike_leg = strike * df_from_rate(rate_dom, expiry)  # PV of paying the strike
    return direction * notional * (foreign_leg - strike_leg)


@njit(cache=True)
def price_fx_vanilla(spot, strike, expiry, rate_dom, rate_for, vol, is_call, notional):
    """Garman-Kohlhagen European FX option PV."""
    # Degenerate market: no time value left, fall back to (undiscounted) intrinsic.
    if expiry <= 0.0 or vol <= 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        return notional * intrinsic

    sqrt_t = math.sqrt(expiry)
    # Standard GK d1/d2 with the (rate_dom - rate_for) drift.
    d1 = (math.log(spot / strike) + (rate_dom - rate_for + 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_dom = df_from_rate(rate_dom, expiry)
    df_for = df_from_rate(rate_for, expiry)

    if is_call:
        return notional * (spot * df_for * norm_cdf(d1) - strike * df_dom * norm_cdf(d2))
    return notional * (strike * df_dom * norm_cdf(-d2) - spot * df_for * norm_cdf(-d1))


@njit(cache=True)
def price_fx_digital(spot, strike, expiry, rate_dom, rate_for, vol, is_call, payout):
    """Cash-or-nothing digital: pays ``payout`` (domestic) if S_T is beyond the strike."""
    df_dom = df_from_rate(rate_dom, expiry)
    # Degenerate market: pay the (discounted) payout iff currently in-the-money.
    if expiry <= 0.0 or vol <= 0.0:
        if is_call:
            return payout * df_dom if spot > strike else 0.0
        return payout * df_dom if spot < strike else 0.0

    sqrt_t = math.sqrt(expiry)
    # Probability of finishing ITM under Q is N(d2) (call) / N(-d2) (put).
    d2 = (math.log(spot / strike) + (rate_dom - rate_for - 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    return payout * df_dom * (norm_cdf(d2) if is_call else norm_cdf(-d2))


@njit(cache=True)
def price_fx_vanilla_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, vol_path, is_call, notional):
    """PV of one vanilla across ``n`` scenarios; market inputs are ``(n,)`` arrays."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_vanilla(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], vol_path[i], is_call, notional,
        )
    return pv


@njit(cache=True)
def price_fx_digital_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, vol_path, is_call, payout):
    """PV of one digital across ``n`` scenarios."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_digital(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], vol_path[i], is_call, payout,
        )
    return pv


@njit(cache=True)
def price_fx_forward_vec(spot_path, strike, expiry, rate_dom_path, rate_for_path, notional, direction):
    """PV of one forward across ``n`` scenarios."""
    n = spot_path.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_fx_forward(
            spot_path[i], strike, expiry, rate_dom_path[i], rate_for_path[i], notional, direction,
        )
    return pv
