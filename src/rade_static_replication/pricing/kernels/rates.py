"""
Rates pricing kernels — par swaps + Bachelier (normal-vol) swaptions, ``@njit`` compiled.

Normal vols are used because swap rates can be negative. As with the FX kernels there are
scalar and ``*_vec`` (scenario-window) variants. Curve inputs are zero-rate pillars
``(curve_tenors, curve_values)``; the annuity and forward swap rate are rebuilt from them
so a shocked curve flows straight through into PnL.
"""
from __future__ import annotations

import math

import numpy as np
from src.rade_static_replication.pricing.kernels._numba import njit
from src.rade_static_replication.pricing.kernels._math import (
    norm_cdf,
    norm_pdf,
    compute_annuity,
    compute_forward_swap_rate,
)


@njit(cache=True)
def price_ir_swap(curve_tenors, curve_values, maturity, fixed_rate, notional, pay_receive_sign, freq):
    """PV of a par swap: ``sign * notional * (par_rate - fixed_rate) * annuity``."""
    annuity = compute_annuity(curve_tenors, curve_values, 0.0, maturity, freq)
    par_rate = compute_forward_swap_rate(curve_tenors, curve_values, 0.0, maturity, freq)
    return pay_receive_sign * notional * (par_rate - fixed_rate) * annuity


@njit(cache=True)
def price_ir_swaption_bachelier(
    curve_tenors, curve_values, option_expiry, swap_tenor, strike, notional, is_payer, normal_vol, freq,
):
    """Bachelier swaption PV on the forward swap rate."""
    annuity = compute_annuity(curve_tenors, curve_values, option_expiry, swap_tenor, freq)
    forward_rate = compute_forward_swap_rate(curve_tenors, curve_values, option_expiry, swap_tenor, freq)

    # Degenerate market: discounted intrinsic on the annuity.
    if normal_vol <= 0.0 or option_expiry <= 0.0:
        intrinsic = max(forward_rate - strike, 0.0) if is_payer else max(strike - forward_rate, 0.0)
        return notional * annuity * intrinsic

    sqrt_t = math.sqrt(option_expiry)
    standardised_moneyness = (forward_rate - strike) / (normal_vol * sqrt_t)
    payer_receiver_sign = 1.0 if is_payer else -1.0
    # Bachelier call/put value per unit annuity: sign*(F-K)*N(sign*d) + vol*sqrt(T)*phi(d).
    option_value = (
        payer_receiver_sign * (forward_rate - strike) * norm_cdf(payer_receiver_sign * standardised_moneyness)
        + normal_vol * sqrt_t * norm_pdf(standardised_moneyness)
    )
    return notional * annuity * option_value


@njit(cache=True)
def price_ir_swap_vec(curve_tenors, curve_paths, maturity, fixed_rate, notional, pay_receive_sign, freq):
    """PV of one swap across ``n`` scenarios; ``curve_paths`` is ``(n, n_pillars)``."""
    n = curve_paths.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_ir_swap(curve_tenors, curve_paths[i], maturity, fixed_rate, notional, pay_receive_sign, freq)
    return pv


@njit(cache=True)
def price_ir_swaption_bachelier_vec(
    curve_tenors, curve_paths, option_expiry, swap_tenor, strike, notional, is_payer, normal_vol_path, freq,
):
    """PV of one swaption across ``n`` scenarios; ``normal_vol_path`` is ``(n,)``."""
    n = curve_paths.shape[0]
    pv = np.empty(n, dtype=np.float64)
    for i in range(n):
        pv[i] = price_ir_swaption_bachelier(
            curve_tenors, curve_paths[i], option_expiry, swap_tenor, strike,
            notional, is_payer, normal_vol_path[i], freq,
        )
    return pv
