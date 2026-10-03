"""
IR pricing kernels — @njit compiled for maximum throughput.

Every function takes only primitive numeric inputs. Curve data is
passed as (tenors, values) arrays — the kernel does its own interp.

Naming convention:
  - price_*      → present value (single trade)
  - greeks_*     → tuple of first-order sensitivities
  - price_*_vec  → vectorised over rate shock scenarios
"""
from __future__ import annotations

import math

import numpy as np
from ._numba import njit

from ._math import (
    norm_cdf,
    norm_pdf,
    df_from_rate,
    compute_annuity,
    compute_forward_swap_rate,
)


# =====================================================================
# IR Swap
# =====================================================================

@njit(cache=True)
def price_ir_swap(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    maturity: float,
    fixed_rate: float,
    notional: float,
    pay_receive_sign: float,
    freq: float,
) -> float:
    """PV of a plain-vanilla interest rate swap.

    PV = sign * notional * (par_rate - fixed_rate) * annuity
    """
    annuity = compute_annuity(curve_tenors, curve_values, 0.0, maturity, freq)
    par_rate = compute_forward_swap_rate(curve_tenors, curve_values, 0.0, maturity, freq)
    return pay_receive_sign * notional * (par_rate - fixed_rate) * annuity


@njit(cache=True)
def greeks_ir_swap(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    maturity: float,
    fixed_rate: float,
    notional: float,
    pay_receive_sign: float,
    freq: float,
) -> tuple:
    """IR swap greeks: (dv01, gamma, vega, theta)."""
    annuity = compute_annuity(curve_tenors, curve_values, 0.0, maturity, freq)
    dv01 = pay_receive_sign * notional * annuity * 0.0001
    theta = pay_receive_sign * notional * fixed_rate * freq / 365.0
    return (dv01, 0.0, 0.0, theta)


@njit(cache=True)
def price_ir_swap_vec(
    curve_tenors: np.ndarray,
    base_curve_values: np.ndarray,
    curve_shocks: np.ndarray,
    maturity: float,
    fixed_rate: float,
    notional: float,
    pay_receive_sign: float,
    freq: float,
) -> np.ndarray:
    """Vectorised swap PV across rate shock scenarios.

    Parameters
    ----------
    curve_shocks : (n_scenarios, n_pillars)
        Additive shocks to the base curve.

    Returns
    -------
    (n_scenarios,) PnL under each scenario.
    """
    n_scenarios = curve_shocks.shape[0]
    result = np.empty(n_scenarios, dtype=np.float64)
    base_pv = price_ir_swap(
        curve_tenors, base_curve_values, maturity, fixed_rate,
        notional, pay_receive_sign, freq,
    )

    shocked_values = np.empty_like(base_curve_values)
    for j in range(n_scenarios):
        for k in range(len(base_curve_values)):
            shocked_values[k] = base_curve_values[k] + curve_shocks[j, k]
        shocked_pv = price_ir_swap(
            curve_tenors, shocked_values, maturity, fixed_rate,
            notional, pay_receive_sign, freq,
        )
        result[j] = shocked_pv - base_pv

    return result


# =====================================================================
# IR Swaption (Bachelier / Black)
# =====================================================================

@njit(cache=True)
def price_ir_swaption_bachelier(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    option_expiry: float,
    swap_tenor: float,
    strike: float,
    notional: float,
    is_payer: bool,
    sigma_normal: float,
    freq: float,
) -> float:
    """Bachelier (normal vol) swaption pricing."""
    annuity = compute_annuity(curve_tenors, curve_values, option_expiry, swap_tenor, freq)
    fwd = compute_forward_swap_rate(curve_tenors, curve_values, option_expiry, swap_tenor, freq)

    if sigma_normal <= 0.0 or option_expiry <= 0.0:
        if is_payer:
            intrinsic = max(fwd - strike, 0.0)
        else:
            intrinsic = max(strike - fwd, 0.0)
        return notional * annuity * intrinsic

    sqrt_T = math.sqrt(option_expiry)
    d = (fwd - strike) / (sigma_normal * sqrt_T)
    sign = 1.0 if is_payer else -1.0

    pv = sign * (fwd - strike) * norm_cdf(sign * d) + sigma_normal * sqrt_T * norm_pdf(d)
    return notional * annuity * pv


@njit(cache=True)
def price_ir_swaption_black(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    option_expiry: float,
    swap_tenor: float,
    strike: float,
    notional: float,
    is_payer: bool,
    sigma_lognormal: float,
    freq: float,
) -> float:
    """Black (lognormal vol) swaption pricing."""
    annuity = compute_annuity(curve_tenors, curve_values, option_expiry, swap_tenor, freq)
    fwd = compute_forward_swap_rate(curve_tenors, curve_values, option_expiry, swap_tenor, freq)

    if sigma_lognormal <= 0.0 or option_expiry <= 0.0 or fwd <= 0.0 or strike <= 0.0:
        if is_payer:
            intrinsic = max(fwd - strike, 0.0)
        else:
            intrinsic = max(strike - fwd, 0.0)
        return notional * annuity * intrinsic

    sqrt_T = math.sqrt(option_expiry)
    d1 = (math.log(fwd / strike) + 0.5 * sigma_lognormal * sigma_lognormal * option_expiry) / (sigma_lognormal * sqrt_T)
    d2 = d1 - sigma_lognormal * sqrt_T

    if is_payer:
        pv = fwd * norm_cdf(d1) - strike * norm_cdf(d2)
    else:
        pv = strike * norm_cdf(-d2) - fwd * norm_cdf(-d1)

    return notional * annuity * pv


@njit(cache=True)
def greeks_ir_swaption_bachelier(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    option_expiry: float,
    swap_tenor: float,
    strike: float,
    notional: float,
    is_payer: bool,
    sigma_normal: float,
    freq: float,
) -> tuple:
    """Bachelier swaption greeks: (delta, gamma, vega, theta)."""
    if sigma_normal <= 0.0 or option_expiry <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    annuity = compute_annuity(curve_tenors, curve_values, option_expiry, swap_tenor, freq)
    fwd = compute_forward_swap_rate(curve_tenors, curve_values, option_expiry, swap_tenor, freq)

    sqrt_T = math.sqrt(option_expiry)
    d = (fwd - strike) / (sigma_normal * sqrt_T)
    n_d = norm_pdf(d)

    if is_payer:
        N_d = norm_cdf(d)
    else:
        N_d = norm_cdf(-d)

    delta = notional * annuity * N_d
    gamma = notional * annuity * n_d / (sigma_normal * sqrt_T)
    vega = notional * annuity * n_d * sqrt_T / 10000.0  # per 1bp normal vol

    return (delta, gamma, vega, 0.0)


# =====================================================================
# IR Cap/Floor (strip of caplets/floorlets)
# =====================================================================

@njit(cache=True)
def price_ir_capfloor_bachelier(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    maturity: float,
    strike: float,
    notional: float,
    is_cap: bool,
    sigma_normal: float,
    freq: float,
) -> float:
    """Bachelier cap/floor pricing as sum of caplets."""
    n_caplets = int(maturity / freq)
    total_pv = 0.0
    sign = 1.0 if is_cap else -1.0

    for i in range(1, n_caplets + 1):
        t_fix = i * freq
        t_pay = t_fix + freq

        # Forward rate for this caplet period
        r_fix = np.interp(t_fix, curve_tenors, curve_values)
        r_pay = np.interp(t_pay, curve_tenors, curve_values)
        df_fix = math.exp(-r_fix * t_fix)
        df_pay = math.exp(-r_pay * t_pay)
        if df_pay == 0.0:
            continue
        fwd = (df_fix / df_pay - 1.0) / freq

        if sigma_normal <= 0.0 or t_fix <= 0.0:
            intrinsic = max(sign * (fwd - strike), 0.0)
            total_pv += notional * freq * df_pay * intrinsic
        else:
            sqrt_T = math.sqrt(t_fix)
            d = (fwd - strike) / (sigma_normal * sqrt_T)
            caplet_pv = sign * (fwd - strike) * norm_cdf(sign * d) + sigma_normal * sqrt_T * norm_pdf(d)
            total_pv += notional * freq * df_pay * caplet_pv

    return total_pv


@njit(cache=True)
def price_ir_capfloor_black(
    curve_tenors: np.ndarray,
    curve_values: np.ndarray,
    maturity: float,
    strike: float,
    notional: float,
    is_cap: bool,
    sigma_lognormal: float,
    freq: float,
) -> float:
    """Black (lognormal) cap/floor pricing as sum of caplets."""
    n_caplets = int(maturity / freq)
    total_pv = 0.0

    for i in range(1, n_caplets + 1):
        t_fix = i * freq
        t_pay = t_fix + freq

        r_fix = np.interp(t_fix, curve_tenors, curve_values)
        r_pay = np.interp(t_pay, curve_tenors, curve_values)
        df_fix = math.exp(-r_fix * t_fix)
        df_pay = math.exp(-r_pay * t_pay)
        if df_pay == 0.0:
            continue
        fwd = (df_fix / df_pay - 1.0) / freq

        if sigma_lognormal <= 0.0 or t_fix <= 0.0 or fwd <= 0.0 or strike <= 0.0:
            if is_cap:
                intrinsic = max(fwd - strike, 0.0)
            else:
                intrinsic = max(strike - fwd, 0.0)
            total_pv += notional * freq * df_pay * intrinsic
        else:
            sqrt_T = math.sqrt(t_fix)
            d1 = (math.log(fwd / strike) + 0.5 * sigma_lognormal * sigma_lognormal * t_fix) / (sigma_lognormal * sqrt_T)
            d2 = d1 - sigma_lognormal * sqrt_T

            if is_cap:
                caplet_pv = fwd * norm_cdf(d1) - strike * norm_cdf(d2)
            else:
                caplet_pv = strike * norm_cdf(-d2) - fwd * norm_cdf(-d1)

            total_pv += notional * freq * df_pay * caplet_pv

    return total_pv
