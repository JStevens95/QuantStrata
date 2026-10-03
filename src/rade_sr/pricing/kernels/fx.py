"""
FX pricing kernels — Garman-Kohlhagen framework, @njit compiled.

Every function takes only primitive numeric inputs (float, bool) and
returns primitive outputs or tuples.  No Python objects cross the JIT
boundary.

Convention  (Garman-Kohlhagen standard)
-----------------------------------------
    S       Spot FX rate expressed as units of *domestic* (numeraire /
            pricing) currency per one unit of *foreign* (asset) currency.
            Example: for EURUSD, S ≈ 1.08 means 1 EUR costs 1.08 USD.

    r_d     Continuously-compounded domestic (numeraire) risk-free rate.
            For EURUSD this is the **USD** rate.

    r_f     Continuously-compounded foreign (asset) risk-free rate.
            For EURUSD this is the **EUR** rate.

    K       Strike in the same quotation as S.
    T       Time to expiry in year fractions (ACT/365 or similar).
    vol     Implied lognormal volatility of the spot FX rate.

IMPORTANT — mapping from FXAsset
    FXAsset uses "domestic_ir" for the *base* (asset) currency and
    "foreign_ir" for the *quote* (numeraire) currency. That is the
    **opposite** of the GK naming used here.  The instrument /
    pricer layer must map:

        r_d  ←  fx_asset.foreign_ir   (quote ccy = numeraire)
        r_f  ←  fx_asset.domestic_ir  (base ccy  = asset)

Function naming
    price_*   → present value (single trade, scalar inputs)
    greeks_*  → tuple of first-order sensitivities
"""
from __future__ import annotations

import math

import numpy as np
from ._numba import njit

from ._math import norm_cdf, norm_pdf, df_from_rate


# =====================================================================
#  FX Forward
# =====================================================================

@njit(cache=True)
def price_fx_forward(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    notional: float,
    phi: float,
) -> float:
    """Present value of an FX forward contract.

    The forward FX rate from covered interest-rate parity:

        F = S · exp((r_d − r_f) · T)

    PV expressed in domestic currency:

        PV = φ · N · [S · exp(−r_f · T) − K · exp(−r_d · T)]

    Parameters
    ----------
    S : Spot rate (domestic per foreign).
    K : Delivery / strike rate.
    T : Time to expiry (years).
    r_d : Domestic (numeraire) risk-free rate.
    r_f : Foreign (asset) risk-free rate.
    notional : Face amount in foreign-currency units.
    phi : +1.0 for long (buy foreign), −1.0 for short.
    """
    df_d = df_from_rate(r_d, T)          # exp(−r_d · T)
    df_f = df_from_rate(r_f, T)          # exp(−r_f · T)
    return phi * notional * (S * df_f - K * df_d)


@njit(cache=True)
def greeks_fx_forward(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    notional: float,
    phi: float,
) -> tuple:
    """First-order sensitivities of an FX forward.

    Returns (delta, gamma, vega, theta, rho_d, rho_f).

    delta  = φ · N · exp(−r_f · T)              per 1-unit spot move
    gamma  = 0                                    linear instrument
    vega   = 0                                    no vol dependence
    theta  = φ · N · [r_f S e^{−r_f T}
             − r_d K e^{−r_d T}] / 365           1-day decay
    rho_d  = φ · N · T · K · e^{−r_d T} / 100    per 1 % dom rate
    rho_f  = −φ · N · T · S · e^{−r_f T} / 100   per 1 % for rate
    """
    df_d = df_from_rate(r_d, T)
    df_f = df_from_rate(r_f, T)

    # Δ = ∂PV/∂S = φ N exp(−r_f T)
    delta = phi * notional * df_f

    # Forward is linear in S → second derivative vanishes
    gamma = 0.0

    # No volatility dependence
    vega = 0.0

    # θ = −∂PV/∂T / 365
    #   ∂PV/∂T = φ N [−r_f S e^{−r_f T} + r_d K e^{−r_d T}]
    theta = phi * notional * (r_f * S * df_f - r_d * K * df_d) / 365.0

    # ρ per 1 % rate shift (÷ 100)
    rho_d = phi * notional * T * K * df_d / 100.0
    rho_f = -phi * notional * T * S * df_f / 100.0

    return (delta, gamma, vega, theta, rho_d, rho_f)


# =====================================================================
#  FX Vanilla Option  (Garman-Kohlhagen)
# =====================================================================

@njit(cache=True)
def price_fx_vanilla(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    is_call: bool,
    notional: float,
) -> float:
    """Garman-Kohlhagen European FX option price.

    Call  PV = N · [S e^{−r_f T} Φ(d₁) − K e^{−r_d T} Φ(d₂)]
    Put   PV = N · [K e^{−r_d T} Φ(−d₂) − S e^{−r_f T} Φ(−d₁)]

    where
        d₁ = [ln(S/K) + (r_d − r_f + σ²/2) T] / (σ √T)
        d₂ = d₁ − σ √T

    At expiry (T ≤ 0) or zero vol the option collapses to intrinsic.
    """
    if T <= 0.0 or vol <= 0.0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return notional * intrinsic

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    df_d = df_from_rate(r_d, T)
    df_f = df_from_rate(r_f, T)

    if is_call:
        pv = S * df_f * norm_cdf(d1) - K * df_d * norm_cdf(d2)
    else:
        pv = K * df_d * norm_cdf(-d2) - S * df_f * norm_cdf(-d1)

    return notional * pv


@njit(cache=True)
def greeks_fx_vanilla(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    is_call: bool,
    notional: float,
) -> tuple:
    """Garman-Kohlhagen first-order greeks.

    Returns (delta, gamma, vega, theta, rho_d, rho_f).

    delta   call = N e^{−r_f T} Φ(d₁)
            put  = N e^{−r_f T} [Φ(d₁) − 1]
    gamma   = N e^{−r_f T} φ(d₁) / (S σ √T)
    vega    = N S e^{−r_f T} φ(d₁) √T / 100          per 1 % vol
    theta   per calendar day (see derivation below)
    rho_d   call =  N K T e^{−r_d T} Φ(d₂) / 100     per 1 % dom rate
            put  = −N K T e^{−r_d T} Φ(−d₂) / 100
    rho_f   call = −N S T e^{−r_f T} Φ(d₁) / 100     per 1 % for rate
            put  =  N S T e^{−r_f T} Φ(−d₁) / 100
    """
    if T <= 0.0 or vol <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * vol * vol) * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    nd1 = norm_pdf(d1)

    df_d = df_from_rate(r_d, T)
    df_f = df_from_rate(r_f, T)

    # ── Delta ───────────────────────────────────────────────────────
    if is_call:
        delta = notional * df_f * Nd1
    else:
        delta = notional * df_f * (Nd1 - 1.0)

    # ── Gamma ───────────────────────────────────────────────────────
    gamma = notional * df_f * nd1 / (S * vol * sqrt_T)

    # ── Vega (per 1 % absolute vol shift) ───────────────────────────
    vega = notional * S * df_f * nd1 * sqrt_T / 100.0

    # ── Theta (1-day PV decay) ──────────────────────────────────────
    #  θ = −∂PV/∂T / 365 where ∂PV/∂T decomposes into:
    #    (a) diffusion bleed:   −S σ e^{−r_f T} φ(d₁) / (2√T)
    #    (b) foreign discount:   r_f S e^{−r_f T} Φ(±d₁)
    #    (c) domestic discount: −r_d K e^{−r_d T} Φ(±d₂)
    theta_diffusion = -S * df_f * nd1 * vol / (2.0 * sqrt_T)
    if is_call:
        theta_drift = r_f * S * df_f * Nd1 - r_d * K * df_d * Nd2
    else:
        theta_drift = (
            r_f * S * df_f * (Nd1 - 1.0) - r_d * K * df_d * (Nd2 - 1.0)
        )
    theta = notional * (theta_diffusion + theta_drift) / 365.0

    # ── Rho domestic (per 1 % rate shift) ───────────────────────────
    if is_call:
        rho_d = notional * K * T * df_d * Nd2 / 100.0
    else:
        rho_d = -notional * K * T * df_d * (1.0 - Nd2) / 100.0

    # ── Rho foreign (per 1 % rate shift) ────────────────────────────
    if is_call:
        rho_f = -notional * S * T * df_f * Nd1 / 100.0
    else:
        rho_f = notional * S * T * df_f * (1.0 - Nd1) / 100.0

    return (delta, gamma, vega, theta, rho_d, rho_f)


# =====================================================================
#  FX Digital  (cash-or-nothing)
# =====================================================================

@njit(cache=True)
def price_fx_digital(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    is_call: bool,
    payout: float,
) -> float:
    """Cash-or-nothing digital option.

    Pays a fixed *payout* H in domestic currency if S_T ≷ K:

        Call  PV = H · e^{−r_d T} · Φ(d₂)
        Put   PV = H · e^{−r_d T} · Φ(−d₂)

    where
        d₂ = [ln(S/K) + (r_d − r_f − σ²/2) T] / (σ √T)
    """
    df_d = df_from_rate(r_d, T)

    if T <= 0.0 or vol <= 0.0:
        if is_call:
            return payout * df_d if S > K else 0.0
        else:
            return payout * df_d if S < K else 0.0

    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r_d - r_f - 0.5 * vol * vol) * T) / (vol * sqrt_T)

    if is_call:
        return payout * df_d * norm_cdf(d2)
    else:
        return payout * df_d * norm_cdf(-d2)


@njit(cache=True)
def greeks_fx_digital(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    vol: float,
    is_call: bool,
    payout: float,
) -> tuple:
    """Cash-or-nothing digital greeks.

    Returns (delta, gamma, vega, theta).

    delta  = w · H · DF_d · φ(d₂) / (S σ √T)
    gamma  = −w · H · DF_d · φ(d₂) · [1 + d₂/(σ√T)] / (S² σ √T)
    vega   = −w · H · DF_d · φ(d₂) · [d₂/σ + √T] / 100   (per 1 % vol)
    theta  simplified to zero (full form involves messy ∂d₂/∂T terms)

    where w = +1 for call, −1 for put.
    """
    if T <= 0.0 or vol <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r_d - r_f - 0.5 * vol * vol) * T) / (vol * sqrt_T)
    nd2 = norm_pdf(d2)
    df_d = df_from_rate(r_d, T)
    w = 1.0 if is_call else -1.0

    # ∂d₂/∂S = 1 / (S σ √T)
    delta = w * payout * df_d * nd2 / (S * vol * sqrt_T)

    # ∂delta/∂S  (second derivative of PV w.r.t. S)
    gamma = -w * payout * df_d * nd2 * (1.0 + d2 / (vol * sqrt_T)) / (S * S * vol * sqrt_T)

    # Full vega via ∂d₂/∂σ = −d₂/σ − √T
    vega = -w * payout * df_d * nd2 * (d2 / vol + sqrt_T) / 100.0

    return (delta, gamma, vega, 0.0)
