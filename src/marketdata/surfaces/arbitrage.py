from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _bs_fx_forward_call_price(
    *,
    forward: float,
    strike: float,
    vol: float,
    expiry: float,
) -> float:
    """
    FX Black-Scholes call *forward* price:
        C_fwd = F N(d1) - K N(d2)

    (This is the domestic-discounted price divided by df_dom.)
    """
    t = float(expiry)
    if t <= 0.0:
        return max(float(forward) - float(strike), 0.0)

    f = float(forward)
    k = float(strike)
    sigma = float(vol)
    if f <= 0.0 or k <= 0.0 or sigma <= 0.0:
        raise ValueError("forward/strike/vol must be > 0.")

    st = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / st
    d2 = d1 - st
    return f * _norm_cdf(d1) - k * _norm_cdf(d2)


def total_variance_grid(expiries: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """w(T,K) = sigma(T,K)^2 * T; vols shape [nT, nK]."""
    t = np.asarray(expiries, dtype=float).reshape(-1, 1)
    v = np.asarray(vols, dtype=float)
    return (v * v) * t


def check_calendar_no_arb_total_variance(
    *,
    expiries: np.ndarray,
    strikes: np.ndarray,
    vols: np.ndarray,
    tol: float = 1e-12,
) -> None:
    """
    Calendar sanity (V2-lite): total variance must be non-decreasing in T for each strike.
    """
    e = np.asarray(expiries, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)
    v = np.asarray(vols, dtype=float)

    if e.size <= 1:
        return

    w = total_variance_grid(e, v)
    dw = np.diff(w, axis=0)

    if np.any(dw < -float(tol)):
        i, j = np.argwhere(dw < -float(tol))[0]
        raise ValueError(
            "Calendar arbitrage sanity check failed (total variance decreased).\n"
            f"  strike={float(k[j])}\n"
            f"  T0={float(e[i])}, w0={float(w[i, j])}\n"
            f"  T1={float(e[i+1])}, w1={float(w[i+1, j])}\n"
        )


def _check_decreasing_and_convex(
    *,
    x: np.ndarray,
    y: np.ndarray,
    tol: float,
    label: str,
) -> None:
    """
    Discrete check: y(x) must be non-increasing and convex in x.

    For call prices vs strike:
      - C(K) decreasing: first differences <= 0
      - C(K) convex: second differences >= 0
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"{label}: x and y length mismatch.")
    if x.size < 3:
        return

    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f"{label}: x must be strictly increasing.")

    dy = np.diff(y)
    if np.any(dy > float(tol)):
        idx = int(np.where(dy > float(tol))[0][0])
        raise ValueError(
            f"{label}: expected non-increasing; found increase at i={idx} "
            f"(y[i]={float(y[idx])} -> y[i+1]={float(y[idx+1])})."
        )

    d2y = y[:-2] - 2.0 * y[1:-1] + y[2:]
    if np.any(d2y < -float(tol)):
        idx = int(np.where(d2y < -float(tol))[0][0]) + 1
        raise ValueError(
            f"{label}: expected convex; found concavity near i={idx} "
            f"(second-diff={float(d2y[idx-1])})."
        )


@dataclass(frozen=True, slots=True)
class FxSurfaceArbitrageConfig:
    """
    Arbitrage validation config for FX GridVolSurface.

    tol:
      Numerical tolerance used in monotonicity/convexity checks.
    check_butterfly:
      If True, compute call prices and enforce decreasing+convex in strike for each expiry.
    """
    tol: float = 1e-10
    check_butterfly: bool = True


def check_fx_grid_surface_no_static_arb(
    *,
    expiries: np.ndarray,
    strikes: np.ndarray,
    vols: np.ndarray,
    spot: float,
    df_domestic: Callable[[float], float],
    df_foreign: Callable[[float], float],
    config: FxSurfaceArbitrageConfig = FxSurfaceArbitrageConfig(),
) -> None:
    """
    V2 checks:
      1) calendar no-arb sanity on total variance
      2) (optional) butterfly no-arb sanity via call price convexity in strike

    Notes
    -----
    - This is a *sanity/guardrail* implementation, not a full production arb-free fit.
    """
    check_calendar_no_arb_total_variance(expiries=expiries, strikes=strikes, vols=vols, tol=config.tol)

    if not config.check_butterfly:
        return

    s0 = float(spot)
    if s0 <= 0.0:
        raise ValueError("spot must be > 0.")

    e = np.asarray(expiries, dtype=float).reshape(-1)
    k = np.asarray(strikes, dtype=float).reshape(-1)
    v = np.asarray(vols, dtype=float)

    for i, t in enumerate(e.tolist()):
        df_d = float(df_domestic(float(t)))
        df_f = float(df_foreign(float(t)))
        if df_d <= 0.0 or df_f <= 0.0:
            raise ValueError("df(t) must be > 0.")

        fwd = s0 * df_f / df_d

        call_fwd = np.array(
            [_bs_fx_forward_call_price(forward=fwd, strike=float(kk), vol=float(v[i, j]), expiry=float(t))
             for j, kk in enumerate(k.tolist())],
            dtype=float,
        )
        _check_decreasing_and_convex(x=k, y=call_fwd, tol=float(config.tol), label=f"Butterfly check @ T={t}")