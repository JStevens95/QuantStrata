from __future__ import annotations

import numpy as np
from typing import Tuple, Literal
from dataclasses import dataclass

DepositCompounding = Literal["simple", "continuous"]


@dataclass(frozen=True, slots=True)
class SpotGbmSpec:
    """
    GBM spot generator spec.

    Notes
    -----
    - If n_time == 1 (single as-of), we keep spot identical across scenarios by default
      (spot[0, :] = initial_level). This matches the test behaviour you just fixed.
    - If you *want* cross-scenario dispersion at t=0, set `initial_dispersion > 0`.
    """
    initial_level: float
    drift: float = 0.0
    vol: float = 0.10
    dt: float = 1.0 / 252.0
    initial_dispersion: float = 0.0  # lognormal dispersion applied at t=0 if > 0


@dataclass(frozen=True, slots=True)
class CurveZeroSpec:
    """
    Zero-rate curve generator spec.

    Produces params shaped [T, S, K, 2] where each [K,2] block is [tenor, zero_rate].
    """
    tenors: np.ndarray
    base_rate: float = 0.02
    slope: float = 0.00          # linear term structure component
    curvature: float = 0.00      # small hump/decay component
    noise_scale: float = 0.0005  # per (time,scenario) noise on rates
    extrapolation: str = "flat"


@dataclass(frozen=True, slots=True)
class CurveBootstrapSpec:
    """
    Synthetic curve generator spec: create deposit + par swap quotes, then bootstrap a curve.

    This becomes a *better default* than a flat curve because:
      - you get a term structure,
      - snapshot() stays simple (still a tenor/rate grid),
      - tests/examples stop relying on flat inputs.

    Parameters
    ----------
    deposit_maturities:
        Short-end maturities in years.
    swap_maturities:
        Longer maturities in years (bootstrapped using par swap identity).
    base_rate:
        Base level for the curve.
    slope:
        Controls how much rates increase with maturity (very simple shape proxy).
    pay_freq:
        Fixed leg payments per year for swaps (1=annual, 2=semi, 4=quarterly).
    compounding:
        Deposit compounding for deposit quotes.
    noise_scale:
        Small per-(time,scenario) additive noise on rates to avoid perfectly static curves.
    """
    deposit_maturities: Tuple[float, ...] = (1.0 / 12.0, 0.25, 0.50)
    swap_maturities: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    base_rate: float = 0.02
    slope: float = 0.006

    pay_freq: int = 2
    compounding: DepositCompounding = "simple"

    noise_scale: float = 0.00025


@dataclass(frozen=True, slots=True)
class VolGridSmileSpec:
    """
    Grid vol surface generator spec (expiry x strike), stored as flattened params.

    We keep absolute strikes for now (your preference).
    """
    expiries: np.ndarray
    strikes: np.ndarray
    atm_vol: float = 0.12
    skew: float = -0.15          # smile skew vs log-moneyness proxy
    smile: float = 0.20          # curvature vs log-moneyness proxy
    term: float = 0.10           # term structure strength
    noise_scale: float = 0.002
    extrapolation: str = "flat"