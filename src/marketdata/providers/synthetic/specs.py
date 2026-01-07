from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from src.marketdata.core.types import BootstrapEngine, DepositCompounding


@dataclass(frozen=True, slots=True)
class SpotGbmSpec:
    """
    GBM spot generator spec.

    Notes
    -----
    - If n_time == 1 (single as-of), we keep spot identical across scenarios by default
      (spot[0, :] = initial_level).
    - If you want cross-scenario dispersion at t=0, set initial_dispersion > 0.
    """
    initial_level: float
    drift: float = 0.0
    vol: float = 0.10
    dt: float = 1.0 / 252.0
    initial_dispersion: float = 0.0  # lognormal dispersion at t=0 if > 0


@dataclass(frozen=True, slots=True)
class CurveZeroSpec:
    """
    Zero-rate curve generator spec.

    Output params format
    --------------------
    Produces params shaped [T, S, K, 2] where each [K,2] block is:
        [tenor, zero_rate]
    """
    tenors: np.ndarray
    base_rate: float = 0.02
    slope: float = 0.00
    curvature: float = 0.00
    noise_scale: float = 0.0005
    extrapolation: str = "flat"


@dataclass(frozen=True, slots=True)
class CurveBootstrapSpec:
    """
    Synthetic curve generator spec: create deposit + par swap quotes then bootstrap a curve.

    Design
    ------
    - Uses your existing bootstrapper API (native today; QuantLib backend later).
    - Provider stores the *resulting* curve as a tenor/zero grid (so snapshot() stays simple).

    Parameters
    ----------
    deposit_maturities:
        Short-end maturities in years (base instruments).
    swap_maturities:
        Longer maturities in years (bootstrapped using par swap identity).
    base_rate, slope:
        Very simple curve-shape proxy.
    pay_freq:
        Fixed leg payments per year (1=annual, 2=semi, 4=quarterly).
    compounding:
        Deposit compounding for deposit quotes.
    noise_scale:
        Small per-(time,scenario) additive noise on rates.
    engine:
        Bootstrap backend:
        - "native"    : your own bootstrapper implementation
        - "quantlib"  : QuantLib-based helpers (when wired into bootstrapper.py)
    """
    deposit_maturities: Tuple[float, ...] = (1.0 / 12.0, 0.25, 0.50)
    swap_maturities: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)

    base_rate: float = 0.02
    slope: float = 0.006

    pay_freq: int = 2
    compounding: DepositCompounding = "simple"

    noise_scale: float = 0.00025
    engine: BootstrapEngine = "native"


@dataclass(frozen=True, slots=True)
class VolGridSmileSpec:
    """
    Grid vol surface generator spec (expiry x strike), stored as flattened params.

    Notes
    -----
    - Strikes are treated as absolute strikes (your current preference).
    - The provider generates a simple parametric smile in log-moneyness proxy + term structure.
    """
    expiries: np.ndarray
    strikes: np.ndarray
    atm_vol: float = 0.12
    skew: float = -0.15
    smile: float = 0.20
    term: float = 0.10
    noise_scale: float = 0.002
    extrapolation: str = "flat"