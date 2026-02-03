"""
Delta hedging simulation and training data.

Provides GBM path generation, Black-Scholes price/delta at rebalancing times,
and dataset construction for ML-based delta hedging (features = state, target = BSM delta).

Used by evaluation/delta_hedging_backtest.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.models.analytic.black_scholes_merton.base import (
    vanilla_greeks,
    vanilla_price,
)


@dataclass
class HedgingPath:
    """
    Single simulated path with spot, option value, and BSM delta at each rebalance time.

    Attributes
    ----------
    times : np.ndarray
        Shape (N+1,). t_0=0, ..., t_N=T.
    spot : np.ndarray
        Shape (N+1,). Spot price at each time.
    option_value : np.ndarray
        Shape (N+1,). BSM option value at each time.
    delta : np.ndarray
        Shape (N+1,). BSM delta at each time.
    """

    times: np.ndarray
    spot: np.ndarray
    option_value: np.ndarray
    delta: np.ndarray

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1


def _option_type_from_int(cp: int) -> str:
    return "call" if cp >= 0 else "put"


def _bsm_price_and_delta(
    spot: float,
    strike: float,
    expiry: float,
    r: float,
    vol: float,
    option_type_int: int,
) -> Tuple[float, float]:
    """Return (BSM price, BSM delta). option_type_int: 1 = call, -1 = put."""
    opt = _option_type_from_int(option_type_int)
    pv = vanilla_price(
        option_type=opt,
        spot=spot,
        strike=strike,
        expiry=max(expiry, 1e-10),
        discount_rate=r,
        carry=r,
        vol=vol,
    )
    greeks = vanilla_greeks(
        option_type=opt,
        spot=spot,
        strike=strike,
        expiry=max(expiry, 1e-10),
        discount_rate=r,
        carry=r,
        vol=vol,
    )
    return pv, greeks["delta"]


def generate_gbm_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate one GBM path at rebalancing times 0, dt, 2*dt, ..., T. Shape (n_steps + 1,)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)
    log_returns = drift + vol_sqrt_dt * rng.standard_normal(n_steps)
    path = np.empty(n_steps + 1)
    path[0] = S0
    path[1:] = S0 * np.exp(np.cumsum(log_returns))
    return path


def simulate_hedging_path(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> HedgingPath:
    """Simulate one path: GBM spot and BSM option value + delta at each rebalance time."""
    spot_path = generate_gbm_path(S0, r, sigma, T, n_steps, seed=seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    option_value = np.zeros(n_steps + 1)
    delta = np.zeros(n_steps + 1)

    for k in range(n_steps + 1):
        t = times[k]
        S = spot_path[k]
        tau = T - t
        pv, d = _bsm_price_and_delta(S, K, tau, r, sigma, option_type)
        option_value[k] = pv
        delta[k] = d

    return HedgingPath(
        times=times,
        spot=spot_path,
        option_value=option_value,
        delta=delta,
    )


def simulate_hedging_paths(
    n_paths: int,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> list[HedgingPath]:
    """Simulate multiple paths with different seeds."""
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_paths)]
    return [
        simulate_hedging_path(S0, K, T, r, sigma, option_type, n_steps, seed=s)
        for s in seeds
    ]


DELTA_HEDGE_FEATURE_NAMES = [
    "moneyness",
    "time_to_expiry",
    "vol",
    "rate",
    "option_type",
]


def path_to_feature_target_arrays(
    path: HedgingPath,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (features, targets) from one path for all rebalance points.
    features: (n_steps+1, 5), targets: (n_steps+1,).
    """
    n = len(path.times)
    features = np.zeros((n, 5), dtype=np.float32)
    for k in range(n):
        t = path.times[k]
        S = path.spot[k]
        tau = max(T - t, 1e-10)
        features[k] = [
            S / K,
            tau,
            sigma,
            r,
            float(option_type),
        ]
    targets = path.delta.astype(np.float32)
    return features, targets


def build_delta_hedging_dataset(
    n_paths: int,
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    option_type: int = 1,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature and target arrays for delta hedging ML.
    Returns features (n_samples, 5), targets (n_samples,).
    """
    paths = simulate_hedging_paths(
        n_paths=n_paths,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        n_steps=n_steps,
        seed=seed,
    )
    feats_list: list[np.ndarray] = []
    tgt_list: list[np.ndarray] = []
    for p in paths:
        f, t = path_to_feature_target_arrays(p, K, T, r, sigma, option_type)
        feats_list.append(f)
        tgt_list.append(t)
    features = np.vstack(feats_list)
    targets = np.concatenate(tgt_list)
    return features, targets


__all__ = [
    "HedgingPath",
    "generate_gbm_path",
    "simulate_hedging_path",
    "simulate_hedging_paths",
    "DELTA_HEDGE_FEATURE_NAMES",
    "path_to_feature_target_arrays",
    "build_delta_hedging_dataset",
]
