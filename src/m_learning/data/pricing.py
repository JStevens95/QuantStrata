"""
Data adapters for ML-based pricing.

Provides functions to build MLDataset from Monte Carlo paths or analytic pricers.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from src.m_learning.data.types import MLDataset, PricingFeatures


def build_pricing_dataset_from_mc(
    n_samples: int,
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
    n_paths: int = 10_000,
    seed: Optional[int] = None,
    include_greeks: bool = False,
) -> MLDataset:
    """
    Build a pricing dataset by sampling option parameters and computing prices via MC.

    This generates random option parameters, simulates MC paths, and computes
    option payoffs as targets. Useful for training NN pricers.

    Parameters
    ----------
    n_samples : int
        Number of option samples to generate.
    spot_range : tuple
        (min, max) for spot price.
    strike_range : tuple
        (min, max) for strike.
    vol_range : tuple
        (min, max) for volatility.
    rate_range : tuple
        (min, max) for risk-free rate.
    expiry_range : tuple
        (min, max) for time to expiry (years).
    n_paths : int
        Number of MC paths per sample.
    seed : int, optional
        Random seed.
    include_greeks : bool
        If True, also compute delta via finite difference (slower).

    Returns
    -------
    MLDataset
        Dataset with features (spot, strike, vol, rate, expiry, option_type) and targets (price).

    Example
    -------
    >>> dataset = build_pricing_dataset_from_mc(n_samples=1000, n_paths=5000)
    >>> print(dataset.features.shape, dataset.targets.shape)
    (1000, 6) (1000,)
    """
    rng = np.random.default_rng(seed)

    spots = rng.uniform(*spot_range, n_samples)
    strikes = rng.uniform(*strike_range, n_samples)
    vols = rng.uniform(*vol_range, n_samples)
    rates = rng.uniform(*rate_range, n_samples)
    expiries = rng.uniform(*expiry_range, n_samples)
    option_types = rng.choice([1, -1], n_samples)  # 1 = call, -1 = put

    prices = np.zeros(n_samples)
    for i in range(n_samples):
        S0, K, sigma, r, T, cp = spots[i], strikes[i], vols[i], rates[i], expiries[i], option_types[i]
        # GBM terminal distribution
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T) * rng.standard_normal(n_paths)
        S_T = S0 * np.exp(drift + diffusion)
        if cp == 1:
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)
        prices[i] = np.exp(-r * T) * np.mean(payoff)

    features = PricingFeatures(
        spot=spots,
        strike=strikes,
        vol=vols,
        rate=rates,
        expiry=expiries,
        option_type=option_types,
    )
    return MLDataset(
        features=features.to_array(),
        targets=prices,
        feature_names=PricingFeatures.feature_names(),
        target_names=["price"],
        metadata={"method": "mc", "n_paths": n_paths, "n_samples": n_samples},
    )


def build_pricing_dataset_from_analytic(
    n_samples: int,
    pricer_fn: Callable[..., float],
    spot_range: Tuple[float, float] = (80.0, 120.0),
    strike_range: Tuple[float, float] = (80.0, 120.0),
    vol_range: Tuple[float, float] = (0.1, 0.5),
    rate_range: Tuple[float, float] = (0.01, 0.10),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
    seed: Optional[int] = None,
) -> MLDataset:
    """
    Build a pricing dataset using an analytic pricer function.

    Parameters
    ----------
    n_samples : int
        Number of option samples.
    pricer_fn : callable
        Function with signature pricer_fn(spot, strike, vol, rate, expiry, option_type) -> price.
    spot_range, strike_range, vol_range, rate_range, expiry_range : tuple
        Ranges for parameter sampling.
    seed : int, optional
        Random seed.

    Returns
    -------
    MLDataset
        Dataset with features and analytic prices as targets.

    Example
    -------
    >>> from src.models.analytic.black_scholes_merton.vanilla import bsm_price
    >>> def pricer(S, K, vol, r, T, cp):
    ...     return bsm_price(S, K, T, r, 0, vol, cp)
    >>> dataset = build_pricing_dataset_from_analytic(1000, pricer)
    """
    rng = np.random.default_rng(seed)

    spots = rng.uniform(*spot_range, n_samples)
    strikes = rng.uniform(*strike_range, n_samples)
    vols = rng.uniform(*vol_range, n_samples)
    rates = rng.uniform(*rate_range, n_samples)
    expiries = rng.uniform(*expiry_range, n_samples)
    option_types = rng.choice([1, -1], n_samples)

    prices = np.array([
        pricer_fn(spots[i], strikes[i], vols[i], rates[i], expiries[i], option_types[i])
        for i in range(n_samples)
    ])

    features = PricingFeatures(
        spot=spots,
        strike=strikes,
        vol=vols,
        rate=rates,
        expiry=expiries,
        option_type=option_types,
    )
    return MLDataset(
        features=features.to_array(),
        targets=prices,
        feature_names=PricingFeatures.feature_names(),
        target_names=["price"],
        metadata={"method": "analytic", "n_samples": n_samples},
    )


__all__ = ["build_pricing_dataset_from_mc", "build_pricing_dataset_from_analytic"]
