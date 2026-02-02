"""
Quasi-Monte Carlo (QMC) Methods.

This module provides Quasi-Monte Carlo sampling using low-discrepancy sequences
for faster convergence than pseudo-random Monte Carlo.

Overview
--------
QMC replaces pseudo-random numbers with deterministic low-discrepancy sequences
that fill the unit hypercube more uniformly. This typically provides:

- Convergence rate: O(1/N) vs O(1/√N) for standard MC
- More stable estimates
- Better performance for low-to-moderate dimensions (d < 20)

Sequences Implemented
---------------------
- Sobol: Most popular for finance, good up to ~1000 dimensions
- Halton: Simple construction, good for low dimensions

Randomization
-------------
To obtain unbiased error estimates, sequences are randomized via:
- Digital shift (for Sobol)
- Random start (for Halton)

References
----------
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
- Jäckel, P. (2002). Monte Carlo Methods in Finance. Wiley.
- Sobol, I.M. (1967). "Distribution of points in a cube and approximate 
  evaluation of integrals."
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm, qmc


# =============================================================================
# Sobol Sequence Generator
# =============================================================================

@dataclass(slots=True)
class SobolRng:
    """
    Sobol sequence generator for Quasi-Monte Carlo.

    Uses scipy's Sobol implementation with optional randomization.

    Parameters
    ----------
    d : int
        Dimension of the sequence.
    seed : int, optional
        Random seed for scrambling/randomization.
    scramble : bool
        If True, use Owen scrambling for unbiased error estimates.

    Examples
    --------
    >>> rng = SobolRng(d=2, seed=42, scramble=True)
    >>> samples = rng.standard_normals(1000)  # Shape (1000, 2)
    """

    d: int
    seed: Optional[int] = None
    scramble: bool = True
    _engine: qmc.Sobol = None

    def __post_init__(self):
        self._engine = qmc.Sobol(d=self.d, scramble=self.scramble, seed=self.seed)

    def reset(self):
        """Reset the sequence to the beginning."""
        self._engine.reset()

    def uniform(self, n: int) -> np.ndarray:
        """
        Generate n points from the Sobol sequence in [0, 1]^d.

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n, d).
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        return self._engine.random(n)

    def standard_normals(self, n: int) -> np.ndarray:
        """
        Generate n points transformed to standard normal distribution.

        Uses inverse CDF (Φ^{-1}) to transform uniform [0,1] to N(0,1).

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Standard normal samples, shape (n, d).
        """
        U = self.uniform(n)
        # Clip to avoid infinities at 0 and 1
        U = np.clip(U, 1e-10, 1 - 1e-10)
        return norm.ppf(U)

    def standard_normals_antithetic(self, n: int) -> np.ndarray:
        """
        Generate n points with antithetic variates.

        Returns 2n points: the original n Sobol points and their negatives.

        Parameters
        ----------
        n : int
            Number of base points (returns 2n total).

        Returns
        -------
        np.ndarray
            Standard normal samples with antithetic pairs, shape (2n, d).
        """
        Z = self.standard_normals(n)
        return np.vstack([Z, -Z])


# =============================================================================
# Halton Sequence Generator
# =============================================================================

@dataclass(slots=True)
class HaltonRng:
    """
    Halton sequence generator for Quasi-Monte Carlo.

    Good for low dimensions (d < 10). For higher dimensions, use Sobol.

    Parameters
    ----------
    d : int
        Dimension of the sequence.
    seed : int, optional
        Random seed for randomization.
    scramble : bool
        If True, use scrambling for better uniformity.
    """

    d: int
    seed: Optional[int] = None
    scramble: bool = True
    _engine: qmc.Halton = None

    def __post_init__(self):
        self._engine = qmc.Halton(d=self.d, scramble=self.scramble, seed=self.seed)

    def reset(self):
        """Reset the sequence to the beginning."""
        self._engine.reset()

    def uniform(self, n: int) -> np.ndarray:
        """
        Generate n points from the Halton sequence in [0, 1]^d.

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Uniform samples, shape (n, d).
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        return self._engine.random(n)

    def standard_normals(self, n: int) -> np.ndarray:
        """
        Generate n points transformed to standard normal distribution.

        Parameters
        ----------
        n : int
            Number of points to generate.

        Returns
        -------
        np.ndarray
            Standard normal samples, shape (n, d).
        """
        U = self.uniform(n)
        U = np.clip(U, 1e-10, 1 - 1e-10)
        return norm.ppf(U)


# =============================================================================
# QMC Option Pricing
# =============================================================================

def qmc_european_call(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 10000,
    seed: Optional[int] = None,
    use_antithetic: bool = True,
) -> tuple[float, float]:
    """
    Price European call using Quasi-Monte Carlo (Sobol).

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of QMC samples.
    seed : int, optional
        Random seed.
    use_antithetic : bool
        Use antithetic variates.

    Returns
    -------
    tuple[float, float]
        (price, std_error) - Note: std_error for QMC is an approximation.
    """
    rng = SobolRng(d=1, seed=seed, scramble=True)

    if use_antithetic:
        Z = rng.standard_normals_antithetic(n_samples).flatten()
    else:
        Z = rng.standard_normals(n_samples).flatten()

    # Simulate terminal spot
    drift = (r - q - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * np.sqrt(maturity)
    S_T = spot0 * np.exp(drift + diffusion * Z)

    # Payoff and price
    payoffs = np.maximum(S_T - strike, 0.0)
    discount = np.exp(-r * maturity)
    price = float(discount * payoffs.mean())

    # Approximate std error (QMC error estimation is complex)
    std_error = float(discount * payoffs.std() / np.sqrt(len(payoffs)))

    return price, std_error


def qmc_european_put(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_samples: int = 10000,
    seed: Optional[int] = None,
    use_antithetic: bool = True,
) -> tuple[float, float]:
    """
    Price European put using Quasi-Monte Carlo (Sobol).

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    strike : float
        Strike price.
    maturity : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    n_samples : int
        Number of QMC samples.
    seed : int, optional
        Random seed.
    use_antithetic : bool
        Use antithetic variates.

    Returns
    -------
    tuple[float, float]
        (price, std_error).
    """
    rng = SobolRng(d=1, seed=seed, scramble=True)

    if use_antithetic:
        Z = rng.standard_normals_antithetic(n_samples).flatten()
    else:
        Z = rng.standard_normals(n_samples).flatten()

    # Simulate terminal spot
    drift = (r - q - 0.5 * sigma ** 2) * maturity
    diffusion = sigma * np.sqrt(maturity)
    S_T = spot0 * np.exp(drift + diffusion * Z)

    # Payoff and price
    payoffs = np.maximum(strike - S_T, 0.0)
    discount = np.exp(-r * maturity)
    price = float(discount * payoffs.mean())
    std_error = float(discount * payoffs.std() / np.sqrt(len(payoffs)))

    return price, std_error


def qmc_path_simulation(
    spot0: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM paths using Quasi-Monte Carlo (Sobol).

    Parameters
    ----------
    spot0 : float
        Initial spot price.
    maturity : float
        Time to maturity.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility.
    n_paths : int
        Number of paths.
    n_steps : int
        Number of time steps.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Simulated paths, shape (n_paths, n_steps + 1).
    """
    dt = maturity / n_steps

    # Generate Sobol sequence for all steps at once
    rng = SobolRng(d=n_steps, seed=seed, scramble=True)
    Z = rng.standard_normals(n_paths)

    # Build paths
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot0

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

    return paths


# =============================================================================
# QMC vs MC Comparison Utility
# =============================================================================

def compare_mc_qmc_convergence(
    spot0: float,
    strike: float,
    maturity: float,
    r: float,
    q: float,
    sigma: float,
    true_price: float,
    sample_sizes: list[int],
    n_trials: int = 10,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare MC and QMC convergence for European call pricing.

    Parameters
    ----------
    spot0, strike, maturity, r, q, sigma : float
        Option parameters.
    true_price : float
        Analytical (Black-Scholes) price for comparison.
    sample_sizes : list[int]
        List of sample sizes to test.
    n_trials : int
        Number of trials for each sample size.
    seed : int, optional
        Base random seed.

    Returns
    -------
    dict
        Dictionary with convergence results.
    """
    from src.models.numeric.monte_carlo.rng import NormalRng

    results = {
        'sample_sizes': sample_sizes,
        'mc_errors': [],
        'qmc_errors': [],
        'mc_std': [],
        'qmc_std': [],
    }

    for n in sample_sizes:
        mc_prices = []
        qmc_prices = []

        for trial in range(n_trials):
            trial_seed = seed + trial if seed is not None else None

            # Standard MC
            rng = NormalRng(seed=trial_seed)
            Z = rng.standard_normals(n, 1, antithetic=True).flatten()
            S_T = spot0 * np.exp((r - q - 0.5 * sigma ** 2) * maturity + sigma * np.sqrt(maturity) * Z)
            mc_price = np.exp(-r * maturity) * np.maximum(S_T - strike, 0).mean()
            mc_prices.append(mc_price)

            # QMC
            qmc_price, _ = qmc_european_call(spot0, strike, maturity, r, q, sigma, n, trial_seed)
            qmc_prices.append(qmc_price)

        mc_prices = np.array(mc_prices)
        qmc_prices = np.array(qmc_prices)

        results['mc_errors'].append(np.abs(mc_prices.mean() - true_price))
        results['qmc_errors'].append(np.abs(qmc_prices.mean() - true_price))
        results['mc_std'].append(mc_prices.std())
        results['qmc_std'].append(qmc_prices.std())

    return results
