"""
Lévy Process Models.

This module provides implementations of Lévy process models for pricing
derivatives where the underlying follows a pure-jump or jump-diffusion process.

Models Implemented
------------------
- VarianceGamma: Time-changed Brownian motion with Gamma subordinator

Mathematical Framework
----------------------
A Lévy process X_t is characterized by:
1. Independent increments
2. Stationary increments
3. Stochastically continuous paths

The Variance Gamma process is constructed as:
    X_t = θ G_t + σ W_{G_t}

where:
    - G_t ~ Gamma(t/ν, 1/ν) is the subordinator
    - W_t is standard Brownian motion
    - θ is the drift parameter
    - σ is the volatility parameter
    - ν controls the variance rate of time (fat tails)

Key Features
------------
- Pure jump process (no diffusion component)
- Finite activity of jumps
- Fat tails controlled by ν parameter
- Skewness controlled by θ parameter
- Semi-closed form for European options (FFT methods)

Usage
-----
>>> from src.models.levy import (
...     VarianceGammaParameters,
...     VarianceGammaDynamics,
...     VarianceGammaSimulation,
... )
>>>
>>> params = VarianceGammaParameters(
...     theta=-0.1,   # Negative drift (skew)
...     sigma=0.2,    # Volatility
...     nu=0.2,       # Variance rate of Gamma time
... )
>>> dynamics = VarianceGammaDynamics(params=params, drift=0.05)
>>> sim = dynamics.simulate(spot0=100, maturity=1.0, n_paths=10000, n_steps=252)
"""

from src.models.levy.variance_gamma import (
    VarianceGammaParameters,
    VarianceGammaDynamics,
    VarianceGammaSimulation,
    vg_european_call,
    vg_european_put,
)

__all__ = [
    "VarianceGammaParameters",
    "VarianceGammaDynamics",
    "VarianceGammaSimulation",
    "vg_european_call",
    "vg_european_put",
]
