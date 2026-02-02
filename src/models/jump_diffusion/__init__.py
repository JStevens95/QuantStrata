"""
Jump-Diffusion Models.

This module provides implementations of jump-diffusion models for pricing
derivatives where the underlying asset experiences both continuous diffusion
and discrete jumps.

Models Implemented
------------------
- MertonJumpDiffusion: Classic Merton (1976) jump-diffusion model
  (GBM + compound Poisson process with log-normal jumps)

Mathematical Framework
----------------------
The general jump-diffusion SDE is:

    dS_t / S_t = (μ - λκ) dt + σ dW_t + dJ_t

where:
    - μ: Drift rate
    - σ: Diffusion volatility
    - W_t: Brownian motion
    - J_t: Compound Poisson process with intensity λ
    - κ = E[J - 1]: Expected relative jump size

Key Features
------------
- Captures fat tails and volatility clustering
- Models market crashes via jump component
- Retains tractability for European options (semi-closed form)
- Allows calibration to market smiles

Usage
-----
>>> from src.models.jump_diffusion import (
...     MertonParameters,
...     MertonDynamics,
...     MertonSimulation,
... )
>>>
>>> params = MertonParameters(
...     sigma=0.2,      # Diffusion volatility
...     lambda_=0.5,    # Jump intensity (0.5 jumps/year on average)
...     mu_j=-0.1,      # Log-jump mean (negative = crash-like)
...     sigma_j=0.2,    # Log-jump std dev
... )
>>> dynamics = MertonDynamics(params=params, drift=0.05)
>>> sim = dynamics.simulate(spot0=100, maturity=1.0, n_paths=10000, n_steps=252)
"""

from src.models.jump_diffusion.merton import (
    MertonParameters,
    MertonDynamics,
    MertonSimulation,
)

__all__ = [
    "MertonParameters",
    "MertonDynamics",
    "MertonSimulation",
]
