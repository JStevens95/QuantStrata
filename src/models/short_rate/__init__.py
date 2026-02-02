"""
Short Rate Models.

This module provides implementations of short rate models for pricing
interest rate derivatives. Short rate models describe the evolution of
the instantaneous interest rate r(t).

Available Models
----------------
- Hull-White (1-factor): Mean-reverting Gaussian short rate
- Black-Karasinski (1-factor): Log-normal short rate (positive rates only)

Mathematical Framework
----------------------
Short rate models specify the dynamics of r(t) under the risk-neutral measure Q:

    dr(t) = μ(r,t) dt + σ(r,t) dW(t)

The zero-coupon bond price is given by the expectation:

    P(t,T) = E^Q[exp(-∫_t^T r(s) ds) | F_t]

For affine models (like Hull-White), this has a closed-form solution.
For non-affine models (like Black-Karasinski), numerical methods are required.

Model Comparison
----------------
| Model            | Rate Distribution | Negative Rates | Bond Formula |
|------------------|-------------------|----------------|--------------|
| Hull-White       | Gaussian          | Yes            | Closed-form  |
| Black-Karasinski | Log-normal        | No             | Numerical    |
"""

from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    HullWhiteDynamics,
    HullWhiteSimulation,
)

from src.models.short_rate.black_karasinski import (
    BlackKarasinskiParameters,
    BlackKarasinskiDynamics,
    BlackKarasinskiSimulation,
)

__all__ = [
    # Hull-White
    "HullWhiteParameters",
    "HullWhiteDynamics",
    "HullWhiteSimulation",
    # Black-Karasinski
    "BlackKarasinskiParameters",
    "BlackKarasinskiDynamics",
    "BlackKarasinskiSimulation",
]
