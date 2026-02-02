"""
Stochastic Volatility Models.

This module provides implementations of stochastic volatility models
for pricing derivatives where volatility itself follows a stochastic process.

Models Implemented
------------------
- Heston: Square-root mean-reverting variance process (CIR for V_t)
- SABR: Stochastic Alpha Beta Rho model (CEV forward + log-normal vol)

Usage
-----
>>> from src.models.stochastic_volatility import (
...     HestonParameters, HestonDynamics,
...     SabrDynamics, SabrSimulation,
... )
"""

from src.models.stochastic_volatility.heston import (
    HestonParameters,
    HestonDynamics,
    HestonSimulation,
)

from src.models.stochastic_volatility.sabr import (
    SabrDynamics,
    SabrSimulation,
    sabr_mc_call,
    sabr_mc_put,
)

# Re-export calibration parameters from calibration module
from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    sabr_implied_vol_vec,
    calibrate_sabr_to_smile,
)

__all__ = [
    # Heston
    "HestonParameters",
    "HestonDynamics",
    "HestonSimulation",
    # SABR
    "SabrParameters",
    "SabrConfig",
    "SabrDynamics",
    "SabrSimulation",
    "sabr_implied_vol",
    "sabr_implied_vol_vec",
    "sabr_mc_call",
    "sabr_mc_put",
    "calibrate_sabr_to_smile",
]
