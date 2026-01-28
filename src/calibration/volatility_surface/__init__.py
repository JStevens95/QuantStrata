"""
Volatility Surface Calibration.

This module provides calibration tools for volatility surfaces:
- **SABR Model**: Parametric smile fitting (Hagan et al., 2002)
- **Dupire Local Volatility**: Non-parametric local vol extraction

Both native Python and QuantLib backends are available:
- Native: Pure Python implementations (default)
- QuantLib: Industry-standard implementations for validation

Usage Example
-------------
>>> from src.calibration.volatility_surface import (
...     calibrate_sabr_to_smile,
...     calibrate_local_vol_from_implied,
...     SabrParameters,
... )
>>> # Native SABR calibration
>>> params = calibrate_sabr_to_smile(
...     forward=1.0,
...     strikes=np.array([0.9, 0.95, 1.0, 1.05, 1.1]),
...     market_vols=np.array([0.22, 0.21, 0.20, 0.19, 0.18]),
...     expiry=1.0,
... )

Author: QuantStrata Team
"""

# Native implementations (always available)
from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    sabr_implied_vol_vec,
    calibrate_sabr_to_smile,
    calibrate_sabr_term_structure,
    create_sabr_vol_surface,
)

from src.calibration.volatility_surface.dupire import (
    DupireConfig,
    DupireCalibrator,
    calibrate_local_vol_from_implied,
)

# QuantLib backends (optional - require QuantLib installation)
# Import these explicitly when needed:
# from src.calibration.volatility_surface.quantlib import (
#     sabr_implied_vol_quantlib,
#     calibrate_sabr_quantlib,
#     calibrate_local_vol_quantlib,
# )

__all__ = [
    # SABR (Native)
    "SabrParameters",
    "SabrConfig",
    "sabr_implied_vol",
    "sabr_implied_vol_vec",
    "calibrate_sabr_to_smile",
    "calibrate_sabr_term_structure",
    "create_sabr_vol_surface",
    # Dupire (Native)
    "DupireConfig",
    "DupireCalibrator",
    "calibrate_local_vol_from_implied",
]
