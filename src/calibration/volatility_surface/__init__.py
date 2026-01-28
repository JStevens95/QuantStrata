"""
Volatility Surface Calibration.

This module provides calibration tools for volatility surfaces:
- SABR model fitting
- Vanna-Volga method
- Generic smile interpolation
- Dupire's formula for local volatility calibration

Author: QuantStrata Team
"""

from src.calibration.volatility_surface.dupire import (
    DupireCalibrator,
    DupireConfig,
    calibrate_local_vol_from_implied,
)
from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    calibrate_sabr_to_smile,
    create_sabr_vol_surface,
)

__all__ = [
    "DupireCalibrator",
    "DupireConfig",
    "calibrate_local_vol_from_implied",
    "SabrParameters",
    "SabrConfig",
    "sabr_implied_vol",
    "calibrate_sabr_to_smile",
    "create_sabr_vol_surface",
]
