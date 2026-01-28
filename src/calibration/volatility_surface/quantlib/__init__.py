"""
QuantLib Backends for Volatility Surface Calibration.

This module provides QuantLib-backed implementations of:
- SABR calibration and implied volatility
- Dupire local volatility extraction

These serve as validation tools and production-grade alternatives to the
native Python implementations.

Author: QuantStrata Team
"""
from src.calibration.volatility_surface.quantlib.sabr_ql import (
    sabr_implied_vol_quantlib,
    calibrate_sabr_quantlib,
)
from src.calibration.volatility_surface.quantlib.dupire_ql import (
    calibrate_local_vol_quantlib,
    DupireQuantLibConfig,
)

__all__ = [
    "sabr_implied_vol_quantlib",
    "calibrate_sabr_quantlib",
    "calibrate_local_vol_quantlib",
    "DupireQuantLibConfig",
]
