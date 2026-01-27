"""
Local Volatility Calibration Module.

This module provides tools for calibrating local volatility surfaces from
market-implied volatilities using Dupire's formula.
"""

from src.calibration.local_volatility.dupire import (
    DupireCalibrator,
    DupireConfig,
    calibrate_local_vol_from_implied,
)

__all__ = [
    "DupireCalibrator",
    "DupireConfig",
    "calibrate_local_vol_from_implied",
]
